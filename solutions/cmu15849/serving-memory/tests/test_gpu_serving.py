"""GPU-part tests: Triton paged attention, Qwen2 forward pieces, engine. Skip without CUDA."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_build_paged_block_tables_are_a_partition_cpu():
    from paged_attn import build_paged
    lens = [5, 17, 33, 64]
    kp, vp, bt, ln = build_paged(lens, 16, 2, 8, dtype=torch.float32, device="cpu")
    used = [int(b) for i, n in enumerate(lens) for b in bt[i, : -(-n // 16)]]
    assert sorted(used) == list(range(kp.shape[0]))


def test_gather_decode_matches_reference_cpu():
    from paged_attn import build_paged, gather_decode, reference_decode
    kp, vp, bt, ln = build_paged([7, 40, 3], 8, 2, 16, dtype=torch.float32, device="cpu")
    q = torch.randn(3, 4, 16)
    assert torch.allclose(gather_decode(q, kp, vp, bt, ln), reference_decode(q, kp, vp, bt, ln), atol=1e-5)


@cuda
@pytest.mark.parametrize("bs", [8, 16, 32, 64])
def test_triton_paged_decode_matches_reference(bs):
    from paged_attn import build_paged, paged_decode, reference_decode
    lens = [1, 15, 16, 17, 100, 1000, 4096]
    kp, vp, bt, ln = build_paged(lens, bs, 2, 64, seed=bs)
    q = torch.randn(len(lens), 14, 64, device="cuda", dtype=torch.bfloat16)
    err = (paged_decode(q, kp, vp, bt, ln).float() - reference_decode(q, kp, vp, bt, ln)).abs().max()
    assert err < 2e-2


def _tiny_qwen(tmp_path):
    """Random tiny Qwen2 checkpoint in HF layout."""
    import json
    from safetensors.torch import save_file
    L, d, H, Hk, D, ff, V = 2, 64, 4, 2, 16, 128, 300
    sd = {"model.embed_tokens.weight": torch.randn(V, d) * 0.1, "model.norm.weight": torch.ones(d)}
    for i in range(L):
        p = f"model.layers.{i}."
        sd |= {p + "input_layernorm.weight": torch.ones(d), p + "post_attention_layernorm.weight": torch.ones(d),
               p + "self_attn.q_proj.weight": torch.randn(H * D, d) * 0.1, p + "self_attn.q_proj.bias": torch.zeros(H * D),
               p + "self_attn.k_proj.weight": torch.randn(Hk * D, d) * 0.1, p + "self_attn.k_proj.bias": torch.zeros(Hk * D),
               p + "self_attn.v_proj.weight": torch.randn(Hk * D, d) * 0.1, p + "self_attn.v_proj.bias": torch.zeros(Hk * D),
               p + "self_attn.o_proj.weight": torch.randn(d, H * D) * 0.1,
               p + "mlp.gate_proj.weight": torch.randn(ff, d) * 0.1, p + "mlp.up_proj.weight": torch.randn(ff, d) * 0.1,
               p + "mlp.down_proj.weight": torch.randn(d, ff) * 0.1}
    save_file(sd, str(tmp_path / "model.safetensors"))
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": L, "hidden_size": d, "num_attention_heads": H, "num_key_value_heads": Hk,
        "intermediate_size": ff, "rms_norm_eps": 1e-6, "vocab_size": V, "rope_theta": 10000.0}))
    return tmp_path


def test_engine_policies_produce_identical_greedy_tokens(tmp_path, monkeypatch):
    """Batching policy must not change the generated tokens (same model, greedy).
    On CPU the Triton decode kernel is replaced by the torch gather reference."""
    pytest.importorskip("safetensors")
    import engine
    from engine import Engine, Req
    from qwen import Qwen2
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        from paged_attn import gather_decode
        monkeypatch.setattr(engine, "paged_decode", gather_decode)
    m = Qwen2(_tiny_qwen(tmp_path), device=dev, dtype=torch.float32, max_pos=1024)
    out = {}
    for pol in ("continuous", "chunked", "static"):
        reqs = [Req(i, 0.01 * i, p, o) for i, (p, o) in enumerate([(5, 4), (40, 6), (130, 5), (9, 8)])]
        eng = Engine(m, kv_bytes=m.kv_bytes_per_token() * 16 * 64, policy=pol, chunk_tokens=32)
        done, steps, _ = eng.run(reqs, vocab_hi=300)
        out[pol] = {r.rid: r.tokens[-r.output:] for r in done}
        assert all(r.generated == r.output for r in done)
        assert len(eng.free) == eng.num_blocks
    for rid in out["continuous"]:
        assert out["continuous"][rid] == out["chunked"][rid]
    # static left-pads to the longest prompt without masking (cost model only); request 0 runs alone
    assert out["static"][0] == out["continuous"][0]
