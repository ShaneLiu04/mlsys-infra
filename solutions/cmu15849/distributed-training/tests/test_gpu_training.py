"""GPU-part tests. CPU-only checks always run; CUDA checks skip without a GPU."""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))

from gpt import ACT_BYTES_PER_SBH, CONFIGS, GPT, Stage, activation_bytes_model, flops_per_token  # noqa: E402

import mlsys_calc  # noqa: E402
from disttrain.pipeline import bubble_fraction, schedule_orders, simulate  # noqa: E402

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")


def test_param_count_matches_formula_and_gpt2():
    for name in ("tiny", "gpt2-124m"):
        if name == "gpt2-124m":
            assert CONFIGS[name].params() == 124_439_808
        else:
            assert GPT(CONFIGS[name]).num_params() == CONFIGS[name].params()


def test_flops_and_activation_formulas():
    c = CONFIGS["gpt2-124m"]
    assert flops_per_token(c, 1024) == pytest.approx(6 * c.params() + 12 * 12 * 768 * 1024)
    assert activation_bytes_model(c, 2, 512, "fp32") == 2 * activation_bytes_model(c, 2, 512, "bf16")
    assert ACT_BYTES_PER_SBH["amp_bf16"] == 36


@pytest.mark.parametrize("mode", ["full", "mlp", "attn"])
def test_checkpoint_modes_bitwise_equal_grads_cpu(mode):
    torch.manual_seed(0)
    m = GPT(CONFIGS["tiny"]).double()
    x = torch.randint(0, 512, (2, 32))
    y = torch.roll(x, -1, 1)
    m(x, y).backward()
    ref = [p.grad.clone() for p in m.parameters()]
    m.zero_grad()
    m.set_ckpt(mode)
    m(x, y).backward()
    for a, b in zip(ref, (p.grad for p in m.parameters())):
        assert torch.equal(a, b)


def test_stages_compose_to_full_model_cpu():
    torch.manual_seed(0)
    m = GPT(CONFIGS["tiny"]).double()
    x = torch.randint(0, 512, (2, 16))
    y = torch.roll(x, -1, 1)
    h = x
    stages = [Stage(m, d, 4) for d in range(4)]
    for d, s in enumerate(stages):
        h = s(h, y if d == 3 else None)
    assert torch.equal(h, m(x, y))
    assert sum(len(s.blocks) for s in stages) == CONFIGS["tiny"].n_layer


def test_replay_with_unit_costs_matches_lab_simulator():
    from train_bench import replay
    for sched in ("gpipe", "1f1b"):
        for m in (1, 4, 8):
            orders = schedule_orders(sched, 4, m)
            dur = {(k, d, i): (1.0 if k == "F" else 2.0) for d in range(4) for (k, i, _) in orders[d]}
            ops = replay(orders, dur)
            mk = max(o["end"] for o in ops)
            busy = sum(o["end"] - o["start"] for o in ops)
            assert 1 - busy / (4 * mk) == pytest.approx(bubble_fraction(simulate(sched, 4, m), 4))
            assert 1 - busy / (4 * mk) == pytest.approx(mlsys_calc.pipeline_bubble(4, m))


@cuda
def test_cuda_amp_step_and_memory_components():
    from train_bench import build, make_opt, train_step
    for mode in ("fp32", "amp_bf16", "bf16", "bf16_master"):
        model = build("tiny", mode)
        opt, pairs = make_opt(model, mode)
        x = torch.randint(0, 512, (2, 64), device="cuda")
        rec = {}
        loss = train_step(model, opt, pairs, mode, x, torch.roll(x, -1, 1), rec)
        assert torch.isfinite(loss)
        assert rec["act"] > 0 and rec["grads"] > 0


@cuda
def test_cuda_pipeline_runs_and_gradients_match():
    from disttrain.pipeline import schedule_orders as so
    from train_bench import run_pipeline
    torch.manual_seed(0)
    m = GPT(CONFIGS["tiny"]).cuda()
    stages = [Stage(m, d, 4) for d in range(4)]
    xs = [torch.randint(0, 512, (2, 32), device="cuda") for _ in range(4)]
    ys = [torch.roll(x, -1, 1) for x in xs]
    grads = {}
    for sched in ("gpipe", "1f1b"):
        m.zero_grad(set_to_none=True)
        run_pipeline(stages, so(sched, 4, 4), xs, ys, record_mem=False)
        grads[sched] = [p.grad.clone() for p in m.parameters()]
    for a, b in zip(grads["gpipe"], grads["1f1b"]):
        assert torch.allclose(a, b, atol=1e-5, rtol=1e-4)
