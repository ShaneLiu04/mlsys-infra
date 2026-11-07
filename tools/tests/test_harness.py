"""The shared harness: devices, timing, provenance, campaign checks, cross-card classification."""

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mlsys_harness.campaign import CampaignError, ExclusiveLock, check_spec, run, verify
from mlsys_harness.compare import collect, predicted_class
from mlsys_harness.devices import UnknownDevice, get, match_device
from mlsys_harness.paths import raw_dir, read_raw
from mlsys_harness.timing import percentile, summarize, time_calls
from mlsys_harness.writer import JsonlWriter

os.environ.setdefault("PYTHONHASHSEED", "0")


def test_device_registry_matches_the_old_5090_literals():
    d = get("rtx5090").legacy_dict()
    assert d["sms"] == 170
    assert d["hbm_bw"] == 1.792e12
    assert d["saxpy_bw"] == 1.55e12
    assert d["bf16_flops"] == 209.5e12
    assert d["fp8_flops"] == 419e12
    assert d["int8_ops"] == 838e12
    assert d["l2_bytes"] == 96 * 2**20
    assert d["smem_per_sm"] == 128 * 1024


def test_device_match_distinguishes_the_two_measured_cards():
    assert match_device("NVIDIA GeForce RTX 5090").slug == "rtx5090"
    assert match_device("NVIDIA GeForce RTX 4080 SUPER").slug == "rtx4080s"
    assert get("rtx4080s").spec_bf16_flops is None
    assert get("rtx4080s").spec_hbm_bw == 736.3e9
    assert get("rtx4080s").smem_per_sm == 101376
    with pytest.raises(UnknownDevice):
        match_device("RTX 4060")


def test_rtx6000d_spec_is_derived_from_bus_width_and_clocks():
    d = match_device("NVIDIA RTX 6000D")
    assert d.slug == "rtx6000d"
    assert d.spec_hbm_bw == pytest.approx(1.398e12, rel=1e-3)
    assert d.spec_bf16_flops is None
    with pytest.raises(ValueError):
        d.legacy_dict()


def test_read_raw_finds_the_existing_trees():
    lab = ROOT / "solutions/cs229s/project2-kv-speculative"
    assert read_raw(lab, "rtx5090").name == "gpu"
    assert read_raw(lab, "rtx4080s").name == "ada4080"


def test_raw_dir_honors_override_and_legacy_env(monkeypatch, tmp_path):
    lab = tmp_path / "lab"
    assert raw_dir(lab) == lab / "artifacts" / "raw" / "gpu"
    monkeypatch.setenv("MLSYS_RAW_DIR", str(tmp_path / "fresh"))
    assert raw_dir(lab) == tmp_path / "fresh"
    monkeypatch.setenv("FP8_OUT", str(tmp_path / "fp8"))
    assert raw_dir(lab, "FP8_OUT") == tmp_path / "fp8"


def test_percentiles_and_host_timer():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([1, 2, 3, 4, 5], 0.1) == pytest.approx(1.4)
    assert percentile([1, 2, 3, 4, 5], 0.9) == pytest.approx(4.6)
    result = summarize([1.0, 3.0, 2.0], "batched")
    assert result.p50_s == 2.0 and result.mode == "batched"
    calls = []
    out = time_calls(lambda: calls.append(1), warmup=2, repeats=4)
    assert out.samples == 4 and len(calls) == 6
    with pytest.raises(ValueError):
        summarize([1.0], "nope")


def test_writer_puts_the_commit_on_the_first_line(tmp_path):
    path = tmp_path / "raw" / "x.jsonl"
    with JsonlWriter(path, cwd=ROOT) as w:
        w.write({"part": "point", "value": 1})
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["kind"] == "run"
    assert len(rows[0]["git_commit"]) == 40
    assert isinstance(rows[0]["git_dirty"], bool)
    assert rows[1] == {"part": "point", "value": 1}


def test_check_rejects_the_fp8_false_success(tmp_path):
    path = tmp_path / "fp8.jsonl"
    path.write_text('{"part": "correctness", "max_abs_err": 234}\n', encoding="utf-8")
    spec = {"id": "err", "file": "fp8.jsonl", "where": {"part": "correctness"},
            "field": "max_abs_err", "max": 1.0}
    with pytest.raises(CampaignError):
        check_spec(tmp_path, spec)
    path.write_text('{"part": "correctness", "max_abs_err": 1.0}\n', encoding="utf-8")
    check_spec(tmp_path, spec)


def test_run_stops_when_the_process_exits_zero_but_the_check_fails(tmp_path):
    script = "open('out.jsonl','w').write('{\"part\": \"correctness\", \"max_abs_err\": 234}\\n')"
    campaign = {
        "name": "t", "lock": None,
        "steps": [{
            "id": "bad", "cwd": ".", "cmd": [sys.executable, "-c", script],
            "checks": [{"file": "out.jsonl", "where": {"part": "correctness"},
                        "field": "max_abs_err", "max": 1.0}],
        }],
    }
    with pytest.raises(CampaignError, match="max"):
        run(campaign, tmp_path)


def test_run_exports_the_lock_flag_and_excludes_a_second_taker(tmp_path):
    flag = tmp_path / "flag.txt"
    campaign = {
        "name": "t", "lock": str(tmp_path / "gpu.lock"),
        "steps": [{
            "id": "env", "cwd": ".",
            "cmd": [sys.executable, "-c",
                    f"open(r'{flag}', 'w').write(__import__('os').environ.get('MLSYS_LOCK_HELD',''))"],
            "checks": [],
        }],
    }
    run(campaign, tmp_path)
    assert flag.read_text(encoding="utf-8") == "1"

    order = []
    lock = tmp_path / "held.lock"

    def hold():
        with ExclusiveLock(lock):
            order.append("in")
            time.sleep(0.3)
            order.append("out")

    thread = threading.Thread(target=hold)
    thread.start()
    time.sleep(0.05)
    t0 = time.perf_counter()
    with ExclusiveLock(lock):
        order.append("second")
    waited = time.perf_counter() - t0
    thread.join()
    assert order == ["in", "out", "second"]
    assert waited >= 0.2


def test_recorded_campaign_and_published_claims():
    campaign = json.loads((ROOT / "campaigns" / "recorded.json").read_text(encoding="utf-8"))
    assert verify(campaign, ROOT) == ["copy", "attention", "fp8", "spec"]
    claims = json.loads((ROOT / "claims.json").read_text(encoding="utf-8"))["claims"]
    for spec in claims:
        check_spec(ROOT, spec)


def test_cross_card_classification_matches_the_reference_each_metric_claims():
    metrics, refs = collect(ROOT)
    assert refs["bandwidth"] == pytest.approx(0.41, abs=0.02)
    assert refs["compute"] == pytest.approx(0.47, abs=0.03)
    for metric in metrics:
        assert predicted_class(metric.ratio, refs) == metric.expected, metric.id
    by_id = {m.id: m for m in metrics}
    assert by_id["spec-g4"].ratio == pytest.approx(1.0, abs=0.01)
    assert by_id["copy"].ratio == pytest.approx(refs["bandwidth"])
