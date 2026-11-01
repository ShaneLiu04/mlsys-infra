import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gpu"))
from bankmodel import access, kernel_patterns, wavefronts  # noqa: E402


def test_basic_patterns():
    assert access(lambda t: t) == 1                 # consecutive words
    assert access(lambda t: 0) == 1                 # broadcast
    assert access(lambda t: 2 * t) == 2             # stride 2 -> 2-way
    assert access(lambda t: 32 * t) == 32           # same bank, all distinct
    assert access(lambda t: 33 * t) == 1            # padded stride, conflict-free
    assert access(lambda t: 4 * t, 16) == 4         # contiguous float4: 4 phases, no conflict


def test_wavefront_phases():
    assert wavefronts([[0]] * 32) == 1
    assert wavefronts([[i * 32] for i in range(8)], 16) == 8


def test_reg2d_conflicts_explain_slowdown():
    p = kernel_patterns()
    assert p["reg2d"]["B load Bs[k*BN+tcol*TN+j]"][0] == 4
    assert p["reg2d"]["A store As[c*BM+r] (transpose)"][0] == 8
    assert all(w == i for w, i in p["smem32"].values())
