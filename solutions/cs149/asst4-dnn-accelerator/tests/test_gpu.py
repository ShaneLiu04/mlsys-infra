"""GPU kernels vs torch. Skipped unless CUDA and nvcc are available (the local CPU run stays green)."""

import math
import shutil
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

GPU_DIR = Path(__file__).resolve().parents[1] / "gpu"
sys.path.insert(0, str(GPU_DIR))
import kernels as K  # noqa: E402

if shutil.which(K.NVCC) is None and not Path(K.NVCC).exists():
    pytest.skip("nvcc not available", allow_module_level=True)
K.build()
F = torch.nn.functional


def _err(c, ref):
    return float((c.double() - ref).abs().max() / ref.abs().max())


@pytest.mark.parametrize("variant", range(11))
@pytest.mark.parametrize("shape", [(256, 256, 256), (384, 512, 136), (200, 136, 72)])
def test_sgemm_variants(variant, shape):
    m, n, k = shape
    if variant in (7, 8, 9) and (m % 128 or n % 128 or k % 8):
        pytest.skip("vec4 kernels need multiples of 128/8")
    a = torch.randn(m, k, device="cuda"); b = torch.randn(k, n, device="cuda"); c = torch.zeros(m, n, device="cuda")
    K.sgemm(variant, a, b, c)
    assert _err(c, a.double() @ b.double()) < 1e-5


@pytest.mark.parametrize("bm,bn", [(32, 32), (64, 256), (256, 128)])
def test_reg2d_tiles(bm, bn):
    a = torch.randn(300, 200, device="cuda"); b = torch.randn(200, 260, device="cuda"); c = torch.zeros(300, 260, device="cuda")
    K.sgemm_tile(bm, bn, a, b, c)
    assert _err(c, a.double() @ b.double()) < 1e-5


@pytest.mark.parametrize("m", [1, 3, 8, 13])
def test_skinny(m):
    a = torch.randn(m, 1000, device="cuda"); b = torch.randn(1000, 777, device="cuda"); c = torch.zeros(m, 777, device="cuda")
    K.sgemm_skinny(a, b, c, 4)
    assert _err(c, a.double() @ b.double()) < 1e-5


@pytest.mark.parametrize("dt", [torch.float16, torch.bfloat16])
def test_wmma(dt):
    a = torch.randn(256, 384, device="cuda").to(dt); b = torch.randn(384, 512, device="cuda").to(dt)
    c = torch.zeros(256, 512, device="cuda")
    K.hgemm_wmma(a, b, c)
    assert _err(c, a.double() @ b.double()) < 1e-5


@pytest.mark.parametrize("c_,h,f,k,s,p", [(3, 32, 16, 7, 2, 3), (16, 14, 24, 3, 1, 1), (32, 9, 40, 1, 1, 0),
                                          (8, 15, 8, 3, 2, 1)])
def test_conv_variants(c_, h, f, k, s, p):
    x = torch.randn(2, c_, h, h, device="cuda"); w = torch.randn(f, c_, k, k, device="cuda") / math.sqrt(c_ * k * k)
    ref = F.conv2d(x.double(), w.double(), stride=s, padding=p)
    y = torch.empty_like(ref, dtype=torch.float32)
    K.conv_direct(x, w, y, s, p)
    assert _err(y, ref) < 1e-5
    K.conv_igemm(x, w, y, s, p)
    assert _err(y, ref) < 1e-5
    oh = K.out_hw(h, k, s, p)
    cols = torch.empty(2, c_ * k * k, oh * oh, device="cuda")
    K.conv_im2col(x, cols, k, k, s, p)
    assert _err(torch.matmul(w.reshape(f, -1), cols).view_as(y), ref) < 1e-5


@pytest.mark.parametrize("c_,h,f", [(8, 16, 12), (16, 14, 8)])
def test_fused_conv_relu_pool(c_, h, f):
    x = torch.randn(2, c_, h, h, device="cuda"); w = torch.randn(f, c_, 3, 3, device="cuda")
    ref = F.max_pool2d(F.relu(F.conv2d(x.double(), w.double(), padding=1)), 2)
    out = torch.empty(2, f, h // 2, h // 2, device="cuda")
    K.conv2x2(True, x, w, out, 1, 1)
    assert _err(out, ref) < 1e-5
    y = torch.empty(2, f, h, h, device="cuda"); y2 = torch.empty_like(y)
    K.conv2x2(False, x, w, y, 1, 1); K.relu(y, y2); K.maxpool2x2(y2, out)
    assert _err(out, ref) < 1e-5
    K.maxpool2x2(y, out, with_relu=True)
    assert _err(out, ref) < 1e-5
