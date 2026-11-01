"""Launch each key kernel exactly once so `ncu` can profile it (see gpu/ncu.sh)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import kernels as K  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = False
S = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
a = torch.randn(S, S, device="cuda"); b = torch.randn(S, S, device="cuda"); c = torch.empty(S, S, device="cuda")
for v in (0, 1, 2, 4, 5, 6, 7, 8):
    K.sgemm(v, a, b, c)
ah, bh = a.half(), b.half()
K.hgemm_wmma(ah, bh, c)
_ = a @ b
_ = ah @ bh
n = 64 * 2**20
x = torch.rand(n, device="cuda"); y = torch.empty_like(x); z = torch.rand(n, device="cuda")
K.bw_triad(y, x, z, 170 * 16)
# fusion layer 3x3 @56 C128, batch 8
xi = torch.randn(8, 128, 56, 56, device="cuda"); w = torch.randn(128, 128, 3, 3, device="cuda") / math.sqrt(1152)
yc = torch.empty(8, 128, 56, 56, device="cuda"); y2 = torch.empty_like(yc); out = torch.empty(8, 128, 28, 28, device="cuda")
K.conv2x2(False, xi, w, yc, 1, 1); K.relu(yc, y2); K.maxpool2x2(y2, out)
K.conv2x2(True, xi, w, out, 1, 1)
K.conv_igemm(xi, w, yc, 1, 1, relu=True)
_ = F.conv2d(xi, w, padding=1)
torch.cuda.synchronize()
print("ok")
