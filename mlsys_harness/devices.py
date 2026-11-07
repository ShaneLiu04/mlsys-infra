"""Datasheet numbers for the GPUs this repo has actually measured.

Measured peaks stay in the jsonl files. This module only holds the published
specs those measurements are compared against, and the rounded reference
bandwidth the existing 5090 plots already draw. Nothing here is read back out
of a result file, so a new run cannot silently change the spec line.
"""

from __future__ import annotations

from dataclasses import dataclass


class UnknownDevice(KeyError):
    """Raised when a torch device name matches no registered card."""


@dataclass(frozen=True)
class Device:
    slug: str
    match: tuple[str, ...]
    sms: int
    spec_bf16_flops: float | None
    spec_fp8_flops: float | None
    spec_fp32_flops: float | None
    spec_int8_ops: float | None
    spec_hbm_bw: float
    reference_copy_bw: float | None
    l2_bytes: int
    smem_per_sm: int
    note: str = ""

    def legacy_dict(self) -> dict:
        """The RTX 5090 dict the compiler lab already indexes by these keys."""
        if self.spec_bf16_flops is None or self.reference_copy_bw is None:
            raise ValueError(f"{self.slug} has no legacy datasheet dict")
        return {
            "name": "RTX 5090" if self.slug == "rtx5090" else self.slug,
            "sms": self.sms,
            "hbm_bw": self.spec_hbm_bw,
            "saxpy_bw": self.reference_copy_bw,
            "bf16_flops": self.spec_bf16_flops,
            "int8_ops": self.spec_int8_ops,
            "fp8_flops": self.spec_fp8_flops,
            "l2_bytes": self.l2_bytes,
            "smem_per_sm": self.smem_per_sm,
        }


# 209.5e12 is the GeForce dense-bf16 rate with fp32 accumulate (half the fp16-accumulate
# datasheet rate). 1.55e12 is the rounded streaming bandwidth the 5090 plots use;
# the jsonl copy measurement is 1.522 TB/s and is not this constant.
_DEVICES = (
    Device(
        slug="rtx5090",
        match=("RTX 5090",),
        sms=170,
        spec_bf16_flops=209.5e12,
        spec_fp8_flops=419e12,
        spec_fp32_flops=104.8e12,
        spec_int8_ops=838e12,
        spec_hbm_bw=1.792e12,
        reference_copy_bw=1.55e12,
        l2_bytes=96 * 2**20,
        smem_per_sm=128 * 1024,
        note="GDDR7 512-bit. GeForce tensor rates are the fp32-accumulate ones.",
    ),
    Device(
        slug="rtx4080s",
        match=("4080 SUPER",),
        sms=80,
        spec_bf16_flops=None,
        spec_fp8_flops=None,
        spec_fp32_flops=52.22e12,
        spec_int8_ops=None,
        spec_hbm_bw=736.3e9,
        reference_copy_bw=None,
        l2_bytes=64 * 2**20,
        smem_per_sm=101376,
        note="Published 4080 SUPER is 16 GB at 736.3 GB/s. Ada shared memory "
             "tops out at 101376 bytes; a 256x256 tile does not compile. "
             "Dense bf16/fp8 tensor rates are not published for this GeForce "
             "part, so efficiency uses the measured GEMM peak. The unit in "
             "this repo reported 32 GB, which the datasheet does not.",
    ),
    Device(
        slug="rtx6000d",
        match=("RTX 6000D",),
        sms=156,
        spec_bf16_flops=None,
        spec_fp8_flops=None,
        spec_fp32_flops=156 * 128 * 2 * 2.43e9,
        spec_int8_ops=None,
        spec_hbm_bw=2 * 12481e6 * 448 / 8,
        reference_copy_bw=None,
        l2_bytes=112 * 2**20,
        smem_per_sm=101376,
        note="Blackwell sm_120, 84 GB GDDR7. No public datasheet for this SKU: bandwidth is "
             "2 x 12481 MHz x 448 bit = 1.398 TB/s and fp32 is 156 SM x 128 lanes x 2 x 2.43 GHz, "
             "both from cudaDeviceGetAttribute / nvidia-smi max clocks. Tensor rates are not "
             "published, so efficiency uses the measured GEMM peak. Shared memory per SM is the "
             "Ada-sized 101376 bytes, not the 5090's 128 KB.",
    ),
)

_BY_SLUG = {d.slug: d for d in _DEVICES}


def get(slug: str) -> Device:
    try:
        return _BY_SLUG[slug]
    except KeyError:
        raise UnknownDevice(slug) from None


def match_device(gpu_name: str) -> Device:
    """Match a torch device name. Longer patterns win, so 4080 SUPER is not a 4080."""
    hits = [d for d in _DEVICES if any(token in gpu_name for token in d.match)]
    if len(hits) != 1:
        raise UnknownDevice(gpu_name)
    return hits[0]


def all_devices() -> tuple[Device, ...]:
    return _DEVICES
