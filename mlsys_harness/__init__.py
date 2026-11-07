"""Shared measurement infrastructure for the labs.

Labs used to carry their own copies of the RTX 5090 datasheet, their own CUDA timers
and their own JSONL writers. This package is the single place for those three, plus
the campaign runner that replaced the hand-rolled GPU shell script.

Importing this package does not import torch. CUDA timing lives in ``mlsys_harness.timing``
and imports torch only when a CUDA clock is requested.
"""

__version__ = "1"
