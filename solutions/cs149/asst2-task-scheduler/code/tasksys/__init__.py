"""Python analogue of CS149 Assignment 2 (task systems and task-graph scheduling)."""

from .systems import (
    IRunnable,
    ITaskSystem,
    TaskSystemParallelSpawn,
    TaskSystemParallelThreadPoolSleeping,
    TaskSystemParallelThreadPoolSpinning,
    TaskSystemSerial,
    make_system,
)

__all__ = [
    "IRunnable",
    "ITaskSystem",
    "TaskSystemSerial",
    "TaskSystemParallelSpawn",
    "TaskSystemParallelThreadPoolSpinning",
    "TaskSystemParallelThreadPoolSleeping",
    "make_system",
]
