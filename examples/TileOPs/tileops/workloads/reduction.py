"""Workload definitions for the reduction op family."""

from tileops.workloads.workload_base import RandnWorkload


class LogSumExpWorkload(RandnWorkload):
    """Workload definition for LogSumExpFwdOp (spec interface: shape + dtype)."""


class ArgmaxWorkload(RandnWorkload):
    """Workload definition for ArgmaxFwdOp (spec interface: shape + dtype)."""
