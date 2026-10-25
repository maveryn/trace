"""EasyR1 reward entry for TRACE certificates (REWARD_NAME "trace_certificate_v1", batch).

EasyR1 loads this file by path (``worker.reward.reward_function =
<checkout>/experiments/certificates/reward_entry.py:compute_score``) without package
context; ``trace_tasks`` must be importable from ``PYTHONPATH`` (this checkout's ``src``).
The logic lives in ``trace_tasks.experimental.certificates.reward`` (DESIGN 5).
"""

from trace_tasks.experimental.certificates.reward import REWARD_NAME, REWARD_TYPE, compute_score

__all__ = ["REWARD_NAME", "REWARD_TYPE", "compute_score"]
