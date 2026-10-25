"""Vendored TRACE scoring code (Apache-2.0).

Source: https://github.com/maveryn/trace, branch ``rlvr``, commit
af32fe0e668eb878e909daef7322ac1c4ae20f64.

| file | upstream path | sha256 (as copied) |
| --- | --- | --- |
| reward_scoring.py | src/trace_tasks/core/reward_scoring.py | 120146d6c4cfc8e8c7fc879472aa0d15c0f024a3ccebb9127042359fb2404f65 |
| reward_contracts.py | src/trace_tasks/core/reward_contracts.py | 970d130bbc54ac96d6b35d6fd8be582fdc19d276147cb15bff6fab605039e297 |
| answer_extraction.py | rlvr/evaluation/trace_validation/answer_extraction.py | 20239d57d5550ec23fd590f345a4413792e7b0870f7e5dc8e503f91705757363 |
| judge_contract.py | excerpt of rlvr/evaluation/trace_validation/judge_extract.py | (excerpt) |

Vendored instead of imported because ``import trace_tasks`` loads the whole
generator (cairosvg / libcairo). ``reward_scoring.py`` is the scorer that the
TRACE GRPO reward (``rlvr/easyr1_backend/examples/reward_function/trace_rlvr.py``)
calls; ``answer_extraction.py`` is the deterministic extractor of the official
validation protocol. Do not edit these files; re-copy them from upstream.
"""
