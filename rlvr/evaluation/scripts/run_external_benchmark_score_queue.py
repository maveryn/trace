#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import re
import shutil
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pandas as pd
import requests
from tqdm import tqdm

from benchmark_queue_lib import (
    BENCHMARK_RUN_SETS,
    DEFAULT_BENCHMARK_ROOT,
    DEFAULT_QUEUE_ROOT,
    REPO_ROOT,
    VLMEVAL_ROOT,
    BenchmarkSpec,
    benchmark_specs_for_run_set,
    build_vlmeval_dataset,
    claim_next_job,
    filter_benchmark_specs,
    json_default,
    local_judge_eval_mode,
    mark_job,
    run_dir,
    score_path,
    spec_by_key,
    write_json,
)
from trace_benchmark_answer_parsing import extract_final_answer  # noqa: E402
from trace_eval_scoring_contract import (  # noqa: E402
    DEDICATED_SCORE_KEYS,
    DIRECT_SCORE_KEYS,
    LLM_EXTRACT_SCORE_KEYS,
)


def _import_vlmeval_runner():
    scripts_root = VLMEVAL_ROOT / "scripts"
    for path in (VLMEVAL_ROOT, scripts_root):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import batched_vlmevalkit_qwen3vl as runner

    return runner, None


def _judge_cache_entry_needs_retry(
    row: dict[str, Any],
    *,
    retry_on_length: bool = True,
) -> bool:
    output = str(row.get("judge_output", "")).strip()
    finish_reason = str(row.get("judge_finish_reason", "")).strip().lower()
    return not output or (retry_on_length and finish_reason in {"length", "max_tokens"})


def _judge_retry_token_limits(initial_max_tokens: int) -> list[int]:
    limit = max(128, int(initial_max_tokens))
    ceiling = max(1024, limit)
    limits = [limit]
    while limits[-1] < ceiling:
        limits.append(min(ceiling, limits[-1] * 2))
    return limits


PERSISTENT_JUDGE_CACHE_CONTRACT_VERSION = "trace-persistent-judge-v2"
DIRECT_JUDGE_CACHE_CONTRACTS = {
    "mathvision_extract": "trace-eval-v1-mathvision-extract-v2",
    "mathvista_extract": "trace-eval-v1-mathvista-extract-v2",
    "mathverse_extract": "trace-eval-v1-mathverse-extract-v2",
    "mathverse_score": "trace-eval-v1-mathverse-vlmevalkit-judgement01-v4",
    "logicvista_extract": "trace-eval-v1-logicvista-option-extract-v1",
    "charxiv_judge": "trace-eval-v1-charxiv-judge-v1",
    "evochart_judge": "trace-eval-v1-evochart-judge-v1",
}


def _nonempty_judge_output(value: Any) -> bool:
    return bool(str(value or "").strip())


class _JudgeAPIRequestError(RuntimeError):
    pass


def _make_judge_api_batches(
    pending: list[tuple[str, str]],
    rendered: dict[str, str],
    *,
    batch_size: int,
    max_batch_chars: int,
) -> list[list[tuple[str, str]]]:
    batches: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []
    current_chars = 0
    for item in pending:
        prompt_chars = len(rendered[str(item[0])])
        if current and (
            len(current) >= batch_size or current_chars + prompt_chars > max_batch_chars
        ):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += prompt_chars
        if prompt_chars >= max_batch_chars:
            batches.append(current)
            current = []
            current_chars = 0
    if current:
        batches.append(current)
    return batches


class PersistentJudge:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.llm = None
        self.tokenizer = None
        self.api_tokenizer = None
        self._api_thread_local = threading.local()
        self._api_sessions: list[requests.Session] = []
        self._api_sessions_lock = threading.Lock()

    @property
    def api_bases(self) -> list[str]:
        return [
            str(item).rstrip("/")
            for item in (getattr(self.args, "judge_api_bases", None) or [])
            if str(item).strip()
        ]

    def _using_api_pool(self) -> bool:
        return bool(self.api_bases)

    def _arg(self, name: str, default: Any) -> Any:
        return getattr(self.args, name, default)

    def _api_session(self) -> requests.Session:
        session = getattr(self._api_thread_local, "session", None)
        if session is None:
            session = requests.Session()
            self._api_thread_local.session = session
            with self._api_sessions_lock:
                self._api_sessions.append(session)
        return session

    def _request_metadata(
        self,
        prompt: str,
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
        contract_version: str,
        backend: str,
        system_prompt: str | None = None,
    ) -> dict[str, str]:
        prompt_identity = prompt
        if system_prompt is not None:
            prompt_identity = json.dumps(
                {"system": system_prompt, "user": prompt},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
        prompt_hash = hashlib.sha256(prompt_identity.encode("utf-8")).hexdigest()
        request_contract = {
            "contract_version": contract_version,
            "prompt_sha256": prompt_hash,
            "backend": backend,
            "judge_model": str(self._arg("judge_model", "Qwen/Qwen3-32B")),
            "judge_api_model": (
                str(self._arg("judge_api_model", "qwen3-32b-judge"))
                if backend == "api"
                else None
            ),
            "judge_api_tokenizer_model": (
                str(self._arg("judge_api_tokenizer_model", "Qwen/Qwen3-32B"))
                if backend == "api"
                else None
            ),
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "chat_template": "user/add_generation_prompt/enable_thinking_false_when_supported",
            "system_prompt_sha256": (
                hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()
                if system_prompt is not None
                else None
            ),
        }
        encoded = json.dumps(
            request_contract, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        return {
            "request_hash": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "contract_version": contract_version,
            "judge_prompt_hash": prompt_hash,
        }

    @staticmethod
    def _cache_entry_needs_retry(
        row: dict[str, Any],
        expected: dict[str, str],
        output_validator: Callable[[str], bool] | None,
        output_validator_by_index: Callable[[str, str], bool] | None = None,
        *,
        retry_on_length: bool = True,
    ) -> bool:
        if any(str(row.get(key, "")) != str(value) for key, value in expected.items()):
            return True
        if _judge_cache_entry_needs_retry(row, retry_on_length=retry_on_length):
            return True
        if output_validator_by_index is not None:
            try:
                return not bool(
                    output_validator_by_index(
                        str(row.get("index", "")),
                        str(row.get("judge_output", "")),
                    )
                )
            except Exception:
                return True
        if output_validator is not None:
            try:
                return not bool(output_validator(str(row.get("judge_output", ""))))
            except Exception:
                return True
        return False

    def _ensure_loaded(self) -> None:
        if self.llm is not None:
            return
        if self.args.gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", self.args.attention_backend)
        os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

        from transformers import AutoTokenizer
        from vllm import LLM

        print(
            "[judge:init] "
            f"model={self.args.judge_model} gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
            f"batch_size={self.args.judge_batch_size}"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.judge_model, trust_remote_code=True
        )
        kwargs: dict[str, Any] = {
            "model": self.args.judge_model,
            "tensor_parallel_size": self.args.judge_tensor_parallel_size,
            "gpu_memory_utilization": self.args.judge_gpu_memory_utilization,
            "max_num_seqs": self.args.judge_max_num_seqs,
            "max_num_batched_tokens": self.args.judge_max_num_batched_tokens,
            "trust_remote_code": True,
            "seed": 0,
        }
        if self.args.judge_max_model_len is not None:
            kwargs["max_model_len"] = self.args.judge_max_model_len
        self.llm = LLM(**kwargs)

    def chat_prompt(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self._ensure_loaded()
        assert self.tokenizer is not None
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    def _ensure_api_tokenizer(self) -> None:
        if self.api_tokenizer is not None:
            return
        from transformers import AutoTokenizer

        tokenizer_model = self._arg("judge_api_tokenizer_model", None) or self._arg(
            "judge_model", "Qwen/Qwen3-32B"
        )
        self.api_tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model, trust_remote_code=True
        )

    def api_chat_prompt(self, prompt: str, *, system_prompt: str | None = None) -> str:
        self._ensure_api_tokenizer()
        assert self.api_tokenizer is not None
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        try:
            return self.api_tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return self.api_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

    @staticmethod
    def _completion_url(base: str) -> str:
        base = base.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/completions"
        if base.endswith("/v1/completions"):
            return base
        return f"{base}/v1/completions"

    def _call_api_completion_batch_once(
        self,
        endpoint: str,
        prompts: list[str],
        *,
        max_tokens: int | None,
        temperature: float,
        top_p: float,
    ) -> list[dict[str, Any]]:
        if not prompts:
            return []
        payload = {
            "model": self._arg("judge_api_model", "qwen3-32b-judge"),
            # Preserve the historical scalar request shape when batching is disabled.
            "prompt": prompts[0] if len(prompts) == 1 else prompts,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": (
                max_tokens
                if max_tokens is not None
                else self._arg("judge_max_tokens", 256)
            ),
        }
        timeout = float(self._arg("judge_api_timeout", 120.0))
        try:
            response = self._api_session().post(
                self._completion_url(endpoint), json=payload, timeout=timeout
            )
            if response.status_code >= 400:
                raise _JudgeAPIRequestError(
                    f"{endpoint} returned HTTP {response.status_code}: {response.text[:500]}"
                )
            data = response.json()
        except _JudgeAPIRequestError:
            raise
        except Exception as exc:
            raise _JudgeAPIRequestError(f"{endpoint} request failed: {exc}") from exc

        choices = data.get("choices")
        if not isinstance(choices, list) or len(choices) != len(prompts):
            raise _JudgeAPIRequestError(
                f"{endpoint} returned {len(choices) if isinstance(choices, list) else 'invalid'} choices "
                f"for {len(prompts)} prompts"
            )
        mapped: list[dict[str, Any] | None] = [None] * len(prompts)
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else {}
        for choice in choices:
            if not isinstance(choice, dict):
                raise _JudgeAPIRequestError(
                    f"{endpoint} returned a non-object completion choice"
                )
            choice_index = choice.get("index")
            if isinstance(choice_index, bool) or not isinstance(choice_index, int):
                raise _JudgeAPIRequestError(
                    f"{endpoint} returned a choice without an integer index"
                )
            if not 0 <= choice_index < len(prompts) or mapped[choice_index] is not None:
                raise _JudgeAPIRequestError(
                    f"{endpoint} returned duplicate or out-of-range choice index {choice_index}"
                )
            choice_usage = (
                choice.get("usage") if isinstance(choice.get("usage"), dict) else {}
            )
            token_count = choice_usage.get(
                "completion_tokens", choice.get("completion_tokens")
            )
            if token_count is None and len(prompts) == 1:
                token_count = usage.get("completion_tokens")
            mapped[choice_index] = {
                "judge_output": str(choice.get("text", "")).strip(),
                "judge_finish_reason": choice.get("finish_reason"),
                "judge_output_token_count": token_count,
                "judge_api_batch_usage": usage,
                "judge_api_batch_prompt_count": len(prompts),
                "judge_api_response_id": data.get("id"),
                "judge_choice_index": choice_index,
                "judge_api_endpoint": endpoint,
            }
        if any(item is None for item in mapped):
            raise _JudgeAPIRequestError(
                f"{endpoint} response did not cover every prompt index"
            )
        return [item for item in mapped if item is not None]

    def _call_api_completion_batch(
        self,
        start_endpoint: int,
        prompts: list[str],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[dict[str, Any]]:
        endpoints = self.api_bases
        max_attempts = max(len(endpoints), int(self._arg("judge_api_max_retries", 5)))
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            endpoint = endpoints[(start_endpoint + attempt) % len(endpoints)]
            try:
                return self._call_api_completion_batch_once(
                    endpoint,
                    prompts,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except _JudgeAPIRequestError as exc:
                last_error = exc
                if attempt + 1 < max_attempts:
                    delay = float(self._arg("judge_api_retry_base_delay", 0.5))
                    if delay > 0:
                        time.sleep(min(8.0, delay * (2**attempt)))
        raise RuntimeError(
            f"judge API batch failed after {max_attempts} attempts; last_error={last_error}"
        )

    def _call_api_completion(
        self,
        endpoint: str,
        prompt: str,
        *,
        max_tokens: int | None,
        temperature: float,
        top_p: float,
    ) -> dict[str, Any]:
        """Compatibility wrapper retained for callers that issue one completion."""

        return self._call_api_completion_batch_once(
            endpoint,
            [prompt],
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )[0]

    def _run_cached_api(
        self,
        *,
        output_dir: Path,
        prompts: list[tuple[str, str]],
        cache_name: str,
        max_tokens: int | None,
        temperature: float,
        top_p: float,
        no_resume: bool,
        desc: str,
        output_validator: Callable[[str], bool] | None,
        output_validator_by_index: Callable[[str, str], bool] | None,
        contract_version: str,
        system_prompt: str | None,
        retry_on_length: bool,
        return_unresolved: bool,
    ) -> dict[str, dict[str, Any]]:
        runner, _ = _import_vlmeval_runner()

        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / cache_name
        if no_resume and cache_path.exists():
            cache_path.unlink()
        existing = {} if no_resume else runner.load_jsonl_by_index(cache_path)
        if len({str(idx) for idx, _ in prompts}) != len(prompts):
            raise ValueError(
                f"Duplicate judge prompt indices are not supported: {cache_path}"
            )
        effective_max_tokens = max(
            128,
            int(
                max_tokens
                if max_tokens is not None
                else self._arg("judge_max_tokens", 256)
            ),
        )
        expected = {
            str(idx): self._request_metadata(
                prompt,
                max_tokens=effective_max_tokens,
                temperature=temperature,
                top_p=top_p,
                contract_version=contract_version,
                backend="api",
                system_prompt=system_prompt,
            )
            for idx, prompt in prompts
        }
        retry_ids = {
            str(idx)
            for idx, _ in prompts
            if str(idx) in existing
            and (
                any(
                    str(existing[str(idx)].get(key, "")) != str(value)
                    for key, value in expected[str(idx)].items()
                )
                or (
                    not return_unresolved
                    and self._cache_entry_needs_retry(
                        existing[str(idx)],
                        expected[str(idx)],
                        output_validator,
                        output_validator_by_index,
                        retry_on_length=retry_on_length,
                    )
                )
            )
        }
        pending = [
            (idx, prompt)
            for idx, prompt in prompts
            if str(idx) not in existing or str(idx) in retry_ids
        ]
        endpoints = self.api_bases
        batch_size = max(1, int(self._arg("judge_api_batch_size", 1)))
        max_batch_chars = max(1, int(self._arg("judge_api_max_batch_chars", 100_000)))
        batches_per_endpoint = max(
            1, int(self._arg("judge_api_batches_per_endpoint", 1))
        )
        print(
            "[judge:api-cached] "
            f"cache={cache_path} rows={len(prompts)} existing={len(existing)} pending={len(pending)} "
            f"retry_incomplete={len(retry_ids)} max_tokens={effective_max_tokens} "
            f"endpoints={len(endpoints)} batch_size={batch_size} max_batch_chars={max_batch_chars}"
        )
        if pending:
            rendered = {
                str(idx): self.api_chat_prompt(prompt, system_prompt=system_prompt)
                for idx, prompt in pending
            }
            batches = _make_judge_api_batches(
                pending,
                rendered,
                batch_size=batch_size,
                max_batch_chars=max_batch_chars,
            )

            def run_batch(
                pos_batch: tuple[int, list[tuple[str, str]]],
            ) -> list[dict[str, Any]]:
                pos, batch = pos_batch
                active = list(batch)
                accepted: dict[str, dict[str, Any]] = {}
                retry_counts = {str(idx): 0 for idx, _ in batch}
                retry_reasons: dict[str, str] = {}
                last_results: dict[str, tuple[dict[str, Any], str, int]] = {}
                limits = _judge_retry_token_limits(effective_max_tokens)
                for retry_round, token_limit in enumerate(limits):
                    rows = self._call_api_completion_batch(
                        pos + retry_round,
                        [rendered[str(idx)] for idx, _ in active],
                        max_tokens=token_limit,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    unresolved: list[tuple[str, str]] = []
                    for (idx, original_prompt), result in zip(active, rows):
                        last_results[str(idx)] = (result, original_prompt, token_limit)
                        output = str(result.get("judge_output", ""))
                        reason = ""
                        if _judge_cache_entry_needs_retry(
                            result, retry_on_length=retry_on_length
                        ):
                            reason = "incomplete"
                        elif output_validator_by_index is not None:
                            try:
                                if not bool(
                                    output_validator_by_index(str(idx), output)
                                ):
                                    reason = "parse_invalid"
                            except Exception:
                                reason = "parse_invalid"
                        elif output_validator is not None:
                            try:
                                if not bool(output_validator(output)):
                                    reason = "parse_invalid"
                            except Exception:
                                reason = "parse_invalid"
                        if reason:
                            retry_counts[str(idx)] += 1
                            retry_reasons[str(idx)] = reason
                            unresolved.append((idx, original_prompt))
                            continue
                        accepted[str(idx)] = {
                            "index": str(idx),
                            **result,
                            **expected[str(idx)],
                            "judge_prompt": original_prompt,
                            "judge_max_tokens_used": token_limit,
                            "judge_retry_count": retry_counts[str(idx)],
                            "judge_retry_reason": retry_reasons.get(str(idx), ""),
                        }
                    active = unresolved
                    if not active:
                        break
                if active:
                    failed = [str(idx) for idx, _ in active]
                    if not return_unresolved:
                        raise RuntimeError(
                            "Judge produced incomplete or parse-invalid output after retries "
                            f"for indices={failed[:10]}"
                        )
                    for idx, _ in active:
                        result, original_prompt, token_limit = last_results[str(idx)]
                        accepted[str(idx)] = {
                            "index": str(idx),
                            **result,
                            **expected[str(idx)],
                            "judge_prompt": original_prompt,
                            "judge_max_tokens_used": token_limit,
                            "judge_retry_count": retry_counts[str(idx)],
                            "judge_retry_reason": retry_reasons.get(str(idx), ""),
                        }
                return [accepted[str(idx)] for idx, _ in batch]

            max_workers = min(
                len(batches),
                max(1, len(endpoints) * batches_per_endpoint),
                max(1, int(self._arg("judge_api_parallelism", 128))),
            )
            print(
                "[judge:api-batches] "
                f"batches={len(batches)} workers={max_workers} batches_per_endpoint={batches_per_endpoint}"
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(run_batch, item) for item in enumerate(batches)]
                for future in tqdm(
                    concurrent.futures.as_completed(futures),
                    total=len(futures),
                    desc=desc,
                ):
                    runner.append_jsonl(cache_path, future.result())
        return runner.load_jsonl_by_index(cache_path)

    def run_cached(
        self,
        *,
        output_dir: Path,
        prompts: list[tuple[str, str]],
        cache_name: str,
        max_tokens: int | None = None,
        temperature: float = 0.0,
        top_p: float = 1.0,
        no_resume: bool = False,
        desc: str = "local judge",
        output_validator: Callable[[str], bool] | None = None,
        output_validator_by_index: Callable[[str, str], bool] | None = None,
        contract_version: str | None = None,
        system_prompt: str | None = None,
        retry_on_length: bool = True,
        return_unresolved: bool = False,
    ) -> dict[str, dict[str, Any]]:
        effective_contract_version = contract_version or str(
            self._arg(
                "judge_cache_contract_version", PERSISTENT_JUDGE_CACHE_CONTRACT_VERSION
            )
        )
        if self._using_api_pool():
            return self._run_cached_api(
                output_dir=output_dir,
                prompts=prompts,
                cache_name=cache_name,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                no_resume=no_resume,
                desc=desc,
                output_validator=output_validator,
                output_validator_by_index=output_validator_by_index,
                contract_version=effective_contract_version,
                system_prompt=system_prompt,
                retry_on_length=retry_on_length,
                return_unresolved=return_unresolved,
            )
        runner, _ = _import_vlmeval_runner()
        from vllm import SamplingParams

        output_dir.mkdir(parents=True, exist_ok=True)
        cache_path = output_dir / cache_name
        if no_resume and cache_path.exists():
            cache_path.unlink()
        existing = {} if no_resume else runner.load_jsonl_by_index(cache_path)
        if len({str(idx) for idx, _ in prompts}) != len(prompts):
            raise ValueError(
                f"Duplicate judge prompt indices are not supported: {cache_path}"
            )
        effective_max_tokens = max(
            128,
            int(
                max_tokens
                if max_tokens is not None
                else self._arg("judge_max_tokens", 256)
            ),
        )
        expected = {
            str(idx): self._request_metadata(
                prompt,
                max_tokens=effective_max_tokens,
                temperature=temperature,
                top_p=top_p,
                contract_version=effective_contract_version,
                backend="local",
                system_prompt=system_prompt,
            )
            for idx, prompt in prompts
        }
        retry_ids = {
            str(idx)
            for idx, _ in prompts
            if str(idx) in existing
            and (
                any(
                    str(existing[str(idx)].get(key, "")) != str(value)
                    for key, value in expected[str(idx)].items()
                )
                or (
                    not return_unresolved
                    and self._cache_entry_needs_retry(
                        existing[str(idx)],
                        expected[str(idx)],
                        output_validator,
                        output_validator_by_index,
                        retry_on_length=retry_on_length,
                    )
                )
            )
        }
        pending = [
            (idx, prompt)
            for idx, prompt in prompts
            if str(idx) not in existing or str(idx) in retry_ids
        ]
        print(
            "[judge:cached] "
            f"cache={cache_path} rows={len(prompts)} existing={len(existing)} pending={len(pending)} "
            f"retry_incomplete={len(retry_ids)} max_tokens={effective_max_tokens}"
        )
        if pending:
            self._ensure_loaded()
            assert self.llm is not None
            judge_batch_size = max(1, int(self._arg("judge_batch_size", 1024)))
            active = list(pending)
            accepted: dict[str, dict[str, Any]] = {}
            retry_counts = {str(idx): 0 for idx, _ in pending}
            retry_reasons: dict[str, str] = {}
            last_results: dict[str, tuple[dict[str, Any], str, int]] = {}
            for token_limit in _judge_retry_token_limits(effective_max_tokens):
                sampling = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=token_limit,
                )
                unresolved: list[tuple[str, str]] = []
                total_batches = math.ceil(len(active) / judge_batch_size)
                for start in tqdm(
                    range(0, len(active), judge_batch_size),
                    total=total_batches,
                    desc=desc,
                ):
                    batch = active[start : start + judge_batch_size]
                    llm_prompts = [
                        self.chat_prompt(prompt, system_prompt=system_prompt)
                        for _, prompt in batch
                    ]
                    outputs = self.llm.generate(
                        llm_prompts, sampling_params=sampling, use_tqdm=False
                    )
                    completed_rows: list[dict[str, Any]] = []
                    for (idx, original_prompt), out in zip(batch, outputs):
                        result = {
                            "judge_output": out.outputs[0].text.strip(),
                            "judge_finish_reason": out.outputs[0].finish_reason,
                            "judge_output_token_count": len(out.outputs[0].token_ids),
                        }
                        last_results[str(idx)] = (result, original_prompt, token_limit)
                        reason = ""
                        if _judge_cache_entry_needs_retry(
                            result, retry_on_length=retry_on_length
                        ):
                            reason = "incomplete"
                        elif output_validator_by_index is not None:
                            try:
                                if not bool(
                                    output_validator_by_index(
                                        str(idx), str(result["judge_output"])
                                    )
                                ):
                                    reason = "parse_invalid"
                            except Exception:
                                reason = "parse_invalid"
                        elif output_validator is not None:
                            try:
                                if not bool(
                                    output_validator(str(result["judge_output"]))
                                ):
                                    reason = "parse_invalid"
                            except Exception:
                                reason = "parse_invalid"
                        if reason:
                            retry_counts[str(idx)] += 1
                            retry_reasons[str(idx)] = reason
                            unresolved.append((idx, original_prompt))
                            continue
                        accepted[str(idx)] = {
                            "index": str(idx),
                            **result,
                            **expected[str(idx)],
                            "judge_prompt": original_prompt,
                            "judge_max_tokens_used": token_limit,
                            "judge_retry_count": retry_counts[str(idx)],
                            "judge_retry_reason": retry_reasons.get(str(idx), ""),
                        }
                        completed_rows.append(accepted[str(idx)])
                    if completed_rows:
                        runner.append_jsonl(cache_path, completed_rows)
                active = unresolved
                if not active:
                    break
            if active:
                failed = [str(idx) for idx, _ in active]
                if not return_unresolved:
                    raise RuntimeError(
                        "Judge produced incomplete or parse-invalid output after retries "
                        f"for indices={failed[:10]}"
                    )
                unresolved_rows = []
                for idx, _ in active:
                    result, original_prompt, token_limit = last_results[str(idx)]
                    unresolved_rows.append(
                        {
                            "index": str(idx),
                            **result,
                            **expected[str(idx)],
                            "judge_prompt": original_prompt,
                            "judge_max_tokens_used": token_limit,
                            "judge_retry_count": retry_counts[str(idx)],
                            "judge_retry_reason": retry_reasons.get(str(idx), ""),
                        }
                    )
                runner.append_jsonl(cache_path, unresolved_rows)
        return runner.load_jsonl_by_index(cache_path)

    def cleanup(self) -> None:
        if self._using_api_pool():
            self.api_tokenizer = None
            with self._api_sessions_lock:
                sessions = list(self._api_sessions)
                self._api_sessions.clear()
            for session in sessions:
                session.close()
            self._api_thread_local = threading.local()
            return
        runner, _ = _import_vlmeval_runner()
        runner.cleanup_vllm_engine(self.llm)
        self.llm = None
        self.tokenizer = None


def _namespace_for_spec(
    args: argparse.Namespace, spec: BenchmarkSpec, output_dir: Path, model_path: str
) -> SimpleNamespace:
    return SimpleNamespace(
        dataset=spec.alias,
        model=model_path,
        output_dir=output_dir,
        no_resume=args.no_resume,
        eval_judge_model=args.eval_judge_model,
        eval_nproc=args.eval_nproc,
        judge_model=args.judge_model,
        judge_gpu=args.gpu,
        judge_tensor_parallel_size=args.judge_tensor_parallel_size,
        judge_gpu_memory_utilization=args.judge_gpu_memory_utilization,
        judge_max_model_len=args.judge_max_model_len,
        judge_max_num_seqs=args.judge_max_num_seqs,
        judge_max_num_batched_tokens=args.judge_max_num_batched_tokens,
        judge_batch_size=args.judge_batch_size,
        judge_max_tokens=args.judge_max_tokens,
        attention_backend=args.attention_backend,
    )


def _sanitize_prediction_table_for_scoring(
    spec: BenchmarkSpec, output_dir: Path
) -> None:
    """Normalize evaluator input cells that some VLMEvalKit scorers assume are strings."""
    if spec.alias != "TableVQABench":
        return
    pred_table = output_dir / f"{spec.alias}_predictions.xlsx"
    if not pred_table.exists():
        return
    data = pd.read_excel(pred_table)
    changed = False
    for col in ("prediction", "answer"):
        if col not in data:
            continue
        fill_value = (
            "__missing_prediction__" if col == "prediction" else "__missing_answer__"
        )
        normalized = data[col].fillna(fill_value).map(str)
        if not normalized.equals(data[col]):
            data[col] = normalized
            changed = True
    if changed:
        data.to_excel(pred_table, index=False)
        print(
            f"[score:sanitize] {spec.alias} cast prediction/answer cells to strings: {pred_table}"
        )


def _run_tablevqabench_local_score(
    spec: BenchmarkSpec,
    model_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Run the pinned TableVQABench parser and four split scorers."""

    _import_vlmeval_runner()
    from vlmeval.dataset.utils.tablevqabench import (
        evaluate_fintabnet,
        evaluate_tabfact,
        evaluate_wtq,
    )
    from vlmeval.dataset.utils.trace_eval_answer_parsing import (
        unwrap_single_answer_block,
    )

    pred_table = output_dir / f"{spec.alias}_predictions.xlsx"
    if not pred_table.exists():
        raise FileNotFoundError(pred_table)
    data = pd.read_excel(pred_table).copy()
    if "prediction" not in data or "answer" not in data or "split" not in data:
        raise ValueError(f"Malformed TableVQABench prediction table: {pred_table}")

    data["raw_prediction"] = data["prediction"]
    data["prediction"] = data["prediction"].map(unwrap_single_answer_block)
    data["prediction"] = data["prediction"].str.replace("^Answer: ", "", regex=True)

    scored_rows: list[dict[str, Any]] = []
    score_table: list[dict[str, Any]] = []
    all_reported_scores: list[float] = []
    data_group = dict(tuple(data.groupby("split")))
    for split in ("fintabnetqa", "vtabfact", "vwtq", "vwtq_syn"):
        group = data_group[split]
        records = group.to_dict(orient="records")
        if split == "fintabnetqa":
            meta = evaluate_fintabnet(records, ["accuracy"])
        elif split == "vtabfact":
            meta = evaluate_tabfact(records, ["accuracy"])
        else:
            meta = evaluate_wtq(records, ["accuracy"])
        values = [float(value) for value in meta.get("average_scores", [])]
        score_table.append({"split": split, "average_scores": values})
        all_reported_scores.extend(values)
        scored_rows.extend(records)

    if len(scored_rows) != len(data) or not all_reported_scores:
        raise ValueError("TableVQABench scorer did not account for every row")
    overall = float(sum(all_reported_scores) / len(all_reported_scores))
    judged_table = output_dir / f"{spec.alias}_trace_final_answer_scored.xlsx"
    pd.DataFrame(scored_rows).to_excel(judged_table, index=False)
    score_csv = output_dir / f"{spec.alias}_predictions_acc.csv"
    pd.DataFrame(score_table).to_csv(score_csv, index=False)
    scores = {"Overall": overall, "table": score_table}
    summary = {
        "dataset": spec.alias,
        "model": model_path,
        "run_name": spec.run_name,
        "harness": "Pinned VLMEvalKit TableVQABench parser and split scorers",
        "rows": int(len(data)),
        "score": overall,
        "scores": scores,
        "aggregation": "macro mean over all official split average_scores values",
        "extraction": {
            "method": "single <answer> unwrap, then pinned VLMEvalKit leading '^Answer: ' removal",
            "changed_predictions": int(
                (data["raw_prediction"] != data["prediction"]).sum()
            ),
        },
        "artifacts": {
            "prediction_table": str(pred_table),
            "judged_table": str(judged_table),
            "score_csv": str(score_csv),
        },
    }
    write_json(output_dir / "scores.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    return summary


def _copy_score_to_benchmark(
    spec: BenchmarkSpec, model_slug: str, run_output_dir: Path, benchmark_root: Path
) -> Path:
    src = run_output_dir / "scores.json"
    if not src.exists():
        raise FileNotFoundError(src)
    dst = score_path(spec, model_slug, benchmark_root)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _archive_scalar(value: Any) -> Any:
    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _archive_first_present(*values: Any) -> Any:
    """Return the first non-null value without treating ordinal zero as missing."""

    for value in values:
        normalized = _archive_scalar(value)
        if normalized is None:
            continue
        if isinstance(normalized, str) and not normalized.strip():
            continue
        return normalized
    return None


def _archive_expected_rows(summary: dict[str, Any]) -> int | None:
    value = _archive_scalar(summary.get("rows"))
    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    if isinstance(value, bool):
        raise RuntimeError(
            f"Archive summary rows must be a non-negative integer, got {value!r}"
        )
    try:
        rows = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError(
            f"Archive summary rows must be a non-negative integer, got {value!r}"
        ) from exc
    if isinstance(value, str):
        is_exact = bool(re.fullmatch(r"\+?\d+", value.strip()))
    else:
        try:
            is_exact = bool(rows == value)
        except Exception:
            is_exact = False
    if rows < 0 or not is_exact:
        raise RuntimeError(
            f"Archive summary rows must be a non-negative integer, got {value!r}"
        )
    return rows


_ARCHIVE_AGGREGATE_ONLY_SCORE_KEYS = frozenset(
    {"tablevqabench", "mathvision", "mathvista"}
)


def _archive_table_candidates(output_dir: Path, summary: dict[str, Any]) -> list[Path]:
    values: list[Any] = []

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                collect(child)
        else:
            values.append(value)

    collect(summary.get("artifacts") or {})
    collect(summary.get("outputs") or {})
    paths: list[Path] = []
    for value in values:
        if not isinstance(value, (str, Path)):
            continue
        path = Path(value)
        if path.suffix.lower() in {".xlsx", ".xls"} and path.exists():
            paths.append(path)
    paths.extend(sorted(output_dir.glob("*.xlsx")))
    return list(dict.fromkeys(paths))


def _archive_best_score_table(
    output_dir: Path, summary: dict[str, Any]
) -> pd.DataFrame:
    best: tuple[int, pd.DataFrame] | None = None
    for path in _archive_table_candidates(output_dir, summary):
        try:
            frame = pd.read_excel(path, keep_default_na=False)
        except Exception:
            continue
        columns = set(frame.columns)
        quality = 0
        lowered = path.name.lower()
        if any(token in lowered for token in ("judged", "scored", "result")):
            quality += 100
        if any(key in columns for key in ("eval_score", "hit", "score", "correct")):
            quality += 80
        if any(
            key in columns
            for key in (
                "eval_pred",
                "res",
                "extract",
                "extract_answer",
                "raw_prediction",
            )
        ):
            quality += 40
        if "prediction" in columns:
            quality += 20
        if "index" in columns:
            quality += 10
        quality += min(len(frame), 10)
        if best is None or quality > best[0]:
            best = (quality, frame)
    if best is None:
        raise FileNotFoundError(
            f"No readable score/prediction table under {output_dir}"
        )
    return best[1]


def _archive_judge_events(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    events: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(output_dir.glob("*.jsonl")):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if (
                not isinstance(row, dict)
                or "judge_output" not in row
                or "index" not in row
            ):
                continue
            events[str(row["index"])].append(
                {
                    "cache": path.name,
                    "prompt": row.get("judge_prompt", ""),
                    "response": row.get("judge_output", ""),
                    "request_hash": row.get("request_hash"),
                    "contract_version": row.get("contract_version"),
                    "finish_reason": row.get("judge_finish_reason"),
                    "retry_count": row.get("judge_retry_count", 0),
                    "max_tokens_used": row.get("judge_max_tokens_used"),
                }
            )
    return events


def _archive_direct_score_slices(
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    model_path: str,
    model_slug: str,
    output_dir: Path,
    summary: dict[str, Any],
) -> tuple[Path | None, Path | None]:
    if not os.environ.get("TRACE_EVAL_HF_SPOOL_ROOT", "").strip():
        return None, None
    from trace_eval_archive_hooks import (
        canonical_json,
        emit_extraction_slice,
        emit_score_slice,
        resolve_model_revision,
        resolve_model_source,
        sanitize_benchmark_source_row,
    )

    frame = _archive_best_score_table(output_dir, summary)
    expected_rows = _archive_expected_rows(summary)
    if expected_rows is not None and len(frame) != expected_rows:
        raise RuntimeError(
            f"Refusing to archive {spec.key}: selected score table has {len(frame)} rows, "
            f"but the score summary declares {expected_rows}"
        )
    judge_events = _archive_judge_events(output_dir)
    source_rows_by_index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_candidates = [
        output_dir / f"{spec.alias}_predictions.xlsx",
        output_dir / "predictions.xlsx",
    ]
    for source_path in source_candidates:
        if not source_path.exists():
            continue
        try:
            source_frame = pd.read_excel(source_path, keep_default_na=False)
        except Exception:
            continue
        if len(source_frame) != len(frame):
            raise RuntimeError(
                f"Refusing to archive {spec.key}: selected score table has {len(frame)} rows, "
                f"but source prediction table {source_path.name} has {len(source_frame)}"
            )
        if expected_rows is not None and len(source_frame) != expected_rows:
            raise RuntimeError(
                f"Refusing to archive {spec.key}: source prediction table {source_path.name} has "
                f"{len(source_frame)} rows, but the score summary declares {expected_rows}"
            )
        for source_ordinal, source_row in enumerate(
            source_frame.to_dict(orient="records")
        ):
            source_rows_by_index[str(source_row.get("index", source_ordinal))].append(
                {**source_row, "_source_ordinal": source_ordinal}
            )
        break
    source_occurrences: defaultdict[str, int] = defaultdict(int)
    extraction_records: list[dict[str, Any]] = []
    score_records: list[dict[str, Any]] = []
    missing_row_scores: list[str] = []
    aggregate_only_scores = spec.key in _ARCHIVE_AGGREGATE_ONLY_SCORE_KEYS
    aggregate_score = _archive_scalar(summary.get("score"))
    if aggregate_only_scores and aggregate_score is None:
        raise RuntimeError(
            f"Refusing to archive {spec.key}: aggregate-only score contract requires summary.score"
        )
    generated_keys = {
        "prediction",
        "raw_prediction",
        "finish_reason",
        "output_token_count",
        "prompt_token_count",
        "request_hash",
        "source_ordinal",
        "source_row_hash",
        "prompt",
        "usage",
        "_source_ordinal",
    }
    for ordinal, raw_row in enumerate(frame.to_dict(orient="records")):
        row = {str(key): _archive_scalar(value) for key, value in raw_row.items()}
        index = str(row.get("index", ordinal))
        occurrence = source_occurrences[index]
        source_occurrences[index] += 1
        source_matches = source_rows_by_index.get(index, [])
        source_row = (
            source_matches[min(occurrence, len(source_matches) - 1)]
            if source_matches
            else {}
        )
        events = judge_events.get(index, [])
        model_response = row.get("raw_prediction", source_row.get("raw_prediction"))
        if model_response in {None, ""}:
            model_response = source_row.get("prediction", row.get("prediction", ""))
        extraction_value = None
        extraction_method = ""
        for key in ("extract_answer", "eval_pred", "res", "extract", "prediction"):
            value = row.get(key)
            if value not in {None, ""}:
                extraction_value = value
                extraction_method = key
                break
        for method_key in (
            "extract_answer_method",
            "eval_pred_method",
            "trace_extraction_method",
        ):
            if row.get(method_key) not in {None, ""}:
                extraction_method = str(row[method_key])
                break
        source_hash = str(
            row.get("source_row_hash") or source_row.get("source_row_hash") or ""
        ).strip()
        if not source_hash:
            safe_source_row = sanitize_benchmark_source_row(
                {
                    key: value
                    for key, value in (source_row or row).items()
                    if key not in generated_keys
                }
            )
            source_hash = hashlib.sha256(
                canonical_json(safe_source_row).encode("utf-8")
            ).hexdigest()
        event_request_hashes = [
            event.get("request_hash") for event in events if event.get("request_hash")
        ]
        extraction_request_hash = hashlib.sha256(
            canonical_json(
                {
                    "contract_version": "trace-eval-v1-direct-extraction-v1",
                    "benchmark": spec.key,
                    "source_row_hash": source_hash,
                    "model_response": model_response,
                    "judge_request_hashes": event_request_hashes,
                    "extraction": extraction_value,
                }
            ).encode("utf-8")
        ).hexdigest()
        common = {
            "source_index": index,
            "source_ordinal": int(
                _archive_first_present(
                    row.get("source_ordinal"),
                    source_row.get("source_ordinal"),
                    source_row.get("_source_ordinal"),
                    ordinal,
                )
            ),
            "source_row_hash": source_hash,
            "question": row.get(
                "question", source_row.get("question", row.get("query"))
            ),
            "ground_truth": row.get("answer", source_row.get("answer")),
            "metadata": {"benchmark_run_name": spec.run_name},
        }
        extraction_records.append(
            {
                **common,
                "request_hash": extraction_request_hash,
                "model_response": model_response,
                "judge_prompt": canonical_json(
                    [event.get("prompt", "") for event in events]
                ),
                "judge_response": canonical_json(
                    [event.get("response", "") for event in events]
                ),
                "normalized_extraction": {
                    "status": (
                        "resolved"
                        if extraction_value is not None and extraction_value != ""
                        else "invalid"
                    ),
                    "value": extraction_value,
                    "method": extraction_method,
                },
                "retries": {
                    "events": events,
                    "total_retries": sum(
                        int(event.get("retry_count") or 0) for event in events
                    ),
                },
            }
        )
        score_value = None
        for key in ("eval_score", "hit", "score", "correct"):
            if row.get(key) not in {None, ""}:
                score_value = row[key]
                break
        if score_value is None and not aggregate_only_scores:
            missing_row_scores.append(index)
        scorer = str(summary.get("harness") or summary.get("run_name") or spec.run_name)
        score_request_hash = hashlib.sha256(
            canonical_json(
                {
                    "contract_version": "trace-eval-v1-score-v1",
                    "extraction_request_hash": extraction_request_hash,
                    "prediction": extraction_value,
                    "score": score_value,
                    "scorer": scorer,
                }
            ).encode("utf-8")
        ).hexdigest()
        score_records.append(
            {
                **common,
                "metadata": {
                    **common["metadata"],
                    "score_contract": (
                        "aggregate_only" if aggregate_only_scores else "per_row"
                    ),
                    **(
                        {"aggregate_score": aggregate_score}
                        if aggregate_only_scores
                        else {}
                    ),
                },
                "request_hash": score_request_hash,
                "prediction": extraction_value,
                "score": score_value,
                "scorer": scorer,
                "excluded": False,
            }
        )

    if missing_row_scores:
        raise RuntimeError(
            f"Refusing to archive {spec.key}: {len(missing_row_scores)} scored rows have no explicit "
            f"per-row score; first indices={missing_row_scores[:10]}"
        )
    identity = {
        "model": resolve_model_source(model_slug, model_path),
        "model_slug": model_slug,
        "model_revision": resolve_model_revision(model_slug, model_path),
        "seed": int(getattr(args, "seed", 0)),
        "benchmark": spec.key,
        "dataset_alias": spec.alias,
        "dataset_split": spec.split or "default",
        "dataset_revision": os.environ.get(
            "TRACE_EVAL_DATASET_REVISION",
            os.environ.get("TRACE_VLMEVALKIT_GIT_COMMIT", "unknown"),
        ),
    }
    extraction_path = emit_extraction_slice(
        records=extraction_records,
        contract_version="trace-eval-v1-direct-extraction-v1",
        aggregate={
            "rows": len(extraction_records),
            "judge_model": getattr(args, "judge_model", None),
        },
        **identity,
    )
    score_path = emit_score_slice(
        records=score_records,
        contract_version="trace-eval-v1-score-v1",
        aggregate=summary,
        **identity,
    )
    return extraction_path, score_path


def _run_direct_vlmeval(
    args: argparse.Namespace, spec: BenchmarkSpec, model_path: str, output_dir: Path
) -> dict[str, Any]:
    runner, _ = _import_vlmeval_runner()
    _sanitize_prediction_table_for_scoring(spec, output_dir)
    ns = _namespace_for_spec(args, spec, output_dir, model_path)
    return runner.run_vlmeval_evaluate(ns)


def _extract_braced_answer(text: Any) -> str:
    text = str(text or "").strip()
    matches = re.findall(r"\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1].strip()
    yes_no = re.findall(r"\b(yes|no)\b", text, flags=re.I)
    if yes_no:
        return yes_no[-1].strip()
    final = list(
        re.finditer(
            r"\b(?:final\s+answer|answer)\b\s*[:：]?\s*(.+)", text, flags=re.I | re.S
        )
    )
    if final:
        return final[-1].group(1).strip().splitlines()[0].strip()
    return text


def _extract_option_letter(text: Any, *, choices: str = "ABCD") -> str:
    text = str(text or "").strip()
    braced = _extract_braced_answer(text)
    answer_patterns = [
        rf"<\s*answer\s*>\s*[:：]?\s*([{choices}])\b",
        rf"\b(?:final\s+answer|answer|option|choice|correct)\b\s*(?:is|:|：)?\s*([{choices}])\b",
    ]
    for candidate in (braced, text):
        stripped = candidate.strip().upper()
        if stripped in set(choices):
            return stripped
        for pattern in answer_patterns:
            match = re.search(pattern, candidate, flags=re.I)
            if match:
                return match.group(1).upper()
    matches = re.findall(rf"\b([{choices}])\b", text, flags=re.I)
    return matches[-1].upper() if matches else ""


def _run_wemath_subset_score(
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    model_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    _import_vlmeval_runner()
    from vlmeval.smp import load

    pred_table = output_dir / f"{spec.alias}_predictions.xlsx"
    if not pred_table.exists():
        raise FileNotFoundError(pred_table)
    data = load(str(pred_table))
    data["eval_pred"] = [
        _extract_option_letter(x, choices="ABCDEFG") for x in data["prediction"]
    ]
    data["eval_gt"] = [
        _extract_option_letter(x, choices="ABCDEFG") for x in data["answer"]
    ]
    data["hit"] = [
        float(p != "" and p == g) for p, g in zip(data["eval_pred"], data["eval_gt"])
    ]

    judged = output_dir / f"{spec.alias}_subset_option_judged.xlsx"
    data.to_excel(judged, index=False)
    overall = float(data["hit"].mean() * 100.0) if len(data) else 0.0
    table: list[dict[str, Any]] = [
        {"Split": "Overall", "Accuracy (%)": overall, "Samples": int(len(data))}
    ]
    for col in ("key", "split", "knowledge concept"):
        if col in data:
            for key, group in data.groupby(col, dropna=False):
                table.append(
                    {
                        "Split": f"{col}/{key}",
                        "Accuracy (%)": float(group["hit"].mean() * 100.0),
                        "Samples": int(len(group)),
                    }
                )

    score_csv = output_dir / f"{spec.alias}_subset_option_score.csv"
    pd.DataFrame(table).to_csv(score_csv, index=False)
    scores = {"accuracy": overall, "Overall": overall, "table": table}
    summary = {
        "dataset": spec.alias,
        "model": model_path,
        "run_name": spec.run_name,
        "harness": "Trace subset-safe WeMath option-letter exact scorer",
        "rows": len(data),
        "score": overall,
        "scores": scores,
        "artifacts": {
            "prediction_table": str(pred_table),
            "judged_table": str(judged),
            "score_csv": str(score_csv),
        },
    }
    write_json(output_dir / "scores.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    return summary


def _run_phyx_option_score(
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    model_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    _import_vlmeval_runner()
    from vlmeval.smp import load

    pred_table = output_dir / f"{spec.alias}_predictions.xlsx"
    if not pred_table.exists():
        raise FileNotFoundError(pred_table)
    data = load(str(pred_table))
    data["eval_pred"] = [
        _extract_option_letter(x, choices="ABCD") for x in data["prediction"]
    ]
    data["eval_gt"] = [
        _extract_option_letter(x, choices="ABCD") for x in data["answer"]
    ]
    data["hit"] = [
        float(p != "" and p == g) for p, g in zip(data["eval_pred"], data["eval_gt"])
    ]
    judged = output_dir / f"{spec.alias}_option_judged.xlsx"
    data.to_excel(judged, index=False)
    overall = float(data["hit"].mean() * 100.0) if len(data) else 0.0
    scores: dict[str, Any] = {"Overall": overall, "Accuracy (%)": overall}
    for col in ("category", "subfield", "reasoning_type"):
        if col in data:
            for key, group in data.groupby(col):
                scores[f"{col}/{key}"] = float(group["hit"].mean() * 100.0)
    summary = {
        "dataset": spec.alias,
        "model": model_path,
        "run_name": spec.run_name,
        "harness": "local option-letter exact scorer",
        "rows": len(data),
        "score": overall,
        "scores": scores,
        "artifacts": {"prediction_table": str(pred_table), "judged_table": str(judged)},
    }
    write_json(output_dir / "scores.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    return summary


def _parse_json_object(text: Any) -> dict[str, Any]:
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    match = re.search(r"\{.*?\}", raw, flags=re.S)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def _resolve_charxiv_extracted_answer(
    item: dict[str, Any],
    parsed: dict[str, Any],
    prediction: Any,
) -> tuple[str, str]:
    extracted = str(
        item.get("extract_answer", parsed.get("extract_answer", ""))
    ).strip()
    if extracted:
        return extracted, "judge"

    if prediction is None or (isinstance(prediction, float) and math.isnan(prediction)):
        return "", "empty_prediction"
    fallback, method = extract_final_answer(prediction)
    if fallback:
        return fallback, f"deterministic_{method}"
    return "", "empty_prediction"


def _parse_charxiv_score_output(value: Any) -> float | None:
    output = str(value or "")
    obj = _parse_json_object(output)
    raw_score = obj.get("score")
    if raw_score is None:
        score_match = re.search(
            r'"?score"?\s*[:=]\s*([01](?:\.\d+)?)', output, flags=re.I
        )
        raw_score = score_match.group(1) if score_match else None
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        return None
    return score if 0.0 <= score <= 1.0 else None


def _charxiv_primary_score(scores: dict[str, float]) -> float:
    return float(scores["Overall"] * 100.0)


def _extract_final_mcq_option(text: Any, choices: list[str]) -> tuple[str, str]:
    """Extract an explicit final MCQ option from verbose CoT-style answers."""

    if not choices:
        return "Z", "no_choices"
    value = str(text or "").replace("\r", "\n")
    labels = "".join(re.escape(label) for label in choices)
    marker_patterns = (
        rf"(?is)(?:\*\*)?\s*(?:final\s+answer|correct\s+answer|answer)"
        rf"\s*(?:is)?\s*:?\s*(?:\*\*)?\s*:?\s*"
        rf"(?:\*\*)?\(?([{labels}])\)?(?:\*\*)?\s*(?:[\.\):\-]|$)",
        rf"(?is)(?:therefore|thus|so),?\s*(?:the\s+)?(?:\*\*)?\s*"
        rf"(?:final\s+)?(?:correct\s+)?answer\s*(?:is)?\s*:?\s*(?:\*\*)?\s*:?\s*"
        rf"(?:\*\*)?\(?([{labels}])\)?(?:\*\*)?\s*(?:[\.\):\-]|$)",
        rf"(?is)(?:final\s+answer|correct\s+answer|answer)[\s\S]{{0,160}}?"
        rf"(?:\*\*)?\(?([{labels}])\)?(?:\*\*)?\s*[\.\)]",
    )
    matches: list[re.Match[str]] = []
    for pattern in marker_patterns:
        matches.extend(re.finditer(pattern, value))
    if matches:
        matches.sort(key=lambda match: match.start())
        return matches[-1].group(1).upper(), "answer_marker"

    tail = value[-1600:]
    line_matches = list(
        re.finditer(
            rf"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?([{labels}])(?:\*\*)?\s*[\.\):]\s+",
            tail,
        )
    )
    if line_matches:
        return line_matches[-1].group(1).upper(), "tail_line_start"

    if len(re.findall(r"\S+", value)) <= 40:
        short_match = re.search(rf"(?<![A-Za-z])([{labels}])\s*[\.\):]\s+", value)
        if short_match:
            return short_match.group(1).upper(), "short_option"

    lines = [line.strip() for line in value.splitlines() if line.strip()]
    for line in reversed(lines[-8:]):
        line_match = re.search(
            rf"^(?:\*\*)?([{labels}])(?:\*\*)?(?:\s*[\.\):]|\s*$)", line
        )
        if line_match:
            return line_match.group(1).upper(), "last_line"
    return "Z", "unparsed"


def _truthy_score(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def _normalize_mathverse_score_table(score_df: pd.DataFrame) -> dict[str, Any]:
    records = json.loads(score_df.to_json(orient="records"))
    scores: dict[str, Any] = {"table": records}
    if records:
        row = records[0]
        for key, value in row.items():
            if key != "split":
                scores[key] = value
    return scores


def _logicvista_label_tokens(value: Any) -> tuple[str, list[str]] | None:
    text = str(value or "").strip().upper()
    if text.endswith(".") and text.count(".") == 1:
        text = text[:-1].rstrip()
    compact = re.sub(r"[\s,;/]+", "", text)
    if re.fullmatch(r"[A-KZ]+", compact):
        return "letter", list(compact)
    if re.fullmatch(r"[1-9]+", compact):
        return "number", list(compact)
    return None


def _normalize_logicvista_judgement(
    judge_output: Any, answer: Any
) -> tuple[str, bool, bool] | None:
    extracted = _logicvista_label_tokens(judge_output)
    target = _logicvista_label_tokens(answer)
    if extracted is None or target is None:
        return None
    extracted_kind, extracted_tokens = extracted
    target_kind, target_tokens = target
    mapped = extracted_kind != target_kind
    if mapped and extracted_tokens == ["Z"]:
        normalized_tokens = ["Z"]
    elif extracted_kind == "number" and target_kind == "letter":
        normalized_tokens = [
            chr(ord("A") + int(token) - 1) for token in extracted_tokens
        ]
    elif extracted_kind == "letter" and target_kind == "number":
        if any(token == "Z" for token in extracted_tokens):
            normalized_tokens = ["Z"]
        else:
            normalized_tokens = [
                str(ord(token) - ord("A") + 1) for token in extracted_tokens
            ]
    else:
        normalized_tokens = extracted_tokens
    normalized = "".join(sorted(normalized_tokens))
    expected = "".join(sorted(target_tokens))
    return normalized, normalized == expected, mapped


_OFFICIAL_MATH_EXTRACTION_ROUTES = frozenset(
    {
        ("mathvision", "mathvision_qwen3_32b_extract.jsonl"),
        ("mathvista", "mathvista_qwen3_32b_extract.jsonl"),
        ("mathverse", "mathverse_qwen3_32b_extract.jsonl"),
    }
)


_MATHVERSE_SCORE_DECISION_RE = re.compile(
    r"(?m)^[ \t]*(?:"
    r"\*\*Judgement[ \t]*:[ \t]*([01])\*\*"
    r"|\*\*Judgement\*\*[ \t]*:[ \t]*(?:\*\*([01])\*\*|([01]))"
    r"|Judgement[ \t]*:[ \t]*(?:\*\*([01])\*\*|([01]))"
    r")(?=$|[ \t\r\n])"
)


def _parse_mathverse_score_output(value: Any) -> int | None:
    output = str(value or "").strip()
    if output in {"0", "1"}:
        return int(output)
    matches = list(_MATHVERSE_SCORE_DECISION_RE.finditer(output))
    if not matches:
        return None
    decisions = {
        int(next(group for group in match.groups() if group is not None))
        for match in matches
    }
    return decisions.pop() if len(decisions) == 1 else None


def _math_like_judge_cache_policy(
    spec_key: str,
    cache_name: str,
) -> tuple[Callable[[str], bool], str]:
    route = (spec_key, cache_name)
    policies: dict[tuple[str, str], tuple[Callable[[str], bool], str]] = {
        ("mathvision", "mathvision_qwen3_32b_extract.jsonl"): (
            _nonempty_judge_output,
            DIRECT_JUDGE_CACHE_CONTRACTS["mathvision_extract"],
        ),
        ("mathvista", "mathvista_qwen3_32b_extract.jsonl"): (
            _nonempty_judge_output,
            DIRECT_JUDGE_CACHE_CONTRACTS["mathvista_extract"],
        ),
        ("mathverse", "mathverse_qwen3_32b_extract.jsonl"): (
            _nonempty_judge_output,
            DIRECT_JUDGE_CACHE_CONTRACTS["mathverse_extract"],
        ),
        ("mathverse", "mathverse_qwen3_32b_score.jsonl"): (
            lambda output: _parse_mathverse_score_output(output) is not None,
            DIRECT_JUDGE_CACHE_CONTRACTS["mathverse_score"],
        ),
        ("logicvista", "logicvista_qwen3_32b_extract.jsonl"): (
            lambda output: _logicvista_label_tokens(output) is not None,
            DIRECT_JUDGE_CACHE_CONTRACTS["logicvista_extract"],
        ),
    }
    if route not in policies:
        raise RuntimeError(
            f"No judge cache contract registered for {spec_key}:{cache_name}"
        )
    return policies[route]


def _run_logicvista_judge_with_official_retries(
    *,
    judge: PersistentJudge,
    call_args: Any,
    prompts: list[tuple[str, str]],
    cache_name: str,
    max_tokens: int | None,
    top_p: float,
    contract_version: str,
) -> dict[str, dict[str, Any]]:
    """Apply VLMEvalKit's five-temperature LogicVista extraction schedule."""

    accepted: dict[str, dict[str, Any]] = {}
    pending = list(prompts)
    if not pending:
        return accepted

    cache_path = Path(cache_name)
    temperatures = (0.0, 0.5, 1.0, 1.5, 2.0)
    for attempt, temperature in enumerate(temperatures):
        attempt_cache = (
            cache_name
            if attempt == 0
            else f"{cache_path.stem}_retry_{attempt}{cache_path.suffix}"
        )
        judged = judge.run_cached(
            output_dir=call_args.output_dir,
            prompts=pending,
            cache_name=attempt_cache,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            no_resume=call_args.no_resume,
            desc=f"{call_args.dataset} local judge attempt {attempt + 1}",
            output_validator=_nonempty_judge_output,
            contract_version=(
                contract_version
                if attempt == 0
                else f"{contract_version}-official-retry{attempt}"
            ),
            retry_on_length=False,
            return_unresolved=True,
        )

        unresolved: list[tuple[str, str]] = []
        for index, prompt in pending:
            item = judged.get(str(index), {})
            if _logicvista_label_tokens(item.get("judge_output", "")) is None:
                unresolved.append((index, prompt))
                continue
            accepted[str(index)] = item
        pending = unresolved
        if not pending:
            return accepted

    failed = [str(index) for index, _ in pending]
    raise RuntimeError(
        "LogicVista judge produced malformed option extraction after five official "
        f"temperature attempts; indices={failed[:10]}"
    )


def _load_logicvista_official_retry_outputs(
    output_dir: Path,
    runner: Any,
    cache_name: str = "logicvista_qwen3_32b_extract.jsonl",
) -> dict[str, dict[str, Any]]:
    """Select the first valid output without mixing retry-cache identities."""

    cache_path = Path(cache_name)
    cache_names = [
        cache_name,
        *(
            f"{cache_path.stem}_retry_{attempt}{cache_path.suffix}"
            for attempt in range(1, 5)
        ),
    ]
    accepted: dict[str, dict[str, Any]] = {}
    for candidate_name in cache_names:
        candidate_path = output_dir / candidate_name
        if not candidate_path.exists():
            continue
        for index, item in runner.load_jsonl_by_index(candidate_path).items():
            index = str(index)
            if index in accepted:
                continue
            if _logicvista_label_tokens(item.get("judge_output", "")) is not None:
                accepted[index] = item
    return accepted


def _normalize_logicvista_option_summary(
    spec: BenchmarkSpec,
    model_path: str,
    output_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    if spec.key != "logicvista":
        return summary
    judged_table = output_dir / f"{spec.alias}_judged_qwen3_32b.xlsx"
    cache_path = output_dir / "logicvista_qwen3_32b_extract.jsonl"
    if not judged_table.exists() or not cache_path.exists():
        return summary

    runner, _ = _import_vlmeval_runner()
    from vlmeval.dataset.utils.logicvista import evaluate_logicvista
    from vlmeval.smp import dump, get_intermediate_file_path

    data = pd.read_excel(judged_table, keep_default_na=False)
    cached = _load_logicvista_official_retry_outputs(output_dir, runner)
    malformed: list[str] = []
    normalized_values: list[str] = []
    hits: list[int] = []
    mapped_rows = 0
    for _, row in data.iterrows():
        index = str(row.get("index"))
        judge_output = cached.get(index, {}).get("judge_output", "")
        normalized = _normalize_logicvista_judgement(
            judge_output, row.get("answer", "")
        )
        if normalized is None:
            malformed.append(index)
            normalized_values.append("")
            hits.append(0)
            continue
        value, hit, mapped = normalized
        normalized_values.append(value)
        hits.append(int(hit))
        mapped_rows += int(mapped)
    if malformed:
        raise RuntimeError(
            f"LogicVista judge produced {len(malformed)} malformed option extractions; "
            f"first indices={malformed[:10]}"
        )

    data["res"] = normalized_values
    data["log"] = "Succeed"
    data["hit"] = hits
    data.to_excel(judged_table, index=False)
    score_df = evaluate_logicvista(str(judged_table))
    dump(score_df, str(get_intermediate_file_path(str(judged_table), "_score", "csv")))
    records = json.loads(score_df.to_json(orient="records"))
    overall_rows = [row for row in records if row.get("Task&Skill") == "Overall"]
    if len(overall_rows) != 1:
        raise RuntimeError(
            f"LogicVista scorer returned no unique Overall row: {records}"
        )
    overall = float(overall_rows[0]["acc"])
    normalized_summary = {
        "dataset": spec.alias,
        "model": model_path,
        "rows": len(data),
        "score": overall,
        "scores": {"table": records, "Overall": overall},
        "artifacts": {
            **(summary.get("artifacts") or {}),
            "judged_table": str(judged_table),
            "logicvista_ordinal_label_mappings": mapped_rows,
            "logicvista_validated_judge_rows": int(len(data)),
        },
    }
    write_json(output_dir / "scores.json", normalized_summary)
    print(
        json.dumps(
            normalized_summary, indent=2, ensure_ascii=False, default=json_default
        )
    )
    return normalized_summary


def _normalize_mathverse_binary_judgement_summary(
    spec: BenchmarkSpec,
    model_path: str,
    output_dir: Path,
    summary: dict[str, Any],
) -> dict[str, Any]:
    if spec.key != "mathverse":
        return summary
    judged_table = output_dir / f"{spec.alias}_judged_qwen3_32b.xlsx"
    if not judged_table.exists():
        return summary

    from vlmeval.dataset.utils.mathverse import MathVerse_acc
    from vlmeval.smp import dump, get_intermediate_file_path

    # Preserve literal judge answers such as "None". Pandas otherwise coerces
    # those Excel strings to NaN and reports a successful extraction as empty.
    df = pd.read_excel(judged_table, keep_default_na=False)
    if "score" not in df or "log_score" not in df:
        return summary

    missing_extract = [
        str(row.get("index"))
        for _, row in df.iterrows()
        if pd.isna(row.get("extract")) or not str(row.get("extract", "")).strip()
    ]
    if missing_extract:
        raise RuntimeError(
            f"MathVerse judge produced {len(missing_extract)} empty extractions; first indices={missing_extract[:10]}"
        )

    normalized_scores: list[bool] = []
    changed = 0
    malformed: list[str] = []
    for _, row in df.iterrows():
        current = _truthy_score(row["score"])
        log_score = str(row.get("log_score", ""))
        if log_score == "Prefetch succeed":
            normalized_scores.append(True)
            continue
        prefix = "Judge output:"
        raw_decision = (
            log_score[len(prefix) :].strip() if log_score.startswith(prefix) else ""
        )
        decision = _parse_mathverse_score_output(raw_decision)
        if decision is None:
            malformed.append(str(row.get("index")))
            normalized_scores.append(current)
            continue
        fixed = decision == 1
        if fixed != current:
            changed += 1
        normalized_scores.append(fixed)

    if malformed:
        raise RuntimeError(
            f"MathVerse judge produced {len(malformed)} malformed score decisions; first indices={malformed[:10]}"
        )

    df["score"] = normalized_scores
    df.to_excel(judged_table, index=False)
    score_df = MathVerse_acc(str(judged_table))
    dump(score_df, str(get_intermediate_file_path(str(judged_table), "_score", "csv")))
    scores = _normalize_mathverse_score_table(score_df)
    normalized_summary = {
        "dataset": spec.alias,
        "model": model_path,
        "rows": len(df),
        "score": float(scores.get("Overall", 0.0)),
        "scores": scores,
        "artifacts": {
            **(summary.get("artifacts") or {}),
            "judged_table": str(judged_table),
            "mathverse_normalized_binary_judgement_rows": changed,
            "mathverse_validated_judge_rows": int(len(df)),
        },
    }
    write_json(output_dir / "scores.json", normalized_summary)
    print(
        json.dumps(
            normalized_summary, indent=2, ensure_ascii=False, default=json_default
        )
    )
    return normalized_summary


def _validate_math_like_judged_table(spec: BenchmarkSpec, output_dir: Path) -> None:
    judged_table = output_dir / f"{spec.alias}_judged_qwen3_32b.xlsx"
    if not judged_table.exists():
        raise FileNotFoundError(judged_table)
    # Preserve literal judge answers such as "None". They are valid non-empty
    # extraction outputs even when the benchmark scorer later marks them wrong.
    data = pd.read_excel(judged_table, keep_default_na=False)
    if spec.key in {"mathvision", "mathvista"}:
        bad = data["res"].isna() | data["res"].astype(str).str.strip().eq("")
        if bad.any():
            indices = data.loc[bad, "index"].astype(str).tolist()
            raise RuntimeError(
                f"{spec.display} judge produced {len(indices)} empty extractions; first indices={indices[:10]}"
            )
    elif spec.key == "logicvista":
        values = data["res"].fillna("").astype(str).str.strip().str.upper()
        bad = values.eq("") | ~values.map(
            lambda value: set(value) <= set("ABCDEFGHIJKZ123456789")
        )
        if bad.any():
            indices = data.loc[bad, "index"].astype(str).tolist()
            raise RuntimeError(
                f"LogicVista judge produced {len(indices)} malformed option extractions; first indices={indices[:10]}"
            )


def _run_local_math_like(
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    model_path: str,
    output_dir: Path,
    judge: PersistentJudge,
) -> dict[str, Any]:
    runner, _ = _import_vlmeval_runner()
    ns = _namespace_for_spec(args, spec, output_dir, model_path)
    old = runner._run_local_text_judge

    def persistent_text_judge(
        call_args, *, prompts, cache_name, max_tokens=None, temperature=0.0, top_p=1.0
    ):
        output_validator, contract_version = _math_like_judge_cache_policy(
            spec.key, cache_name
        )
        if (
            spec.key == "logicvista"
            and cache_name == "logicvista_qwen3_32b_extract.jsonl"
        ):
            return _run_logicvista_judge_with_official_retries(
                judge=judge,
                call_args=call_args,
                prompts=prompts,
                cache_name=cache_name,
                max_tokens=max_tokens,
                top_p=top_p,
                contract_version=contract_version,
            )
        if spec.key == "mathverse" and cache_name == "mathverse_qwen3_32b_score.jsonl":
            accepted: dict[str, dict[str, Any]] = {}
            pending = list(prompts)
            if not pending:
                return accepted
            cache_path = Path(cache_name)
            for attempt in range(5):
                attempt_cache = (
                    cache_name
                    if attempt == 0
                    else f"{cache_path.stem}_retry_{attempt}{cache_path.suffix}"
                )
                judged = judge.run_cached(
                    output_dir=call_args.output_dir,
                    prompts=pending,
                    cache_name=attempt_cache,
                    max_tokens=max_tokens,
                    temperature=attempt * 0.5,
                    top_p=top_p,
                    no_resume=call_args.no_resume,
                    desc=f"{call_args.dataset} local judge score attempt {attempt + 1}",
                    output_validator=_nonempty_judge_output,
                    contract_version=f"{contract_version}-attempt{attempt + 1}",
                )
                unresolved = []
                for index, prompt in pending:
                    item = judged.get(str(index), {})
                    if output_validator(item.get("judge_output", "")):
                        accepted[str(index)] = item
                    else:
                        unresolved.append((index, prompt))
                pending = unresolved
                if not pending:
                    return accepted
            failed = [str(index) for index, _ in pending]
            raise RuntimeError(
                f"MathVerse judge produced non-0/1 output after 5 attempts; indices={failed[:10]}"
            )
        return judge.run_cached(
            output_dir=call_args.output_dir,
            prompts=prompts,
            cache_name=cache_name,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            no_resume=call_args.no_resume,
            desc=f"{call_args.dataset} local judge",
            output_validator=output_validator,
            contract_version=contract_version,
            retry_on_length=(spec.key, cache_name)
            not in _OFFICIAL_MATH_EXTRACTION_ROUTES,
        )

    runner._run_local_text_judge = persistent_text_judge
    try:
        mode = local_judge_eval_mode(spec)
        if mode == "mathv_local_judge":
            summary = runner.run_mathv_local_judge(ns)
        elif mode == "mathvista_local_judge":
            summary = runner.run_mathvista_local_judge(ns)
        elif mode == "mathverse_local_judge":
            summary = runner.run_mathverse_local_judge(ns)
            summary = _normalize_mathverse_binary_judgement_summary(
                spec, model_path, output_dir, summary
            )
        elif mode == "logicvista_local_judge":
            summary = runner.run_logicvista_local_judge(ns)
            summary = _normalize_logicvista_option_summary(
                spec, model_path, output_dir, summary
            )
        else:
            raise ValueError(
                f"Unsupported math-like local judge mode for {spec.key}: {mode}"
            )
        _validate_math_like_judged_table(spec, output_dir)
        return summary
    finally:
        runner._run_local_text_judge = old


def _run_charxiv_local_judge(
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    model_path: str,
    output_dir: Path,
    judge: PersistentJudge,
) -> dict[str, Any]:
    runner, _ = _import_vlmeval_runner()
    from vlmeval.dataset import build_dataset
    from vlmeval.dataset.charxiv import qid2category
    from vlmeval.smp import load

    pred_table = output_dir / f"{spec.alias}_predictions.xlsx"
    if not pred_table.exists():
        raise FileNotFoundError(pred_table)
    data = load(str(pred_table))
    _ = build_dataset(spec.alias)
    judge_jsonl = output_dir / "judge_qwen3_32b.jsonl"
    if args.no_resume and judge_jsonl.exists():
        judge_jsonl.unlink()
    existing = {} if args.no_resume else runner.load_jsonl_by_index(judge_jsonl)
    print(
        f"[charxiv judge] dataset={spec.alias} rows={len(data)} existing={len(existing)}"
    )
    prompts: list[tuple[str, str]] = []
    for _, row in data.iterrows():
        prompts.append(
            (
                str(row["index"]),
                str(row["grading_query"]).replace(
                    "{PREDICTION}", str(row["prediction"])
                ),
            )
        )
    raw = judge.run_cached(
        output_dir=output_dir,
        prompts=prompts,
        cache_name="judge_qwen3_32b.jsonl",
        max_tokens=args.judge_max_tokens,
        temperature=0.0,
        top_p=1.0,
        no_resume=False,
        desc=f"{spec.alias} judge",
        output_validator=lambda output: _parse_charxiv_score_output(output) is not None,
        contract_version=DIRECT_JUDGE_CACHE_CONTRACTS["charxiv_judge"],
    )
    judged_map = {**existing, **raw}
    prediction_by_index = {
        str(row["index"]): row.get("prediction") for _, row in data.iterrows()
    }
    malformed: list[str] = []
    for idx, item in list(judged_map.items()):
        output = str(item.get("judge_output", ""))
        parsed_score = item.get("score")
        if parsed_score is None:
            parsed_score = _parse_charxiv_score_output(output)
        try:
            score = float(parsed_score)
        except (TypeError, ValueError):
            malformed.append(str(idx))
            continue
        if not 0.0 <= score <= 1.0:
            malformed.append(str(idx))
            continue
        parsed = runner._parse_charxiv_judge(output)
        extract_answer, extraction_method = _resolve_charxiv_extracted_answer(
            item,
            parsed,
            prediction_by_index.get(str(idx)),
        )
        if not extract_answer and extraction_method != "empty_prediction":
            malformed.append(str(idx))
            continue
        if extraction_method == "empty_prediction" and score != 0.0:
            malformed.append(str(idx))
            continue
        item["score"] = score
        item["extract_answer"] = extract_answer
        item["extract_answer_method"] = extraction_method
        judged_map[idx] = item
    if malformed:
        raise RuntimeError(
            f"CharXiv judge produced {len(malformed)} malformed decisions; first indices={malformed[:10]}"
        )
    data["score"] = [float(judged_map[str(x)]["score"]) for x in data["index"]]
    data["extract_answer"] = [
        judged_map[str(x)]["extract_answer"] for x in data["index"]
    ]
    judged_xlsx = output_dir / f"{spec.alias}_judged_qwen3_32b.xlsx"
    data.to_excel(judged_xlsx, index=False)

    mode = "descriptive" if "descriptive" in spec.alias else "reasoning"
    category_map, index_col = qid2category(mode)
    buckets: dict[str, list[float]] = defaultdict(list)
    for _, row in data.iterrows():
        buckets[str(category_map[row[index_col]])].append(float(row["score"]))
    scores = {k: sum(v) / len(v) for k, v in sorted(buckets.items())}
    scores["Overall"] = sum(float(x) for x in data["score"]) / len(data)
    summary = {
        "dataset": spec.alias,
        "model": model_path,
        "judge_model": args.judge_model,
        "rows": len(data),
        "score": _charxiv_primary_score(scores),
        "scores": scores,
        "judge": {
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": args.judge_max_tokens,
            "thinking": "disabled via chat template enable_thinking=False when supported",
        },
        "artifacts": {
            "prediction_table": str(pred_table),
            "judge_jsonl": str(judge_jsonl),
            "judged_xlsx": str(judged_xlsx),
        },
    }
    write_json(output_dir / "scores.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    return summary


def _run_evochart_local_score(
    spec: BenchmarkSpec,
    model_path: str,
    output_dir: Path,
) -> dict[str, Any]:
    _import_vlmeval_runner()
    from vlmeval.smp import get_intermediate_file_path

    pred_table = output_dir / f"{spec.alias}_predictions.xlsx"
    if not pred_table.exists():
        raise FileNotFoundError(pred_table)
    dataset = build_vlmeval_dataset(spec)
    result = dataset.evaluate(str(pred_table))
    rows = json.loads(result.to_json(orient="records"))
    overall_rows = [row for row in rows if row.get("split") == "Overall"]
    if len(overall_rows) != 1:
        raise RuntimeError(f"EvoChart scorer returned no unique Overall row: {rows}")
    overall = float(overall_rows[0]["acc"])
    judged_xlsx = Path(get_intermediate_file_path(str(pred_table), "_results"))
    score_csv = Path(get_intermediate_file_path(str(pred_table), "_acc", "csv"))
    summary = {
        "dataset": spec.alias,
        "model": model_path,
        "run_name": spec.run_name,
        "harness": "Local deterministic EvoChart extension scorer",
        "rows": int(overall_rows[0]["tot"]),
        "score": overall,
        "scores": {"Overall": overall, "table": rows},
        "artifacts": {
            "prediction_table": str(pred_table),
            "judged_table": str(judged_xlsx),
            "score_csv": str(score_csv),
        },
    }
    write_json(output_dir / "scores.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=json_default))
    return summary


def _run_score_for_spec(
    args: argparse.Namespace,
    spec: BenchmarkSpec,
    model_path: str,
    model_slug: str,
    judge: PersistentJudge,
) -> dict[str, Any]:
    output_dir = run_dir(spec, model_slug, args.run_root)
    mode = local_judge_eval_mode(spec)
    if args.run_set == "trace_eval_v1" and spec.key in LLM_EXTRACT_SCORE_KEYS:
        raise RuntimeError(
            f"{spec.display} requires the official saved-workbook route; "
            "the generic direct scorer is intentionally disabled for this benchmark"
        )
    if args.run_set == "trace_eval_v1" and spec.key in DEDICATED_SCORE_KEYS:
        raise RuntimeError(
            f"{spec.display} requires run_mme_reasoning_eval.py; "
            "the generic direct scorer is intentionally disabled for this benchmark"
        )
    if spec.key == "tablevqabench":
        summary = _run_tablevqabench_local_score(spec, model_path, output_dir)
    elif spec.key == "phyx_mini_mc":
        summary = _run_phyx_option_score(args, spec, model_path, output_dir)
    elif spec.key == "evochart":
        summary = _run_evochart_local_score(spec, model_path, output_dir)
    elif mode == "charxiv_local_judge":
        summary = _run_charxiv_local_judge(args, spec, model_path, output_dir, judge)
    elif mode in {
        "mathv_local_judge",
        "mathvista_local_judge",
        "mathverse_local_judge",
        "logicvista_local_judge",
    }:
        summary = _run_local_math_like(args, spec, model_path, output_dir, judge)
    elif mode == "wemath_local_judge":
        summary = _run_wemath_subset_score(args, spec, model_path, output_dir)
    else:
        summary = _run_direct_vlmeval(args, spec, model_path, output_dir)
    dst = _copy_score_to_benchmark(spec, model_slug, output_dir, args.benchmark_root)
    summary["benchmark_score_path"] = str(dst)
    archive_paths = _archive_direct_score_slices(
        args,
        spec,
        model_path,
        model_slug,
        output_dir,
        summary,
    )
    if any(archive_paths):
        summary["archive_descriptors"] = [
            str(path) for path in archive_paths if path is not None
        ]
        write_json(output_dir / "scores.json", summary)
        shutil.copy2(output_dir / "scores.json", dst)
    return summary


def run_worker(args: argparse.Namespace) -> None:
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", args.attention_backend)
    os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")

    specs = benchmark_specs_for_run_set(args.run_set, model_slug=args.model_slug)
    specs = filter_benchmark_specs(specs, only=args.only, exclude=args.exclude)
    if args.run_set == "trace_eval_v1":
        unsupported = [spec.key for spec in specs if spec.key not in DIRECT_SCORE_KEYS]
        if args.only and unsupported:
            raise ValueError(
                "The trace_eval_v1 direct scorer only accepts DIRECT_SCORE_KEYS: "
                f"{unsupported}"
            )
        specs = [spec for spec in specs if spec.key in DIRECT_SCORE_KEYS]
        print(
            "[score-worker:trace-eval-route] "
            f"direct={len(specs)} llm_extract={len(LLM_EXTRACT_SCORE_KEYS)} "
            f"dedicated={len(DEDICATED_SCORE_KEYS)}"
        )
    queue_path = (
        args.queue_root
        / f"score_{args.queue_name or args.model_slug + '_' + args.run_set}.json"
    )
    jobs = [
        (spec.key, score_path(spec, args.model_slug, args.benchmark_root))
        for spec in specs
    ]
    print(
        "[score-worker:init] "
        f"model={args.model} slug={args.model_slug} gpu={os.environ.get('CUDA_VISIBLE_DEVICES', '')} "
        f"run_set={args.run_set} jobs={len(jobs)} queue={queue_path}"
    )
    judge = PersistentJudge(args)
    try:
        while True:
            job_id = claim_next_job(
                queue_path=queue_path,
                jobs=jobs,
                worker_id=args.worker_id,
                stale_after_sec=args.stale_after_sec,
                max_attempts=args.max_attempts,
            )
            if job_id is None:
                print("[score-worker:done] no remaining score jobs")
                break
            spec = spec_by_key(job_id)
            try:
                print(f"[score-worker:claim] {job_id}")
                summary = _run_score_for_spec(
                    args, spec, args.model, args.model_slug, judge
                )
                mark_job(
                    queue_path,
                    job_id,
                    "done",
                    worker=args.worker_id,
                    output_dir=str(run_dir(spec, args.model_slug, args.run_root)),
                    rows=summary.get("rows"),
                )
            except Exception as exc:
                mark_job(
                    queue_path,
                    job_id,
                    "failed",
                    worker=args.worker_id,
                    output_dir=str(run_dir(spec, args.model_slug, args.run_root)),
                    error=repr(exc),
                )
                if args.stop_on_error:
                    raise
                print(
                    f"[score-worker:error] {job_id} failed and will be skipped by this worker: {exc!r}",
                    file=sys.stderr,
                )
    finally:
        judge.cleanup()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument(
        "--run-set",
        choices=BENCHMARK_RUN_SETS,
        default="trace_eval_v1",
    )
    parser.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    parser.add_argument("--worker-id", default=f"score-{os.getpid()}")
    parser.add_argument("--queue-name", default="")
    parser.add_argument(
        "--seed", type=int, default=int(os.environ.get("TRACE_EVAL_SEED", "42"))
    )
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument(
        "--run-root",
        type=Path,
        default=REPO_ROOT / "rlvr" / "evaluation" / ".work" / "runs",
    )
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--eval-judge-model", default="exact_matching")
    parser.add_argument("--eval-nproc", type=int, default=16)
    parser.add_argument("--judge-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--judge-batch-size", type=int, default=1024)
    parser.add_argument("--judge-tensor-parallel-size", type=int, default=1)
    parser.add_argument("--judge-gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--judge-max-model-len", type=int, default=8192)
    parser.add_argument("--judge-max-num-seqs", type=int, default=1024)
    parser.add_argument("--judge-max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--judge-max-tokens", type=int, default=256)
    parser.add_argument("--judge-api-base", action="append", dest="judge_api_bases")
    parser.add_argument("--judge-api-model", default="qwen3-32b-judge")
    parser.add_argument("--judge-api-tokenizer-model", default="Qwen/Qwen3-32B")
    parser.add_argument("--judge-api-parallelism", type=int, default=128)
    parser.add_argument(
        "--judge-api-batch-size",
        type=int,
        default=1,
        help="Prompts per /completions request; one preserves the historical request shape.",
    )
    parser.add_argument("--judge-api-batches-per-endpoint", type=int, default=1)
    parser.add_argument("--judge-api-max-batch-chars", type=int, default=100_000)
    parser.add_argument("--judge-api-timeout", type=float, default=120.0)
    parser.add_argument("--judge-api-max-retries", type=int, default=5)
    parser.add_argument(
        "--judge-cache-contract-version",
        default=PERSISTENT_JUDGE_CACHE_CONTRACT_VERSION,
    )
    parser.add_argument("--attention-backend", default="FLASH_ATTN")
    parser.add_argument("--stale-after-sec", type=float, default=900)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    run_worker(args)


if __name__ == "__main__":
    main()
