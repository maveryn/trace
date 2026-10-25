#!/usr/bin/env python
"""Check on the server that our tool messages render as intended for this checkpoint.

1. Renders tool conversations through the checkpoint's own processor
   (transformers ``apply_chat_template``):
   - Qwen3-VL: the native ``tools=`` + ``role: tool`` path and our flattened form
     must be identical.
   - Qwen3.5: the same, with XML tool calls (native: structured ``tool_calls``;
     ours: the text the model writes) and thinking off (``enable_thinking=False``).
     Two conversations: a tool body with whitespace at its edges (the template trims
     it), and a forced-mode one (budget statement in the system prompt, the
     calls-left notice, then the final-answer notice, each inside the last
     ``<tool_response>``).
   - Qwen2.5-VL (no tool support in its template): our form must equal Qwen-Agent's
     NousFnCallPrompt layout, which is what Qwen2.5-VL agents are prompted with.
2. Optionally (``--endpoint``) sends the flattened form, with the family's
   ``chat_template_kwargs``, to vLLM's ``/tokenize`` + ``/detokenize`` and compares
   with (1) after collapsing ``<|image_pad|>`` runs. Every case is checked. An HTTP
   4xx answer is a FAILURE (the server rejected the request; its body is printed),
   except a 404 that is not a vLLM error (the server has no such endpoint). That, a
   connection error or an HTTP 5xx skips the check for that case (SKIPPED, with the
   response body when there is one).

The processor renders deep copies: transformers' ``apply_chat_template`` rewrites
``image_url`` parts in place, and the server must get the OpenAI parts we send.
Exit status: 0 when every case is OK or SKIPPED, 1 on any MISMATCH or FAILURE.

    python scripts/check_template.py --model Qwen/Qwen2.5-VL-7B-Instruct
    python scripts/check_template.py --model Qwen/Qwen3.5-9B --revision c202236235762e1c871ad0ccb60c8ee5ba337b9a
    python scripts/check_template.py --model Qwen/Qwen2.5-VL-7B-Instruct \
        --endpoint http://127.0.0.1:18100 --served qwen2.5-vl-7b-instruct
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import urllib.error
import urllib.request

from PIL import Image

from trace_tool.models import FAMILIES, QWEN25_VL, chat_template_kwargs, detect_family, get_family
from trace_tool.prompting import (
    BUDGET_STATEMENT,
    CALLS_LEFT_NOTICE,
    FINAL_ANSWER_NOTICE,
    TOOLS_FOOTER,
    TOOLS_HEADER,
    encode_png_data_url,
    hf_tojson,
    system_message,
    text_part,
    tool_response_content,
    user_prompt_content,
)
from trace_tool.tools import build_tools


def _parts(parts, url):
    return [{"type": "text", "text": p["text"]} if p["type"] == "text" else {"type": "image_url", "image_url": {"url": url}} for p in parts]


def _post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.loads(response.read())


def _render(processor, messages, **kwargs) -> str:
    """apply_chat_template on a deep copy: it rewrites image_url parts in place."""
    return processor.apply_chat_template(copy.deepcopy(messages), tokenize=False, add_generation_prompt=True, **kwargs)


def _http_error(exc: urllib.error.HTTPError) -> tuple[bool, str]:
    """(is a failure, response body). A 4xx is a failure: the server rejected the request.
    The exception is a 404 that is not a vLLM error response (FastAPI answers an unknown
    route with {"detail": "Not Found"}): the server has no such endpoint. A 404 from vLLM
    itself, e.g. for an unknown --served name, is a failure."""
    try:
        body = exc.read().decode("utf-8", "replace")
    except Exception as read_error:  # e.g. the connection dropped mid-body
        body = f"(unreadable: {type(read_error).__name__}: {read_error})"
    try:
        payload = json.loads(body)
    except ValueError:
        payload = None
    vllm_error = isinstance(payload, dict) and ("error" in payload or payload.get("object") == "error")
    missing_route = exc.code == 404 and not vllm_error
    if len(body) > 4000:
        body = body[:4000] + f"... [{len(body) - 4000} more characters]"
    return 400 <= exc.code < 500 and not missing_route, body


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id or local path of the checkpoint")
    p.add_argument("--endpoint", help="vLLM server root, e.g. http://127.0.0.1:18100")
    p.add_argument("--served", help="served model name (for --endpoint)")
    p.add_argument("--revision", help="checkpoint revision")
    p.add_argument("--family", choices=sorted(FAMILIES), help="model family (default: detected from --model)")
    args = p.parse_args()
    family = get_family(args.family) if args.family else detect_family(args.model)
    template_kwargs = chat_template_kwargs(family) or {}  # Qwen3.5: {"enable_thinking": False}

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, revision=args.revision)
    image = Image.new("RGB", (320, 320), (200, 30, 30))
    url = encode_png_data_url(image)
    schemas = [t.schema() for t in build_tools(["zoom", "calculator"], family).values()]
    sys_text = "You are a helpful assistant."
    user = _parts(user_prompt_content('<image>Count the boxes.\n{"answer":2}', [(image, 0)]), url)
    zoom_body = [{"type": "image", "image": image, "index": 1}]

    # A case is (label, system prompt, turns); a turn is (assistant text as the model writes it,
    # the same assistant message for the native path, tool bodies).
    if family.tool_format == "qwen3_xml":
        a1 = (
            "<tool_call>\n<function=image_zoom_in_tool>\n<parameter=bbox_2d>\n[0, 0, 500, 500]\n</parameter>\n"
            "<parameter=label>\nbox\n</parameter>\n<parameter=img_idx>\n0\n</parameter>\n</function>\n</tool_call>"
        )
        native1 = {"role": "assistant", "content": "", "tool_calls": [
            {"type": "function", "function": {"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 500, 500], "label": "box", "img_idx": 0}}}]}
        a2 = "I will compute it.\n\n<tool_call>\n<function=calculator>\n<parameter=expression>\n6 * 7\n</parameter>\n</function>\n</tool_call>"
        native2 = {"role": "assistant", "content": "I will compute it.", "tool_calls": [
            {"type": "function", "function": {"name": "calculator", "arguments": {"expression": "6 * 7"}}}]}
        # forced mode with a 2-call budget, so that the second tool turn is the forcing one
        forced_system = sys_text + "\n\n" + BUDGET_STATEMENT.format(budget=2, per_turn=2)
        cases = [
            ("tool body with edge whitespace", sys_text, [(a1, native1, [zoom_body, [text_part(" \n42\n ")]])]),
            ("forced mode", forced_system, [
                (a1, native1, [zoom_body + [text_part("\n\n" + CALLS_LEFT_NOTICE.format(left=1))]]),
                (a2, native2, [[text_part("42"), text_part("\n\n" + FINAL_ANSWER_NOTICE)]]),
            ]),
        ]
    else:
        a1 = '<tool_call>\n{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 500, 500], "label": "box", "img_idx": 0}}\n</tool_call>'
        cases = [("", sys_text, [(a1, {"role": "assistant", "content": a1}, [zoom_body, [text_part("42")]])])]

    with_kwargs = "".join(f", with {k}={v!r}" for k, v in template_kwargs.items())
    rendered = []
    failed = False
    for label, system, turns in cases:
        where = f" ({label})" if label else ""
        native = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        ours = [system_message(system, schemas, family=family), {"role": "user", "content": user}]
        for text, native_assistant, bodies in turns:
            native.append(native_assistant)
            native += [{"role": "tool", "content": _parts(b, url)} for b in bodies]
            ours.append({"role": "assistant", "content": text})
            ours.append({"role": "user", "content": _parts(tool_response_content(bodies, family=family), url)})

        ours_text = _render(processor, ours, **template_kwargs)
        if family is QWEN25_VL:
            vision = "<|vision_start|><|image_pad|><|vision_end|>"
            block = TOOLS_HEADER + "".join("\n" + hf_tojson(x) for x in schemas) + TOOLS_FOOTER
            reference = (
                f"<|im_start|>system\n{sys_text}\n\n{block}<|im_end|>\n"
                f'<|im_start|>user\n{vision}Count the boxes.\n{{"answer":2}}<|im_end|>\n'
                f"<|im_start|>assistant\n{a1}<|im_end|>\n"
                f"<|im_start|>user\n<tool_response>\n{vision}\n</tool_response>\n<tool_response>\n42\n</tool_response><|im_end|>\n"
                "<|im_start|>assistant\n"
            )
            ref_label = "Qwen-Agent's NousFnCallPrompt layout"
        else:
            reference = _render(processor, native, tools=schemas, **template_kwargs)
            ref_label = "the native tools= / role: tool path"
        rendered.append((where, ours, ours_text))
        if reference != ours_text:
            print(f"MISMATCH{where} against {ref_label}\n--- reference\n" + reference + "\n--- ours\n" + ours_text)
            failed = True
            continue
        if template_kwargs and _render(processor, ours) == ours_text:
            print(f"MISMATCH{where}: the template ignores chat_template_kwargs={template_kwargs}")
            failed = True
            continue
        print(f"OK{where}: the processor renders our messages identically to {ref_label}{with_kwargs}")

    if args.endpoint:
        root = args.endpoint.rstrip("/").removesuffix("/v1")
        collapse = lambda s: re.sub(r"(<\|image_pad\|>)+", "<|image_pad|>", s)
        for where, ours, ours_text in rendered:
            payload = {"model": args.served, "messages": ours, "add_generation_prompt": True}
            if template_kwargs:
                payload["chat_template_kwargs"] = template_kwargs
            try:
                tokens = _post(f"{root}/tokenize", payload)["tokens"]
                server_text = _post(f"{root}/detokenize", {"model": args.served, "tokens": tokens})["prompt"]
            except urllib.error.HTTPError as exc:
                is_failure, body = _http_error(exc)
                if is_failure:
                    print(f"FAILURE server check{where}: {exc.url} answered HTTP {exc.code}\n--- response body\n{body}")
                    failed = True
                else:
                    print(f"SKIPPED server check{where} (HTTP {exc.code} from {exc.url}); compare a logged prompt manually if needed\n--- response body\n{body}")
                continue
            except Exception as exc:
                print(f"SKIPPED server check{where} ({type(exc).__name__}: {exc}); compare a logged prompt manually if needed")
                continue
            if collapse(server_text) != collapse(ours_text):
                print(f"MISMATCH{where} between vLLM rendering and processor rendering\n--- vllm\n" + server_text + "\n--- processor\n" + ours_text)
                failed = True
                continue
            print(f"OK{where}: vLLM renders the flattened messages identically")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
