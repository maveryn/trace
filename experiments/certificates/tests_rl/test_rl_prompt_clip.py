"""Prompt clip ratio 0: the fast length arithmetic equals EasyR1's processor path, and every frozen RL /
val prompt fits max_prompt_length (EasyR1 right-truncates silently)."""

from __future__ import annotations

import io

import pytest


def png(width, height):
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (width, height), (240, 240, 240)).save(buf, format="PNG")
    return {"bytes": buf.getvalue(), "path": None}


@pytest.mark.parametrize("size", [(450, 450), (300, 900), (1280, 640), (2500, 2500), (919, 920), (513, 511)])
def test_resize_matches_process_image(trl, specs, rlvr_env, size):
    paths = trl.Paths(specs["3b"])
    with trl.backend_on_path(paths.backend):
        from verl.utils.dataset import process_image
    release = trl.release_easyr1(specs["3b"], paths)
    lo, hi = release["data"]["min_pixels"], release["data"]["max_pixels"]
    image = process_image(png(*size), lo, hi)
    assert trl.easyr1_resized(*size, lo, hi) == image.size


def test_fast_length_equals_exact_processor_path(trl, specs, releases, base_snapshot):
    spec, release = specs["3b"], releases["3b"]
    paths = trl.Paths(spec)
    meter = trl.PromptMeter(base_snapshot, paths.backend, prompt_key="prompt_cert", system_prompt_file=trl.SYSTEM_PROMPT,
                            min_pixels=release["data"]["min_pixels"], max_pixels=release["data"]["max_pixels"],
                            max_prompt_length=3072)
    prompt = ("<image>The figure shows a lever. What is the torque on the left side?\nImage size: 300x900 pixels. "
              'End your response with one JSON object with exactly the keys "annotation", "certificate" and "answer".')
    for size in [(450, 450), (300, 900), (1280, 640), (2500, 2500), (919, 920)]:
        row = {"prompt_cert": prompt, "images": [png(*size)], "answer_gt": "{}", "instance_id": "x"}
        fast = meter.fast(prompt, [size])
        full, kept = meter.exact(row)
        assert fast == full, size
        assert kept == min(full, 3072)
    # a prompt longer than the cap is cut by EasyR1 (the reason the check exists)
    long_prompt = prompt + " filler" * 3000
    row = {"prompt_cert": long_prompt, "images": [png(640, 640)], "answer_gt": "{}", "instance_id": "x"}
    full, kept = meter.exact(row)
    assert full > 3072 and kept == 3072


def test_frozen_rl_prompts_fit(trl, specs, releases, data_ready, base_snapshot):
    spec = specs["3b"]
    paths = trl.Paths(spec)
    data = trl.check_data(spec, paths, trl.ShaCache(paths.cache / "sha256.json"))
    result = trl.check_prompt_clip(spec, releases["3b"], paths, data, base_snapshot)
    assert result["ok"] and result["max"] <= 3072
    for role in ("train", "val"):
        assert result[role]["over_limit"] == 0 and result[role]["at_limit"] == 0 and result[role]["clip_ratio"] == 0.0
        assert result[role]["rows"] == data[role]["rows"]
        assert result[role]["image_size_mismatch_vs_image_sizes"] == 0
