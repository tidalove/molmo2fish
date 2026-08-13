"""vLLM batch inference over a Molmo2 dataset.

Turns dataset examples into vLLM inputs, runs them in chunks, and writes a
predictions.json. The interesting part is multi-turn: a correction example is a
whole conversation (pre-correction tracks, a correction instruction, the repaired
tracks, possibly several rounds of that), and it has to be rendered with exactly
the turn structure training used or the model sees a prompt shape it was never
trained on. build_multi_turn_chat does that by running every turn back through
the training DataFormatter.

Entry point: launch_scripts/hf_eval.py.
"""
import json
import logging
import os
import random
import re
from pathlib import Path

from olmo.preprocessing.data_formatter import (
    GENERAL_PROMPTS_V1, apply_keyword_prompt, build_prompt_for_inference,
)

log = logging.getLogger(__name__)

# Mirrors the released Molmo2 checkpoint's training config. The prompt text the
# model was tuned on depends on all of these, so they are not free knobs.
MOLMO2_FORMATTER_KWARGS = dict(
    prompt_templates="uber_model_v2",
    message_format="qwen3",
    system_prompt="demo_or_style_v2",
    pointing_format="html-v2",
    points_decimal_places=1,
    timestamp_mode="50-percent-seconds",
    output_timestamp_mode="seconds",
    seconds_decimal_places=1,
    debug=True,
)

_TEXT_FORMATTER = None


def get_text_formatter():
    """Lazily build the DataFormatter matching the released model's training config."""
    global _TEXT_FORMATTER
    if _TEXT_FORMATTER is None:
        from olmo.preprocessing.data_formatter import DataFormatter
        _TEXT_FORMATTER = DataFormatter(**MOLMO2_FORMATTER_KWARGS)
    return _TEXT_FORMATTER


def build_multi_turn_chat(raw, video_path=None, max_frames=None,
                          frame_sample_mode=None, max_fps=None, sampling_fps=None):
    """Render a multi_turn_messages example (the correction format) into a chat list.

    Each turn goes through DataFormatter.get_user_prompt, which yields the same
    (prompt, response) pair training would have produced for it — so the closed
    turns carry the model's own track syntax rather than a paraphrase. Non-final
    turns emit user+assistant; the final turn emits user only, leaving the model
    to generate the corrected tracks.

    The video is attached to the first user turn only, matching get_message()'s
    (text, video) content ordering. Later turns are text: the model is expected
    to still have the video in context, exactly as in training.
    """
    formatter = get_text_formatter()
    rng = random.Random(0)          # deterministic prompt variation across runs
    turns = raw["multi_turn_messages"]
    chat = []
    for i, turn in enumerate(turns):
        prompt, response, _ = formatter.get_user_prompt(
            turn, is_training=False, for_inference=False, rng=rng)
        content = [dict(type="text", text=prompt, style=turn.get("style", "demo"))]
        if i == 0 and video_path is not None:
            content.append(dict(type="video", video=video_path,
                                **_video_kwargs(max_frames, frame_sample_mode,
                                                max_fps, sampling_fps)))
        chat.append({"role": "user", "content": content})
        if i < len(turns) - 1:
            chat.append({"role": "assistant",
                         "content": [dict(type="text", text=response)]})
    return chat


def _video_kwargs(max_frames, frame_sample_mode, max_fps, sampling_fps):
    kwargs = {"max_frames": max_frames, "frame_sample_mode": frame_sample_mode}
    if max_fps is not None:
        kwargs["max_fps"] = max_fps
    if sampling_fps is not None:
        kwargs["sampling_fps"] = sampling_fps
    return kwargs


def get_message(images=None, video_path=None, max_frames=384, frame_sample_mode="fps",
                max_fps=None, sampling_fps=None, input_text="", style="demo"):
    """Single-turn chat list: one user message with text, then any images/video."""
    content = [dict(type="text", text=input_text, style=style)]
    if images:
        content.extend(dict(type="image", image=image) for image in images)
    if video_path:
        content.append(dict(type="video", video=video_path,
                            **_video_kwargs(max_frames, frame_sample_mode,
                                            max_fps, sampling_fps)))
    return [{"role": "user", "content": content}]


def get_prompt_text(style, prompt_override, example=None):
    """Prompt text: explicit override > style template > the dataset example itself."""
    if prompt_override:
        return prompt_override
    if style:
        templates = GENERAL_PROMPTS_V1[style]
        keywords = sorted(re.findall("{([^{}]+)}", templates[0]))
        if keywords and example:
            return apply_keyword_prompt(templates, example, None, dbg=True)
        return templates[0]
    if example and 'message_list' in example:
        return build_prompt_for_inference(example['message_list'][0])
    if example and 'multi_turn_messages' in example:
        return build_prompt_for_inference(example['multi_turn_messages'][0])
    if example and 'question' in example:
        return example['question']
    raise ValueError("No prompt source: provide a style or an explicit prompt")


def build_examples(dataset=None, video_dir=None, max_examples=None,
                   shard_index=None, num_shards=None):
    """List of {raw, video, example_id} from a dataset, or of bare videos from a dir."""
    if video_dir:
        from glob import glob
        videos = sorted(glob(os.path.join(video_dir, "*.mp4")))
        if not videos:
            raise ValueError(f"No .mp4 files found in {video_dir}")
        examples = [{"video": v, "example_id": Path(v).stem} for v in videos]
    else:
        examples = []
        for i in range(len(dataset)):
            ex = dataset.get(i, None)
            examples.append({
                "raw": ex,
                "video": ex.get("video", None),
                "example_id": ex.get('metadata', {}).get('example_id', str(i)),
            })

    if max_examples:
        examples = examples[:max_examples]
    if num_shards is not None:
        # interleaved rather than contiguous, so shards get comparable work
        examples = examples[shard_index::num_shards]
        log.info(f"Shard {shard_index}/{num_shards}: {len(examples)} examples")
    return examples


def build_vllm_input(example, processor, style=None, prompt_override=None,
                     max_fps_override=None):
    """One vLLM input dict from one example."""
    from molmo_utils import process_vision_info

    raw = example.get("raw", None)
    prompt_text = get_prompt_text(style, prompt_override, raw)

    effective_style = style
    if not effective_style and raw:
        for key in ('message_list', 'multi_turn_messages'):
            if key in raw:
                effective_style = raw[key][0].get('style', 'demo')
                break
    effective_style = effective_style or 'demo'

    video_path = example.get("video", None)
    if video_path is None:
        # Text-only (e.g. cfc_hf_text): still needs the full multi-turn rendering
        if raw and "multi_turn_messages" in raw:
            messages = build_multi_turn_chat(raw)
        else:
            messages = [{"role": "user",
                         "content": [dict(type="text", text=prompt_text,
                                          style=effective_style)]}]
        return {
            "prompt": processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True),
            "multi_modal_data": {},
            "mm_processor_kwargs": {},
        }

    video_cfg = processor.video_processor
    max_fps = max_fps_override if max_fps_override is not None else video_cfg.max_fps
    sampling_fps = (raw or {}).get('sampling_fps')
    if sampling_fps is None:
        sampling_fps = video_cfg.sampling_fps

    shared = dict(max_frames=video_cfg.num_frames,
                  frame_sample_mode=video_cfg.frame_sample_mode,
                  max_fps=max_fps, sampling_fps=sampling_fps)
    if raw and "multi_turn_messages" in raw:
        messages = build_multi_turn_chat(raw, video_path=video_path, **shared)
    else:
        messages = get_message(video_path=video_path, input_text=prompt_text,
                               style=effective_style, **shared)

    images, videos_inputs, video_kwargs = process_vision_info(messages)
    multi_modal_data = {}
    if images:
        multi_modal_data["image"] = images
    if videos_inputs:
        multi_modal_data["video"] = videos_inputs

    return {
        "prompt": processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True),
        "multi_modal_data": multi_modal_data,
        "mm_processor_kwargs": video_kwargs,
    }


def collect_metadata(example):
    """Scalar metadata fields worth carrying into predictions.json."""
    result = {"example_id": example["example_id"], "video": example.get("video", None)}
    raw = example.get("raw")
    if raw:
        for k, v in raw.get("metadata", {}).items():
            if isinstance(v, (str, int, float, bool)):
                result[k] = v
    return result


def filter_done(examples, output_path):
    """Drop examples already present in an existing predictions file (--resume)."""
    if not os.path.exists(output_path):
        return examples, []
    with open(output_path) as f:
        existing = json.load(f)
    done_ids = {p["example_id"] for p in existing}
    remaining = [ex for ex in examples if ex["example_id"] not in done_ids]
    log.info(f"Resume: {len(examples) - len(remaining)} already done, "
             f"{len(remaining)} remaining")
    return remaining, existing


def generate_predictions(llm, processor, examples, output_path, sampling_params,
                         chunk_size=4, style=None, prompt_override=None,
                         max_fps_override=None, existing=()):
    """Generate over `examples` in chunks, saving after each chunk.

    Saving incrementally (rather than once at the end) is what makes --resume
    useful on a long run that gets pre-empted.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    all_predictions = list(existing)

    for chunk_start in range(0, len(examples), chunk_size):
        chunk = examples[chunk_start:chunk_start + chunk_size]
        chunk_idx = chunk_start // chunk_size + 1
        total_chunks = (len(examples) + chunk_size - 1) // chunk_size
        log.info(f"Processing chunk {chunk_idx}/{total_chunks} ({len(chunk)} examples)")

        vllm_inputs, valid_examples = [], []
        for ex in chunk:
            try:
                vllm_inputs.append(build_vllm_input(
                    ex, processor, style=style, prompt_override=prompt_override,
                    max_fps_override=max_fps_override))
                valid_examples.append(ex)
            except Exception as e:
                log.error(f"Failed to build input for {ex['example_id']}: {e}")
        if not vllm_inputs:
            continue

        outputs = llm.generate(vllm_inputs, sampling_params=sampling_params)

        for inp, ex, output in zip(vllm_inputs, valid_examples, outputs):
            result = collect_metadata(ex)
            result["prediction"] = output.outputs[0].text
            result["input"] = inp["prompt"]
            all_predictions.append(result)

        with open(output_path, "w") as f:
            json.dump(all_predictions, f, indent=2)
        log.info(f"Saved {len(all_predictions)} predictions to {output_path}")

    return all_predictions
