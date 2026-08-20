"""Evaluate a HuggingFace-format Molmo2 checkpoint on a registered task.

    python launch_scripts/hf_eval.py <hf_model_dir> <task>

Runs vLLM inference over the task's dataset and then scores the result with the
task's own evaluators, writing predictions.json and metrics.json side by side.

Examples:
    python launch_scripts/hf_eval.py Molmo2-8B-HF cfc_hf_track_eval_2fps
    python launch_scripts/hf_eval.py Molmo2-8B-HF \\
        cfc_hf_synthetic_correction_full_eval_2fps --max_examples 16
    # score a predictions.json produced earlier, no GPU needed
    python launch_scripts/hf_eval.py - cfc_hf_track_eval_2fps \\
        --predictions results/cfc_hf_track_eval_2fps/validation/predictions.json
"""
import argparse
import logging
import os
from os.path import join

# vLLM env has to be set before the import below pulls vllm in
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")   # deterministic scheduling
os.environ.setdefault("VLLM_VIDEO_LOADER_BACKEND", "molmo2")

from olmo.data.get_dataset import get_dataset_by_name
from olmo.eval.standalone_eval import run_eval
from olmo.util import prepare_cli_environment

log = logging.getLogger(__name__)

SPLIT_PREFERENCE = ["validation", "test", "train"]


def resolve_split(task):
    """Pick a default split from the task's dataset class SPLIT_MAP."""
    from olmo.data.cfc_hf_datasets import CFC_HF_DATASETS
    if task not in CFC_HF_DATASETS:
        raise SystemExit(
            f"--split is required for {task!r}: only the CFC hub tasks publish a "
            f"SPLIT_MAP this can read")
    cls, _ = CFC_HF_DATASETS[task]
    for split in SPLIT_PREFERENCE:
        if split in cls.SPLIT_MAP:
            return split
    raise SystemExit(f"{task!r} has no split among {SPLIT_PREFERENCE}: "
                     f"{sorted(cls.SPLIT_MAP)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run a Molmo2 HF checkpoint on a task and score it")
    parser.add_argument("model_dir", help="HF-format model directory (use '-' with --predictions)")
    parser.add_argument("task", help="Registered task name, e.g. cfc_hf_track_eval_2fps")
    parser.add_argument("--split", default=None,
                        help=f"Dataset split (default: first of {SPLIT_PREFERENCE} the task has)")
    parser.add_argument("--save_dir", default=None,
                        help="Output directory (default: results/<task>/<split>)")
    parser.add_argument("--predictions", default=None,
                        help="Score this predictions.json instead of generating")
    parser.add_argument("--no_eval", action="store_true",
                        help="Generate predictions but skip scoring")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite an existing metrics.json")
    parser.add_argument("--resume", action="store_true",
                        help="Skip examples already in the output predictions file")
    parser.add_argument("--max_examples", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=4,
                        help="Examples per vLLM generate call")
    parser.add_argument("--max_tokens", type=int, default=38000,
                        help="Generation budget. CFC track outputs are long (a dense nusagak "
                             "clip runs past 15k tokens), and truncation silently costs HOTA, "
                             "so this is a high ceiling rather than a typical length")
    parser.add_argument("--max_fps", type=float, default=None,
                        help="Override video max_fps (default: from the processor config)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--use_float32", action="store_true")
    parser.add_argument("--shard_index", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=None)
    args = parser.parse_args()

    if (args.shard_index is None) != (args.num_shards is None):
        parser.error("--shard_index and --num_shards must be used together")

    prepare_cli_environment()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    split = args.split or resolve_split(args.task)
    save_dir = args.save_dir or join("results", args.task, split)

    if args.predictions:
        run_eval(args.predictions, args.task, split, overwrite=args.overwrite,
                 out_dir=args.save_dir)
        return

    import torch
    from transformers import AutoProcessor
    from vllm import LLM
    from vllm.sampling_params import SamplingParams

    from olmo.eval.vllm_runner import build_examples, filter_done, generate_predictions

    log.info(f"Loading dataset: {args.task} split={split}")
    dataset = get_dataset_by_name(args.task, split)
    log.info(f"Dataset size: {len(dataset)}")

    examples = build_examples(dataset, max_examples=args.max_examples,
                              shard_index=args.shard_index, num_shards=args.num_shards)
    log.info(f"Total examples: {len(examples)}")

    name = ("predictions.json" if args.num_shards is None
            else f"predictions_shard{args.shard_index}.json")
    output_path = join(save_dir, name)

    existing = []
    if args.resume:
        examples, existing = filter_done(examples, output_path)

    if examples:
        log.info(f"Loading model from {args.model_dir}")
        llm = LLM(
            model=args.model_dir,
            trust_remote_code=True,
            tensor_parallel_size=torch.cuda.device_count(),
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="float32" if args.use_float32 else "bfloat16",
            limit_mm_per_prompt={"image": 6, "video": 1},
            max_num_batched_tokens=36864,
        )
        processor = AutoProcessor.from_pretrained(
            args.model_dir, trust_remote_code=True, dtype="auto",
            device_map="auto", padding_side="left")

        generate_predictions(
            llm, processor, examples, output_path,
            SamplingParams(max_tokens=args.max_tokens, temperature=0),
            chunk_size=args.chunk_size, max_fps_override=args.max_fps,
            existing=existing)
    else:
        log.info("No examples left to process")

    if args.no_eval:
        return
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()

    run_eval(output_path, args.task, split, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
