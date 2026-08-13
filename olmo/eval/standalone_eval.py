"""Score an existing predictions.json with the task's own evaluators.

Decoupled from inference: given a predictions file and a task name, this rebuilds
the per-example metadata (GT points, masks, frame size, cadence) by instantiating
the dataset, matches it to the predictions by example_id, and runs whatever
evaluators olmo/eval/eval_utils.py:get_evaluator assigns to that task.

Nothing here reads a hand-built json: metadata comes from the dataset classes,
and the masks come from the MasksRLE/ tree that download() writes.
"""
import json
import logging
import os
from os.path import join

import torchmetrics

from olmo.eval.eval_utils import get_evaluator
from olmo.eval.evaluators import SavePredictions
from olmo.eval.object_tracking_utils import points_from_masks
from olmo.util import log_metrics_to_console

log = logging.getLogger(__name__)

CFC_VIDEO_FPS = 6


def build_metadata_from_masks_rle(masks_dir, video_fps=CFC_VIDEO_FPS, sampling_fps=None):
    """Per-example metadata straight from precomputed MasksRLE files.

    Walks {masks_dir}/{example_id}/*.json. GT points are bbox centroids of each
    RLE mask, so points and masks share object-slot order. Useful when the
    predictions cover examples that no longer resolve through a dataset class
    (a renamed task, a hand-assembled prediction set); the dataset path below is
    the normal one.
    """
    from glob import glob as _glob

    metadata_by_id = {}
    n_skipped = 0
    example_dirs = sorted(d for d in _glob(join(masks_dir, "*")) if os.path.isdir(d))

    for vdir in example_dirs:
        example_id = os.path.basename(vdir)
        for mask_file in sorted(_glob(join(vdir, "*.json"))):
            with open(mask_file) as f:
                masks = json.load(f)

            # Any non-None RLE gives the frame size
            sample_rle = next(
                (rle for frame_list in masks.values()
                 for rle in frame_list if rle is not None), None)
            if sample_rle is None:
                n_skipped += 1
                continue
            height, width = sample_rle['size']
            points = points_from_masks(masks, video_fps)

            metadata_by_id[example_id] = {
                'example_id': example_id,
                'w': width,
                'h': height,
                'video_fps': video_fps,
                'sampling_fps': sampling_fps,
                'video': example_id,
                'points': points,
                'initial_points': points,
                'masks': masks,
                'mask_id': [str(i) for i in range(len(masks))],
            }

    log.info(f"Built metadata for {len(metadata_by_id)} entries from {masks_dir} "
             f"({n_skipped} mask files skipped — all-None)")
    return metadata_by_id


def build_metadata_from_dataset(task, split):
    """Per-example metadata by instantiating the dataset class.

    Forces is_eval=True so the GT block (points/masks/mask_id) is attached on any
    split, including train — scoring a train split is legitimate, the hub carries
    masks for it, and the alternative is a silently empty metric set.
    """
    from olmo.data.get_dataset import get_dataset_by_name

    log.info(f"Loading dataset: {task} split={split}")
    try:
        dataset = get_dataset_by_name(task, split, is_eval=True)
    except TypeError:
        # datasets that don't take the kwarg derive is_eval from the split; it is
        # only read inside get(), so setting it here still takes effect
        dataset = get_dataset_by_name(task, split)
        dataset.is_eval = True
    log.info(f"Dataset size: {len(dataset)}")

    metadata_by_id = {}
    for i in range(len(dataset)):
        ex = dataset.get(i, None)
        ex_id = ex.get('metadata', {}).get('example_id', str(i))
        metadata_by_id[ex_id] = ex['metadata']
    return metadata_by_id


def run_eval(predictions_path, task, split="validation", overwrite=False,
             masks_dir=None, out_dir=None, sampling_fps=None):
    """Run the task's evaluators over a predictions file. Returns the metrics dict."""
    with open(predictions_path) as f:
        predictions_json = json.load(f)
    log.info(f"Loaded {len(predictions_json)} predictions from {predictions_path}")

    evaluator_config = get_evaluator(task)
    inf_evaluator = evaluator_config.build(default_save_dir=None)
    evaluators = [m for m in inf_evaluator.metrics if not isinstance(m, SavePredictions)]
    if not evaluators:
        log.warning(f"No evaluators for task {task}, skipping")
        return {}

    if masks_dir:
        metadata_by_id = build_metadata_from_masks_rle(masks_dir, sampling_fps=sampling_fps)
    else:
        metadata_by_id = build_metadata_from_dataset(task, split)

    matched_metadatas, matched_preds = [], []
    n_unmatched = 0
    for pred_entry in predictions_json:
        meta = metadata_by_id.get(pred_entry['example_id'])
        if meta is None:
            n_unmatched += 1
            continue
        matched_metadatas.append(meta)
        matched_preds.append(pred_entry['prediction'])
    if n_unmatched:
        log.warning(f"{n_unmatched} predictions had no metadata and were skipped")
    log.info(f"Matched {len(matched_preds)}/{len(predictions_json)} predictions to metadata")

    if not matched_preds:
        log.warning("No predictions matched metadata, skipping evaluation")
        return {}

    predictions = {"predictions": matched_preds, "predictions_text": matched_preds}

    all_metrics = {}
    for metric in evaluators:
        all_metrics.update(metric(matched_metadatas, predictions, step=None, tokenizer=None))

    resolved_metrics = {}
    for k in sorted(all_metrics):
        v = all_metrics[k]
        if isinstance(v, (int, float)):
            resolved_metrics[k] = v
        elif isinstance(v, torchmetrics.Metric):
            resolved_metrics[k] = v.compute().item()
        else:
            # HtmlTable, List, ... — not a number, nothing to write out
            log.info(f"Skipping non-numeric metric {k}: {type(v).__name__}")

    # nMAE is a ratio of two global sums, so it can only be formed after the
    # components are reduced (mirrors InfEvaluator).
    _resolve_ratios(resolved_metrics)

    log_metrics_to_console(task, resolved_metrics)

    metrics_path = join(out_dir or os.path.dirname(predictions_path), "metrics.json")
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    if overwrite or not os.path.exists(metrics_path):
        with open(metrics_path, 'w') as f:
            json.dump(resolved_metrics, f, indent=2, sort_keys=True)
        log.info(f"Wrote {metrics_path}")
    else:
        log.info(f"Metrics file {metrics_path} already exists, skipping overwrite")

    return resolved_metrics


def _resolve_ratios(metrics):
    """Turn every nMAE_numerator[_river]/nMAE_denominator[_river] pair into nMAE[_river]."""
    for key in [k for k in metrics if k.startswith("nMAE_numerator")]:
        suffix = key[len("nMAE_numerator"):]
        num = metrics.get(key)
        den = metrics.get(f"nMAE_denominator{suffix}")
        metrics[f"nMAE{suffix}"] = (num / den if num is not None and den else float("nan"))
