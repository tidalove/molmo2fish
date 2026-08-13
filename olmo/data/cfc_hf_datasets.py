"""CFC datasets backed by the HuggingFace release instead of local source jsons.

Loads the hub repo built by scripts/build_cfc_hf_dataset.py (one config per
dataset, train/validation splits, 2 fps annotations over 6 fps videos, inline
RLE masks). Each class supplies load() plus the download plumbing on top of the
standalone bases in olmo/data/cfc_base.py, so this path shares no CFC code with
academic_video_track_datasets.py and keeps working against an upstream copy of
that file — see the clean-repo section of docs/cfc_hf_dataset.md. The local
cfc_* datasets there are unaffected counterparts, not parents.

download() caches the hub annotations, rehydrates MasksRLE/ files
(write-if-missing), stages frames into
$MOLMO_DATA_DIR/video_datasets/video_track/CFC/{SourceFrames,JPEGImages}/ from
the raw-frame release perona-lab/cfc26 (see olmo/data/cfc_frames.py; set
CFC_STAGE_FRAMES=0 to skip when frames are already placed by hand) and encodes
videos/{video_id}.mp4 with ffmpeg. Staging every config in one pass over the
river tarballs is cheaper than one pass per class: scripts/download_cfc_frames.py.
"""
import json
import logging
import multiprocessing
import os
from os.path import exists, join

from tqdm import tqdm

from olmo.data.academic_video_track_datasets import _encode_frames_to_video_worker
from olmo.data.cfc_base import (
    CFC_VIDEO_HOME,
    CFCDatasetBase,
    CFCMultiTurnBase,
    _load_hf_dataset,
    get_candidate_sampling_fps,
)
from olmo.data.cfc_frames import ensure_frames

log = logging.getLogger(__name__)

DEFAULT_CFC_HF_REPO = "tidalove/cfc-track-instruction"


class CFCHFMixin:
    """Shared HF plumbing. Must precede the local CFC class in the MRO."""

    HF_SOURCE = os.environ.get("CFC_HF_REPO", DEFAULT_CFC_HF_REPO)
    HF_CONFIG: str = None
    ANNOTATION_FPS = 2
    NEEDS_VIDEO = True
    SPLIT_MAP = {
        "train-v2": "train",
        "validation-v2": "validation",
        "train": "train",
        "validation": "validation",
    }

    # ── Annotation loading ────────────────────────────────────────────────

    @classmethod
    def _load_hf_rows(cls, data_split, overwrite_cache=False):
        return _load_hf_dataset(
            cls.HF_SOURCE, data_split,
            local_name=join("CFC", "hf_annotations", cls.HF_CONFIG, data_split),
            config=cls.HF_CONFIG, overwrite_cache=overwrite_cache)

    def _get_candidate_fps(self, video_fps):
        # Annotations only exist at ANNOTATION_FPS-divisible rates; the inherited
        # default would offer 3/6 fps sampling with no stored points.
        candidates = [
            c for c in get_candidate_sampling_fps(video_fps, self.sampling_fps or 1)
            if c <= self.ANNOTATION_FPS and self.ANNOTATION_FPS % c == 0
        ]
        if not candidates:
            raise ValueError(
                f"No sampling fps <= {self.ANNOTATION_FPS} compatible with "
                f"video_fps={video_fps}, sampling_fps={self.sampling_fps}")
        return candidates

    # ── Download: annotations + MasksRLE + frame staging + video encode ────

    @classmethod
    def download(cls, n_procs=1):
        for data_split in sorted(set(cls.SPLIT_MAP.values())):
            rows = cls._load_hf_rows(data_split)
            cls._rehydrate_masks(rows, data_split)
            if cls.NEEDS_VIDEO:
                cls._stage_frames(rows, data_split, n_procs)
                cls._encode_videos(rows, data_split, n_procs)

    @classmethod
    def _stage_frames(cls, rows, data_split, n_procs=1):
        """Fetch missing frames from perona-lab/cfc26 into JPEGImages/."""
        if os.environ.get("CFC_STAGE_FRAMES", "1") == "0":
            log.info(f"[{cls.DATASET_NAME}] CFC_STAGE_FRAMES=0, skipping frame staging")
            return
        video_dir = cls._get_video_dir(data_split)
        # a video that is already encoded needs no frames
        videos = {video_id: n_frames
                  for video_id, n_frames in zip(rows["video"], rows["n_frames"])
                  if not exists(join(video_dir, f"{video_id}.mp4"))}
        if not videos:
            return
        log.info(f"[{cls.DATASET_NAME}] {len(videos)} videos need frames ({data_split}); "
                 f"staging all configs at once is cheaper: scripts/download_cfc_frames.py")
        home = cls._get_home(data_split)
        ensure_frames(
            videos,
            jpeg_dir_fn=lambda vid: cls._get_frames_dir(data_split, vid),
            source_root=join(home, "SourceFrames"),
            n_procs=n_procs)

    @classmethod
    def _mask_path(cls, row):
        """Local MasksRLE path for a row. Correction rows key by example id;
        track rows key by video/qid (overridden below)."""
        return join(cls.VIDEO_HOME, "MasksRLE", row["id"], "0.json")

    @staticmethod
    def _inline_masks_to_local(row):
        """Hub inline masks -> local MasksRLE format {slot: [rle|None] * n_frames},
        None-padded to native frame count (kept entry i -> native frame i * step)."""
        n_frames = row["n_frames"]
        step = row["fps"] // row["sampling_fps"]
        out = {}
        for entry in row["masks"]:
            per_frame = [None] * n_frames
            for i, rle in enumerate(entry["masks"]):
                per_frame[i * step] = rle
            out[entry["object_id"]] = per_frame
        return out

    @classmethod
    def _rehydrate_masks(cls, rows, data_split):
        if "masks" not in rows.column_names:
            return
        n_written = n_skipped = 0
        for row in tqdm(rows, desc=f"[{cls.DATASET_NAME}] MasksRLE ({data_split})"):
            out_path = cls._mask_path(row)
            if exists(out_path):
                n_skipped += 1
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "w") as f:
                json.dump(cls._inline_masks_to_local(row), f)
            n_written += 1
        log.info(f"[{cls.DATASET_NAME}] MasksRLE ({data_split}): "
                 f"{n_written} written, {n_skipped} already existed")

    @classmethod
    def _encode_videos(cls, rows, data_split, n_procs=1):
        video_dir = cls._get_video_dir(data_split)
        work = []
        seen = set()
        for video_id, fps in zip(rows["video"], rows["fps"]):
            if video_id in seen:
                continue
            seen.add(video_id)
            out_path = join(video_dir, f"{video_id}.mp4")
            if exists(out_path):
                continue
            frames_dir = cls._get_frames_dir(data_split, video_id)
            if not exists(frames_dir):
                log.warning(f"[{cls.DATASET_NAME}] no frames for {video_id} "
                            f"({frames_dir}); skipping encode")
                continue
            work.append(dict(frames_dir=frames_dir, output_path=out_path, fps=fps))
        if not work:
            log.info(f"[{cls.DATASET_NAME}] videos ({data_split}): all "
                     f"{len(seen)} present")
            return
        log.info(f"[{cls.DATASET_NAME}] encoding {len(work)} videos ({data_split})")
        if n_procs > 1:
            with multiprocessing.Pool(n_procs) as pool:
                results = list(tqdm(
                    pool.imap_unordered(_encode_frames_to_video_worker, work),
                    total=len(work)))
        else:
            results = [_encode_frames_to_video_worker(kw) for kw in tqdm(work)]
        failed = [path for path, ok, _ in results if not ok]
        if failed:
            log.warning(f"[{cls.DATASET_NAME}] {len(failed)} encodes failed, "
                        f"e.g. {failed[:3]}")


# ── Track family ───────────────────────────────────────────────────────────

class CFCTrackHF(CFCHFMixin, CFCDatasetBase):
    DATASET_NAME = "cfc_hf_track"
    HF_CONFIG = "cfc_track"

    @classmethod
    def _mask_path(cls, row):
        return join(cls.VIDEO_HOME, "MasksRLE", row["video"], f"{row['qid']}.json")

    def load(self):
        rows = self._load_hf_rows(self.data_split)
        if "masks" in rows.column_names:
            rows = rows.remove_columns("masks")  # eval reads rehydrated MasksRLE files
        data = []
        for row in rows:
            data.append({
                "id": row["id"],
                "video": row["video"],
                "expression": row["expression"],
                "height": row["height"],
                "width": row["width"],
                "fps": row["fps"],
                "sampling_fps": row["sampling_fps"],
                "mask_id": row["mask_id"],
                "obj_id": row["obj_id"],
                "anno_id": row["anno_id"],
                "qid": row["qid"],
                "frame_trajectories": row["frame_trajectories"],
                "prepend": row["prepend"],
            })
        self.data_lookup = {ex["id"]: i for i, ex in enumerate(data)}
        log.info(f"[{self.DATASET_NAME}] Loaded {len(data)} examples "
                 f"for split={self.data_split}")
        return data


class CFCTargetedHF(CFCTrackHF):
    """Per-query CFC tracking. Same row shape as cfc_hf_track; the hub config
    carries one example per query instead of one per video."""
    DATASET_NAME = "cfc_hf_target"
    HF_CONFIG = "cfc_target"


# ── Correction family (multi-turn) ─────────────────────────────────────────

class CFCCorrectionHFBase(CFCHFMixin, CFCMultiTurnBase):
    def load(self):
        rows = self._load_hf_rows(self.data_split)
        if "masks" in rows.column_names:
            rows = rows.remove_columns("masks")
        data = []
        for row in rows:
            native_fps = row["fps"]
            prompts_list, points_list = [], []
            for turn in sorted(row["turns"], key=lambda t: t["correction_step"]):
                # hub prompts are raw/frame-indexed; rewrite to time like
                # CFCMultiTurn.load does for the local jsonls
                prompts_list.append(
                    self.replace_frames_with_time(turn["prompt"], native_fps))
                points_list.append([
                    {"frame": ft["frame"], "time": ft["time"],
                     "points": {int(p["id"]): {"point": p["point"],
                                               "occluded": p["occluded"]}
                                for p in ft["points"]}}
                    for ft in turn["frame_trajectories"]
                ])
            data.append({
                "id": row["id"],
                "video": row["video"],
                "expression": row["expression"],
                "height": row["height"],
                "width": row["width"],
                "fps": native_fps,
                "sampling_fps": row["sampling_fps"],
                "mask_id": row["mask_id"],
                "obj_id": row["obj_id"],
                "qid": row["qid"],
                "prompts_list": prompts_list,
                "points_list": points_list,
            })
        self.data_lookup = {ex["id"]: i for i, ex in enumerate(data)}
        log.info(f"[{self.DATASET_NAME}] Loaded {len(data)} trajectories "
                 f"for split={self.data_split}")
        return data


class CFCSyntheticCorrectionFullHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_synthetic_correction_full"
    HF_CONFIG = "cfc_synthetic_correction_full"


class CFCSyntheticCorrectionVagueHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_synthetic_correction_vague"
    HF_CONFIG = "cfc_synthetic_correction_vague"


class CFCSyntheticCorrectionWrongOnlyHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_synthetic_correction_wrong_only"
    HF_CONFIG = "cfc_synthetic_correction_wrong_only"


class CFCSyntheticCorrectionNoInfoHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_synthetic_correction_no_info"
    HF_CONFIG = "cfc_synthetic_correction_no_info"


class CFCSyntheticCorrectionIncompleteHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_synthetic_correction_incomplete"
    HF_CONFIG = "cfc_synthetic_correction_incomplete"


class CFCCorrectionRealFullEasyHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_real_full_easy"
    HF_CONFIG = "cfc_correction_real_full_easy"


class CFCCorrectionRealWrongOnlyEasyHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_real_wrong_only_easy"
    HF_CONFIG = "cfc_correction_real_wrong_only_easy"
    # this tier has no val-easy jsonl -> no validation split on the hub
    SPLIT_MAP = {"train-v2": "train", "train": "train"}


class CFCCorrectionRealVagueEasyHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_real_vague_easy"
    HF_CONFIG = "cfc_correction_real_vague_easy"


class CFCCorrectionRealNoInfoEasyHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_real_no_info_easy"
    HF_CONFIG = "cfc_correction_real_no_info_easy"


class CFCCorrectionRealFullHardHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_real_full_hard"
    HF_CONFIG = "cfc_correction_real_full_hard"


class CFCCorrectionRealWrongOnlyHardHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_real_wrong_only_hard"
    HF_CONFIG = "cfc_correction_real_wrong_only_hard"


class CFCCorrectionRealVagueHardHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_real_vague_hard"
    HF_CONFIG = "cfc_correction_real_vague_hard"


class CFCCorrectionRealNoInfoHardHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_real_no_info_hard"
    HF_CONFIG = "cfc_correction_real_no_info_hard"


# YOLO-SORT step-0 tracks vs COCO GT — validation only
_YOLO_SPLIT_MAP = {"validation-v2": "validation", "validation": "validation"}


class CFCCorrectionRealYoloFullHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_real_yolo_full"
    HF_CONFIG = "cfc_correction_real_yolo_full"
    SPLIT_MAP = _YOLO_SPLIT_MAP


class CFCCorrectionRealYoloWrongOnlyHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_real_yolo_wrong_only"
    HF_CONFIG = "cfc_correction_real_yolo_wrong_only"
    SPLIT_MAP = _YOLO_SPLIT_MAP


class CFCCorrectionRealYoloVagueHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_real_yolo_vague"
    HF_CONFIG = "cfc_correction_real_yolo_vague"
    SPLIT_MAP = _YOLO_SPLIT_MAP


class CFCCorrectionRealYoloNoInfoHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_real_yolo_no_info"
    HF_CONFIG = "cfc_correction_real_yolo_no_info"
    SPLIT_MAP = _YOLO_SPLIT_MAP


# ── Text-only corrections ──────────────────────────────────────────────────

class CFCTextHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_text"
    HF_CONFIG = "cfc_text"
    NEEDS_VIDEO = False  # text-only; no video input, no masks on the hub
    HAS_VIDEO = False    # examples carry no video path


CFC_HF_CLASSES = [
    CFCTrackHF,
    CFCTargetedHF,
    CFCSyntheticCorrectionFullHF,
    CFCSyntheticCorrectionVagueHF,
    CFCSyntheticCorrectionWrongOnlyHF,
    CFCSyntheticCorrectionNoInfoHF,
    CFCSyntheticCorrectionIncompleteHF,
    CFCCorrectionRealFullEasyHF,
    CFCCorrectionRealWrongOnlyEasyHF,
    CFCCorrectionRealVagueEasyHF,
    CFCCorrectionRealNoInfoEasyHF,
    CFCCorrectionRealFullHardHF,
    CFCCorrectionRealWrongOnlyHardHF,
    CFCCorrectionRealVagueHardHF,
    CFCCorrectionRealNoInfoHardHF,
    CFCCorrectionRealYoloFullHF,
    CFCCorrectionRealYoloWrongOnlyHF,
    CFCCorrectionRealYoloVagueHF,
    CFCCorrectionRealYoloNoInfoHF,
    CFCTextHF,
]


def _registry():
    """{registered name: (class, kwargs)} for get_dataset_by_name.

    Every dataset gets a plain name and an `_eval_2fps` name; the suffix is what
    olmo/eval/eval_utils.py dispatches the CFC evaluators on. cfc_hf_target is
    the one dataset whose plain name also samples at 2 fps.
    """
    out = {}
    for cls in CFC_HF_CLASSES:
        name = cls.DATASET_NAME
        eval_name = ("cfc_hf_target_track_eval_2fps" if name == "cfc_hf_target"
                     else f"{name}_eval_2fps")
        plain_kwargs = dict(task="track")
        if name == "cfc_hf_target":
            plain_kwargs["sampling_fps"] = 2
        out[name] = (cls, plain_kwargs)
        out[eval_name] = (cls, dict(task="track", sampling_fps=2))
    return out


CFC_HF_DATASETS = _registry()
