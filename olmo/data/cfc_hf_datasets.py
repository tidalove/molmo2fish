"""CFC datasets backed by the HuggingFace release instead of local source jsons.

Loads tidalove/cfc-track-instruction (one config per dataset, train/validation
splits, 2 fps annotations over 6 fps videos, inline RLE masks). Each class
supplies load() plus the download plumbing on top of the standalone bases in
olmo/data/cfc_base.py, so this path shares no CFC code with
academic_video_track_datasets.py and depends only on symbols that are unchanged
from upstream Molmo2.

download() caches the hub annotations, rehydrates MasksRLE/ files
(write-if-missing), stages frames into
$MOLMO_DATA_DIR/video_datasets/video_track/CFC/{SourceFrames,JPEGImages}/ from
the raw-frame release perona-lab/cfc26 (see olmo/data/cfc_frames.py; set
CFC_STAGE_FRAMES=0 to skip when frames are already placed by hand) and encodes
videos/{video_id}.mp4 with ffmpeg. All splits are staged in one pass over the
river tarballs; staging every config at once is cheaper still, which is what
`python -m scripts.download_datasets cfc` does.

The encoded videos define the usable rows: load() drops any row whose mp4 is not
on disk (CFC_REQUIRE_VIDEO=0 opts out) and download() reports what will be
dropped, so a subset download, an interrupted one, or an annotated clip the
frame release does not carry all degrade to a smaller dataset instead of a
missing-file crash. The cached annotations are never pruned, so staging more
videos later restores their rows.
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
        "train": "train",
        "validation": "validation",
    }

    # ── Annotation loading ────────────────────────────────────────────────

    @classmethod
    def _load_hf_rows(cls, data_split, overwrite_cache=False, require_video=False):
        rows = _load_hf_dataset(
            cls.HF_SOURCE, data_split,
            local_name=join("CFC", "hf_annotations", cls.HF_CONFIG, data_split),
            config=cls.HF_CONFIG, overwrite_cache=overwrite_cache)
        if require_video and cls.NEEDS_VIDEO and os.environ.get("CFC_REQUIRE_VIDEO", "1") != "0":
            rows = cls._drop_unencoded(rows, data_split)
        return rows

    @classmethod
    def _encoded_videos(cls, data_split):
        """Video ids that actually have an mp4 on disk (one listdir, not a stat per row)."""
        video_dir = cls._get_video_dir(data_split)
        if not exists(video_dir):
            return set()
        return {f[:-len(".mp4")] for f in os.listdir(video_dir) if f.endswith(".mp4")}

    @classmethod
    def _unencoded_videos(cls, rows, data_split):
        present = cls._encoded_videos(data_split)
        return sorted({v for v in rows["video"] if v not in present})

    @classmethod
    def _drop_unencoded(cls, rows, data_split):
        """Drop rows whose video was never encoded.

        Reasons this happens: a deliberate subset download, an interrupted one, or
        annotations for a clip the CFC26 frame release does not carry. The rows are
        unusable either way, and dropping them here keeps the dataloader from opening
        a file that is not there. The cached annotations stay complete, so staging the
        videos later brings the rows back with no refetch.
        """
        missing = cls._unencoded_videos(rows, data_split)
        if not missing:
            return rows
        missing_set = set(missing)
        keep = [i for i, v in enumerate(rows["video"]) if v not in missing_set]
        log.warning(
            f"[{cls.DATASET_NAME}] {data_split}: dropping {len(rows) - len(keep)} of "
            f"{len(rows)} rows with no encoded video ({len(missing)} videos), e.g. "
            f"{missing[:3]} — run download() to stage them, or ignore if intentional")
        return rows.select(keep)

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
        rows_by_split = {data_split: cls._load_hf_rows(data_split)
                         for data_split in sorted(set(cls.SPLIT_MAP.values()))}
        for data_split, rows in rows_by_split.items():
            cls._rehydrate_masks(rows, data_split)
        if cls.NEEDS_VIDEO:
            # one staging pass for every split: frames are flat on disk, and a per-split
            # pass would refetch each multi-GB river tarball once per split
            cls._stage_frames(rows_by_split, n_procs)
            for data_split, rows in rows_by_split.items():
                cls._encode_videos(rows, data_split, n_procs)
        cls._report_reconciliation(rows_by_split)

    @classmethod
    def _stage_frames(cls, rows_by_split, n_procs=1):
        """Fetch missing frames from perona-lab/cfc26 into JPEGImages/."""
        if os.environ.get("CFC_STAGE_FRAMES", "1") == "0":
            log.info(f"[{cls.DATASET_NAME}] CFC_STAGE_FRAMES=0, skipping frame staging")
            return
        # a video that is already encoded needs no frames
        videos = {}
        for data_split, rows in rows_by_split.items():
            encoded = cls._encoded_videos(data_split)
            videos.update({video_id: n_frames
                           for video_id, n_frames in zip(rows["video"], rows["n_frames"])
                           if video_id not in encoded})
        if not videos:
            return
        splits = "+".join(sorted(rows_by_split))
        log.info(f"[{cls.DATASET_NAME}] {len(videos)} videos need frames ({splits}); "
                 f"staging all configs at once is cheaper: "
                 f"python -m scripts.download_datasets cfc")
        # JPEGImages/ and SourceFrames/ are shared by every split, so any split's
        # paths are the right ones (see CFCDatasetBase._get_frames_dir)
        any_split = next(iter(rows_by_split))
        report = ensure_frames(
            videos,
            jpeg_dir_fn=lambda vid: cls._get_frames_dir(any_split, vid),
            source_root=join(cls._get_home(any_split), "SourceFrames"),
            n_procs=n_procs)
        if report.incomplete or report.missing_clips or report.skipped_unavailable:
            log.warning(
                f"[{cls.DATASET_NAME}] frame staging left {len(report.incomplete)} videos "
                f"incomplete and skipped {report.skipped_unavailable} known-unavailable ones "
                f"({len(report.missing_clips)} source clips missing from the CFC26 release); "
                f"their rows will be dropped at load time. Clips the release does not carry "
                f"are recorded, so the next config will not refetch their river tarball")
        return report

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
        n_unstaged = 0
        for video_id, fps, n_frames in zip(rows["video"], rows["fps"], rows["n_frames"]):
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
                n_unstaged += 1
                continue
            # a partial dir would encode into a short mp4 that still looks present but
            # no longer lines up with the annotations — leave it unencoded instead
            n_staged = len(os.listdir(frames_dir))
            if n_staged != n_frames:
                log.warning(f"[{cls.DATASET_NAME}] {video_id}: {n_staged} frames staged "
                            f"but {n_frames} annotated; skipping encode")
                n_unstaged += 1
                continue
            work.append(dict(frames_dir=frames_dir, output_path=out_path, fps=fps))
        if not work:
            log.info(f"[{cls.DATASET_NAME}] videos ({data_split}): "
                     f"{len(seen) - n_unstaged}/{len(seen)} present"
                     + (f", {n_unstaged} lack complete frames" if n_unstaged else ""))
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

    @classmethod
    def _report_reconciliation(cls, rows_by_split):
        """End-of-download summary: what load() will actually see, and what it won't.

        Reports only — the cached annotations are left whole so a later download can
        fill the gaps. See _drop_unencoded.
        """
        for data_split, rows in rows_by_split.items():
            n_videos = len(set(rows["video"]))
            if not cls.NEEDS_VIDEO:
                log.info(f"[{cls.DATASET_NAME}] {data_split}: {len(rows)} rows "
                         f"(no video needed)")
                continue
            missing = cls._unencoded_videos(rows, data_split)
            if not missing:
                log.info(f"[{cls.DATASET_NAME}] {data_split}: {len(rows)} rows, "
                         f"{n_videos} videos, all encoded")
                continue
            missing_set = set(missing)
            n_dropped = sum(1 for v in rows["video"] if v in missing_set)
            log.warning(
                f"[{cls.DATASET_NAME}] {data_split}: {len(rows)} rows, {n_videos} videos, "
                f"{n_videos - len(missing)} encoded — load() will drop {n_dropped} rows "
                f"for {len(missing)} unencoded videos: {missing[:10]}"
                + (f" (+{len(missing) - 10} more)" if len(missing) > 10 else ""))


# ── Track family ───────────────────────────────────────────────────────────

class CFCTrackHF(CFCHFMixin, CFCDatasetBase):
    DATASET_NAME = "cfc_hf_track"
    HF_CONFIG = "cfc_track"

    @classmethod
    def _mask_path(cls, row):
        return join(cls.VIDEO_HOME, "MasksRLE", row["video"], f"{row['qid']}.json")

    def load(self):
        rows = self._load_hf_rows(self.data_split, require_video=True)
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
        rows = self._load_hf_rows(self.data_split, require_video=True)
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


class CFCCorrectionMolmoHighFullHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_molmo_high_full"
    HF_CONFIG = "cfc_correction_molmo_high_full"


class CFCCorrectionMolmoHighWrongOnlyHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_molmo_high_wrong_only"
    HF_CONFIG = "cfc_correction_molmo_high_wrong_only"
    # this tier has no validation split on the hub
    SPLIT_MAP = {"train": "train"}


class CFCCorrectionMolmoHighVagueHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_molmo_high_vague"
    HF_CONFIG = "cfc_correction_molmo_high_vague"


class CFCCorrectionMolmoHighNoInfoHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_molmo_high_no_info"
    HF_CONFIG = "cfc_correction_molmo_high_no_info"


class CFCCorrectionMolmoLowFullHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_molmo_low_full"
    HF_CONFIG = "cfc_correction_molmo_low_full"


class CFCCorrectionMolmoLowWrongOnlyHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_molmo_low_wrong_only"
    HF_CONFIG = "cfc_correction_molmo_low_wrong_only"


class CFCCorrectionMolmoLowVagueHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_molmo_low_vague"
    HF_CONFIG = "cfc_correction_molmo_low_vague"


class CFCCorrectionMolmoLowNoInfoHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_molmo_low_no_info"
    HF_CONFIG = "cfc_correction_molmo_low_no_info"


# YOLO-SORT step-0 tracks vs COCO GT — validation only
_YOLO_SPLIT_MAP = {"validation": "validation"}


class CFCCorrectionYoloFullHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_yolo_full"
    HF_CONFIG = "cfc_correction_yolo_full"
    SPLIT_MAP = _YOLO_SPLIT_MAP


class CFCCorrectionYoloWrongOnlyHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_yolo_wrong_only"
    HF_CONFIG = "cfc_correction_yolo_wrong_only"
    SPLIT_MAP = _YOLO_SPLIT_MAP


class CFCCorrectionYoloVagueHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_yolo_vague"
    HF_CONFIG = "cfc_correction_yolo_vague"
    SPLIT_MAP = _YOLO_SPLIT_MAP


class CFCCorrectionYoloNoInfoHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_correction_yolo_no_info"
    HF_CONFIG = "cfc_correction_yolo_no_info"
    SPLIT_MAP = _YOLO_SPLIT_MAP


# ── Text-only corrections ──────────────────────────────────────────────────

class CFCTextHF(CFCCorrectionHFBase):
    DATASET_NAME = "cfc_hf_text"
    HF_CONFIG = "cfc_text"
    NEEDS_VIDEO = False
    HAS_VIDEO = False 


CFC_HF_CLASSES = [
    CFCTrackHF,
    CFCTargetedHF,
    CFCSyntheticCorrectionFullHF,
    CFCSyntheticCorrectionVagueHF,
    CFCSyntheticCorrectionWrongOnlyHF,
    CFCSyntheticCorrectionNoInfoHF,
    CFCSyntheticCorrectionIncompleteHF,
    CFCCorrectionMolmoHighFullHF,
    CFCCorrectionMolmoHighWrongOnlyHF,
    CFCCorrectionMolmoHighVagueHF,
    CFCCorrectionMolmoHighNoInfoHF,
    CFCCorrectionMolmoLowFullHF,
    CFCCorrectionMolmoLowWrongOnlyHF,
    CFCCorrectionMolmoLowVagueHF,
    CFCCorrectionMolmoLowNoInfoHF,
    CFCCorrectionYoloFullHF,
    CFCCorrectionYoloWrongOnlyHF,
    CFCCorrectionYoloVagueHF,
    CFCCorrectionYoloNoInfoHF,
    CFCTextHF,
]


def _registry():
    """{registered name: (class, kwargs)} for get_dataset_by_name.    """
    out = {}
    for cls in CFC_HF_CLASSES:
        name = cls.DATASET_NAME
        eval_name = ("cfc_hf_target_track_eval_2fps" if name == "cfc_hf_target"
                     else f"{name}_eval_2fps")
        out[name] = (cls, dict(task="track", sampling_fps=2))
        out[eval_name] = (cls, dict(task="track", sampling_fps=2))
    return out


CFC_HF_DATASETS = _registry()
CFC_HF_TASKS = frozenset(CFC_HF_DATASETS)
CFC_HF_CORRECTION_TASKS = frozenset(
    name for name, (cls, _) in CFC_HF_DATASETS.items()
    if issubclass(cls, CFCMultiTurnBase))
