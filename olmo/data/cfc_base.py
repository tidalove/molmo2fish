"""Standalone dataset bases for the CFC hub path.

Deliberately independent of the tracking dataset classes in
academic_video_track_datasets.py: inheriting from them would tie the CFC path to
a 4.6k-line file that is otherwise unchanged from upstream Molmo2. The only
symbols imported from there are ones that are byte-identical to upstream.

Everything the CFC hub path needs lives in olmo/data/cfc_{base,frames,hf_datasets}.py.
Data root is $MOLMO_DATA_DIR/video_datasets/video_track/CFC/.
"""
import json
import logging
import re
from os.path import exists, join
from pycocotools import mask as mask_utils

from olmo.data.dataset import Dataset, VIDEO_DATA_HOME
# Upstream-identical symbols only. Do not add CFC/LocalTrackingDataset here.
from olmo.data.academic_video_track_datasets import (
    TRACKING_TASKS,
    _load_hf_dataset,
    get_candidate_sampling_fps,
)

log = logging.getLogger(__name__)

CFC_VIDEO_HOME = join(VIDEO_DATA_HOME, "video_track", "CFC")

__all__ = ["CFC_VIDEO_HOME", "CFCDatasetBase", "CFCMultiTurnBase",
           "points_from_masks", "_load_hf_dataset", "get_candidate_sampling_fps"]


def points_from_masks(masks, video_fps):
    """Build GT track points from a per-object RLE mask dict, so the points are
    object-slot consistent with the masks used for validation.

    Each object's per-frame point is the centroid of its mask bbox (via
    `pycocotools.mask.toBbox`); the object index is the mask dict key.

    Args:
        masks: {mask_idx_str: [rle_or_None per frame]} (all lists same length).
        video_fps: frames-per-second used to set each frame's `time`.

    Returns:
        [{'frame', 'time', 'points': {obj_idx: {'point': [x, y], 'occluded': False}}}],
        one entry per frame, or [] if `masks` is empty.
    """
    if not masks:
        return []
    n_frames = len(next(iter(masks.values())))
    points = []
    for frame_idx in range(n_frames):
        frame_points = {}
        for mask_idx_str, frame_list in masks.items():
            rle = frame_list[frame_idx]
            if rle is None:
                continue
            bx, by, bw, bh = mask_utils.toBbox(rle).tolist()
            frame_points[int(mask_idx_str)] = {"point": [bx + bw / 2, by + bh / 2],
                                               "occluded": False}
        points.append({"frame": frame_idx, "time": frame_idx / video_fps,
                       "points": frame_points})
    return points


class CFCDatasetBase(Dataset):
    """Single-turn CFC tracking dataset (point tracks over a 6 fps sonar clip)."""

    DATASET_NAME = None
    VIDEO_HOME = CFC_VIDEO_HOME
    VIDEO_FPS = 6
    TASKS = ["track"]
    EXPRESSION = "fish"
    SPLIT_MAP = {}

    def __init__(self, split, task, sampling_fps=None, use_fps_sampling=True,
                 is_eval=None):
        assert task in TRACKING_TASKS, f"Invalid task: {task}"
        assert task in self.TASKS, \
            f"Task '{task}' not supported for {self.DATASET_NAME}. Available: {self.TASKS}"
        assert split in self.SPLIT_MAP, \
            f"Invalid split: {split}. Available: {list(self.SPLIT_MAP.keys())}"
        self.split = split
        self.task = task
        self.sampling_fps = sampling_fps
        self.use_fps_sampling = use_fps_sampling
        # Controls only whether get() attaches GT (masks/points/mask_id) to
        # metadata, never the prompt. Defaults to "eval on anything but train",
        # but scoring a train split is legitimate (the masks are on the hub for
        # every split), so callers can force it — see olmo/eval/standalone_eval.py.
        self.is_eval = (split != "train") if is_eval is None else is_eval
        self.data_split = self.SPLIT_MAP[split]
        self.video_home = self._get_home(self.data_split)
        self.video_dir = self._get_video_dir(self.data_split)
        self.data_lookup = {}
        self.data = self.load()

    def load(self):
        raise NotImplementedError()

    def __len__(self):
        return len(self.data)

    def get_by_example_id(self, example_id):
        idx = self.data_lookup.get(example_id)
        if idx is not None:
            return self.get(idx, None)
        log.warning(f"Example ID '{example_id}' not found.")
        return None

    # ── Paths ──────────────────────────────────────────────────────────────

    @classmethod
    def _get_home(cls, data_split):
        return cls.VIDEO_HOME

    @classmethod
    def _get_frames_dir(cls, data_split, video_name):
        # Frames are flat — all videos in one JPEGImages/ dir, not split by split
        return join(cls._get_home(data_split), "JPEGImages", video_name)

    @classmethod
    def _get_video_dir(cls, data_split):
        return join(cls._get_home(data_split), "videos")

    def _read_masks(self, *parts):
        """RLE masks written by download(), or {} when there are none."""
        masks_path = join(self.video_home, "MasksRLE", *parts)
        if exists(masks_path):
            with open(masks_path, 'r') as f:
                return json.load(f)
        return {}

    # ── Messages ───────────────────────────────────────────────────────────

    def _get_style(self):
        if self.task == "track":
            return "video_point_track_per_frame"
        elif self.task == "ground":
            return "video_point_ground_start_end"
        elif self.task == "single_point_track":
            return "video_single_point_track_per_frame"
        else:
            raise ValueError(f"Unknown task: {self.task}")

    def _get_candidate_fps(self, video_fps):
        return get_candidate_sampling_fps(video_fps, self.sampling_fps or 1)

    def _create_message_list(self, ex):
        """One message, with points organized by frame."""
        object_id_to_idx = {obj_id: idx for idx, obj_id in enumerate(ex['mask_id'])}

        message_list = [{
            "style": self._get_style(),
            "label": ex['expression'],
            "sampling_fps": ex['sampling_fps'],
            "width": ex['width'],
            "height": ex['height'],
        }]

        if 'frame_trajectories' in ex:
            point_frames = []
            for frame_data in ex['frame_trajectories']:
                points_out = {}
                for p in frame_data['points']:
                    obj_key = str(p['id'])
                    if obj_key in object_id_to_idx:
                        points_out[object_id_to_idx[obj_key]] = {
                            'point': p['point'],
                            'occluded': p['occluded'],
                        }
                point_frames.append({
                    'frame': frame_data['frame'],
                    'time': frame_data['time'],
                    'points': points_out,
                })
            point_frames.sort(key=lambda x: x['frame'])
            message_list[0]['points'] = point_frames or None

        message_list[0]['prepend'] = ex.get('prepend')
        return message_list

    # ── Examples ───────────────────────────────────────────────────────────

    def get(self, idx, rng):
        ex = dict(self.data[idx])  # shallow copy — don't mutate cached data

        # Fall back to the row's own cadence
        sampling_fps = self.sampling_fps or ex.get('sampling_fps')
        ex['sampling_fps'] = sampling_fps  # _create_message_list reads this

        video_path = join(self.video_dir, ex['video'] + '.mp4')
        message_list = self._create_message_list(ex)

        metadata = {
            'example_id': ex['id'],
            'task': self.task,
            'expression': ex['expression'],
            'w': ex['width'],
            'h': ex['height'],
            'video_fps': ex.get('fps', self.VIDEO_FPS),
            'sampling_fps': sampling_fps,
            'video': ex['video'],
        }

        if self.use_fps_sampling:
            metadata['sampler_overrides'] = {
                'frame_sample_mode': 'fps',
                'candidate_sampling_fps': self._get_candidate_fps(
                    ex.get('fps', self.VIDEO_FPS)),
                'min_fps': sampling_fps or 1,
            }

        item = {
            'video': video_path,
            'message_list': message_list,
            'sampling_fps': sampling_fps,
            'metadata': metadata,
            'fps': str(sampling_fps) if sampling_fps else None,
            'label': ex['expression']
        }

        if self.is_eval:
            if "points" in item['message_list'][0]:
                item['metadata']['points'] = item['message_list'][0]['points']
            item['metadata']['mask_id'] = ex['mask_id']
            item['metadata']['masks'] = self._read_masks(
                ex['video'], f"{ex.get('qid', '0')}.json")

        return item


class CFCMultiTurnBase(CFCDatasetBase):
    """Multi-turn CFC correction dataset: one message per correction step."""

    HAS_VIDEO = True

    # Range regex tried first so single-frame regex doesn't eat half a range.
    # Captures separator so "between frames N and M" stays "between Ns and Ms".
    _FRAME_RANGE_RE = re.compile(
        r'\bframes?\s+(\d+)\s+(to|and|through|until)\s+(\d+)\b',
        re.I,
    )
    # Dash range accepts ASCII hyphen plus en/em dash ("frames N–M").
    _FRAME_RANGE_DASH_RE = re.compile(r'\bframes?\s+(\d+)\s*[-–—]\s*(\d+)\b', re.I)
    # Comma list ("frames N, M, ..., and K") — needs >=1 comma so it doesn't
    # overlap the single/range-with-"and" cases. Tried before the single regex.
    _FRAME_LIST_RE = re.compile(
        r'\bframes\s+\d+(?:\s*,\s*\d+)+(?:\s*,?\s*and\s+\d+)?', re.I)
    _FRAME_SINGLE_RE = re.compile(r'\bframes?\s+(\d+)\b', re.I)

    @classmethod
    def replace_frames_with_time(cls, text, fps):
        """Rewrite frame references as time, nearest integer second, 'Ns' format.

        Handles single ('frame N', 'frames N'), range ('frames N to M',
        'frame N and M', 'frames N-M', 'frames N–M', 'frames N through M') and
        comma-list ('frames N, M, and K') phrasings.
        """
        def _secs(d):
            return f"{round(int(d.group()) / fps)}s"

        def _range_sub(m):
            n = round(int(m.group(1)) / fps)
            sep = m.group(2)
            k = round(int(m.group(3)) / fps)
            return f"{n}s {sep} {k}s"

        def _range_dash_sub(m):
            n = round(int(m.group(1)) / fps)
            k = round(int(m.group(2)) / fps)
            return f"{n}s-{k}s"

        def _list_sub(m):
            # Drop the "frames" word, convert every number, keep ", "/"and" seps.
            body = re.sub(r'^frames\s+', '', m.group(), flags=re.I)
            return re.sub(r'\d+', _secs, body)

        def _single_sub(m):
            n = round(int(m.group(1)) / fps)
            return f"{n}s"

        text = cls._FRAME_RANGE_RE.sub(_range_sub, text)
        text = cls._FRAME_RANGE_DASH_RE.sub(_range_dash_sub, text)
        text = cls._FRAME_LIST_RE.sub(_list_sub, text)
        text = cls._FRAME_SINGLE_RE.sub(_single_sub, text)
        return text

    def _create_message_list(self, ex):
        """Multi-turn message list from a correction trajectory."""
        prompts_list = ex['prompts_list']
        points_list = ex['points_list']
        message_list = []
        for i in range(len(prompts_list)):
            message_list.append(dict(
                width=ex['width'],
                height=ex['height'],
                question=prompts_list[i],
                points=points_list[i],
                label=ex['expression'],
                sampling_fps=ex['sampling_fps'],
                style="video_point_track_per_frame",
            ))
        return message_list

    def get(self, idx, rng):
        ex = dict(self.data[idx])  # shallow copy — don't mutate cached data
        video_fps = ex.get("fps", self.VIDEO_FPS)
        # Same rule as CFCDatasetBase.get: constructor value wins, row is the
        # fallback. Correction rows always carry their own sampling_fps.
        sampling_fps = self.sampling_fps or ex.get('sampling_fps')
        ex['sampling_fps'] = sampling_fps
        message_list = self._create_message_list(ex)

        metadata = {
            'example_id': ex['id'],
            'task': self.task,
            'expression': ex['expression'],
            'w': ex['width'],
            'h': ex['height'],
            'video_fps': video_fps,
            'sampling_fps': sampling_fps,
        }
        if self.HAS_VIDEO:
            metadata['video'] = ex['video']

        if self.use_fps_sampling:
            metadata['sampler_overrides'] = {
                'frame_sample_mode': 'fps',
                'candidate_sampling_fps': self._get_candidate_fps(video_fps),
                'min_fps': sampling_fps,
            }

        if self.is_eval:
            points_list = ex['points_list']
            metadata['points'] = points_list[-1]
            metadata['initial_points'] = points_list[0]
            metadata['mask_id'] = ex['mask_id']
            metadata['video'] = ex['video']
            metadata['masks'] = self._read_masks(ex['id'], "0.json")

            # Derive eval GT points from the masks so their object-slot order matches
            # the masks used for validation.
            if metadata['masks']:
                metadata['points'] = points_from_masks(metadata['masks'], video_fps)

        item = {
            'multi_turn_messages': message_list,
            'sampling_fps': sampling_fps,
            'metadata': metadata,
            'fps': str(sampling_fps) if sampling_fps else None,
            'label': ex['expression']
        }
        if self.HAS_VIDEO:
            item['video'] = join(self.video_dir, ex['video'] + '.mp4')
        return item
