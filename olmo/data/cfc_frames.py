"""Stage CFC frames from the raw-frame hub release (perona-lab/cfc26).

The track-instruction release (tidalove/cfc-track-instruction) ships annotations
and masks but no pixels. This module fetches the per-river frame tarballs from
the public CFC26 release and materializes exactly the clip directories the CFC
tasks draw from:

    CFC/SourceFrames/{river}/{clip}_{i}.jpg      one real single-channel file
    CFC/JPEGImages/{video_id}/{video_id}_{j}.jpg hardlinks into the above

Notes on the source format, all verified against the hub:

- Tar members are `{river}/{clip}/{clip}_{i}.jpg` (GNU long names, frames in
  arbitrary order within a clip), where river is one of kenai (left-bank
  cameras), rightbank, nushagak, elwha.
- Frames are 3-channel JPEGs whose channels differ; channel 1 (G) is the
  background-subtracted plane and the only one used here, written back out as a
  single-channel ("L") JPEG.
- A `video_id` is either a whole clip or a window `{clip}_f{a}-{b}` of one, in
  which case its frame j is source frame a + j. Windows over the same clip
  overlap heavily (1.63 links per source frame overall), hence the hardlinks.
- ~27% of clips are named with an end frame one lower on the hub than in the
  local annotations (`RB_..._2700_2999` vs `RB_..._2700_3000`), so extraction
  matches on both spellings and always writes the local one.
"""
import logging
import multiprocessing
import os
import re
import shutil
import tarfile
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from os.path import basename, dirname, exists, join

import numpy as np
from PIL import Image
from tqdm import tqdm

log = logging.getLogger(__name__)

CFC26_REPO = "perona-lab/cfc26"
RIVERS = ("kenai", "rightbank", "nushagak", "elwha")
JPEG_QUALITY = 95

WINDOW_RE = re.compile(r"^(?P<clip>.+)_f(?P<a>\d+)-(?P<b>\d+)$")
TAIL_RE = re.compile(r"^(?P<stem>.+_)(?P<start>\d+)_(?P<end>\d+)$")
# The frame index is the last _-separated token of the basename
FRAME_RE = re.compile(r"^(?P<clip>.+)_(?P<index>\d+)\.jpg$", re.IGNORECASE)


def split_window(video_id):
    """(source clip, frame offset) for a video id. Whole clips get offset 0."""
    m = WINDOW_RE.match(video_id)
    if m is None:
        return video_id, 0
    return m.group("clip"), int(m.group("a"))


def river_of(clip):
    """Which CFC26 river tarball holds a clip's frames."""
    if clip.startswith("Elwha"):
        return "elwha"
    if clip.startswith("RB_"):
        return "nushagak"
    if re.search(r"_(LeftFar|LeftNear)_", clip):
        return "kenai"
    if re.search(r"_(RightFar|RightNear)_", clip):
        return "rightbank"
    # kenai-channel and eel (ARIS_*) clips live in other tarballs and are not
    # part of the track-instruction release; fail loudly rather than guess.
    raise ValueError(f"Cannot map clip to a CFC26 river tarball: {clip}")


def hub_clip_candidates(clip):
    """Spellings of a local clip name that may appear as a hub clip directory."""
    out = [clip]
    m = TAIL_RE.match(clip)
    if m:
        out.append(f"{m.group('stem')}{m.group('start')}_{int(m.group('end')) - 1}")
    return out


def source_frame_path(source_root, river, clip, index):
    return join(source_root, river, f"{clip}_{index}.jpg")


def to_g_channel_jpeg(data):
    """3-channel source JPEG bytes -> single-channel (G) JPEG bytes."""
    arr = np.asarray(Image.open(BytesIO(data)))
    if arr.ndim == 3:
        if arr.shape[2] < 2:
            raise ValueError(f"expected >=2 channels, got shape {arr.shape}")
        arr = arr[:, :, 1]
    elif arr.ndim != 2:
        raise ValueError(f"unexpected frame shape {arr.shape}")
    buf = BytesIO()
    Image.fromarray(arr, mode="L").save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def _convert_worker(item):
    out_path, data = item
    try:
        payload = to_g_channel_jpeg(data)
        tmp = f"{out_path}.tmp{os.getpid()}"
        with open(tmp, "wb") as f:
            f.write(payload)
        os.replace(tmp, out_path)
        return out_path, None
    except Exception as e:  # keep going; the caller reports what is missing
        return out_path, f"{type(e).__name__}: {e}"


@dataclass
class StageReport:
    already_done: int = 0                       # videos whose dirs were complete
    staged: int = 0                             # videos linked in this run
    incomplete: dict = field(default_factory=dict)   # video_id -> n missing frames
    frames_written: int = 0
    frames_reused: int = 0                      # source frames already on disk
    links_created: int = 0
    links_copied: int = 0
    missing_clips: set = field(default_factory=set)  # clips absent from the tars
    wanted_by_river: dict = field(default_factory=dict)

    def log_summary(self, prefix=""):
        log.info(f"{prefix}videos: {self.staged} staged, {self.already_done} already complete, "
                 f"{len(self.incomplete)} incomplete")
        log.info(f"{prefix}frames: {self.frames_written} converted, {self.frames_reused} reused; "
                 f"links: {self.links_created} hardlinked, {self.links_copied} copied")
        if self.missing_clips:
            sample = sorted(self.missing_clips)[:5]
            log.warning(f"{prefix}{len(self.missing_clips)} clips not found in the CFC26 "
                        f"tarballs, e.g. {sample}")
        if self.incomplete:
            sample = sorted(self.incomplete.items())[:5]
            log.warning(f"{prefix}{len(self.incomplete)} videos could not be completed, "
                        f"e.g. {sample}")


def _plan(videos, jpeg_dir_fn, source_root, report):
    """Work out which source frames are missing. Returns (todo, wanted).

    todo:   [(video_id, clip, offset, n_frames, river)] for videos needing work
    wanted: {river: {hub_or_local_clip_name: {index: out_path}}}
    """
    todo = []
    wanted = defaultdict(lambda: defaultdict(dict))
    required = set()
    for video_id, n_frames in videos.items():
        frames_dir = jpeg_dir_fn(video_id)
        if exists(frames_dir) and len(os.listdir(frames_dir)) == n_frames:
            report.already_done += 1
            continue
        clip, offset = split_window(video_id)
        river = river_of(clip)
        todo.append((video_id, clip, offset, n_frames, river))
        for j in range(n_frames):
            index = offset + j
            if (clip, index) in required:
                continue
            required.add((clip, index))
            out_path = source_frame_path(source_root, river, clip, index)
            if exists(out_path):
                continue
            wanted[river][clip][index] = out_path
    for river, clips in wanted.items():
        report.wanted_by_river[river] = sum(len(v) for v in clips.values())
    report.frames_reused = len(required) - sum(report.wanted_by_river.values())
    return todo, wanted


def _resolve_hub_names(clips):
    """{hub clip dir name -> local clip name} for one river's wanted clips."""
    lookup = {}
    for clip in clips:
        for cand in hub_clip_candidates(clip):
            other = lookup.get(cand)
            if other is not None and other != clip:
                raise AssertionError(
                    f"hub clip name {cand} maps to both {other} and {clip}")
            lookup[cand] = clip
    return lookup


def _download_tar(river, tar_dir):
    from huggingface_hub import hf_hub_download
    os.makedirs(tar_dir, exist_ok=True)
    log.info(f"[cfc26] downloading frames/{river}.tar.gz")
    return hf_hub_download(CFC26_REPO, f"frames/{river}.tar.gz",
                           repo_type="dataset", local_dir=tar_dir)


def _extract_river(river, clips, tar_path, n_procs, report):
    """Convert every wanted frame of one river out of its tarball."""
    lookup = _resolve_hub_names(clips)
    remaining = {clip: dict(idx) for clip, idx in clips.items()}
    n_wanted = sum(len(v) for v in remaining.values())
    # every output for a river lands in the same directory
    os.makedirs(dirname(next(iter(next(iter(clips.values())).values()))), exist_ok=True)

    batch, batch_size = [], max(8, n_procs * 8)
    pool = multiprocessing.Pool(n_procs) if n_procs > 1 else None
    pbar = tqdm(desc=f"[cfc26] {river}", unit=" members")

    def flush():
        if not batch:
            return
        results = pool.map(_convert_worker, batch) if pool else \
            [_convert_worker(item) for item in batch]
        for out_path, err in results:
            if err is None:
                report.frames_written += 1
            else:
                log.warning(f"[cfc26] convert failed for {out_path}: {err}")
        batch.clear()

    try:
        with tarfile.open(tar_path, mode="r|gz") as tf:
            for i, member in enumerate(tf, start=1):
                if i % 5000 == 0:
                    tf.members.clear()   # do not accumulate millions of TarInfo
                    pbar.update(5000)
                if not member.isfile():
                    continue
                base = basename(member.name)
                if base.startswith("._"):
                    continue
                m = FRAME_RE.match(base)
                if m is None:
                    continue
                local_clip = lookup.get(m.group("clip"))
                if local_clip is None:
                    continue
                index = int(m.group("index"))
                out_path = remaining.get(local_clip, {}).pop(index, None)
                if out_path is None:
                    continue
                batch.append((out_path, tf.extractfile(member).read()))
                if len(batch) >= batch_size:
                    flush()
                if not any(remaining.values()):
                    break
        flush()
    finally:
        if pool:
            pool.close()
            pool.join()
        pbar.close()

    still = {clip: idx for clip, idx in remaining.items() if idx}
    for clip in still:
        report.missing_clips.add(clip)
    log.info(f"[cfc26] {river}: {n_wanted - sum(len(v) for v in still.values())}"
             f"/{n_wanted} wanted frames extracted"
             + (f", {len(still)} clips incomplete" if still else ""))


def _link_video(video_id, clip, offset, n_frames, river, frames_dir, source_root, report):
    os.makedirs(frames_dir, exist_ok=True)
    missing = 0
    for j in range(n_frames):
        src = source_frame_path(source_root, river, clip, offset + j)
        dst = join(frames_dir, f"{video_id}_{j}.jpg")
        if exists(dst):
            continue
        if not exists(src):
            missing += 1
            continue
        try:
            os.link(src, dst)
            report.links_created += 1
        except OSError:
            # different filesystem or link-count limit; fall back to a real copy
            shutil.copy2(src, dst)
            report.links_copied += 1
    if missing:
        report.incomplete[video_id] = missing
    else:
        report.staged += 1


def ensure_frames(videos, jpeg_dir_fn, source_root, n_procs=1, rivers=None,
                  keep_tars=False, tar_dir=None, dry_run=False):
    """Make sure every video's frame directory exists and is complete.

    Args:
        videos: {video_id: n_frames}. n_frames is the native (6 fps) count, as
            stored on the hub rows, and is the number of files the video's
            JPEGImages dir must hold.
        jpeg_dir_fn: video_id -> frames directory to populate.
        source_root: root for the single real copy of each source frame
            (CFC/SourceFrames).
        n_procs: workers for JPEG decode/re-encode.
        rivers: restrict work to these rivers (default: whatever is needed).
        keep_tars: keep the downloaded tarballs instead of deleting them.
        tar_dir: where to download tarballs (default: alongside source_root).
        dry_run: report what would be fetched, change nothing.
    """
    report = StageReport()
    todo, wanted = _plan(videos, jpeg_dir_fn, source_root, report)
    if not todo:
        log.info(f"[cfc26] all {report.already_done} video frame dirs already complete")
        report.log_summary(prefix="[cfc26] ")
        return report

    if rivers is not None:
        rivers = set(rivers)
        wanted = {r: c for r, c in wanted.items() if r in rivers}
        todo = [t for t in todo if t[-1] in rivers]

    n_frames_wanted = sum(report.wanted_by_river.get(r, 0) for r in wanted)
    log.info(f"[cfc26] {len(todo)} videos need frames; {n_frames_wanted} source frames "
             f"to fetch from {sorted(wanted)} "
             f"({report.wanted_by_river}); {report.frames_reused} already staged")
    if dry_run:
        report.log_summary(prefix="[cfc26] dry run, ")
        return report

    tar_dir = tar_dir or join(dirname(source_root.rstrip("/")), "cfc26_tars")
    for river in sorted(wanted):
        clips = {c: idx for c, idx in wanted[river].items() if idx}
        if not clips:
            continue
        tar_path = _download_tar(river, tar_dir)
        try:
            _extract_river(river, clips, tar_path, n_procs, report)
        except Exception:
            # keep the tarball so a retry (e.g. the next config in a group
            # download) resumes instead of refetching tens of GB
            log.error(f"[cfc26] extraction failed; keeping {tar_path} for a retry")
            raise
        if not keep_tars and exists(tar_path):
            os.remove(tar_path)

    for video_id, clip, offset, n_frames, river in tqdm(
            todo, desc="[cfc26] linking", unit=" videos"):
        _link_video(video_id, clip, offset, n_frames, river,
                    jpeg_dir_fn(video_id), source_root, report)

    report.log_summary(prefix="[cfc26] ")
    return report
