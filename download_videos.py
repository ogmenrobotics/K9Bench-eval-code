#!/usr/bin/env python3
"""Download YouTube videos for the K9Bench/K9Bench dataset and save them as
``{scene_name}.mp4``. ``scene_name`` is the 11-character YouTube video id
extracted from the dataset's ``video_url`` column.

Usage:
    python download_videos.py --output_dir ./k9bench_videos
    python download_videos.py --output_dir ./k9bench_videos --max_samples 10
"""

import argparse
import os
import re
import subprocess
import sys

from datasets import load_dataset


YT_ID_RE = re.compile(r"(?:v=|youtu\.be/|/embed/|/v/|/shorts/)([\w-]{11})")


def extract_scene_name(video_url: str) -> str:
    m = YT_ID_RE.search(video_url)
    if not m:
        raise ValueError(f"Could not extract YouTube id from URL: {video_url}")
    return m.group(1)


def get_ffmpeg_exe() -> str:
    """Return path to an ffmpeg binary. Prefer system ffmpeg, else the
    bundled one from the ``imageio-ffmpeg`` pip package."""
    from shutil import which
    sys_ff = which("ffmpeg")
    if sys_ff:
        return sys_ff
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as e:
        raise RuntimeError(
            "ffmpeg is required but not found. Install it system-wide or "
            "`pip install imageio-ffmpeg`."
        ) from e


def remux_for_decoder_compat(in_path: str, out_path: str, ffmpeg_exe: str) -> None:
    """Stream-copy the file into a clean, seekable mp4. yt-dlp output sometimes
    confuses decord/torchvision and causes ``process_vision_info`` to hang;
    a passthrough remux with ``+faststart`` reliably fixes it (~1-2s/video)."""
    cmd = [
        ffmpeg_exe, "-y", "-loglevel", "error",
        "-i", in_path,
        "-c", "copy", "-movflags", "+faststart",
        out_path,
    ]
    subprocess.run(cmd, check=True)


def download_video(video_url: str, output_path: str, ffmpeg_exe: str) -> bool:
    """Download a single YouTube video to ``output_path`` using yt-dlp, then
    remux for decoder compatibility. Skips if the final file already exists."""
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return True

    raw_path = output_path + ".raw.mp4"
    cmd = [
        "yt-dlp",
        "-f", "best[ext=mp4][height<=720]/bestvideo[ext=mp4][height<=720]/bestvideo[height<=720]/best",
        "--remux-video", "mp4",
        "--no-playlist", "--no-warnings", "--quiet", "--retries", "3",
        "-o", raw_path, video_url,
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL yt-dlp] {video_url}: {e}")
        if os.path.exists(raw_path):
            os.remove(raw_path)
        return False

    if not os.path.exists(raw_path):
        # yt-dlp occasionally writes a slightly different filename
        base = os.path.basename(raw_path)
        cands = [p for p in os.listdir(os.path.dirname(output_path) or ".")
                 if p.startswith(base)]
        if not cands:
            return False
        raw_path = os.path.join(os.path.dirname(output_path), cands[0])

    try:
        remux_for_decoder_compat(raw_path, output_path, ffmpeg_exe)
    except subprocess.CalledProcessError as e:
        print(f"  [FAIL ffmpeg] {video_url}: {e}")
        return False
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)
    return True


def main():
    parser = argparse.ArgumentParser(description="Download K9Bench videos from YouTube.")
    parser.add_argument("--dataset_name", default="K9Bench/K9Bench")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output_dir", default="./k9bench_videos",
                        help="Directory where {scene_name}.mp4 files will be saved.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="If > 0, only download videos for the first N dataset rows.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Loading dataset {args.dataset_name} (split={args.split})...")
    ds = load_dataset(args.dataset_name, split=args.split)
    if args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))

    # de-duplicate URLs by scene_name (multiple QAs can share a video)
    seen: dict[str, str] = {}
    for ex in ds:
        scene = extract_scene_name(ex["video_url"])
        if scene not in seen:
            seen[scene] = ex["video_url"]

    ffmpeg_exe = get_ffmpeg_exe()
    print(f"Using ffmpeg: {ffmpeg_exe}")
    print(f"Need to download {len(seen)} unique videos.")
    n_ok = n_fail = n_skip = 0
    for i, (scene, url) in enumerate(seen.items(), 1):
        out_path = os.path.join(args.output_dir, f"{scene}.mp4")
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            n_skip += 1
            print(f"[{i}/{len(seen)}] SKIP (exists) {scene}")
            continue
        print(f"[{i}/{len(seen)}] {scene} <- {url}")
        ok = download_video(url, out_path, ffmpeg_exe)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nDone. ok={n_ok} skipped={n_skip} failed={n_fail}")
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
