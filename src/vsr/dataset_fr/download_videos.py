import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np


def get_video_source(video_id, max_height=720):
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            "--ignore-config",
            "--no-playlist",
            "--skip-download",
            "--dump-single-json",
            "--no-warnings",
            "-f",
            f"bestvideo[ext=mp4][protocol=https][height<={max_height}]/bestvideo[protocol=https][height<={max_height}]",
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    info = json.loads(result.stdout)
    return {
        "url": info["url"],
        "width": int(info["width"]),
        "height": int(info["height"]),
        "headers": info.get("http_headers", {}),
    }


def get_cuts_from_pckl(video_id, pckl_path):
    """Read (start_frame, inclusive_end_frame) pairs from pckl_path/video_id/*.pckl."""
    import pickle

    cuts = []
    for segment_path in sorted((Path(pckl_path) / video_id).glob("*.pckl")):
        with segment_path.open("rb") as f:
            data = pickle.load(f)
        start_frame, end_frame = data["frame"]
        cuts.append((int(start_frame), int(end_frame)))
    return cuts
