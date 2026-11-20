#!/usr/bin/env python3
"""TorchTPU Video Diffusion async benchmark & MP4 quality-verification harness.

Benchmarks OpenAI-compatible `/v1/videos` endpoints served by `sglang-torchtpu`
(supports both T2V and I2V `image_path` conditioning) and verifies every generated
MP4 using `ffprobe` (resolution, frame count, and codec).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any


def verify_mp4_with_ffprobe(video_path: Path) -> dict[str, Any]:
    if not video_path.exists() or video_path.stat().st_size == 0:
        return {"exists": False, "valid_video": False}
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name,nb_frames",
        "-of", "json", str(video_path),
    ]
    out = json.loads(subprocess.check_output(cmd, text=True))
    streams = out.get("streams") or []
    if not streams:
        return {"exists": True, "bytes": video_path.stat().st_size, "valid_video": False}
    s = streams[0]
    return {
        "exists": True,
        "bytes": video_path.stat().st_size,
        "width": int(s.get("width", 0)),
        "height": int(s.get("height", 0)),
        "frames": int(s.get("nb_frames", 0)) if str(s.get("nb_frames", "")).isdigit() else None,
        "codec": s.get("codec_name"),
        "valid_video": int(s.get("width", 0)) > 0 and int(s.get("height", 0)) > 0,
    }


def generate_video(
    base_url: str,
    model: str,
    prompt: str,
    width: int,
    height: int,
    frames: int,
    steps: int,
    fps: int,
    image_path: str | None,
    output_mp4: Path,
    timeout_s: float = 900.0,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "size": f"{width}x{height}",
        "num_frames": frames,
        "num_inference_steps": steps,
        "fps": fps,
    }
    if image_path:
        payload["input_reference"] = image_path

    t0 = time.perf_counter()
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/videos",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        job = json.loads(resp.read().decode("utf-8"))
    vid_id = job["id"]

    status_obj = job
    while time.perf_counter() - t0 < timeout_s:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/videos/{vid_id}", timeout=30) as resp:
            status_obj = json.loads(resp.read().decode("utf-8"))
        st = status_obj.get("status", "")
        if st in ("completed", "succeeded"):
            break
        if st in ("failed", "error"):
            raise RuntimeError(f"Video generation failed: {status_obj}")
        time.sleep(1.0)
    else:
        raise TimeoutError(f"Timed out polling video {vid_id} after {timeout_s}s")

    with urllib.request.urlopen(f"{base_url.rstrip('/')}/v1/videos/{vid_id}/content", timeout=120) as resp:
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        output_mp4.write_bytes(resp.read())

    e2e_s = time.perf_counter() - t0
    return {
        "model": model,
        "width": width,
        "height": height,
        "frames": frames,
        "steps": steps,
        "e2e_s": round(e2e_s, 3),
        "inference_s": status_obj.get("inference_time_s"),
        "peak_memory_mb": status_obj.get("peak_memory_mb"),
        "quality": verify_mp4_with_ffprobe(output_mp4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark TorchTPU video diffusion servers.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", default="A cinematic aerial shot gliding across a misty alpine valley at sunrise.")
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--image-path", default=None, help="Conditioning PNG path for I2V / Animate models")
    parser.add_argument("--output-mp4", type=Path, default=Path("/tmp/output.mp4"))
    args = parser.parse_args()

    res = generate_video(
        args.base_url, args.model, args.prompt,
        args.width, args.height, args.frames, args.steps, args.fps,
        args.image_path, args.output_mp4,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
