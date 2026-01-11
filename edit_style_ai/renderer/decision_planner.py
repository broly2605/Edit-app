import math
import os
from typing import Any, Dict, List

import cv2
import numpy as np

from analysis.scoring import (
    beat_alignment_score,
    brightness_score,
    describe_candidate,
    duration_fit_score,
    final_score,
    motion_score,
)


def _video_duration(path: str) -> float:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    if fps <= 1e-3:
        return 0.0
    return frames / fps


def _sample_metrics(path: str, start: float, end: float, sample_frames: int = 3) -> Dict[str, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"motion": 0.0, "brightness": 0.0}

    # Downscale for faster scoring reads only (does not affect output render)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 180)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = total_frames / fps if fps > 0 else 0
    if duration <= 0:
        cap.release()
        return {"motion": 0.0, "brightness": 0.0}

    start = max(0.0, min(start, duration))
    end = max(start, min(end, duration))
    if end <= start:
        cap.release()
        return {"motion": 0.0, "brightness": 0.0}

    frame_idxs = [int((start + (end - start) * (i / max(sample_frames - 1, 1))) * fps) for i in range(sample_frames)]
    prev_gray = None
    motions, brights = [], []
    for idx in frame_idxs:
        if idx >= total_frames:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brights.append(float(np.mean(gray)))
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            motions.append(float(np.mean(diff)))
        prev_gray = gray
    cap.release()

    motion_raw = float(np.mean(motions)) if motions else 0.0
    bright_raw = float(np.mean(brights)) if brights else 0.0
    return {"motion": motion_score(motion_raw), "brightness": brightness_score(bright_raw)}


def _image_metrics(path: str) -> Dict[str, float]:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"motion": 0.0, "brightness": 0.0}
    bright_raw = float(np.mean(img))
    return {"motion": 0.0, "brightness": brightness_score(bright_raw)}


def _rank_assets(videos: List[str], images: List[str]) -> Dict[str, List[str]]:
    return {"videos": sorted(videos), "images": sorted(images)}


def _enumerate_video_candidates(video: str, shot_len: float, beats: List[float], style_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    dur = _video_duration(video)
    if dur <= 0 or shot_len <= 0:
        return []

    seg_len = min(max(shot_len, 0.4), min(shot_len * 1.2, 5.0))
    stride = max(seg_len * 1.0, 0.4)  # fewer segments, faster; keeps ordering stable
    candidates: List[Dict[str, Any]] = []

    pos = 0.0
    while pos + 0.2 <= dur:
        start = pos
        end = min(start + seg_len, dur)
        metrics = _sample_metrics(video, start, end)
        beat = beat_alignment_score(start, beats)
        fit = duration_fit_score(end - start, shot_len)
        score = final_score(metrics["motion"], metrics["brightness"], beat, fit, style_context)
        candidates.append({
            "asset": video,
            "name": os.path.basename(video),
            "type": "video",
            "start": start,
            "end": end,
            "duration": end - start,
            "score": score,
            "metrics": {
                "motion": metrics["motion"],
                "brightness": metrics["brightness"],
                "beat": beat,
                "duration_fit": fit,
            },
        })
        pos += stride

    return candidates


def _enumerate_image_candidates(image: str, shot_len: float, beats: List[float], style_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    metrics = _image_metrics(image)
    beat = beat_alignment_score(0.0, beats)
    fit = duration_fit_score(shot_len, shot_len)
    score = final_score(metrics["motion"], metrics["brightness"], beat, fit, style_context)
    return [{
        "asset": image,
        "name": os.path.basename(image),
        "type": "image",
        "duration": shot_len,
        "effect": "ken_burns_slow",
        "score": score,
        "metrics": {
            "motion": metrics["motion"],
            "brightness": metrics["brightness"],
            "beat": beat,
            "duration_fit": fit,
        },
    }]


def plan_timeline(blueprint: Dict[str, Any], videos: List[str], images: List[str], style_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    shot_lengths = blueprint.get("shot_lengths", [])
    beats = blueprint.get("beats", [])
    assets = _rank_assets(videos, images)
    image_usage = (style_context or {}).get("image_usage", "medium")
    image_allowed = image_usage != "none"  # allow images for low/medium/high

    timeline: List[Dict[str, Any]] = []
    log_lines: List[str] = []
    shot_start = 0.0
    use_counts: Dict[str, int] = {}
    last_asset: str = ""

    for shot_idx, duration in enumerate(shot_lengths):
        candidates: List[Dict[str, Any]] = []

        for vf in assets["videos"]:
            if os.path.isfile(vf):
                candidates.extend(_enumerate_video_candidates(vf, duration, beats, style_context))

        # Allow images only for shots >= 1s and when style allows images
        if image_allowed and duration >= 1.0:
            for img in assets["images"]:
                if os.path.isfile(img):
                    candidates.extend(_enumerate_image_candidates(img, duration, beats, style_context))

        if not candidates:
            log_lines.append(f"[Shot {shot_idx}] No candidates found.")
            timeline.append({
                "asset": None,
                "type": "video",
                "start": 0.0,
                "end": duration,
                "duration": duration,
                "score": 0.0,
                "metrics": {},
                "style": style_context.get("style", "default"),
            })
            shot_start += duration
            continue

        # Diversify: apply a deterministic reuse penalty to avoid one asset dominating.
        diversified = []
        for cand in candidates:
            asset = cand.get("asset", "")
            base = cand.get("score", 0.0)
            reuse_penalty = 1.0 / (1 + use_counts.get(asset, 0))
            consecutive_penalty = 0.05 if asset == last_asset else 0.0
            adjusted = base * reuse_penalty - consecutive_penalty
            ccopy = dict(cand)
            ccopy["score"] = adjusted
            ccopy.setdefault("metrics", {})["reuse_penalty"] = reuse_penalty
            ccopy["base_score"] = base
            diversified.append(ccopy)

        diversified.sort(key=lambda c: c.get("score", -math.inf), reverse=True)
        top3 = diversified[:3]
        chosen = top3[0]

        timeline.append({
            "asset": chosen.get("asset"),
            "type": chosen.get("type"),
            "start": chosen.get("start", 0.0),
            "end": chosen.get("end", chosen.get("duration", duration)),
            "duration": chosen.get("duration", duration),
            "score": chosen.get("score", 0.0),
            "metrics": chosen.get("metrics", {}),
            "style": style_context.get("style", "default"),
            "effect": chosen.get("effect"),
        })

        asset_key = chosen.get("asset", "")
        if asset_key:
            use_counts[asset_key] = use_counts.get(asset_key, 0) + 1
            last_asset = asset_key

        log_lines.append(
            f"[Shot {shot_idx}] winner: {chosen.get('name','?')} score={chosen.get('score',0):.3f} "
            f"base={chosen.get('base_score',0):.3f} reuse_penalty={chosen.get('metrics',{}).get('reuse_penalty',1):.2f}"
        )
        for idx, cand in enumerate(top3, start=1):
            log_lines.append("   " + describe_candidate(idx, cand))

        shot_start += duration

    for line in log_lines:
        print(line)

    return timeline