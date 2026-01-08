import math
import os
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


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


def _sample_segment_scores(path: str, target_duration: float, beat_density: float) -> Tuple[float, Dict[str, Any]]:
    # Sample a handful of evenly spaced windows and score motion/brightness.
    dur = _video_duration(path)
    if dur <= 0.0 or dur < target_duration:
        return -math.inf, {"reason": "too_short", "duration": dur}

    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames_total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or (dur * fps)

    windows = 5
    hop = max(1, int((frames_total - target_duration * fps) / max(1, windows)))
    best_score = -math.inf
    best_meta: Dict[str, Any] = {}

    for i in range(windows):
        start_frame = i * hop
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frames_to_read = int(target_duration * fps)
        motions = []
        brightness = []
        prev_gray = None
        read_frames = 0

        while read_frames < frames_to_read:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness.append(float(np.mean(gray)))
            if prev_gray is not None:
                flow = cv2.absdiff(gray, prev_gray)
                motions.append(float(np.mean(flow)))
            prev_gray = gray
            read_frames += 1

        if not motions:
            continue

        motion_score = np.mean(motions)
        brightness_score = np.mean(brightness)
        beat_sync_score = -abs((read_frames / max(1, fps)) - target_duration)

        # Weight beat density: more beats want more motion.
        score = motion_score * (1 + beat_density) + 0.2 * brightness_score + beat_sync_score

        if score > best_score:
            best_score = score
            best_meta = {
                "start": start_frame / max(1, fps),
                "duration": read_frames / max(1, fps),
                "motion": motion_score,
                "brightness": brightness_score,
                "score": score,
            }

    cap.release()
    if best_score == -math.inf:
        return -math.inf, {"reason": "no_windows", "duration": dur}
    return best_score, best_meta


def _score_image(path: str, target_duration: float) -> Tuple[float, Dict[str, Any]]:
    img = cv2.imread(path)
    if img is None:
        return -math.inf, {"reason": "unreadable"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    score = brightness * 0.4 + contrast * 0.6 - abs(target_duration - 2.0)
    return score, {"brightness": brightness, "contrast": contrast, "duration": target_duration}


def _rank_assets(videos: List[str], images: List[str]) -> Dict[str, List[str]]:
    videos_sorted = sorted(videos)
    images_sorted = sorted(images)
    return {"videos": videos_sorted, "images": images_sorted}


def plan_timeline(blueprint: Dict[str, Any], videos: List[str], images: List[str], style_context: Dict[str, Any]) -> List[Dict[str, Any]]:
    shot_lengths = blueprint.get("shot_lengths", [])
    beats = blueprint.get("beats", [])
    beat_density = len(beats) / max(1, blueprint.get("duration", len(shot_lengths)))

    assets = _rank_assets(videos, images)
    timeline: List[Dict[str, Any]] = []

    video_idx = 0
    image_idx = 0

    for i, duration in enumerate(shot_lengths):
        use_image = False
        if duration <= 1.2 and image_idx < len(assets["images"]):
            use_image = True

        if use_image:
            path = assets["images"][image_idx % len(assets["images"])]
            image_idx += 1
            score, meta = _score_image(path, duration)
            timeline.append({
                "asset": path,
                "type": "image",
                "duration": duration,
                "score": score,
                "meta": meta,
                "style": style_context.get("style", "default"),
                "effect": style_context.get("image_effect", "zoom_in"),
            })
        else:
            if not assets["videos"]:
                # Fallback to any image if no videos exist.
                if not assets["images"]:
                    break
                path = assets["images"][image_idx % len(assets["images"])]
                image_idx += 1
                score, meta = _score_image(path, duration)
                timeline.append({
                    "asset": path,
                    "type": "image",
                    "duration": duration,
                    "score": score,
                    "meta": meta,
                    "style": style_context.get("style", "default"),
                    "effect": style_context.get("image_effect", "zoom_in"),
                })
                continue

            path = assets["videos"][video_idx % len(assets["videos"])]
            video_idx += 1
            score, meta = _sample_segment_scores(path, duration, beat_density)
            timeline.append({
                "asset": path,
                "type": "video",
                "duration": duration,
                "start": meta.get("start", 0.0),
                "end": meta.get("start", 0.0) + meta.get("duration", duration),
                "score": score,
                "meta": meta,
                "style": style_context.get("style", "default"),
            })

    # Log top picks for transparency.
    ranked = sorted(timeline, key=lambda x: x.get("score", -math.inf), reverse=True)
    print("[Layer2] Top picks:")
    for shot in ranked[:3]:
        asset = os.path.basename(shot.get("asset", ""))
        print(f"  {shot.get('type')} {asset} score={shot.get('score'):.2f} dur={shot.get('duration'):.2f}s")

    return timeline