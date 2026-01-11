import math
from typing import Dict, List

from analysis import learning


# Style-specific weights (sum to 1.0 across core metrics; bias is additive)
STYLE_WEIGHTS: Dict[str, Dict[str, float]] = {
    "fast": {"motion": 0.40, "brightness": 0.20, "beat": 0.30, "duration": 0.10, "bias": 0.05},
    "cinematic": {"motion": 0.15, "brightness": 0.30, "beat": 0.15, "duration": 0.40, "bias": 0.05},
    "beat_sync": {"motion": 0.25, "brightness": 0.20, "beat": 0.40, "duration": 0.15, "bias": 0.05},
    "story": {"motion": 0.15, "brightness": 0.25, "beat": 0.20, "duration": 0.40, "bias": 0.05},
}


def style_weights(style_context: Dict) -> Dict[str, float]:
    return learning.get_adaptive_weights(style_context, STYLE_WEIGHTS)


def motion_score(avg_frame_diff: float) -> float:
    # Normalize motion by a heuristic scale (30 mean absolute diff ~ high motion) and clamp to [0,1].
    if avg_frame_diff is None:
        return 0.0
    return max(0.0, min(1.0, avg_frame_diff / 30.0))


def brightness_score(avg_luma: float) -> float:
    # Normalize luminance 0-255 to 0-1 and softly penalize extremes.
    if avg_luma is None:
        return 0.0
    base = max(0.0, min(1.0, avg_luma / 255.0))
    # Penalize very dark (<0.2) or very bright (>0.9) regions.
    penalty = 0.0
    if base < 0.2:
        penalty = (0.2 - base) * 0.5
    elif base > 0.9:
        penalty = (base - 0.9) * 0.8
    return max(0.0, min(1.0, base - penalty))


def beat_alignment_score(segment_start: float, beats: List[float]) -> float:
    if not beats:
        return 0.5  # neutral when no beats
    nearest = min(beats, key=lambda b: abs(b - segment_start))
    dist = abs(nearest - segment_start)
    if dist <= 0.05:
        return 1.0
    if dist <= 0.10:
        return 0.7
    if dist <= 0.20:
        return 0.3
    return 0.0


def duration_fit_score(segment_duration: float, target_duration: float) -> float:
    if target_duration <= 0:
        return 0.0
    raw = 1.0 - abs(segment_duration - target_duration) / target_duration
    return max(0.0, min(1.0, raw))


def final_score(motion: float, brightness: float, beat_align: float, duration_fit: float, style_context: Dict) -> float:
    w = style_weights(style_context)
    return (
        w["motion"] * motion
        + w["brightness"] * brightness
        + w["beat"] * beat_align
        + w["duration"] * duration_fit
        + w.get("bias", 0.0)
    )


def describe_candidate(idx: int, cand: Dict) -> str:
    m = cand.get("metrics", {})
    return (
        f"cand{idx}: {cand.get('type')} {cand.get('name','?')} score={cand.get('score',0):.3f} "
        f"m={m.get('motion',0):.2f} b={m.get('brightness',0):.2f} "
        f"beat={m.get('beat',0):.2f} fit={m.get('duration_fit',0):.2f}"
    )