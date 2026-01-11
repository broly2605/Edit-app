import json
import os
from typing import Dict


STATE_PATH = os.path.join("output", "learning_state.json")
MAX_DELTA = 0.10  # clamp per-weight adjustment
BIAS_MAX = 0.20
LR = 0.02  # small learning rate for stability


def _load_state() -> Dict:
    if not os.path.exists(STATE_PATH):
        return {}
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_state(state: Dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _clamp_adjustments(adj: Dict[str, float]) -> Dict[str, float]:
    out = {}
    for k, v in adj.items():
        if k == "bias":
            out[k] = max(-BIAS_MAX, min(BIAS_MAX, v))
        else:
            out[k] = max(-MAX_DELTA, min(MAX_DELTA, v))
    return out


def get_adaptive_weights(style_context: Dict, base_weights: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    style = (style_context or {}).get("style", "fast")
    user_id = (style_context or {}).get("user_id", "global")
    base = base_weights.get(style, base_weights.get("fast"))

    state = _load_state()
    user_state = state.get(user_id, {})
    adj = user_state.get(style, {})

    # Merge adjustments, clamp, renormalize core weights to sum=1
    motion = base["motion"] + adj.get("motion", 0.0)
    brightness = base["brightness"] + adj.get("brightness", 0.0)
    beat = base["beat"] + adj.get("beat", 0.0)
    duration = base["duration"] + adj.get("duration", 0.0)
    bias = base.get("bias", 0.0) + adj.get("bias", 0.0)

    motion = max(0.01, motion)
    brightness = max(0.01, brightness)
    beat = max(0.01, beat)
    duration = max(0.01, duration)

    total = motion + brightness + beat + duration
    motion /= total
    brightness /= total
    beat /= total
    duration /= total
    bias = max(0.0, min(BIAS_MAX, bias))

    return {
        "motion": motion,
        "brightness": brightness,
        "beat": beat,
        "duration": duration,
        "bias": bias,
    }


def register_feedback(style_context: Dict, decision_metrics: Dict[str, float], signal: str, confidence: float = 1.0) -> None:
    """
    Update adjustments based on outcome signals.
    signal: "exported" (positive), "regenerated"/"replaced" (negative)
    decision_metrics: winner metrics dict with keys motion, brightness, beat, duration_fit
    """
    style = (style_context or {}).get("style", "fast")
    user_id = (style_context or {}).get("user_id", "global")

    if not decision_metrics:
        return

    state = _load_state()
    user_state = state.setdefault(user_id, {})
    adj = user_state.setdefault(style, {"motion": 0.0, "brightness": 0.0, "beat": 0.0, "duration": 0.0, "bias": 0.0})

    # Determine direction
    if signal == "exported":
        direction = 1.0
    elif signal in ("regenerated", "replaced", "rejected"):
        direction = -0.5
    else:
        return

    total = sum([
        decision_metrics.get("motion", 0.0),
        decision_metrics.get("brightness", 0.0),
        decision_metrics.get("beat", 0.0),
        decision_metrics.get("duration_fit", 0.0),
    ])
    if total <= 0:
        return

    # Distribute small adjustments proportional to metric contributions
    for key, metric_key in [("motion", "motion"), ("brightness", "brightness"), ("beat", "beat"), ("duration", "duration_fit")]:
        share = decision_metrics.get(metric_key, 0.0) / total
        delta = direction * LR * confidence * share
        adj[key] = adj.get(key, 0.0) + delta

    # Bias small nudge in same direction
    adj["bias"] = adj.get("bias", 0.0) + direction * LR * 0.5 * confidence

    adj = _clamp_adjustments(adj)
    user_state[style] = adj
    state[user_id] = user_state
    _save_state(state)

    print(f"[Layer4] Updated weights for user={user_id} style={style}: {adj}")