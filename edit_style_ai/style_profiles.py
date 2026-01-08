# Static, deterministic style profiles for Layer 1 control.

STYLE_PROFILES = {
    "fast": {
        "style": "fast",
        "shot_length_bias": "short",           # 0.3–0.8s target range (enforced downstream)
        "beat_snap_strength": "high",          # strict snap to beats
        "motion_priority": "high",             # aggressive motion/zoom
        "image_usage": "low",                  # prefer videos over images
        "version": "v1",
    },
    "cinematic": {
        "style": "cinematic",
        "shot_length_bias": "long",            # 1.5–3.0s target
        "beat_snap_strength": "low",           # loose snap
        "motion_priority": "low",              # gentle motion
        "image_usage": "medium",               # allow images with slow zoom
        "version": "v1",
    },
    "beat_sync": {
        "style": "beat_sync",
        "shot_length_bias": "medium",          # 0.8–1.5s
        "beat_snap_strength": "very_high",     # prioritize beat alignment
        "motion_priority": "medium",
        "image_usage": "low",
        "version": "v1",
    },
    "story": {
        "style": "story",
        "shot_length_bias": "varied",          # narrative pacing
        "beat_snap_strength": "medium",
        "motion_priority": "low",
        "image_usage": "high",                 # images OK with gentle moves
        "version": "v1",
    },
}


def get_style_context(name: str) -> dict:
    """Return the immutable style context for a supported style."""
    key = name.strip().lower()
    if key not in STYLE_PROFILES:
        raise ValueError(f"Unsupported style '{name}'. Supported: {list(STYLE_PROFILES.keys())}")
    # Return a shallow copy to avoid mutations downstream.
    return dict(STYLE_PROFILES[key])
