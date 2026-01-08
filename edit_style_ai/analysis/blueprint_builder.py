import numpy as np


def _snap_cuts_to_beats(cuts, beats, window=0.12):
    """Snap each cut to the nearest beat within a time window."""
    if not beats:
        return cuts[:]

    snapped = []
    for cut in cuts:
        nearest = min(beats, key=lambda b: abs(b - cut))
        if abs(nearest - cut) <= window:
            snapped.append(nearest)
        else:
            snapped.append(cut)
    return snapped


def _shot_lengths_from_cuts(cuts):
    if len(cuts) < 2:
        return []
    return [cuts[i + 1] - cuts[i] for i in range(len(cuts) - 1)]


def build_blueprint(cuts, beats, snap_window=0.12):
    """Build the edit blueprint with optional cut snapping and rhythm stats."""
    # Ensure cuts are sorted and start at 0.0
    cuts = sorted(set([max(0.0, c) for c in cuts]))
    if cuts and cuts[0] > 0.05:
        cuts = [0.0] + cuts

    snapped_cuts = _snap_cuts_to_beats(cuts, beats, window=snap_window)

    shot_lengths = _shot_lengths_from_cuts(snapped_cuts)

    beat_cut_matches = 0
    for cut in snapped_cuts:
        for beat in beats:
            if abs(cut - beat) < 0.1:
                beat_cut_matches += 1
                break

    ratio = beat_cut_matches / max(len(snapped_cuts), 1)

    rhythm_stats = {
        "count": len(shot_lengths),
        "mean": float(np.mean(shot_lengths)) if shot_lengths else 0.0,
        "std": float(np.std(shot_lengths)) if shot_lengths else 0.0,
        "median": float(np.median(shot_lengths)) if shot_lengths else 0.0,
    }

    return {
        "cuts_original": cuts,
        "cuts": snapped_cuts,
        "shot_lengths": shot_lengths,
        "beats": beats,
        "cut_on_beat_ratio": round(ratio, 2),
        "snap_window": snap_window,
        "rhythm_stats": rhythm_stats,
    }
