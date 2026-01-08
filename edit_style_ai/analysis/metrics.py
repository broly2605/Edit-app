import numpy as np


def cut_timing_accuracy(ref_cuts, gen_cuts, tolerance=0.1):
    """Percent of generated cuts within tolerance of any reference cut."""
    if not gen_cuts or not ref_cuts:
        return 0.0
    matches = 0
    for g in gen_cuts:
        if any(abs(g - r) <= tolerance for r in ref_cuts):
            matches += 1
    return matches / len(gen_cuts)


def beat_alignment_score(cuts, beats, tolerance=0.1):
    """Percent of cuts that land within tolerance of a beat."""
    if not cuts or not beats:
        return 0.0
    hits = 0
    for c in cuts:
        if any(abs(c - b) <= tolerance for b in beats):
            hits += 1
    return hits / len(cuts)


def rhythm_similarity(ref_shots, gen_shots, bins=12):
    """Similarity of shot-length distributions (histogram cosine similarity)."""
    if not ref_shots or not gen_shots:
        return 0.0

    ref = np.array(ref_shots, dtype=float)
    gen = np.array(gen_shots, dtype=float)

    # Normalize lengths by their median to reduce scale sensitivity
    def _norm(x):
        med = np.median(x)
        return x / med if med > 0 else x

    ref = _norm(ref)
    gen = _norm(gen)

    hist_ref, _ = np.histogram(ref, bins=bins, range=(0, max(ref.max(), gen.max(), 1)))
    hist_gen, _ = np.histogram(gen, bins=bins, range=(0, max(ref.max(), gen.max(), 1)))

    def _cos(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        return float(np.dot(a, b) / denom) if denom > 0 else 0.0

    return _cos(hist_ref, hist_gen)


def final_edit_match_score(ref_bp, gen_bp, weights=None):
    """Weighted score combining cut timing, beat alignment, and rhythm similarity."""
    if weights is None:
        weights = {"cuts": 0.4, "beats": 0.3, "rhythm": 0.3}

    cuts_score = cut_timing_accuracy(ref_bp.get("cuts", []), gen_bp.get("cuts", []))
    beat_score = beat_alignment_score(gen_bp.get("cuts", []), ref_bp.get("beats", []))
    rhythm_score = rhythm_similarity(ref_bp.get("shot_lengths", []), gen_bp.get("shot_lengths", []))

    return (
        weights.get("cuts", 0) * cuts_score
        + weights.get("beats", 0) * beat_score
        + weights.get("rhythm", 0) * rhythm_score
    )
