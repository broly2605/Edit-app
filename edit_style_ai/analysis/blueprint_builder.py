def build_blueprint(cuts, beats):
    shot_lengths = [
        cuts[i+1] - cuts[i] for i in range(len(cuts)-1)
    ]

    beat_cut_matches = 0
    for cut in cuts:
        for beat in beats:
            if abs(cut - beat) < 0.1:
                beat_cut_matches += 1
                break

    ratio = beat_cut_matches / max(len(cuts), 1)

    return {
        "cuts": cuts,
        "shot_lengths": shot_lengths,
        "beats": beats,
        "cut_on_beat_ratio": round(ratio, 2)
    }
