import random


def map_media(shot_lengths, video_files, image_files, style_context=None):
    media = video_files + image_files
    random.shuffle(media)

    style_name = style_context.get("style") if style_context else None

    timeline = []
    for i, length in enumerate(shot_lengths):
        asset = media[i % len(media)]
        timeline.append({
            "asset": asset,
            "duration": round(length, 2),
            "type": "image" if asset.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")) else "video",
            "style": style_name,
        })

    return timeline
