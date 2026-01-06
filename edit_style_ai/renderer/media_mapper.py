import random

def map_media(shot_lengths, video_files, image_files):
    media = video_files + image_files
    random.shuffle(media)

    timeline = []
    for i, length in enumerate(shot_lengths):
        asset = media[i % len(media)]
        timeline.append({
            "asset": asset,
            "duration": round(length, 2),
            "type": "image" if asset.endswith((".jpg", ".png")) else "video"
        })

    return timeline
