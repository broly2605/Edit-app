import json
import os
from renderer.media_mapper import map_media
from renderer.render_video import render
from analysis.cut_detector import detect_cuts
from analysis.beat_detector import detect_beats
from analysis.blueprint_builder import build_blueprint

# Path to the input video file
VIDEO = "input/d1.mp4"
VIDEO = "input/d2.mp4"

cuts = detect_cuts(VIDEO)
beats = detect_beats(VIDEO)

blueprint = build_blueprint(cuts, beats)

with open("output/edit_blueprint.json", "w") as f:
    json.dump(blueprint, f, indent=2)

# Load user videos and images from input directories
videos = [f"input/user_videos/{v}" for v in os.listdir("input/user_videos/")]
images = [f"input/user_images/{i}" for i in os.listdir("input/user_images/")]

# Map media assets to the blueprint based on shot lengths
timeline = map_media(blueprint["shot_lengths"], videos, images)

# Render the final video
render(timeline)
print("Blueprint generated successfully.")
