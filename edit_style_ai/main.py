import argparse
import json
import os
from renderer.media_mapper import map_media
from renderer.render_video import render
from analysis.cut_detector import detect_cuts
from analysis.beat_detector import detect_beats
from analysis.blueprint_builder import build_blueprint


def parse_args():
    parser = argparse.ArgumentParser(description="Edit-style transfer pipeline")
    parser.add_argument(
        "--video",
        default="input/d1.mp4",
        help="Reference edited video path (used for analysis and audio)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video

    cuts = detect_cuts(video_path)
    beats = detect_beats(video_path)

    blueprint = build_blueprint(cuts, beats)

    os.makedirs("output", exist_ok=True)
    with open("output/edit_blueprint.json", "w") as f:
        json.dump(blueprint, f, indent=2)

    # Load user videos and images from input directories
    videos = [f"input/user_videos/{v}" for v in os.listdir("input/user_videos/")]
    images = [f"input/user_images/{i}" for i in os.listdir("input/user_images/")]

    # Map media assets to the blueprint based on shot lengths
    timeline = map_media(blueprint["shot_lengths"], videos, images)

    # Render the final video using the reference audio track
    render(timeline, audio_path=video_path)
    print(f"Blueprint generated successfully for {video_path}.")


if __name__ == "__main__":
    main()
