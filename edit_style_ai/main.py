import argparse
import json
import os
from renderer.render_video import render
from renderer.decision_planner import plan_timeline
from analysis.cut_detector import detect_cuts
from analysis.beat_detector import detect_beats
from analysis.blueprint_builder import build_blueprint
from style_profiles import get_style_context


def parse_args():
    parser = argparse.ArgumentParser(description="Edit-style transfer pipeline (Layer 1: explicit style)")
    parser.add_argument(
        "--video",
        required=True,
        help="Path to reference edited video for analysis and audio (explicit)",
    )
    parser.add_argument(
        "--style",
        required=True,
        help="Explicit style name (e.g., fast, cinematic, beat_sync, story). No auto mode.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    video_path = args.video
    style_context = get_style_context(args.style)

    print(f"[Layer1] Using style: {style_context['style']} (version {style_context['version']})")

    cuts = detect_cuts(video_path)
    beats = detect_beats(video_path)

    blueprint = build_blueprint(cuts, beats)

    os.makedirs("output", exist_ok=True)
    with open("output/edit_blueprint.json", "w") as f:
        json.dump(blueprint, f, indent=2)

    videos = [f"input/user_videos/{v}" for v in os.listdir("input/user_videos/")]
    images = [f"input/user_images/{i}" for i in os.listdir("input/user_images/")]

    timeline = plan_timeline(blueprint, videos, images, style_context)

    render(timeline, audio_path=video_path)
    print(f"[Layer1/2] Completed with style '{style_context['style']}' and reference '{video_path}'")


if __name__ == "__main__":
    main()
