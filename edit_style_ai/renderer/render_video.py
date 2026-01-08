import os
import shutil
import subprocess
import tempfile
from shutil import which

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None


def _scale_filter():
    # Fit to cover 1080x1920: scale up preserving AR, then center-crop.
    return "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"


def _run(cmd):
    ffmpeg_bin = None

    # Prefer bundled ffmpeg from imageio-ffmpeg if available
    if imageio_ffmpeg is not None:
        try:
            ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_bin = None

    # Fallback to PATH
    if ffmpeg_bin is None:
        ffmpeg_bin = which("ffmpeg")

    if cmd and cmd[0] == "ffmpeg":
        if ffmpeg_bin:
            cmd = [ffmpeg_bin] + cmd[1:]
        else:
            raise FileNotFoundError(
                "ffmpeg not found. Install it, add to PATH, or install imageio-ffmpeg."
            )

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("\nFFmpeg stderr:\n", res.stderr)
        raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=res.stderr)


def render(timeline, output="output/final.mp4", audio_path=None):
    """
    FFmpeg-based render: per-shot transcode to 1080x1920 MP4, then concat, then mux audio if provided.
    MoviePy is avoided for rendering to prevent Windows handle issues.
    """

    os.makedirs(os.path.dirname(output) or "./", exist_ok=True)
    tmpdir = tempfile.mkdtemp(prefix="render_ffmpeg_")
    shot_files = []

    try:
        # Per-shot render
        for idx, shot in enumerate(timeline):
            asset = shot["asset"]
            if not os.path.exists(asset):
                raise FileNotFoundError(f"Asset not found: {asset}")

            duration = max(float(shot.get("duration", 0)), 0.01)
            out_path = os.path.join(tmpdir, f"shot_{idx}.mp4")

            if shot["type"] == "image":
                cmd = [
                    "ffmpeg", "-y",
                    "-loop", "1",
                    "-i", asset,
                    "-t", str(duration),
                    "-vf", _scale_filter(),
                    "-r", "30",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    out_path,
                ]
            else:
                cmd = [
                    "ffmpeg", "-y",
                    "-i", asset,
                    "-t", str(duration),
                    "-vf", _scale_filter(),
                    "-r", "30",
                    "-pix_fmt", "yuv420p",
                    "-an",
                    out_path,
                ]

            _run(cmd)
            shot_files.append(out_path)

        if not shot_files:
            print("No clips were processed successfully")
            return

        # Concat list file
        list_file = os.path.join(tmpdir, "files.txt")
        with open(list_file, "w", encoding="utf-8") as f:
            for p in shot_files:
                f.write(f"file '{p}'\n")

        concat_out = os.path.join(tmpdir, "concat.mp4")
        _run([
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", list_file,
            "-c", "copy",
            concat_out,
        ])

        if audio_path:
            # Mux reference audio, re-encode audio only
            cmd = [
                "ffmpeg", "-y",
                "-i", concat_out,
                "-i", audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0?",
                "-c:v", "copy",
                "-c:a", "aac",
                "-shortest",
                output,
            ]
            try:
                _run(cmd)
                print(f"Video rendered successfully: {output}")
                return
            except subprocess.CalledProcessError as e:
                print(f"Warning: audio mux failed ({e}); writing video without audio")

        # Fallback: move concat video as final
        shutil.move(concat_out, output)
        print(f"Video rendered successfully: {output}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
