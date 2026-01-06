import numpy as np
import librosa
from moviepy.editor import VideoFileClip


def detect_beats(video_path, target_sr=44100):
    """Detect beat timestamps from the video's audio track.

    Steps:
    - Extract audio with moviepy (requires ffmpeg on PATH).
    - Convert to mono and normalize.
    - Run librosa beat tracker.
    Returns: list of beat times in seconds.
    """

    clip = VideoFileClip(video_path)

    if clip.audio is None:
        clip.close()
        return []

    try:
        # Gather audio chunks to avoid stack errors on generators
        chunks = list(clip.audio.iter_chunks(fps=target_sr, quantize=True, nbytes=2, chunksize=2048))
    finally:
        clip.close()

    if not chunks:
        return []

    audio = np.vstack(chunks)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Guard against extremely short audio
    if len(audio) < target_sr * 0.5:
        return []

    audio = audio.astype(np.float32)

    tempo, beats = librosa.beat.beat_track(y=audio, sr=target_sr)
    beat_times = librosa.frames_to_time(beats, sr=target_sr)
    return beat_times.tolist()
