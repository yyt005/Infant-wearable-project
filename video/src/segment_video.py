# segment_video.py
import os
from pathlib import Path
import ffmpeg

def segment_video(input_video_path, segment_dir, segment_length_sec=60):
    os.makedirs(segment_dir, exist_ok=True)
    input_video = Path(input_video_path)

    print(f"Segmenting {input_video.name} into {segment_length_sec}-second chunks...")
    ffmpeg.input(str(input_video)).output(
        f"{segment_dir}/segment_%03d.mp4",
        f="segment",
        segment_time=segment_length_sec,
        reset_timestamps=1,
        c="copy"
    ).run(quiet=True)
    print(f"Segments saved to {segment_dir}")
