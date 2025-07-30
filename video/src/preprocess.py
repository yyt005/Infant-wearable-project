import os
import subprocess

def split_video(input_path: str, segment_time: int = 60):
    """
    Split a video into smaller segments of segment_time seconds using ffmpeg.

    Args:
        input_path (str): Path to the input video.
        segment_time (int): Segment duration in seconds (default 60).

    Returns:
        List of chunk file paths created in 'data/processed/<video_name>/'
    """
    video_name = os.path.splitext(os.path.basename(input_path))[0]
    output_dir = os.path.join("data", "processed", video_name)
    os.makedirs(output_dir, exist_ok=True)
    output_pattern = os.path.join(output_dir, f"{video_name}_%03d.mp4")

    command = [
        "ffmpeg",
        "-i", input_path,
        "-c", "copy",
        "-map", "0",
        "-segment_time", str(segment_time),
        "-f", "segment",
        output_pattern
    ]

    print(f"Splitting video {input_path} into {segment_time}s segments...")
    subprocess.run(command, check=True)
    print(f"Split videos saved to {output_dir}")

    # Return list of chunk paths sorted by name
    chunk_files = sorted(os.listdir(output_dir))
    return [os.path.join(output_dir, f) for f in chunk_files]
