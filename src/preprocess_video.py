import ffmpeg
from pathlib import Path

def convert_to_5fps(input_path, output_path):
    """Convert video to 5 fps."""
    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), r=5, vcodec="libx264", crf=18, preset="fast")
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path

def stabilize_video(input_path, output_path):
    """Stabilize video using ffmpeg vidstab plugin."""
    transforms_file = Path(output_path).with_suffix(".trf")

    # Step 1: detect transforms
    (
        ffmpeg
        .input(str(input_path))
        .output("null", vf="vidstabdetect=shakiness=10:accuracy=15", f="null")
        .overwrite_output()
        .run(quiet=True)
    )

    # Step 2: apply transforms
    (
        ffmpeg
        .input(str(input_path))
        .output(str(output_path), vf="vidstabtransform=smoothing=30", vcodec="libx264", crf=18, preset="fast")
        .overwrite_output()
        .run(quiet=True)
    )

    return output_path
