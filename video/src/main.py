import os
from preprocess import split_video
from face_detect import run_face_detection
from object_detect import run_object_detection

if __name__ == "__main__":
    data_dir = "data"
    segment_length = 60  # seconds

    video_files = os.listdir(data_dir)

    for video_file in video_files:
        video_path = os.path.join(data_dir, video_file)
        # print(f" Processing video: {video_file}")

        # Step 1: Split video into chunks under data/processed/<video_name>/
        chunk_paths = split_video(video_path, segment_length)

        # Step 2: Run detection on each chunk
        for chunk_path in chunk_paths:
            chunk_name = os.path.basename(chunk_path)
            # print(f" Running detection on chunk {chunk_name}")
            run_face_detection(chunk_path)
            run_object_detection(chunk_path)

        print(f"Finished processing {video_file}")

    print("All videos processed.")
