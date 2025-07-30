import os
from ultralytics import YOLO

def run_object_detection(video_path: str, output_dir: str = "results"):
    model = YOLO("yolov8n.pt")
    results = model(video_path, save=True, project=f"{output_dir}/runs", name="object_detect", stream=True)

    os.makedirs(f"{output_dir}/csv", exist_ok=True)
    csv_path = f"{output_dir}/csv/object_detect_results.csv"

    with open(csv_path, "w") as f:
        f.write("frame_number,object_class,confidence_score,x1,y1,x2,y2\n")
        for frame_idx, r in enumerate(results):
            for box, conf, cls in zip(r.boxes.xyxy.cpu().numpy(),
                                      r.boxes.conf.cpu().numpy(),
                                      r.boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = box
                f.write(f"{frame_idx},{int(cls)},{conf},{x1},{y1},{x2},{y2}\n")

    print("Object detection complete. Results saved.")
