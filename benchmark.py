from ultralytics import YOLO

def generate_benchmarks(model_path: str) -> None:
    print("Working on model:", model_path)
    try:
        model = YOLO(model_path)
        results = model.benchmark(data="coco8.yaml", imgsz=640)
        print(f"Benchmark for {model_path} completed.")
    except Exception as e:
        print(f"Error with model {model_path}: {e}")

model_names = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov9t.pt",
    "yolov9s.pt",
    "yolov10n.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
]

model_folder = "./models"

for model_name in model_names:
    model_path = f"{model_folder}/{model_name}"
    generate_benchmarks(model_path)
