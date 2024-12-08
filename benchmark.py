from ultralytics import YOLO

# Load a YOLO11n PyTorch model
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
    print("Working on model:", model_name)
    model = YOLO(f"{model_folder}/model_name")

    # Benchmark YOLO11n speed and accuracy on the COCO8 dataset for all all export formats
    results = model.benchmark(data="coco8.yaml", imgsz=640)
