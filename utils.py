from ultralytics import YOLO
import pickle
from pathlib import Path
from typing import Dict

from logger import logger


# Load camera calibration data
def load_camera_calibration(calibration_file: str):
    if not Path(calibration_file).exists():
        raise FileNotFoundError(f"Calibration file {calibration_file} not found.")

    with open(calibration_file, "rb") as f:
        calib_data = pickle.load(f)
        cam_matrix = calib_data["cam_matrix"]
        dist_coeffs = calib_data["dist_coeffs"]

    logger.info("Camera calibration data loaded successfully")
    return cam_matrix, dist_coeffs


# Load the YOLO model
def load_model(model_path: str):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model path file {model_path} not found")
    try:
        model = YOLO(model_path)
    except Exception as e:
        raise RuntimeError(f"Error loading model {model_path}: {e}")
    logger.info("Model loaded successfully. Model type: %s", model.model_name)
    return model


# Function to detect objects using YOLO and get the bounding box
def detect_objects_with_yolo(image, model):
    # Perform inference (detection)
    results = model(image)

    # Results are in a list format; each result corresponds to a batch image
    result = results[0]  # Get the first result from the list

    # Extract bounding boxes, labels, and confidences
    boxes = (
        result.boxes.xywh.cpu().numpy()
    )  # Format: [x_center, y_center, width, height]
    confidences = result.boxes.conf.cpu().numpy()  # Confidences of detections
    class_ids = result.boxes.cls.cpu().numpy()  # Class IDs for each detected object
    labels = result.names  # Class names from the model

    return boxes, labels, confidences, class_ids, image, result.boxes.xyxy.cpu().numpy()


# Function to calculate the distance to the object
def calculate_distance_to_object(
    object_width_pixels, known_object_width_meters, cam_matrix, label
):
    # Focal length in pixels (from the camera matrix)
    focal_length_pixels = cam_matrix[0, 0]  # fx (focal length in pixels)
    if label not in known_object_width_meters or object_width_pixels <= 0:
        # If the label is not found, set distance to infinity
        return -1.0  # Unmeasurable
    real_width = known_object_width_meters[label]
    # Calculate distance using the formula: distance = (object width * focal length) / object width in pixels
    return (real_width * focal_length_pixels) / object_width_pixels
