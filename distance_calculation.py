import cv2
import numpy as np
from ultralytics import YOLO
import pickle
from pathlib import Path
from typing import Dict


# Load camera calibration data
def load_camera_calibration(calibration_file: str):
    if not Path(calibration_file).exists():
        raise FileNotFoundError(f"Calibration file {calibration_file} not found\t\t❌")

    with open(calibration_file, "rb") as f:
        calib_data = pickle.load(f)
        cam_matrix = calib_data["cam_matrix"]
        dist_coeffs = calib_data["dist_coeffs"]

    print("Camera calibration data loaded successfully\t\t✅")
    return cam_matrix, dist_coeffs


# Function to detect objects using YOLO and get the bounding box
def detect_objects_with_yolo(image, model_path):
    # Load the YOLO model
    model = YOLO(model_path)

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

    return boxes, labels, confidences, class_ids, image


# Function to calculate the distance to the object
def calculate_distance_to_object(
    object_width_pixels, known_object_width_meters, cam_matrix, label
):
    # Focal length in pixels (from the camera matrix)
    focal_length_pixels = cam_matrix[0, 0]  # fx (focal length in pixels)
    try:
        print("L is ", label)
        x = known_object_width_meters[label]
        # Calculate distance using the formula: distance = (object width * focal length) / object width in pixels
        distance_meters = (x * focal_length_pixels) / object_width_pixels
    except:
        distance_meters = -1
    return distance_meters


def distance_calculation(
    calibration_file: str,
    model_path: str,
    known_object_width_meters: Dict[str, float],
):

    # Load the camera calibration data
    cam_matrix, dist_coeffs = load_camera_calibration(calibration_file)

    # Open the camera feed
    camera = cv2.VideoCapture(0)
    assert camera.isOpened(), "Error reading video file"
    print("Camera opened successfully\t\t✅")

    try:
        print("Initiating live preview")
        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame\t\t❌")
                break

            # Detect objects in the frame using YOLO
            boxes, labels, confidences, class_ids, img_with_boxes = (
                detect_objects_with_yolo(frame, model_path)
            )

            # Check if we detected any objects
            if len(boxes) == 0:
                print("No objects detected.")
                continue

            distances = []  # To store distances for all detected objects

            for i, box in enumerate(boxes):
                # print(box)
                object_width_pixels = box[2]  # Width of the detected object
                class_id = class_ids[i]  # Get the class ID
                label = labels[int(class_id)]  # Get the label from class ID
                confidence = confidences[i]  # Confidence score of the detection

                print(f"Detected object: {label} with confidence: {confidence:.2f}")

                # Calculate the distance to the object
                distance_meters = calculate_distance_to_object(
                    object_width_pixels, known_object_width_meters, cam_matrix, label
                )
                distances.append((label, distance_meters))

                # Draw the bounding box around the detected object and display the distance
                x1 = int(box[0] - box[2] / 2)
                y1 = int(box[1] - box[3] / 2)
                x2 = int(box[0] + box[2] / 2)
                y2 = int(box[1] + box[3] / 2)

                # Draw the bounding box on the image
                cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    img_with_boxes,
                    f"{label}: {distance_meters:.2f}m",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            # Print all detected objects with their distances
            for label, distance in distances:
                print(f"{label}: {distance:.2f} meters")

            # Display the frame with the bounding box and distance on the screen
            cv2.imshow("Camera Live Feed", img_with_boxes)

            # Check for the ESC key to exit the loop
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # 27 is the ASCII code for ESC
                print("Escape key pressed. Exiting...")
                break

    finally:
        # Release the camera and close any OpenCV windows
        camera.release()
        cv2.destroyAllWindows()


# Path to the calibration file and YOLO model
calibration_file = "calibration.pkl"  # Path to your camera calibration file
model_path = "./models/yolo11n.pt"  # Path to your YOLO model file
known_object_width_meters = {
    "chair": 0.4064,
    "person": 0.4064,
}
# Known real-world width of the object (e.g., a car) in meters

distance_calculation(
    calibration_file=calibration_file,
    model_path=model_path,
    known_object_width_meters=known_object_width_meters,
)
