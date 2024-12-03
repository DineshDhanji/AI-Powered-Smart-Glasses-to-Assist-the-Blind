import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, List, Tuple
from utils import (
    load_camera_calibration,
    detect_objects_with_yolo,
    calculate_distance_to_object,
)

# Constants
ESC_KEY = 27  # ASCII for the ESC key
DEFAULT_FRAME_WIDTH = 640  # Default camera frame width
DEFAULT_FRAME_HEIGHT = 480  # Default camera frame height


def navigation_decision(
    detections: np.ndarray,
    distances: List[Tuple[str, float]],
    frame_width: int,
    total_regions: int,
    threshold_distance: float,
) -> str:
    """
    Decide navigation direction based on detected objects and their distances.

    Args:
        detections (np.ndarray): Detected object bounding boxes.
        distances (List[Tuple[str, float]]): List of objects and their distances.
        frame_width (int): Width of the frame.
        total_regions (int): Number of regions for navigation.
        threshold_distance (float): Threshold for considering an object as too close.

    Returns:
        str: Navigation decision as a direction string.
    """
    safety_status = {"left": True, "center": True, "right": True}

    for i in range(detections.shape[0]):
        x_center = (detections[i][0] + detections[i][2]) / 2
        distance = distances[i][1]

        # Map object location to frame sections
        if x_center <= frame_width / total_regions:
            if distance < threshold_distance:
                safety_status["left"] = False
        elif x_center <= 2 * frame_width / total_regions:
            if distance < threshold_distance:
                safety_status["center"] = False
        else:
            if distance < threshold_distance:
                safety_status["right"] = False

    # Determine navigation based on safety status
    if (
        safety_status["left"]
        and not safety_status["center"]
        and not safety_status["right"]
    ):
        return "Move left."
    elif (
        safety_status["right"]
        and not safety_status["center"]
        and not safety_status["left"]
    ):
        return "Move right."
    elif (
        safety_status["center"]
        and not safety_status["left"]
        and not safety_status["right"]
    ):
        return "Move forward."
    elif safety_status["left"] and safety_status["right"]:
        return "Move to the side with more space."
    else:
        return "Stop and turn."


def draw_guidance(frame: np.ndarray, decision: str) -> np.ndarray:
    """
    Overlay navigation decision onto the frame.

    Args:
        frame (np.ndarray): Video frame.
        decision (str): Navigation decision.

    Returns:
        np.ndarray: Frame with decision overlay.
    """
    text_position = (10, 30)
    cv2.putText(
        frame,
        f"Decision: {decision}",
        text_position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return frame


def navigation(
    calibration_file: str,
    model_path: str,
    known_object_width_meters: Dict[str, float],
    total_regions: int,
    threshold_distance: float,
):
    """
    Main navigation function for detecting objects, estimating distances, and guiding the user.

    Args:
        calibration_file (str): Path to the camera calibration file.
        model_path (str): Path to the YOLO model file.
        known_object_width_meters (Dict[str, float]): Real-world widths of known objects in meters.
        total_regions (int): Number of sections for navigation.
        threshold_distance (float): Threshold for considering an object as too close.
    """
    try:
        # Load camera calibration data
        cam_matrix, dist_coeffs = load_camera_calibration(calibration_file)

        # Open the camera feed
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise IOError("Error opening the camera.")

        print("Camera opened successfully.")

        while True:
            ret, frame = camera.read()
            if not ret:
                print("Failed to grab frame.")
                break

            # Detect objects using YOLO
            boxes, labels, confidences, class_ids, img_with_boxes, detections = (
                detect_objects_with_yolo(frame, model_path)
            )
            if len(boxes) == 0:
                print("No objects detected.")
                continue

            # Calculate distances to detected objects
            distances = []
            for i, box in enumerate(boxes):
                label = labels[class_ids[i]]
                object_width_pixels = box[2]  # Detected object width in pixels
                distance_meters = calculate_distance_to_object(
                    object_width_pixels, known_object_width_meters, cam_matrix, label
                )
                distances.append((label, distance_meters))

                # Draw bounding boxes with distances
                x1 = int(box[0] - box[2] / 2)
                y1 = int(box[1] - box[3] / 2)
                x2 = int(box[0] + box[2] / 2)
                y2 = int(box[1] + box[3] / 2)
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

            # Make navigation decision
            decision = navigation_decision(
                detections=detections,
                distances=distances,
                frame_width=frame.shape[1],
                total_regions=total_regions,
                threshold_distance=threshold_distance,
            )
            print("Navigation decision: ", decision)

            # Overlay decision on the frame
            frame_with_guidance = draw_guidance(img_with_boxes, decision)

            # Display the frame
            cv2.imshow("Navigation Feed", frame_with_guidance)

            # Exit if ESC key is pressed
            if cv2.waitKey(1) & 0xFF == ESC_KEY:
                print("Escape key pressed. Exiting...")
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        # Release the camera and close windows
        if "camera" in locals() and camera.isOpened():
            camera.release()
        cv2.destroyAllWindows()


# Configuration
calibration_file = "calibration.pkl"
model_path = "./models/yolo11n.pt"
known_object_width_meters = {"chair": 0.4064, "person": 0.4064}
total_regions = 3
threshold_distance = 3.0

navigation(
    calibration_file=calibration_file,
    model_path=model_path,
    known_object_width_meters=known_object_width_meters,
    total_regions=total_regions,
    threshold_distance=threshold_distance,
)
