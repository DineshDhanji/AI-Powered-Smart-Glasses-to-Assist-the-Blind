import numpy as np
import cv2

from typing import Dict, List, Tuple

from utils import (
    load_camera_calibration,
    load_model,
    detect_objects_with_yolo,
    calculate_distance_to_object,
)
from constants import *
from logger import logger


class Navigation_Module:

    def __init__(
        self,
        calibration_file: str,
        model_path: str,
        known_object_width_meters: Dict[str, float],
        total_regions: int,
        threshold_distance: float,
    ):
        self.calibration_file = calibration_file
        self.model_path = model_path
        self.known_object_width_meters = known_object_width_meters
        self.total_regions = total_regions
        self.threshold_distance = threshold_distance

    def navigation_decision(
        self,
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
            x1, x2 = (detections[i][0], detections[i][2])
            frame = frame_width / total_regions
            distance = distances[i][1]
            if distance < 0 or distance > threshold_distance:
                continue

            # Map object location to frame sections
            if x1 <= frame and x2 <= frame:
                safety_status["left"] = False
            elif x2 <= 2 * frame:
                safety_status["center"] = False
                if x1 <= frame:
                    safety_status["left"] = False
            else:
                safety_status["right"] = False
                if x1 <= frame:
                    safety_status["left"] = False
                    safety_status["center"] = False
                elif x1 <= 2 * frame:
                    safety_status["center"] = False

        print("Safety status: ", safety_status)

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
            return "Stop and turn. No way ahead."

    def draw_guidance(self, frame: np.ndarray, decision: str) -> np.ndarray:
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
            0.7,
            (255, 0, 0),
            2,
        )
        return frame

    def draw_shaded_regions(
        self, frame: np.ndarray, frame_with_guidance: np.ndarray
    ) -> None:
        """
        Applies shaded regions with specified colors and opacity to the input frame.
        This function divides the input frame into three vertical regions (left, center, and right),
        and overlays each region with a specific color (red, green, and blue respectively) with a
        specified opacity. The shading is applied directly to the input frame.
        Args:
            frame (np.ndarray): The original frame to which the shaded regions will be applied.
                                This array is modified in place.
            frame_with_guidance (np.ndarray): A copy of the original frame used as a base for
                                            creating the overlay.
        Returns:
            None: The function modifies the `frame` in place and does not return any value.
        Notes:
            - The colors for the regions are defined in BGR format:
                - Left region: Red (0, 0, 255)
                - Center region: Green (0, 255, 0)
                - Right region: Blue (255, 0, 0)
            - The opacity of the overlay is controlled by the `alpha` parameter, which is set to 0.4
            by default (40% opacity).
            - The frame is divided into three equal vertical regions based on its width.
        """
        # Shading of regions
        _, width, _ = frame.shape
        region_width = width // 3
        # Define color shades for the regions (BGR format)
        left_color = (0, 0, 255)  # Red
        center_color = (0, 255, 0)  # Green
        right_color = (255, 0, 0)  # Blue

        # Define alpha for blending (opacity: 0 = fully transparent, 1 = fully opaque)
        alpha = 0.4  # You can adjust this value (e.g., 0.4 for 40% opacity)

        # Create an overlay for the frame with the same dimensions as the original frame
        overlay = frame_with_guidance.copy()

        # Apply opacity to each region by blending
        # Top region (apply red shade with opacity)
        overlay_left = overlay[:, :region_width].copy()
        overlay_left[:, :] = left_color  # Assign red to the region
        frame[:, :region_width] = cv2.addWeighted(
            overlay_left, alpha, frame[:, :region_width], 1 - alpha, 0
        )

        # Middle region (apply green shade with opacity)
        overlay_center = overlay[:, region_width : 2 * region_width].copy()
        overlay_center[:, :] = center_color  # Assign green to the region
        frame[:, region_width : 2 * region_width] = cv2.addWeighted(
            overlay_center,
            alpha,
            frame[:, region_width : 2 * region_width],
            1 - alpha,
            0,
        )

        # Bottom region (apply blue shade with opacity)
        overlay_right = overlay[:, 2 * region_width :].copy()
        overlay_right[:, :] = right_color  # Assign blue to the region
        frame[:, 2 * region_width :] = cv2.addWeighted(
            overlay_right, alpha, frame[:, 2 * region_width :], 1 - alpha, 0
        )

    def initiate_navigation(
        self,
        shaded_regions: bool = False,
        render: bool = False,
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
            cam_matrix, _ = load_camera_calibration(calibration_file)

            # Open the camera feed
            camera = cv2.VideoCapture(0)
            if not camera.isOpened():
                raise IOError("Error opening the camera.")

            logger.info("Camera opened successfully.")

            # Load the YOLO model
            model = load_model(model_path)

            while True:
                ret, frame = camera.read()
                if not ret:
                    logger.error("Failed to grab frame.")
                    break

                # Detect objects using YOLO
                boxes, labels, confidences, class_ids, img_with_boxes, detections = (
                    detect_objects_with_yolo(image=frame, model=model)
                )

                if len(boxes) == 0:
                    logger.debug("No objects detected.")
                    continue

                # Calculate distances to detected objects
                distances = []

                for i in range(len(boxes)):
                    box = boxes[i]

                    try:
                        class_id = int(class_ids[i])
                        label = labels[class_id]
                    except IndexError as e:
                        logger.error(
                            f"Index error on class_id or label at index {i}: {e}"
                        )
                        continue

                    object_width_pixels = box[2]  # Detected object width in pixels
                    distance_meters = calculate_distance_to_object(
                        object_width_pixels,
                        known_object_width_meters,
                        cam_matrix,
                        label,
                    )
                    distances.append((label, distance_meters))

                    # Draw bounding boxes with distances
                    x1 = int(box[0] - box[2] / 2)
                    y1 = int(box[1] - box[3] / 2)
                    x2 = int(box[0] + box[2] / 2)
                    y2 = int(box[1] + box[3] / 2)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        img_with_boxes,
                        f"{label}: {distance_meters:.2f}m",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                # Make navigation decision
                decision = self.navigation_decision(
                    detections=detections,
                    distances=distances,
                    frame_width=frame.shape[1],
                    total_regions=self.total_regions,
                    threshold_distance=self.threshold_distance,
                )
                logger.debug("Navigation decision: ", decision)

                if render:
                    # Overlay decision on the frame
                    frame_with_guidance = self.draw_guidance(img_with_boxes, decision)

                    if shaded_regions:
                        self.draw_shaded_regions(
                            frame=frame, frame_with_guidance=frame_with_guidance
                        )

                    # Display the frame
                    cv2.namedWindow("Navigation Feed", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Navigation Feed", WIDTH, HEIGHT)
                    cv2.imshow("Navigation Feed", frame_with_guidance)

                # Exit if ESC key is pressed
                if cv2.waitKey(1) & 0xFF == ESC_KEY:
                    logger.debug("Escape key pressed. Exiting...")
                    break

        except Exception as e:
            logger.error(f"Error: {e}")

        finally:
            # Release the camera and close windows
            if "camera" in locals() and camera.isOpened():
                camera.release()
            cv2.destroyAllWindows()
