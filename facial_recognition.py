import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
import pickle

import matplotlib.pyplot as plt
from constants import facial_recog_model_path, facial_recog_feature_scalar
from logger import logger
from utils import detect_objects_with_yolo, load_model


class Facial_Recognition:
    def __init__(self, model_path=facial_recog_model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.radius = 1
        self.n_points = 8 * self.radius
        self.label_names = {0: "Aneeq", 1: "Hamza"}
        self.padding_ratio = 0.4

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )
        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]

        pad_w = int(w * self.padding_ratio)
        pad_h = int(h * self.padding_ratio)

        x1 = max(x - pad_w, 0)
        y1 = max(y - pad_h, 0)
        x2 = min(x + w + pad_w, image.shape[1])
        y2 = min(y + h + pad_h, image.shape[0])

        face = image[y1:y2, x1:x2]
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    def rgb_2_gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def apply_ellipsoidal_mask(self, image):
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w // 2, h // 2)
        axes = (int(w * 0.4), int(h * 0.55))
        angle = 0
        start_angle = 0
        end_angle = 360
        cv2.ellipse(mask, center, axes, angle, start_angle, end_angle, 255, -1)
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        return masked_image

    def apply_lbp_resize(self, image):
        image_resized = cv2.resize(image, (48, 48))
        image_squeeze = (image_resized.squeeze() * 255).astype(np.uint8)
        lbp = local_binary_pattern(
            image_resized, self.n_points, self.radius, method="uniform"
        )
        lbp_cropped = lbp[:46, :46]
        lbp_cropped = lbp_cropped.astype(np.float32)
        return lbp_cropped

    def predict_image(self, image):
        model = tf.keras.models.load_model("./models/FYP_FR_Model_v1.keras")

        try:
            with open(facial_recog_feature_scalar, "rb") as f:
                scaler = pickle.load(f)

            scaled_image = scaler.transform(image.reshape(1, -1)).reshape(1, 46, 46, 1)
            prediction = model.predict(scaled_image, verbose=0)
            confidence = np.max(prediction) * 100
            predicted_class = np.argmax(prediction)
            true_prediction = self.label_names.get(predicted_class, "Unknown")

            if confidence >= 70:
                final_prediction = true_prediction
            else:
                final_prediction = "Unknown"

            print(
                f"Original Prediction: {true_prediction}\nConfidence: {confidence:.2f}%\nFinal Prediction: {final_prediction}"
            )
            message = f"Prediction: {final_prediction}"
            return final_prediction

        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Error in prediction"

    def identify_person(self, person_image):
        img_color = self.detect_face(person_image)
        if img_color is None:
            print("Face not found")
            return None

        img_gray = self.rgb_2_gray(img_color)

        img_masked = self.apply_ellipsoidal_mask(img_gray)

        img_lbp = self.apply_lbp_resize(img_masked)

        plt.figure(figsize=(7.5, 2.5))

        plt.subplot(1, 4, 1)
        plt.imshow(img_color)
        plt.title("Detected Face")
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.imshow(img_gray, cmap="gray")
        plt.title("Grayscale")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.imshow(img_masked, cmap="gray")
        plt.title("Masked")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.imshow(img_lbp, cmap="gray")
        plt.title("LBP")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

        prediction = self.predict_image(img_lbp)
        return prediction

    def initialize_recognition(self, yolo_model, tts_engine):
        # Open the camera feed
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise IOError("Error opening the camera.")

        logger.info("Camera opened successfully.")

        # Load the YOLO model
        model = yolo_model

        try:
            while True:
                ret, frame = camera.read()
                if not ret:
                    logger.error("Failed to grab frame.")
                    break

                # Display the current frame in a window
                cv2.imshow("Camera Feed", frame)

                # Wait for key press
                key = cv2.waitKey(1) & 0xFF

                if key == ord(" "):  # If the user presses the spacebar
                    print("Spacebar pressed, detecting person...")
                    # Detect objects using YOLO
                    (
                        boxes,
                        labels,
                        confidences,
                        class_ids,
                        img_with_boxes,
                        detections,
                    ) = detect_objects_with_yolo(image=frame, model=model)

                    print("Total detections:", len(boxes))

                    if len(boxes) == 0:
                        logger.debug("No objects detected.")
                    else:
                        for i in range(len(boxes)):
                            print(labels[i], confidences[i])
                            if labels[i] == "person":
                                x, y, w, h = detections[i]
                                x = int(x)
                                y = int(y)
                                w = int(w)
                                h = int(h)
                                cropped_person = frame[
                                    y : y + h, x : x + w
                                ]  # Crop person from frame
                                # # Render the cropped image in a new window
                                # cv2.imshow(
                                #     "Cropped Person", cropped_person
                                # )  # Display the cropped image

                                result = self.identify_person(
                                    cropped_person
                                )  # Call face recognition pipeline
                                print("Prediction Result:", result)
                                if result != "Unknown":
                                    tts_engine.speak(f"Found {result}")
                                else:
                                    tts_engine.speak(
                                        "Unknown person detected, please verify."
                                    )
                    # Wait for user input before continuing
                    cv2.waitKey(
                        500
                    )  # Delay to allow the user to see the detection before continuing

                elif (
                    cv2.waitKey(1) & 0xFF == 27
                ):  # Break the loop if the user presses 'q'
                    break

        except Exception as e:
            logger.error(f"Error: {e}")

        finally:
            # Release the camera and close windows
            if "camera" in locals() and camera.isOpened():
                camera.release()
            cv2.destroyAllWindows()
