import cv2
import numpy as np
import tensorflow as tf
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler


from logger import logger
from utils import detect_objects_with_yolo, load_model


class Facial_Recognition:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = StandardScaler()
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.radius = 1
        self.n_points = 8 * self.radius
        self.label_names = {0: "Aneeq", 1: "Hamza"}

    def detect_face(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )
        if len(faces) == 0:
            return None
        x, y, w, h = faces[0]
        face = image[y : y + h, x : x + w]
        return cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    def rgb_2_gray(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def apply_eppisoidal(self, image):
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
        lbp = local_binary_pattern(
            image_resized, self.n_points, self.radius, method="uniform"
        )
        lbp_cropped = lbp[:46, :46]
        lbp_cropped = lbp_cropped.astype(np.float32) / lbp_cropped.max()
        lbp_cropped = np.expand_dims(lbp_cropped, axis=-1)
        lbp_cropped = np.expand_dims(lbp_cropped, axis=0)
        return lbp_cropped

    def scale_image(self, image):
        flat = image.reshape(1, -1)
        scaled = self.scaler.fit_transform(flat).reshape(1, 46, 46, 1)
        return scaled

    def predict_image(self, image):
        prediction = self.model.predict(image)
        confidence = np.max(prediction) * 100
        true_prediction = self.label_names[np.argmax(prediction)]
        if confidence >= 70:
            final_prediction = true_prediction
        else:
            final_prediction = "Unknown"
        message = f"Original Prediction: {true_prediction}\nConfidence: {confidence:.2f}%\nFinal Prediction: {final_prediction}"
        return message, true_prediction

    def identify_person(self, person_image):
        color_face = self.detect_face(person_image)
        if color_face is None:
            return "Face not found"
        gray_face = self.rgb_2_gray(color_face)
        epp_gray_face = self.apply_eppisoidal(gray_face)
        epp_lbp_gray_face = self.apply_lbp_resize(epp_gray_face)
        scaled_image = self.scale_image(epp_lbp_gray_face)
        prediction, person = self.predict_image(scaled_image)
        return prediction, person

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

                                result, _ = self.identify_person(
                                    cropped_person
                                )  # Call face recognition pipeline
                                print("Prediction Result:", result)
                                if _ != "Unknown":
                                    tts_engine.speak(
                                        f"Found {_} with confidence {result}"
                                    )
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
