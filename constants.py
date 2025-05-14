# Paths
calibration_file = "./calibration.pkl"
model_path = "./models/yolo11n.pt"
facial_recog_model_path = "./models/FYP_FR_Model_v1.keras"

# Constants
## Main Interface Window
WIDTH = 800  # Width of the main interface window
HEIGHT = 600  # Height of the main interface window

## Navigation Window
ESC_KEY = 27  # ASCII for the ESC key
DEFAULT_FRAME_WIDTH = 640  # Default camera frame width
DEFAULT_FRAME_HEIGHT = 480  # Default camera frame height

## Model
MODEL_CONFIDENCE_THRESHOLD = 0.5

## Navigation
TOTAL_REGIONS = 3
THRESHOLD_DISTANCE = 3.0  # Threshold distance for navigation decision (in meters)

# Configuration
known_object_width_meters = {"chair": 0.4064, "person": 0.4064}
