# Paths
calibration_file = "./calibration.pkl"
model_path = "./models/yolo11n.pt"

# Constants
ESC_KEY = 27  # ASCII for the ESC key
DEFAULT_FRAME_WIDTH = 640  # Default camera frame width
DEFAULT_FRAME_HEIGHT = 480  # Default camera frame height
MODEL_CONFIDENCE_THRESHOLD = 0.5

TOTAL_REGIONS = 3
THRESHOLD_DISTANCE = 3.0  # Threshold distance for navigation decision (in meters)

# Configuration
known_object_width_meters = {"chair": 0.4064, "person": 0.4064}
