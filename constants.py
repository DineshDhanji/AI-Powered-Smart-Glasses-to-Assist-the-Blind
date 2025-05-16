# Paths
calibration_file = "./data/calibration.pkl"
model_path = "./models/yolo11n.pt"

facial_recog_model_path = "./models/FYP_FR_Model_v1.keras"
facial_recog_feature_scalar = "./data/feature_scalar.pkl"

scene_desc_model_path = "./models/scene_desc_epoch200.h5"
scene_desc_captions_path = "./data/captions.txt"

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

## TTS
VOICE_ID = 1
VOICE_RATE = 250
VOICE_VOLUME = 1.0  # Default volume level (0.0 to 1.0)

# Configuration
known_object_width_meters = {"chair": 0.4064, "person": 0.4064}
