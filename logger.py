import logging

# Basic configuration
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("navigation.log"),  # Log to file
        logging.StreamHandler(),  # Also log to console
    ],
)

logger = logging.getLogger(__name__)
