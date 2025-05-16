# AI-Powered Smart Glasses to Assist the Blind

## Overview

This project focuses on developing an AI-powered smart glasses system designed to assist visually impaired individuals. The system leverages advanced computer vision and machine learning techniques to provide real-time navigation and scene description, helping users navigate through their environment with ease.

The smart glasses integrate a camera that captures real-time images of the user's surroundings. AI models process the images to detect objects, analyze scenes, and provide auditory feedback to the user. This project aims to make technology more inclusive and help visually impaired individuals gain greater independence in their daily lives.

## Features

- Real-time Navigation Assistance: Detects and calculates the distance to objects and provides navigational guidance to avoid obstacles.
- Scene Description: Identifies objects and describes the environment in real-time through auditory feedback.
- Voice Interaction: Allows users to interact with the system through voice commands to switch between modes.
- Lightweight and Affordable: Designed to run on affordable and easily available hardware, making the technology accessible to a larger audience.

## Table of Contents

- [AI-Powered Smart Glasses to Assist the Blind](#ai-powered-smart-glasses-to-assist-the-blind)
  - [Overview](#overview)
  - [Features](#features)
  - [Table of Contents](#table-of-contents)
    - [Technologies Used](#technologies-used)
    - [Installation](#installation)
    - [Usage](#usage)
    - [Project Structure](#project-structure)
    - [Contributing](#contributing)

### Technologies Used

This project uses the following technologies:

- **Python:** The core programming language for building the AI models and backend logic.
- OpenCV: Used for real-time image processing and computer vision tasks.
- **TensorFlow / Keras:** For building and deploying machine learning models for object detection and scene recognition.
- **Tkinter:** A Python library for creating graphical user interfaces (GUIs) for mode selection.
- **Text-to-Speech (TTS):** For voice interaction and providing auditory feedback to the user.
- **Raspberry Pi 4:** The main hardware platform, providing affordability and accessibility for the system.

### Installation

To set up and run the project, follow these steps:

1. Clone the repository:

```
git clone https://github.com/DineshDhanji/AI-Powered-Smart-Glasses-to-Assist-the-Blind.git
cd AI-Powered-Smart-Glasses-to-Assist-the-Blind
```

2. Install the required dependencies:
   The project uses Python 3.11. You can install the dependencies via pip:

```
uv pip install -r requirements.txt
or
pip install -r requirements.txt

```

3. Install the necessary hardware dependencies:

   - For running the camera, make sure the opencv-python package is installed. You can test the camera with OpenCV by running:
     ```
     python -c "import cv2; print(cv2.__version__)"
     ```

4. Hardware Setup:
   - Set up a Raspberry Pi 4 with a compatible camera module.
   - Ensure that the camera is connected and enabled in the Raspberry Pi settings.

### Usage

Once the project is set up, you can run the main application using:

```
python main.py
```

This will launch the graphical user interface (GUI) with three options:

- Navigation Mode: This mode processes the camera input and provides real-time navigation assistance.

- Scene Description Mode: This mode identifies objects in the environment and provides a verbal description of the scene.

- Facial Recognition Mode: (Coming Soon) This mode will identify faces in the camera feed and notify the user.

- Voice Interaction
  You can switch between different modes by selecting the buttons in the GUI or by using voice commands. The system provides feedback through the text-to-speech feature, making it more user-friendly.

### Project Structure

The project follows a modular structure:

```
AI-Powered-Smart-Glasses-to-Assist-the-Blind/
├── main.py                    # Main script to launch the GUI and handle mode selection
├── navigation.py              # Navigation module for real-time navigation assistance
├── scene_description.py       # Scene description module
├── facial_recognition.py      # (Upcoming) Facial recognition module
├── models/                    # Directory to store pre-trained AI models
│   └── yolo11n.pt    # Example pre-trained model for object detection
│   └── FYP_FR_Model_v1.keras    # Example pre-trained model for facial recognition
├── utils.py                     # Utility functions (image processing, voice interaction, etc.)
├── requirements.txt           # Python dependencies
└── README.md                  # This README file
```

`main.py:` The entry point of the application, initializing the GUI and handling user inputs.

`navigation.py:` The module responsible for real-time object detection, distance calculation, and navigation guidance.

`scene_description.py:` A module for identifying objects and providing scene descriptions through voice feedback.

`facial_recognition.py:` (In progress) Module for detecting and recognizing faces in the camera feed.

### Contributing

We welcome contributions to improve this project! If you would like to contribute, please fork the repository and submit a pull request. Ensure that all new code is well-tested and includes appropriate documentation.

Steps for Contributing:

1. Fork the repository

2. Create a new branch for your changes (`git checkout -b feature/your-feature`)

3. Commit your changes (`git commit -m 'Add your feature'`)

4. Push to your branch (`git push origin feature/your-feature`)

5. Submit a pull request with a detailed description of the changes
