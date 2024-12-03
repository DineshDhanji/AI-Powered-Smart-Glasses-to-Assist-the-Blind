# Distance Calculation for Object Detection
This repository is designed for implementing object detection and calculating the distance of detected objects from a camera. The project is tailored for deployment on low-end devices, ensuring lightweight computation without requiring additional sensors.

# Calibration
`calibration.py` does all the calibration related work. It use the photos from `calibration_photos` folder, then create a pickle based file to store them. This `calibration.pkl` will be used by other file such as distance_calculation for measuring the distance between camera and detected object.

# Distance Calculation
`distance_calculation.py` loads the model and calibration file to identify object and measure the distance. This is done in real time. 

