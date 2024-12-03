from pathlib import Path
import cv2
import numpy as np

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

import pickle


def calibrate(
    images_folder_path: str = "./calibration_photos/",
    calibration_filename: str = "calibration.pkl",
    silent: bool = True,
):
    folder_path = Path(images_folder_path)
    if not folder_path.exists():
        raise FileNotFoundError(f"Folder {images_folder_path} not found")

    n_x = 9  # number of inner corners per row
    n_y = 6  # number of inner corners per column

    # Setup object points (3D points in real world space)
    objp = np.zeros((n_y * n_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:n_x, 0:n_y].T.reshape(-1, 2)
    image_points = []  # 2D points in image plane
    object_points = []  # 3D points in real world

    # Criteria for refining corner locations
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # List of image file extensions you want to match
    image_extensions = {
        ".jpg",
        ".jpeg",
    }
    image_files = [
        file
        for file in folder_path.rglob("*")
        if file.is_file() and file.suffix.lower() in image_extensions
    ]

    print("Calibrating camera \t\t🟣")
    # Loop through all calibration images
    for image_file in image_files:
        # Read the image and convert to grayscale
        img = mpimg.imread(image_file)
        img = np.array(img)  # Convert to a writable NumPy array
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Find the chessboard corners
        found, corners = cv2.findChessboardCorners(img_gray, (n_x, n_y))

        if found:
            # Refine the corner locations
            cv2.drawChessboardCorners(img, (n_x, n_y), corners, found)
            corners2 = cv2.cornerSubPix(img_gray, corners, (11, 11), (-1, -1), criteria)

            # Append the object points and image points
            image_points.append(corners2)
            object_points.append(objp)

            if not silent:
                plt.imshow(img)
                plt.show()

    # Perform camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, img_gray.shape[::-1], None, None
    )

    print("Calibrating done \t\t✅")

    img_size = img.shape
    calib_data = {"cam_matrix": mtx, "dist_coeffs": dist, "img_size": img_size}

    print("Saving calibration data \t", end="")
    try:
        # Save the calibration data to a pickle file
        with open(calibration_filename, "wb") as f:
            pickle.dump(calib_data, f)
        print("✅")
    except Exception as e:
        print("❌")
        print(e)
    plt.show()

    return mtx, dist


calibrate(calibration_filename="calibration.pkl", silent=True)
