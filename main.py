from navigation import Navigation_Module
from constants import *


def main():

    navigation_module = Navigation_Module(
        calibration_file=calibration_file,
        model_path=model_path,
        known_object_width_meters=known_object_width_meters,
        total_regions=TOTAL_REGIONS,
        threshold_distance=THRESHOLD_DISTANCE,
    )
    navigation_module.initiate_navigation(
        shaded_regions=True,
        render=True,
    )


if __name__ == "__main__":
    main()
