import tkinter as tk
from tkinter import font
from constants import *
from utils import load_model
from navigation import Navigation_Module
from facial_recognition import Facial_Recognition

# Load the YOLO model
model = load_model(model_path)


def center_window(window, width, height):
    window.update_idletasks()
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")


def run_navigation(root, navigation_module):
    root.withdraw()  # Hide the main window

    navigation_module.initiate_navigation(
        shaded_regions=True,
        render=True,
        model=model,
    )

    root.deiconify()  # Show again after navigation module ends


def run_recognition(root, facial_recognition_module):
    root.withdraw()  # Hide the main window

    facial_recognition_module.initialize_recognition(
        yolo_model=model,
    )

    root.deiconify()  # Show again after navigation module ends


def main():

    navigation_module = Navigation_Module(
        calibration_file=calibration_file,
        model_path=model_path,
        known_object_width_meters=known_object_width_meters,
        total_regions=TOTAL_REGIONS,
        threshold_distance=THRESHOLD_DISTANCE,
    )

    facial_recognition_module = Facial_Recognition(model_path=facial_recog_model_path)

    root = tk.Tk()
    root.title("AI Assistant Control Panel")
    WIDTH, HEIGHT = 800, 500
    center_window(root, WIDTH, HEIGHT)
    root.resizable(False, False)

    custom_font = font.Font(family="Helvetica", size=12, weight="bold")
    btn_style = {
        "font": custom_font,
        "bg": "#005f73",
        "fg": "white",
        "activebackground": "#0a9396",
        "activeforeground": "white",
        "width": 25,
        "height": 2,
        "bd": 0,
    }

    # Outer frame to center everything
    outer_frame = tk.Frame(root, bg="#e0fbfc")
    outer_frame.pack(expand=True, fill="both")

    # Inner frame that holds all widgets
    content_frame = tk.Frame(outer_frame, bg="#e0fbfc")
    content_frame.place(relx=0.5, rely=0.5, anchor="center")

    # Title
    title = tk.Label(
        content_frame,
        text="AI Assistant Control Panel",
        font=("Helvetica", 21, "bold"),
        bg="#e0fbfc",
        fg="#001219",
        pady=20,
    )
    title.pack()

    # Buttons
    modules = [
        ("Navigation Module", lambda: run_navigation(root, navigation_module)),
        (
            "Facial Recognition Module",
            lambda: run_recognition(root, facial_recognition_module),
        ),
        (
            "Scene Description Module",
            lambda: print("Scene Description module placeholder."),
        ),
    ]

    for text, cmd in modules:
        tk.Button(
            content_frame,
            text=text,
            command=cmd,
            **btn_style,
        ).pack(pady=10)

    # Footer
    footer = tk.Label(
        content_frame,
        text="© 2025 AI Assistant",
        font=("Helvetica", 10),
        bg="#e0fbfc",
        fg="#555",
        pady=20,
    )
    footer.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
