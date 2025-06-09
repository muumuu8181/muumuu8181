import time
import sys

import pyautogui
import keyboard


def main():
    target_image = "target_image.png"  # replace with your image file
    print("Press ESC to stop.")
    while True:
        if keyboard.is_pressed("esc"):
            print("ESC pressed. Exiting.")
            break

        location = pyautogui.locateCenterOnScreen(target_image, confidence=0.8)
        if location:
            pyautogui.click(location)

        time.sleep(0.2)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
