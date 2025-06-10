import time
import sys

import pyautogui
import keyboard


def click_until_escape(target_image: str = "target_image.png", interval: float = 0.2, confidence: float = 0.8, auto_capture: bool = False) -> None:
    """Repeatedly search and click ``target_image`` until ESC is pressed.

    If ``auto_capture`` is True and the image is not found, a screenshot is
    taken and saved to ``target_image`` for calibration on new machines.
    """
    print("Press ESC to stop.")
    while True:
        if keyboard.is_pressed("esc"):
            print("ESC pressed. Exiting.")
            break

        location = pyautogui.locateCenterOnScreen(target_image, confidence=confidence)
        if location:
            pyautogui.click(location)
        else:
            if auto_capture:
                pyautogui.screenshot(target_image)
                print(f"Captured reference image to {target_image}")
                auto_capture = False
        time.sleep(interval)


def main():
    click_until_escape(auto_capture=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
