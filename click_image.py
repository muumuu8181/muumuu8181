import time
import sys

import pyautogui
import keyboard


def click_until_escape(target_image: str = "target_image.png", interval: float = 0.2) -> None:
    """Repeatedly search and click ``target_image`` until ESC is pressed."""
    print("Press ESC to stop.")
    while True:
        if keyboard.is_pressed("esc"):
            print("ESC pressed. Exiting.")
            break

        location = pyautogui.locateCenterOnScreen(target_image, confidence=0.8)
        if location:
            pyautogui.click(location)

        time.sleep(interval)


def main():
    click_until_escape()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
