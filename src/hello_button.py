import time
import os
import tempfile
import pyautogui

def take_screenshot():
    timestamp = int(time.time())
    temp_dir = tempfile.gettempdir()
    filename = os.path.join(temp_dir, f'screenshot_{timestamp}.png')
    try:
        screenshot = pyautogui.screenshot()
        screenshot.save(filename)
        print(f"Screenshot saved: {filename}")
        return filename
    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return None

def main():
    print("Starting screenshot monitoring (1-minute intervals)")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            take_screenshot()
            time.sleep(60)  # 1分待機
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
