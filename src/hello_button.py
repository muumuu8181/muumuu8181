import time
import os
import tempfile
import pyautogui
from PIL import ImageGrab

def take_screenshot():
    """スクリーンショットを取得し、一時ファイルとして保存"""
    timestamp = int(time.time())
    temp_dir = tempfile.gettempdir()
    filename = os.path.join(temp_dir, f'screenshot_{timestamp}.png')
    try:
        # PyAutoGUIの代わりにPILのImageGrabを使用
        screenshot = ImageGrab.grab()
        screenshot.save(filename)
        print(f"Screenshot saved: {filename}")
        return filename
    except Exception as e:
        print(f"Error taking screenshot: {e}")
        return None

def display_in_devin(filename):
    """デヴィンのインターフェースに画像を表示"""
    if filename and os.path.exists(filename):
        print(f"Displaying screenshot: {filename}")
        # ここでデヴィンのメッセージ送信コマンドを使用
        return True
    return False

def main():
    print("Starting screenshot monitoring (1-minute intervals)")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            filename = take_screenshot()
            if filename:
                display_in_devin(filename)
            time.sleep(60)  # 1分待機
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
