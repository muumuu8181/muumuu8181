import sys
import unittest
from unittest import mock

sys.modules['pyautogui'] = mock.MagicMock()
sys.modules['keyboard'] = mock.MagicMock()

import auto_click

class AutoClickTests(unittest.TestCase):
    def test_auto_capture_triggers_and_clicks(self):
        with mock.patch('auto_click.keyboard.is_pressed', side_effect=[False, False, True]), \
             mock.patch('auto_click.pyautogui.locateCenterOnScreen', side_effect=[None, (5, 5)]), \
             mock.patch('auto_click.pyautogui.click') as mock_click, \
             mock.patch('auto_click.pyautogui.screenshot') as mock_screen, \
             mock.patch('auto_click.time.sleep'):
            auto_click.click_until_escape('img.png', interval=0, auto_capture=True)
            mock_screen.assert_called_once_with('img.png')
            mock_click.assert_called_once_with((5, 5))

    def test_no_auto_capture_when_disabled(self):
        with mock.patch('auto_click.keyboard.is_pressed', side_effect=[False, True]), \
             mock.patch('auto_click.pyautogui.locateCenterOnScreen', return_value=None), \
             mock.patch('auto_click.pyautogui.click') as mock_click, \
             mock.patch('auto_click.pyautogui.screenshot') as mock_screen, \
             mock.patch('auto_click.time.sleep'):
            auto_click.click_until_escape('img.png', interval=0, auto_capture=False)
            mock_screen.assert_not_called()
            mock_click.assert_not_called()

if __name__ == '__main__':
    unittest.main()
