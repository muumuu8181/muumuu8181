import sys
import unittest
from unittest import mock

sys.modules['pyautogui'] = mock.MagicMock()
sys.modules['keyboard'] = mock.MagicMock()

import click_image

class ClickImageTests(unittest.TestCase):
    def test_click_invoked_when_image_found(self):
        with mock.patch('click_image.keyboard.is_pressed', side_effect=[False, True]), \
             mock.patch('click_image.pyautogui.locateCenterOnScreen', side_effect=[(10, 10), None]), \
             mock.patch('click_image.pyautogui.click') as mock_click, \
             mock.patch('click_image.time.sleep'):
            click_image.click_until_escape('img.png', interval=0)
            mock_click.assert_called_once_with((10, 10))

    def test_loop_stops_on_escape(self):
        with mock.patch('click_image.keyboard.is_pressed', side_effect=[True]), \
             mock.patch('click_image.pyautogui.locateCenterOnScreen') as mock_loc, \
             mock.patch('click_image.pyautogui.click') as mock_click, \
             mock.patch('click_image.time.sleep'):
            click_image.click_until_escape('img.png', interval=0)
            mock_loc.assert_not_called()
            mock_click.assert_not_called()

if __name__ == '__main__':
    unittest.main()
