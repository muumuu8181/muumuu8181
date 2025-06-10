import datetime
import unittest

import fortune


class FortuneTests(unittest.TestCase):
    def test_deterministic_for_same_date(self):
        date = datetime.date(2024, 1, 1)
        first = fortune.get_daily_fortune(date)
        second = fortune.get_daily_fortune(date)
        self.assertEqual(first, second)
        self.assertIn(first, fortune.FORTUNES)


if __name__ == "__main__":
    unittest.main()
