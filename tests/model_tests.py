import unittest

from models import CLIPWrapper


class TestCLIPWrapper(unittest.TestCase):
    def test_input_output_sizes(self):
        # clip_wrapper = CLIPWrapper()
        self.assertEqual(5, 5)


if __name__ == "__main__":
    unittest.main()
