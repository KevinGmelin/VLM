import unittest
from models.blocks.ConvBlock import *
from models.blocks.UpBlock import *

import torch

class TestConvBlock(unittest.TestCase):
    def test_input_output_sizes(self):        
        # ConvBlock test 1        
        test_in = torch.randn(1, 6, 320, 320)
        test_model = ConvBlock(6, 64, 1, False)
        test_out = test_model(test_in)
        self.assertEqual(test_out.shape[1], 64)
        self.assertEqual(test_out.shape[2], 320)
        self.assertEqual(test_out.shape[3], 320)

if __name__ == "__main__":
    unittest.main()