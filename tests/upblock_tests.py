import unittest
from models.blocks.ConvBlock import *
from models.blocks.UpBlock import *

import torch

class TestUpBlock(unittest.TestCase):
    def test_input_output_sizes(self):        
        # UpBlock test 1        
        test_in1 = torch.randn(1, 512, 14, 14)
        test_in2 = torch.randn(1, 512, 28, 28)
        test_model = UpBlock(1024, 256, True, False)
        test_out = test_model(test_in1, test_in2)
        self.assertEqual(test_out.shape[1], 256)
        self.assertEqual(test_out.shape[2], 28)
        self.assertEqual(test_out.shape[3], 28)

if __name__ == "__main__":
    unittest.main()