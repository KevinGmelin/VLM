import unittest
import torch

from models import CLIPWrapper, SpatialStream

device = "cpu"

# class TestCLIPWrapper(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         super(TestCLIPWrapper, self).__init__(*args, *kwargs)
#         self.clip_wrapper = CLIPWrapper.CLIPWrapper(device)

#     def test_forward_output_size(self):
#         input = torch.randn(1, 3, 320, 320)
#         output = self.clip_wrapper(input)
#         self.assertEqual(len(output), 4, "Expect that CLIPWrapper outputs 4 elements")
#         layer1, layer2, layer3, layer4 = output
#         self.assertEqual(layer1, torch.Size(1, 256, 56, 56))
#         self.assertEqual(layer2, torch.Size(1, 512, 28, 28))
#         self.assertEqual(layer3, torch.Size(1, 1024, 14, 14))
#         self.assertEqual(layer4, torch.Size(1, 2048, 7, 7))

#     # def test_embed_sentence_output_size(self):
#     #     test_command = ["Test sentence"]
#     #     embedding = self.clip_wrapper.embed_sentence(test_command)
#     #     self.assertEqual()

class TestSpatialStream(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestSpatialStream, self).__init__(*args, *kwargs)
        self.model1 = SpatialStream.SpatialStream(channels_in=6, channels_out=1)
        self.model2 = SpatialStream.SpatialStream(channels_in=6, channels_out=3)

    def test_forward_output_size(self):
        input = torch.randn(1, 6, 224, 224)
        output1, output1_lat = self.model1(input)
        output2, output2_lat = self.model2(input)
        self.assertEqual(output1.shape, torch.Size((1, 1, 224, 224)))
        self.assertEqual(output2.shape, torch.Size((1, 3, 224, 224)))
        self.assertEqual(len(output1_lat), 7)
        self.assertEqual(len(output2_lat), 7)
        self.assertEqual(output1_lat[0].shape, torch.Size((1, 1024, 14, 14)))
        self.assertEqual(output2_lat[0].shape, torch.Size((1, 1024, 14, 14)))
        self.assertEqual(output1_lat[1].shape, torch.Size((1, 512, 28, 28)))
        self.assertEqual(output2_lat[1].shape, torch.Size((1, 512, 28, 28)))
        self.assertEqual(output1_lat[2].shape, torch.Size((1, 256, 56, 56)))
        self.assertEqual(output2_lat[2].shape, torch.Size((1, 256, 56, 56)))
        self.assertEqual(output1_lat[3].shape, torch.Size((1, 128, 112, 112)))
        self.assertEqual(output2_lat[3].shape, torch.Size((1, 128, 112, 112)))
        self.assertEqual(output1_lat[4].shape, torch.Size((1, 64, 224, 224)))
        self.assertEqual(output2_lat[4].shape, torch.Size((1, 64, 224, 224)))
        self.assertEqual(output1_lat[5].shape, torch.Size((1, 32, 448, 448)))
        self.assertEqual(output2_lat[5].shape, torch.Size((1, 32, 448, 448)))
        self.assertEqual(output1_lat[6].shape, torch.Size((1, 1, 224, 224)))
        self.assertEqual(output2_lat[6].shape, torch.Size((1, 3, 224, 224)))

if __name__ == "__main__":
    unittest.main()
