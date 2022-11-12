import unittest
import torch

from models import CLIPWrapper, SpatialStream, SemanticStream, SpatialSemanticStream, PickModel, PlaceModel

device = "cuda"

class TestCLIPWrapper(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestCLIPWrapper, self).__init__(*args, *kwargs)
        self.clip_wrapper = CLIPWrapper.CLIPWrapper(device)

    def test_forward_output_size(self):
        input = torch.randn(1, 3, 224, 224).to(device)
        output = self.clip_wrapper(input)
        self.assertEqual(len(output), 4, "Expect that CLIPWrapper outputs 4 elements")
        layer1, layer2, layer3, layer4 = output
        self.assertEqual(layer1.shape, torch.Size((1, 256, 56, 56)))
        self.assertEqual(layer2.shape, torch.Size((1, 512, 28, 28)))
        self.assertEqual(layer3.shape, torch.Size((1, 1024, 14, 14)))
        self.assertEqual(layer4.shape, torch.Size((1, 2048, 7, 7)))

    def test_embed_sentence_output_size(self):
        test_command = ["Test sentence 1", "Test sentence 2"]
        embedding = self.clip_wrapper.embed_sentence(test_command)
        self.assertEqual(embedding.shape, torch.Size((2, 1024)))

lateral_connection_sizes = [torch.Size((1, 1024, 14, 14)),
                            torch.Size((1, 512, 28, 28)),
                            torch.Size((1, 256, 56, 56)),
                            torch.Size((1, 128, 112, 112)),
                            torch.Size((1, 64, 224, 224)),
                            torch.Size((1, 32, 448, 448))]

class TestSpatialStream(unittest.TestCase):
    def test_forward_output_size(self):
        input = torch.randn(1, 6, 224, 224).to(device)
        for channels_out in [1,3]:
            self.model = SpatialStream.SpatialStream(channels_in=6, channels_out=channels_out).to(device)
            output, output_lat = self.model(input)
            self.assertEqual(output.shape, torch.Size((1, channels_out, 224, 224)))
            self.assertEqual(len(output_lat), 6)
            for lat, expected_size in zip(output_lat, lateral_connection_sizes):
                self.assertEqual(lat.shape, expected_size)


class TestSemanticStream(unittest.TestCase):
    def test_forward_output_size(self):
        input_img = torch.randn(1, 3, 224, 224).to(device)
        language_command = ["Test language command"]
        lateral_outs = [torch.randn(lat_size).to(device) for lat_size in lateral_connection_sizes]


        for channels_out in [1,3]:
            self.model = SemanticStream.SemanticStream(channels_out=channels_out).to(device)
            output = self.model(input_img, language_command, lateral_outs)
            self.assertEqual(output.shape, torch.Size((1, channels_out, 224, 224)))


class TestSpatialSemanticStream(unittest.TestCase):
    def test_forward_output_size(self):
        rgb_ddd_img = torch.randn(1, 6, 224, 224).to(device)
        language_command = ["Test language command"]

        self.pick_model = SpatialSemanticStream.SpatialSemanticStream(channels_in=6, pick=True, batchnorm=False).to(device)
        output = self.pick_model(rgb_ddd_img, language_command)
        self.assertEqual(output.shape, torch.Size((1, 1, 224, 224)))

        self.place_model = SpatialSemanticStream.SpatialSemanticStream(channels_in=6, pick=False, batchnorm=False).to(device)
        output = self.place_model(rgb_ddd_img, language_command)
        self.assertEqual(output.shape, torch.Size((1, 3, 224, 224)))

class TestPickModel(unittest.TestCase):
    def test_forward_output_size(self):
        rgb_ddd_img = torch.randn(1, 6, 224, 224).to(device)
        language_command = ["Test language command"]

        self.model = PickModel.PickModel(num_rotations=16, batchnorm=False).to(device)
        output = self.model(rgb_ddd_img, language_command)
        self.assertEqual(output.shape, torch.Size((1, 16, 224, 224)))

class TestPlaceModel(unittest.TestCase):
    def test_forward_output_size(self):
        rgb_ddd_img = torch.randn(1, 6, 224, 224).to(device)
        language_command = ["Test language command"]
        pick_locations = [(0,0), (224,224), (100, 100)]

        self.model = PlaceModel.PlaceModel(num_rotations=16, crop_size=9, batchnorm=False).to(device)
        for pick_location in pick_locations:
            output = self.model(rgb_ddd_img, language_command, pick_location)
            self.assertEqual(output.shape, torch.Size((1, 16, 224, 224)))


if __name__ == "__main__":
    unittest.main()
