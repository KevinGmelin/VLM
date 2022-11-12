import torch
import torch.nn as nn
import torch.nn.functional as F
import clip

from blocks.ConvBlock import ConvBlock
from blocks.UpBlock import UpBlock

class CLIPWrapper(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("RN50")
        modified_resnet = list(self.clip_model.children())[0]
        self.modified_resnet = torch.nn.Sequential(*list(modified_resnet.children())[:-1])
        self.device = device

        for param in self.clip_model.parameters():
            param.requires_grad = False
    
    def forward(self, rgb_image):
        head = self.modified_resnet[0:10](rgb_image.half())
        layer1 = self.modified_resnet[10](head)
        layer2 = self.modified_resnet[11](layer1)
        layer3 = self.modified_resnet[12](layer2)
        layer4 = self.modified_resnet[13](layer3)
        return layer1.float(), layer2.float(), layer3.float(), layer4.float()
    
    def embed_sentence(self, language_commands):
        tokens = torch.cat([clip.tokenize(c for c in language_commands)]).to(self.device)
        sentence_embedding = self.clip_model.encode_text(tokens)
        return sentence_embedding.float()

class SemanticStream(nn.Module):
    def __init__(self, channels_out, batchnorm = False):
        super().__init__()
        self.clip_model = CLIPWrapper()

        self.channels_out = channels_out
        self.batchnorm = batchnorm

        self.conv1 = nn.Conv2d(2048, 1024, 3, 1, 1)

        self.linear1 = nn.Linear(1024, 1024)
        self.linear2 = nn.Linear(1024, 512)
        self.linear3 = nn.Linear(1024, 256)

        self.up_block1 = UpBlock(2048,512)
        self.up_block2 = UpBlock(1024,256)
        self.up_block3 = UpBlock(512,128)

        self.fuse_lat_0 = nn.Conv2d(512+1024, 512, 1)
        self.fuse_lat_1 = nn.Conv2d(256+512, 256, 1)
        self.fuse_lat_2 = nn.Conv2d(128+256, 128, 1)
        self.fuse_lat_3 = nn.Conv2d(128+128, 128, 1)
        self.fuse_lat_4 = nn.Conv2d(64+64, 64, 1)
        self.fuse_lat_5 = nn.Conv2d(32+32, 32, 1)

        self.batchnorm = True
        self.decoder4 = nn.Sequential(
          ConvBlock(128, 128, 1, self.batchnorm),
          ConvBlock(128, 128, 1, self.batchnorm, residual=True),
          nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.decoder5 = nn.Sequential(
            ConvBlock(128, 64, 1, self.batchnorm),
            ConvBlock(64, 64, 1, self.batchnorm, residual=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )

        self.decoder6 = nn.Sequential(
            ConvBlock(64, 32, 1, self.batchnorm),
            ConvBlock(32, 32, 1, self.batchnorm, residual=True),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        
        # 1x conv & identity & downsample
        self.down = nn.Sequential(
            ConvBlock(32, 16, 1, self.batchnorm, self.channels_out),
            ConvBlock(self.channels_out, 16, 1, self.batchnorm, self.channels_out, residual=True),
        )
    
    def forward(self, rgb_image, language_command, lateral_outs):
        sentence_embedding = self.clip_model.embed_sentence(language_command)

        sentence_embedding = self.clip_model.embed_sentence(language_command)
        sentence_embedding = sentence_embedding.unsqueeze(1).unsqueeze(1)

        lang_tile1 = torch.tile(self.linear1(sentence_embedding), (1,7,7,1))
        lang_tile1 = lang_tile1.permute(0,3,1,2)
        lang_tile2 = torch.tile(self.linear2(sentence_embedding), (1,14,14,1))
        lang_tile2 = lang_tile2.permute(0,3,1,2)
        lang_tile3 = torch.tile(self.linear3(sentence_embedding), (1,28,28,1))
        lang_tile3 = lang_tile3.permute(0,3,1,2)

        # Need to do transpose
        
        layer1, layer2, layer3, layer4 = self.clip_model(rgb_image.half())

        x = self.conv1(layer4)
        x = x * lang_tile1
        x = self.up_block1(x, layer3)

        x = x * lang_tile2
        x = torch.cat([x, lateral_outs[0]], dim=1)
        x = self.fuse_lat_0(x)
        x = self.up_block2(x, layer2)

        x = x * lang_tile3
        x = torch.cat([x, lateral_outs[1]], dim=1)
        x = self.fuse_lat_1(x)
        x = self.up_block3(x, layer1)

        x = torch.cat([x, lateral_outs[2]], dim=1)
        x = self.fuse_lat_2(x)
        x = self.decoder4(x)

        x = torch.cat([x, lateral_outs[3]], dim=1)
        x = self.fuse_lat_3(x)
        x = self.decoder5(x)

        x = torch.cat([x, lateral_outs[4]], dim=1)
        x = self.fuse_lat_4(x)
        x = self.decoder6(x)

        x = torch.cat([x, lateral_outs[5]], dim=1)
        x = self.fuse_lat_5(x)
        x = self.down(x)
        x = F.interpolate(x, size=(x.shape[-2]//2, x.shape[-1]//2), mode='bilinear')

        return x
