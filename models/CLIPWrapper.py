import torch
import torch.nn as nn
import clip

class CLIPWrapper(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("RN50", device=device)
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
