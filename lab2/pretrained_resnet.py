import torch
from torch import nn
from transformers import ResNetForImageClassification

class PretrainedResnet(nn.Module):
    def __init__(self):
        super().__init__()

        basemodel = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

        self.resnet = basemodel.resnet
        self.classifier = nn.Linear(in_features=2048, out_features=21, bias=True)

    def forward(self, images: torch.Tensor):
        features = self.resnet(images).pooler_output
        features = features.squeeze(-1).squeeze(-1)
        logits = self.classifier(features)

        return logits
