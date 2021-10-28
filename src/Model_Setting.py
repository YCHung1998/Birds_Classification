import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ResNet_classifier(nn.Module):
        """# Reference:
        - [Deep Residual Learning for Image Recognition](
            https://arxiv.org/abs/1512.03385) (CVPR 2016 Best Paper Award)
        """
    def __init__(self, final_tuning = False):
        super().__init__()
        self.model = models.resnet50(pretrained=True)
        self.final_tuning  = final_tuning
        if self.final_tuning:
            for p in self.model.parameters():
                p.requires_grad = False
        last_layer_inputs = self.model.fc.in_features
        self.model.fc = nn.Linear(last_layer_inputs, 200, bias=True)  

    def forward(self, x):
        x = self.model.forward(x)

        return x


class Densenet121(nn.Module):
    def __init__(self, final_tuning = False):
        super().__init__()
        self.model = models.densenet121(pretrained=True)
        self.final_tuning  = final_tuning
        if self.final_tuning:
            for p in self.model.parameters():
                p.requires_grad = False
        last_layer_inputs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(last_layer_inputs, 200, bias=True)  

    def forward(self, x):
        x = self.model.forward(x)

        return x