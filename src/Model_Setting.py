import torch
import torch.nn as nn
from torchvision import models

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class ResNet_classifier(nn.Module):
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


if __name__=='__main__':
    model = ResNet_classifier() # Densenet121() 
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum =0.9)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
