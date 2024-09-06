import torch
import torch.nn as nn
from torchvision import models
import timm

import config

class DogCatClassifier(nn.Module):
    def __init__(self, model_name: str = 'resnet18', pretrained: bool = True, freeze_base: bool = False):
        super().__init__()
        self.model_name = model_name
        
        if model_name == 'resnet18':
            self.base_model = models.resnet18(pretrained=pretrained)
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'resnet50':
            self.base_model = models.resnet50(pretrained=pretrained)
            num_ftrs = self.base_model.fc.in_features
            self.base_model.fc = nn.Identity()
        elif model_name == 'densenet121':
            self.base_model = models.densenet121(pretrained=pretrained)
            num_ftrs = self.base_model.classifier.in_features
            self.base_model.classifier = nn.Identity()
        elif model_name == 'efficientnet_b0':
            self.base_model = models.efficientnet_b0(pretrained=pretrained)
            num_ftrs = self.base_model.classifier[1].in_features
            self.base_model.classifier = nn.Identity()
        elif model_name == 'vit_base_patch16_224':
            self.base_model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
            num_ftrs = self.base_model.num_features
        else:
            raise ValueError(f"Model {model_name} not supported")
        
        if freeze_base:
            for param in self.base_model.parameters():
                param.requires_grad = False
        
        self.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, config.NUM_CLASSES)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.base_model(x)
        return self.classifier(features)

def get_model(model_name: str = 'resnet18', pretrained: bool = True, freeze_base: bool = False) -> DogCatClassifier:
    model = DogCatClassifier(model_name, pretrained, freeze_base).to(config.DEVICE)
    return model