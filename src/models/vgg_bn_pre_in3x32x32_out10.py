import os
import torch.nn as nn

import torch
from torchvision import models

path_models = "materials/weights"
model_filename = "vgg16_bn-6c64b313.pth"
model_filename = os.path.join(path_models, model_filename)


class VggBnPre3x32x32(nn.Module):
    def __init__(self):
        super(VggBnPre3x32x32, self).__init__()

        self.vgg16 = models.vgg16_bn()
        self.vgg16.load_state_dict(torch.load(model_filename))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 10)
        )

        # freeze all layers
        for param in self.vgg16.features.parameters():
            param.require_grad = False

    def forward(self, x):
        x = self.vgg16.forward(x)
        return x
