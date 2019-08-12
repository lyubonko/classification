import torch.nn as nn
import torch.nn.functional as F


class VggBb3x32x32(nn.Module):
    def __init__(self):
        super(VggBb3x32x32, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4_1 = nn.Conv2d(256, 512, 3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.conv5_1 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, 3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.bn1(self.pool(F.relu(self.conv1_2(self.conv1_1(x)))))
        x = self.bn2(self.pool(F.relu(self.conv2_2(self.conv2_1(x)))))
        x = self.bn3(self.pool(F.relu(self.conv3_3(self.conv3_2(self.conv3_1(x))))))
        x = self.bn4(self.pool(F.relu(self.conv4_3(self.conv4_2(self.conv4_1(x))))))
        x = self.bn5(self.pool(F.relu(self.conv5_3(self.conv5_2(self.conv5_1(x))))))
        x = x.view(-1, 512)
        x = F.relu(self.fc(x))
        return x

