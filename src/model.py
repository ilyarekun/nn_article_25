import torch
from torch import nn
import torch.nn.functional as F

class BrainCNN(nn.Module):
    def __init__(self):
        self.conv1 = nn.conv2d(3, 64, (7,7), padding = 1)
        self.conv2 = nn.conv2d(64, 128, (7,7), padding = 1)
        self.conv3 = nn.conv2d(128, 128, (7,7), padding = 1)
        self.conv4 = nn.conv2d(128, 256, (7,7), padding = 1)
        self.conv5 = nn.conv2d(256, 256, (7,7), padding = 1)
        self.conv6 = nn.conv2d(256, 512, (7,7), padding = 1)
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(256)
        self.bn6 = nn.BatchNorm2d(512)
        
        self.fc1 = nn.Linear(512 *3*3, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 4)
        
    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.bn1(out)
        out = F.max_pool2d(out, (2,2))
        
        out = self.conv2(out)
        out = F.relu(out)
        out = self.bn2(out)
        out = F.max_pool2d(out, (2,2))
        
        out = self.conv3(out)
        out = F.relu(out)
        out = self.bn3(out)
        out = F.max_pool2d(out, (2,2))
        
        out = self.conv4(out)
        out = F.relu(out)
        out = self.bn4(out)
        out = F.max_pool2d(out, (2,2))
        
        out = self.conv5(out)
        out = F.relu(out)
        out = self.bn5(out)
        out = F.max_pool2d(out, (2,2))
        
        out = self.conv6(out)
        out = F.relu(out)
        out = self.bn6(out)
        out = F.max_pool2d(out, (2,2))
        
        out = out.view(out.shape[0], -1)     # 5. Flatten
        out = self.fc1(out)
        out = F.dropout(out, 0.25)
        out = self.fc2(out)
        out = F.dropout(out, 0.25)
        out = self.fc3(out)
        
        return out