from torch import nn

class BrainCNN(nn.Module):
    def __init__(self):
        super(BrainCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=0),   # valid -> padding=0
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),  # same -> padding=3
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 128, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 256, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2)
        )
        self.fc_layers = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.25),
            
            nn.Linear(512, 4),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)     # Flatten
        out = self.fc_layers(out)
        return out
