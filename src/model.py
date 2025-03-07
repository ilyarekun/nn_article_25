from torch import nn

class BrainCNN(nn.Module):
    def __init__(self):
        super(BrainCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),   
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=7, stride=1, padding=3),  
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
        )
        
    def forward(self, x):
        out = self.conv_layers(x)
        out = out.view(out.size(0), -1)     # Flatten
        out = self.fc_layers(out)
        return out


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)