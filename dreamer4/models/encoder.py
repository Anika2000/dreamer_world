import torch
import torch.nn as nn 

#input → conv1 (→ smaller spatial) → conv2 → conv3 → conv4 -> flatten → linear → embedding
class Encoder(nn.Module): 
    def __init__(self, channels=3, embedding_dim=1024) :
        super().__init__()
        #channels: each layer increases the number of feature maps
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(4*4*256, embedding_dim)
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x