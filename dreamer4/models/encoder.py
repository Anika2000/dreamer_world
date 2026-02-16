import torch
import torch.nn as nn 

#input → conv1 (→ smaller spatial) → conv2 → conv3 → conv4 -> flatten → linear → embedding
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        channels = config["env"]["channels"]
        enc_channels = config["model"]["encoder_channels"]
        embedding_dim = config["model"]["embedding_dim"]

        layers = []
        in_ch = channels
        for out_ch in enc_channels:
            layers.append(nn.Conv2d(in_ch, out_ch, 3, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_ch = out_ch

        self.conv = nn.Sequential(*layers)

        final_size = enc_channels[-1] * 4 * 4  # for 64x64 input
        self.fc = nn.Linear(final_size, embedding_dim)

    def forward(self, x):
        x = self.conv(x) #go through the convolution later which is conv -> relu -> conv -> relu ... etc
        x = torch.flatten(x, start_dim=1) #flatten from (B, a, b, c) -> (B, a*b*c)
        x = self.fc(x) #commpresses more? 
        return x
