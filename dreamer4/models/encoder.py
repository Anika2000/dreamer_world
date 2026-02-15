import torch
import torch.nn as nn 

#input → conv1 (→ smaller spatial) → conv2 → conv3 → conv4 -> flatten → linear → embedding

# Environment
#     ↓
# Replay Buffer
#     ↓
# World Model
#     ↓
# Imagination
#     ↓
# Actor-Critic


# - so the envionement is to collect the data from simulation (use mujoco and all that)
# - goal is using world model to control so-101 arm (yes i need to get this asap)

class Encoder: 
    def __init__(self, channels, width, height):
        pass
    def forward():
