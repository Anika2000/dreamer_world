import torch
from dreamer4.models.encoder import Encoder

config = {
    "env": {
        "channels": 3
    },
    "model": {
        "encoder_channels": [32, 64, 128, 256],
        "embedding_dim": 1024
    }
}

encoder = Encoder(config)

x = torch.randn(8, 3, 64, 64)
embedding = encoder(x)

print(embedding.shape)  # should be (8, 1024)
