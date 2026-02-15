import torch
from dreamer4.models.encoder import Encoder  # or whatever your package path is

encoder = Encoder()
x = torch.randn(8, 3, 64, 64)  # batch of 8 dummy images
embedding = encoder(x)
print(embedding.shape)  # should be (8, 1024)
