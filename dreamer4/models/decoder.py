import torch
import torch.nn as nn 

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        model_cfg = config["model"]
        hidden_dim = model_cfg["hidden_dim"]
        latent_dim = model_cfg["latent_dim"]
        categories = model_cfg["categories"]
        dec_channels = model_cfg["decoder_channels"]
        out_channels = config["env"]["channels"]
        
        #Step 3: 
        #Start with a small 4 by 4 image (like a tiny “seed” image) with dec_channels[0] channels.
        #Then we’ll upsample it back to the original image size using ConvTranspose2d.
        self.fc = nn.Linear(
            hidden_dim + latent_dim * categories,
            dec_channels[0] * 4 * 4
        )

        #Step 5: Upsampling with Transposed Convolutions
        #ConvTranspose2d = “deconvolution” = upsample + convolution
        #Each layer doubles the spatial size because stride=2
        #Last layer outputs out_channels (e.g., 3 for RGB images)
        layers = []
        in_ch = dec_channels[0]
        for out_ch in dec_channels[1:]:
            layers.append(nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1))
            layers.append(nn.ReLU())
            in_ch = out_ch

        layers.append(nn.ConvTranspose2d(in_ch, out_channels, 4, stride=2, padding=1))

        self.deconv = nn.Sequential(*layers)
    
    #overall: Flatten z → concat with h → FC → reshape → deconv
    def forward(self, h, z):
        #Step 1: original z shape: (B, latent_dim, categories)
        #after flatten: (B, latent_dim * categories)
        #we flatten so we can combine it with the hidden state h 
        z = z.view(z.size(0), -1) 
        #Step 2: h shape: (B, hidden_dim)
        #z_flat shape: (B, latent_dim * categories)
        #After concat: (B, hidden_dim + latent_dim * categories)
        x = torch.cat([h, z], dim=-1)
        #Step 3: This takes the combined vector and maps it into a tensor that can be reshaped into a small feature map.
        #The self.fc is defined in the __init__ function
        #After fc: x shape = (B, dec_channels[0]*4*4)
        x = self.fc(x)
        #Step 4: x is currently a flat list of numbers (a vector) for each item in the batch.
        #This line reshapes that flat list into a 4×4 grid — like turning a list into a tiny square image.
        x = x.view(x.size(0), -1, 4, 4)
        #step 5:
        x = self.deconv(x)
        #Output: (B, channels, 64, 64) → a reconstructed image
        return x
