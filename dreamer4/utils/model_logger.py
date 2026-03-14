import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from datetime import datetime

class ModelLogger:
    def __init__(self, log_dir="logs"):
        self.log_dir = log_dir
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def log_image(self, img_tensor, step, prefix="input"):
        """
        Logs an image (or observation) from the environment as a PNG.
        """
        img = img_tensor.cpu().numpy().transpose(1, 2, 0)  # Convert to HWC
        img = (img * 255).astype(np.uint8)  # Rescale to 0-255 if it's normalized

        # Create a filename
        img_filename = f"{self.log_dir}/{prefix}_step_{step}.png"
        plt.imshow(img)
        plt.axis('off')
        plt.savefig(img_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        print(f"Image saved at: {img_filename}")

    def log_latent(self, latent, step, prefix="latent"):
        """
        Logs the latent variable at a specific training step by saving it to a file.
        
        Args:
            latent: The latent tensor to log.
            step: The current training step.
        """
        # Detach the tensor from the computation graph
        latent = latent.detach()

        # Save it as a .npy file
        latent_filename = f"{self.log_dir}/latent_step_{step}.npy"
        np.save(latent_filename, latent.cpu().numpy())  # Now it's safe to call .numpy()
        print(f"Latent saved at: {latent_filename}")
    
    def log_reward(self, reward, step):
        """
        Log predicted rewards.
        """
        reward_filename = f"{self.log_dir}/reward_step_{step}.txt"
        with open(reward_filename, 'w') as f:
            f.write(str(reward))
        print(f"Reward saved at: {reward_filename}")
    
    def log_action(self, action, step):
        """
        Log the actions taken.
        """
        action_filename = f"{self.log_dir}/action_step_{step}.txt"
        with open(action_filename, 'w') as f:
            f.write(str(action))
        print(f"Action saved at: {action_filename}")
