import torch
from dreamer4.models.world_model import WorldModel

def test_world_model():
    # ---- Configuration ----
    config = {
        "env": {
            "channels": 3,
            "image_size": 64,
            "action_dim": 6,
        },
        "model": {
            "embedding_dim": 1024,
            "hidden_dim": 200,
            "latent_dim": 32,
            "categories": 32,
            "encoder_channels": [32, 64, 128, 256],
            "decoder_channels": [256, 128, 64, 32],
        }
    }

    batch_size = 8
    # ---- Dummy input ----
    obs = torch.randn(batch_size, config["env"]["channels"], 
                      config["env"]["image_size"], config["env"]["image_size"])
    prev_a = torch.randn(batch_size, config["env"]["action_dim"])
    prev_h = torch.zeros(batch_size, config["model"]["hidden_dim"])
    prev_z = torch.zeros(batch_size, config["model"]["latent_dim"], config["model"]["categories"])
    prev_z[:, :, 0] = 1  # simple one-hot initialization

    # ---- Initialize world model ----
    wm = WorldModel(config)

    # ---- Forward pass ----
    out = wm(obs, prev_a, prev_h, prev_z, use_relaxed=True, temperature=0.67)

    # ---- Print shapes ----
    print("Shapes of world model outputs:")
    print("h:", out["h"].shape)
    print("z:", out["z"].shape)
    print("z_prior:", out["z_prior"].shape)
    print("reconstructed:", out["reconstructed"].shape)
    print("reward:", out["reward"].shape)
    print("value:", out["value"].shape)

    # ---- Gradient check ----
    total = out["reconstructed"].sum() + out["reward"].sum() + out["value"].sum()
    total.backward()
    print("\nBackward pass succeeded, gradients flow correctly!")

if __name__ == "__main__":
    test_world_model()
