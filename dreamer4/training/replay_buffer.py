import numpy as np
import torch

class ReplayBuffer:
    """
    Replay buffer for storing sequences of observations, actions, and rewards.
    Now supports storing actions taken by a policy (actor).
    """
    def __init__(self, max_size, seq_len, obs_shape, action_dim, device="cpu"):
        self.max_size = max_size
        self.seq_len = seq_len
        self.device = device

        self.obs_buf = np.zeros((max_size, seq_len, *obs_shape), dtype=np.uint8)
        self.action_buf = np.zeros((max_size, seq_len, action_dim), dtype=np.float32)
        self.reward_buf = np.zeros((max_size, seq_len, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def add_sequence(self, obs_seq, action_seq, reward_seq):
        """
        Add a full sequence of (obs, action, reward) to the buffer.
        obs_seq: (seq_len, C, H, W) numpy array
        action_seq: (seq_len, action_dim)
        reward_seq: (seq_len, 1)
        """
        self.obs_buf[self.ptr] = obs_seq
        self.action_buf[self.ptr] = action_seq
        self.reward_buf[self.ptr] = reward_seq

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        """
        Sample a batch of sequences and return as PyTorch tensors.
        Returns: obs, actions, rewards (all on self.device)
        """
        idxs = np.random.randint(0, self.size, size=batch_size)
        obs_batch = torch.tensor(self.obs_buf[idxs], dtype=torch.float32, device=self.device) / 255.0
        action_batch = torch.tensor(self.action_buf[idxs], dtype=torch.float32, device=self.device)
        reward_batch = torch.tensor(self.reward_buf[idxs], dtype=torch.float32, device=self.device)
        return obs_batch, action_batch, reward_batch

    def __len__(self):
        return self.size