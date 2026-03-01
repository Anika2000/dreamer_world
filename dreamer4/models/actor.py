import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
# Purpose of Actor:
# Takes the latent state (h_t, z_t) from the RSSM/world model and produces actions.
# This is your policy network.: A policy network is just a neural network that decides what action to take in a given situation.
#Why separate from the world model?
#World model learns the dynamics and predicts rewards;
#Actor decides what to do, either in imagination rollouts or in the real environment.
class Actor(nn.Module):
    #Inputs: 
    #h_t → deterministic hidden state from RSSM
    #z_t → stochastic latent
    #Outputs: 
    #a_t → action to take (continuous or discrete, depending on env)
    def __init__(self, hidden_dim, latent_dim, categories, action_dim):
        super().__init__()
        input_dim = hidden_dim + latent_dim * categories
        self.net = nn.Sequential(
            nn.Linear(input_dim, 400),
            nn.ReLU(),
        )
        self.mean = nn.Linear(400, action_dim)
        self.log_std = nn.Linear(400, action_dim)

        self.LOG_STD_MIN = -5
        self.LOG_STD_MAX = 2

    def forward(self, h, z):
        z = z.view(z.size(0), -1)
        x = torch.cat([h, z], dim=-1)
        x = self.net(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        raw_action = dist.rsample()
        action = torch.tanh(raw_action)
        log_prob = dist.log_prob(raw_action)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob
