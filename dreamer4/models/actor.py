import torch
import torch.nn as nn 
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
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim * categories, 400),
            nn.ReLU(),
            nn.Linear(400, action_dim)  # output mean of actions
        )

    def forward(self, h, z):
        z = z.view(z.size(0), -1)
        x = torch.cat([h, z], dim=-1)
        action = self.fc(x)
        return action
