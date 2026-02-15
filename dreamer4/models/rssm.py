#Recurrent model
#Representation model 
#Transition predictor

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import OneHotCategorical, RelaxedOneHotCategorical

class RSSM(nn.Module):
    def __init__(self, action_dim, embedding_dim, hidden_dim=200, latent_dim=32, categories=32):
        #the action dim is the dimension of the action a1, a2 etc.
        #embedding_dim is the size of the encoder output vector x_t 
        #hidden_dim is the size of the GRU hidden state (purple sqaure in the image). If 200, its 
        # a 200-element vector that summarizes the past. 
        #latent_dim is the number of stochastic latent variable (green circle z_t) - e.g., 32 different 
        #"features" of the world
        #categories is the number of discreate states each latent variable can take (32 categories per latent).
        super().__init__()
        self.latent_dim = latent_dim #green circles z_t 
        self.categories = categories #the 32 classes and 32 categories 
        self.hidden_dim = hidden_dim #purple squares 
        
        # GRU for deterministic hidden state, this is the current hidden state, 
        # h_t depending on the previous z_{t-1} and concatenated with the dim of a_{t-1}
        self.gru = nn.GRUCell(latent_dim * categories + action_dim, hidden_dim)
        
        # Posterior network q(z_t | h_t, x_t)
        #nn.Sequential is a shortcut for stacking layers in order. 
        #Think of it as: "Take input -> pass through first Linear layer -> pass 
        # through ReLU -> pass through second Linear layer -> output"
        #here we are just setting things up (blueprint) ... in the forward function we pass in the actual vectors/input variables
        #here we are taking in the observation x_t so thats why we have extra + embedding_dim that is not in the prior
        self.post_net = nn.Sequential(
            #hidden_dim + embedding_dim vector gets turned into a (B, 400) dim vector
            nn.Linear(hidden_dim + embedding_dim, 400),
            nn.ReLU(),
            #takes (B, 400)-dim vector -> outputs (B, 1024) logits (flat)
            nn.Linear(400, latent_dim * categories),
        )
        # a question might be what is so special about 400 and why 400?
        #answer: its just an intermediate layer (400)
        #it gives the model more capacity and allows it to learn a more complex nonlinear mapping
        #makes the posterior more expressive 
        # Prior network p(z_t | h_t)
        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 400),
            nn.ReLU(),
            nn.Linear(400, latent_dim * categories),
        )
    
    def forward(self, prev_h, prev_z, prev_a, embed, use_relaxed=False, temperature=0.67):
        # prev_z: shape (B, latent_dim, categories) one-hot or logits
        # flatten categorical latent for GRU input
        # so for example if prev_z.shape returned (8, 4, 3), after the below line
        # is executed then it will returnr (8, 3*4=12) -> (8, 12)
        prev_z_flat = prev_z.view(prev_z.size(0), -1)
        #the below line will concat prev_z_flat with prev_a, so for example, 
        #if we had prev_a.shape return (8, 12) then the new gru_input size, 
        #after invoking the below line is (8, 14)
        gru_input = torch.cat([prev_z_flat, prev_a], dim=-1)
        
        # deterministic hidden state update
        #let prev_h shape be (B, hidden_dim) = (8, 5)
        #output shape: (B, hidden_dim) = (8,5)
        h = self.gru(gru_input, prev_h)
        
        # posterior logits (combine h and encoder embedding)
        #let embed.shape = (b, embedding_dim) = (8, 6)
        # so torch.cat([h, embed], dim=-1) --> shape: (B, hidden_dim + embedding_dim) = (8, 11)
        post_logits = self.post_net(torch.cat([h, embed], dim=-1))
        #after going through post_net, we get (B, 1024) which is then reshaped to (B, 32, 32) here
        post_logits = post_logits.view(-1, self.latent_dim, self.categories)
        #the below line turns the (32,32) logits into a categorical distribution,
        #and then the sampling gives your concrete latent vector z
        if use_relaxed: #for training because during training, you need gradients to flow
            #through z_t so the encoder and RSSM can improve together 
            post_dist = RelaxedOneHotCategorical(temperature=temperature, logits=post_logits)
            z = post_dist.rsample()
        else:
            post_dist = OneHotCategorical(logits=post_logits) 
            z = post_dist.sample()  # relaxed sample
        
        # prior logits (from h only)
        prior_logits = self.prior_net(h)
        prior_logits = prior_logits.view(-1, self.latent_dim, self.categories)
        if use_relaxed: 
            prior_dist = RelaxedOneHotCategorical(temperature=temperature, logits=prior_logits)
            z_prior = prior_dist.rsample()
        else:
            prior_dist = OneHotCategorical(logits=prior_logits)
            z_prior = prior_dist.sample()
        return h, z, z_prior, post_dist, prior_dist
