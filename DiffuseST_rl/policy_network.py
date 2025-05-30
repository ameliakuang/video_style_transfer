import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in1   = nn.InstanceNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.in2   = nn.InstanceNorm2d(channels)

    def forward(self, x):
        identity = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        return F.relu(out + identity)

class LatentPolicy(nn.Module):
    def __init__(self, base_ch=16):
        super().__init__() # the latent from the last timestep diffusion 1 x 4 x 64 x 113
        latent_channels = 4
        in_ch = 2 * latent_channels  # we concat z_t and z_{t+1}

        self.conv1 = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(base_ch)
        
        self.conv2 = nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(base_ch)

        self.res = ResBlock(base_ch)

        self.mu_head     = nn.Conv2d(base_ch, latent_channels, kernel_size=1)
        self.logstd_head = nn.Conv2d(base_ch, latent_channels, kernel_size=1)
    
        nn.init.xavier_uniform_(self.mu_head.weight, gain=0.01)
        nn.init.constant_(self.mu_head.bias, 0)
        nn.init.xavier_uniform_(self.logstd_head.weight, gain=0.01)
        nn.init.constant_(self.logstd_head.bias, -1)  # start with small variance
        
    def forward(self, z_t, z_tp1):
        x = torch.cat([z_t, z_tp1], dim=1)
        x = F.relu(self.in1(self.conv1(x)))
        x = F.relu(self.in2(self.conv2(x)))
        x = self.res(x)

        mu = self.mu_head(x)
        logstd = self.logstd_head(x).clamp(-5, 2)
        std = logstd.exp()
        return mu, std
    def sample(self, z_t, z_tp1):
        mu, std = self.forward(z_t, z_tp1)
        dist = Normal(mu, std)
        delta = dist.rsample()
        logp = dist.log_prob(delta).sum(dim=[1,2,3])     

        return delta, logp
