# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.jit import jit_fuser

# Trying to apply @jit_fuser / @torch.compile to XIELU class causes issues with sharded_state_dict naming
@jit_fuser
def compiled_xielu(x, alpha_p, alpha_n, beta=0.5, eps=-1e-6):
    return torch.where(x > 0,
                      alpha_p * x * x + beta * x,
                      alpha_n * torch.expm1(torch.min(x, eps)) - alpha_n * x + beta * x)

class XIELU(nn.Module):
    def __init__(self, alpha_p_init=0.8, alpha_n_init=0.8, beta=0.5, eps=-1e-6):
        super().__init__() # config=config
        # self.config = config
        self.alpha_p = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_p_init, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.alpha_n = nn.Parameter(torch.log(torch.exp(torch.tensor(alpha_n_init - beta, dtype=torch.bfloat16)) - 1.0).unsqueeze(0))
        self.beta = beta
        self.eps = torch.tensor(eps, dtype=torch.bfloat16, device='cuda')

    def forward(self, x):
        alpha_p = F.softplus(self.alpha_p)
        alpha_n = self.beta + F.softplus(self.alpha_n)
        return compiled_xielu(x, alpha_p, alpha_n, self.beta, self.eps)
