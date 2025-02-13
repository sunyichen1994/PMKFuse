import torch

from mamba_ssm import Mamba
from torch import nn


class PCVM(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=8, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 2,  # Adjusted for 8-channel processing
            d_state=d_state,
            d_conv=d_conv,  # Increased d_conv from 4 to 8
            expand=expand,
        )
        self.proj = nn.Linear(input_dim, output_dim)
        self.skip_scale = nn.Parameter(torch.ones(1))

    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.input_dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)

        # Split into 8 parts
        chunks = 2
        x_parts = torch.chunk(x_norm, chunks, dim=2)

        # Process each part through Mamba and sum them up with skip connection
        x_mambas = [self.mamba(part) for part in x_parts]
        x_mamba_sum = sum(x_mambas) + self.skip_scale * x_parts[0]

        # Concatenate all parts together
        x_mamba = torch.cat(x_mambas, dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)

        return out







