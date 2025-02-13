from mamba_ssm import Mamba
from torch import nn

import torch

class PVMLayer4(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 4,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
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

        x1, x2, x3, x4 = torch.chunk(x_norm, 4, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

class PVMLayer8(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 8,  # 由于我们要分成8块，所以这里是 input_dim // 8
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
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

        # 将输入分成8块
        x1, x2, x3, x4, x5, x6, x7, x8 = torch.chunk(x_norm, 8, dim=2)
        x_mamba1 = self.mamba(x1) + self.skip_scale * x1
        x_mamba2 = self.mamba(x2) + self.skip_scale * x2
        x_mamba3 = self.mamba(x3) + self.skip_scale * x3
        x_mamba4 = self.mamba(x4) + self.skip_scale * x4
        x_mamba5 = self.mamba(x5) + self.skip_scale * x5
        x_mamba6 = self.mamba(x6) + self.skip_scale * x6
        x_mamba7 = self.mamba(x7) + self.skip_scale * x7
        x_mamba8 = self.mamba(x8) + self.skip_scale * x8
        x_mamba = torch.cat([x_mamba1, x_mamba2, x_mamba3, x_mamba4, x_mamba5, x_mamba6, x_mamba7, x_mamba8], dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

class PVMLayer16(nn.Module):
    def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.norm = nn.LayerNorm(input_dim)
        self.mamba = Mamba(
            d_model=input_dim // 16,  # 由于我们要分成16块，所以这里是 input_dim // 16
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
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

        # 将输入分成16块
        x_chunks = torch.chunk(x_norm, 16, dim=2)
        x_mamba_chunks = [self.mamba(chunk) + self.skip_scale * chunk for chunk in x_chunks]
        x_mamba = torch.cat(x_mamba_chunks, dim=2)

        x_mamba = self.norm(x_mamba)
        x_mamba = self.proj(x_mamba)
        out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
        return out

    import torch
    import torch.nn as nn

    class PVMLayer2(nn.Module):
        def __init__(self, input_dim, output_dim, d_state=16, d_conv=4, expand=2):
            super().__init__()
            self.input_dim = input_dim
            self.output_dim = output_dim
            self.norm = nn.LayerNorm(input_dim)
            self.mamba = Mamba(
                d_model=input_dim // 2,  # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,  # Local convolution width
                expand=expand,  # Block expansion factor
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

            x1, x2 = torch.chunk(x_norm, 2, dim=2)
            x_mamba1 = self.mamba(x1) + self.skip_scale * x1
            x_mamba2 = self.mamba(x2) + self.skip_scale * x2
            x_mamba = torch.cat([x_mamba1, x_mamba2], dim=2)

            x_mamba = self.norm(x_mamba)
            x_mamba = self.proj(x_mamba)
            out = x_mamba.transpose(-1, -2).reshape(B, self.output_dim, *img_dims)
            return out

