import torch
import torch.nn as nn
from .kagn_conv_v2 import KAGNConv2DLayerV2


class KAGNParallelConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, skip_scale=1.0, **norm_kwargs):
        super().__init__()

        branch_output_dim = output_dim // 4
        self.branches = nn.ModuleList([
            KAGNConv2DLayerV2(input_dim, branch_output_dim, kernel_size, degree, groups, padding, stride, dilation,
                              dropout, norm_layer, **norm_kwargs)
            for _ in range(4)
        ])

        self.skip_scale = skip_scale
        self.projection = nn.Conv2d(input_dim, output_dim, kernel_size=1) if input_dim != output_dim else None

    def forward(self, x):
        out = torch.cat([branch(x) for branch in self.branches], dim=1)

        if self.projection is not None:
            x = self.projection(x)

        if self.skip_scale != 0:
            out += self.skip_scale * x

        return out