import torch
import torch.nn as nn


def compute_gram_coeffs(degree, beta_weights):
    """Compute and cache the coefficients for the Gram polynomials."""
    coeffs = [1., 0.]  # Initial coefficients for P0 and P1.
    for i in range(2, degree + 1):
        new_coeff = -beta_weights[i - 1] * coeffs[-2]
        coeffs.append(new_coeff)
    return coeffs

class KAGNConvNDLayerV2(nn.Module):
    def __init__(self, conv_class, norm_class, input_dim, output_dim, degree, kernel_size,
                 groups=1, padding=0, stride=1, dilation=1, dropout: float = 0.0, ndim: int = 2,
                 **norm_kwargs):
        super(KAGNConvNDLayerV2, self).__init__()
        self.inputdim = input_dim
        self.outdim = output_dim
        self.degree = degree
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.base_activation = nn.SiLU()
        self.conv_w_fun = None
        self.ndim = ndim
        self.dropout = None
        self.norm_kwargs = norm_kwargs
        self.p_dropout = dropout

        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            elif ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            elif ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)

        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if input_dim % groups != 0:
            raise ValueError('input_dim must be divisible by groups')
        if output_dim % groups != 0:
            raise ValueError('output_dim must be divisible by groups')

        self.base_conv = conv_class(input_dim,
                                    output_dim,
                                    kernel_size,
                                    stride,
                                    padding,
                                    dilation,
                                    groups=groups,
                                    bias=False)

        self.layer_norm = norm_class(output_dim, **norm_kwargs)

        self.poly_conv = conv_class(input_dim * (degree + 1),
                                    output_dim,
                                    kernel_size,
                                    stride,
                                    padding,
                                    dilation,
                                    groups=groups,
                                    bias=False)

        self.beta_weights = nn.Parameter(torch.zeros(degree + 1, dtype=torch.float32))

        nn.init.kaiming_uniform_(self.base_conv.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.poly_conv.weight, nonlinearity='linear')

        nn.init.normal_(
            self.beta_weights,
            mean=0.0,
            std=1.0 / ((kernel_size ** ndim) * self.inputdim * (self.degree + 1.0)),
        )

    def beta(self, n, m):
        return (
            ((m + n) * (m - n) * n ** 2) / (m ** 2 / (4.0 * n ** 2 - 1.0))
        ) * self.beta_weights[n]


    def gram_poly(self, x, degree):
        # Get the cached coefficients for the Gram polynomials.
        coeffs = compute_gram_coeffs(degree, self.beta_weights)

        # Compute the Gram polynomials for each element in x using the cached coefficients.
        grams_basis = [torch.ones_like(x), x]  # Start with P0(x) = 1 and P1(x) = x.

        for n in range(2, degree + 1):
            curr_poly = x * grams_basis[-1] - coeffs[n] * grams_basis[-2]
            grams_basis.append(curr_poly)

        # Concatenate all Gram polynomials along the feature dimension.
        grams_basis = torch.stack(grams_basis, dim=1)

        # Ensure the dimensions match the expected input shape for the following convolution.
        # Assuming the input shape before the gram_poly operation was [batch_size, channels, height, width],
        # the output shape should be [batch_size, (degree+1)*channels, height, width].
        batch_size, _, height, width = x.shape
        grams_basis = grams_basis.view(batch_size, -1, height, width)

        return grams_basis


    def forward_kag(self, x):
        basis = self.base_conv(self.base_activation(x))

        # Normalizing x for stable Legendre polynomial computation
        x = torch.tanh(x).contiguous()

        if self.dropout is not None:
            x = self.dropout(x)

        grams_basis = self.base_activation(self.gram_poly(x, self.degree))

        y = self.poly_conv(grams_basis)

        y = self.base_activation(self.layer_norm(y + basis))

        return y

    def forward(self, x):
        return self.forward_kag(x)


class KAGNConv3DLayerV2(KAGNConvNDLayerV2):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm3d, **norm_kwargs):
        super(KAGNConv3DLayerV2, self).__init__(nn.Conv3d, norm_layer,
                                                input_dim, output_dim,
                                                degree, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=3, dropout=dropout, **norm_kwargs)


class KAGNConv2DLayerV2(KAGNConvNDLayerV2):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm2d, **norm_kwargs):
        super(KAGNConv2DLayerV2, self).__init__(nn.Conv2d, norm_layer,
                                                input_dim, output_dim,
                                                degree, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=2, dropout=dropout, **norm_kwargs)


class KAGNConv1DLayerV2(KAGNConvNDLayerV2):
    def __init__(self, input_dim, output_dim, kernel_size, degree=3, groups=1, padding=0, stride=1, dilation=1,
                 dropout: float = 0.0, norm_layer=nn.InstanceNorm1d, **norm_kwargs):
        super(KAGNConv1DLayerV2, self).__init__(nn.Conv1d, norm_layer,
                                                input_dim, output_dim,
                                                degree, kernel_size,
                                                groups=groups, padding=padding, stride=stride, dilation=dilation,
                                                ndim=1, dropout=dropout, **norm_kwargs)
