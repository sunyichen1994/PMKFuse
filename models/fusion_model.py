import torch
import torch.nn as nn
import os
from .PKAGN import KAGNParallelConv
from .PCrossVim import PCVM


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


class Local_Encoder(nn.Module):
    def __init__(self):
        super(Local_Encoder, self).__init__()

        self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=3, out_channels=4, stride=1, padding=1)
        self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=3, out_channels=4, stride=1, padding=1)


        self.vi_kagn2 = KAGNParallelConv(input_dim=4, output_dim=8, kernel_size=1)
        self.ir_kagn2 = KAGNParallelConv(input_dim=4, output_dim=8, kernel_size=1)

        self.vi_kagn3 = KAGNParallelConv(input_dim=8, output_dim=16, kernel_size=1)
        self.ir_kagn3 = KAGNParallelConv(input_dim=8, output_dim=16, kernel_size=1)

        self.vi_kagn4 = KAGNParallelConv(input_dim=16, output_dim=32, kernel_size=1)
        self.ir_kagn4 = KAGNParallelConv(input_dim=16, output_dim=32, kernel_size=1)

    def forward(self, y_vi_image, ir_image):
        activate = nn.GELU()
        activate = activate.cuda()
        vi_out = self.vi_conv1(y_vi_image)
        ir_out = self.vi_conv1(ir_image)
        vi_out, ir_out = activate(self.vi_kagn2(vi_out)), activate(self.ir_kagn2(ir_out))
        vi_out, ir_out = activate(self.vi_kagn3(vi_out)), activate(self.ir_kagn3(ir_out))
        vi_out, ir_out = activate(self.vi_kagn4(vi_out)), activate(self.ir_kagn4(ir_out))

        return vi_out, ir_out


# class Local_Encoder(nn.Module):
#     def __init__(self):
#
#         super(Local_Encoder, self).__init__()
#
#         self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=3, out_channels=4, stride=1, padding=1)
#         self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=3, out_channels=4, stride=1, padding=1)
#
#         self.vi_conv2 = AKConv(inc=4, outc=8, num_param=3, stride=1, bias=None)
#         self.ir_conv2 = AKConv(inc=4, outc=8, num_param=3, stride=1, bias=None)
#
#         self.vi_conv3 = AKConv(inc=8, outc=16, num_param=3, stride=1, bias=None)
#         self.ir_conv3 = AKConv(inc=8, outc=16, num_param=3, stride=1, bias=None)
#
#         self.vi_conv4 = AKConv(inc=16, outc=32, num_param=3, stride=1, bias=None)
#         self.ir_conv4 = AKConv(inc=16, outc=32, num_param=3, stride=1, bias=None)
#
#         self.vi_conv2 = AKConv(inc=4, outc=8, num_param=3, stride=1, bias=None)
#         self.ir_conv2 = AKConv(inc=4, outc=8, num_param=3, stride=1, bias=None)
#
#         self.vi_conv3 = AKConv(inc=8, outc=16, num_param=3, stride=1, bias=None)
#         self.ir_conv3 = AKConv(inc=8, outc=16, num_param=3, stride=1, bias=None)
#
#         self.vi_conv4 = AKConv(inc=16, outc=32, num_param=3, stride=1, bias=None)
#         self.ir_conv4 = AKConv(inc=16, outc=32, num_param=3, stride=1, bias=None)
#
#
#     def forward(self, y_vi_image, ir_image):
#
#         activate = nn.GELU()
#         activate = activate.cuda()
#         vi_out = self.vi_conv1(y_vi_image)
#         ir_out = self.vi_conv1(ir_image)
#         vi_out, ir_out = activate(self.vi_conv2(vi_out)), activate(self.ir_conv2(ir_out))
#         vi_out, ir_out = activate(self.vi_conv3(vi_out)), activate(self.ir_conv3(ir_out))
#         vi_out, ir_out = activate(self.vi_conv4(vi_out)), activate(self.ir_conv4(ir_out))
#
#         return vi_out, ir_out

class Global_Encoder(nn.Module):
            def __init__(self):
                super(Global_Encoder, self).__init__()

                self.vi_conv1 = nn.Conv2d(in_channels=1, kernel_size=3, out_channels=4, stride=1, padding=1)
                self.ir_conv1 = nn.Conv2d(in_channels=1, kernel_size=3, out_channels=4, stride=1, padding=1)

                self.vi_pcvm2 = PCVM(input_dim=4, output_dim=8)
                self.ir_pcvm2 = PCVM(input_dim=4, output_dim=8)

                self.vi_pcvm3 = PCVM(input_dim=8, output_dim=16)
                self.ir_pcvm3 = PCVM(input_dim=8, output_dim=16)

                self.vi_pcvm4 = PCVM(input_dim=16, output_dim=32)
                self.ir_pcvm4 = PCVM(input_dim=16, output_dim=32)

            def forward(self, y_vi_image, ir_image):
                activate = nn.GELU()
                activate = activate.cuda()
                vi_out = self.vi_conv1(y_vi_image)
                ir_out = self.vi_conv1(ir_image)

                vi_out, ir_out = (activate(self.vi_pcvm2(vi_out)), activate(self.ir_pcvm2(ir_out)))
                vi_out, ir_out = (activate(self.vi_pcvm3(vi_out)), activate(self.ir_pcvm3(ir_out)))
                vi_out, ir_out = (activate(self.vi_pcvm4(vi_out)), activate(self.ir_pcvm4(ir_out)))

                return vi_out, ir_out


# class Local_Decoder(nn.Module):
#
#     def __init__(self):
#         super(Local_Decoder, self).__init__()
#
#         self.ak1 = AKConv(inc=64, outc=32, num_param=3, stride=1, bias=None)
#         self.ak2 = AKConv(inc=32, outc=16, num_param=3, stride=1, bias=None)
#         self.ak3 = AKConv(inc=16, outc=8, num_param=3, stride=1, bias=None)
#         # self.conv4 = AKConv(inc=16, outc=1, num_param=3, stride=1, bias=None)
#         self.conv4 = nn.Conv2d(in_channels=8, kernel_size=3, out_channels=1, stride=1, padding=1)
#
#
#     def forward(self, x):
#         activate = nn.GELU()
#         activate = activate.cuda()
#         x = activate(self.ak1(x))
#         x = activate(self.ak2(x))
#         x = activate(self.ak3(x))
#         x = nn.Tanh()(self.conv4(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]
#
#         return x


class Local_Decoder(nn.Module):

    def __init__(self):
        super(Local_Decoder, self).__init__()

        self.kagn1 = KAGNParallelConv(input_dim=64, output_dim=32, kernel_size=1)
        self.kagn2 = KAGNParallelConv(input_dim=32, output_dim=16, kernel_size=1)
        self.kagn3 = KAGNParallelConv(input_dim=16, output_dim=8, kernel_size=1)
        # self.conv4 = AKConv(inc=16, outc=1, num_param=3, stride=1, bias=None)
        self.conv4 = nn.Conv2d(in_channels=8, kernel_size=3, out_channels=1, stride=1, padding=1)


    def forward(self, x):
        activate = nn.GELU()
        activate = activate.cuda()
        x = activate(self.kagn1(x))
        x = activate(self.kagn2(x))
        x = activate(self.kagn3(x))
        x = nn.Tanh()(self.conv4(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]

        return x

class Global_Decoder(nn.Module):
    def __init__(self):
        super(Global_Decoder, self).__init__()

        # self.pvm_layer1 = PCVM(input_dim=256, output_dim=128)
        # self.pvm_layer2 = PCVM(input_dim=128, output_dim=64)
        # self.pvm_layer3 = PCVM(input_dim=64, output_dim=32
        # self.conv1 = nn.Conv2d(in_channels=32, kernel_size=3, out_channels=1, stride=1, padding=1)


        self.pcvm1 = PCVM(input_dim=64, output_dim=32)
        self.pcvm2 = PCVM(input_dim=32, output_dim=16)
        self.pcvm3 = PCVM(input_dim=16, output_dim=8)
        self.conv4 = nn.Conv2d(in_channels=8, kernel_size=3, out_channels=1, stride=1, padding=1)


    def forward(self, x):
        activate = nn.GELU()
        activate = activate.cuda()

        x = activate(self.pcvm1(x))
        x = activate(self.pcvm2(x))
        x = activate(self.pcvm3(x))

        x = nn.Tanh()(self.conv4(x)) / 2 + 0.5  # 将范围从[-1,1]转换为[0,1]

        return x


def Fusion(vi_out, ir_out):
    return torch.cat([vi_out, ir_out], dim=1)

class PMKFuse(nn.Module):
    def __init__(self):
        super(PMKFuse, self).__init__()

        self.global_encoder = Global_Encoder()
        self.local_encoder = Local_Encoder()

        self.global_decoder = Global_Decoder()
        self.local_decoder = Local_Decoder()


    def forward(self, y_vi_image, ir_image):

        vi_global_encoder, ir_global_encoder = self.global_encoder(y_vi_image, ir_image)
        vi_local_encoder, ir_local_encoder = self.local_encoder(y_vi_image, ir_image)

        fused_global_output = Fusion(vi_global_encoder, ir_global_encoder)
        fused_local_output = Fusion(vi_local_encoder, ir_local_encoder)

        fused_global_image = self.global_decoder(fused_global_output)
        fused_local_image = self.local_decoder(fused_local_output)

        fused_image = fused_global_image * fused_local_image

        # fused_vi_output = Fusion(vi_global_encoder, vi_local_encoder)
        # fused_ir_output = Fusion(ir_global_encoder, ir_local_encoder)
        #
        #
        # fused_vi_image = self.global_decoder(fused_vi_output)
        # fused_ir_image = self.local_decoder(fused_ir_output)
        #
        #
        # fused_image = fused_vi_image * fused_ir_image

        return fused_image




