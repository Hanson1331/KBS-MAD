from evolution_modules import *
import torch
from torch import nn


class Evo_Block(nn.Module):
    def __init__(self, in_channels, out_channels, input_resolution=None, model_type=None,
                 mlp_ratio=8., drop=0.0, drop_path=0.0, layer_i=0):
        super(Evo_Block, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        model_type = model_type.lower() if model_type is not None else 'gsta'

        if model_type == 'gsta':
            self.block = GASubBlock(
                in_channels, kernel_size=21, mlp_ratio=mlp_ratio,
                drop=drop, drop_path=drop_path, act_layer=nn.GELU)
        elif model_type == 'convmixer':
            self.block = ConvMixerSubBlock(in_channels, kernel_size=11, activation=nn.GELU)
        elif model_type == 'convnext':
            self.block = ConvNeXtSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'uniformer':
            block_type = 'MHSA' if in_channels == out_channels and layer_i > 0 else 'Conv'
            self.block = UniformerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop,
                drop_path=drop_path, block_type=block_type)
        elif model_type == 'vit':
            self.block = ViTSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        elif model_type == 'poolformer':
            self.block = PoolFormerSubBlock(
                in_channels, mlp_ratio=mlp_ratio, drop=drop, drop_path=drop_path)
        else:
            assert False and "Error"

        if in_channels != out_channels:
            self.reduction = nn.Conv2d(
                in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        z = self.block(x)
        return z if self.in_channels == self.out_channels else self.reduction(z)

class Spatio_temporal_evolution(nn.Module):
    def __init__(self, channel_in, channel_hid, N2,
                 input_resolution=None, model_type=None,
                 mlp_ratio=4., drop=0.0, drop_path=0.1):
        super(Spatio_temporal_evolution, self).__init__()
        assert N2 >= 2 and mlp_ratio > 1
        self.N2 = N2
        dpr = [x.item() for x in torch.linspace(1e-2, drop_path, self.N2)]

        # down-sampling
        enc_layers = [Evo_Block(
            channel_in, channel_hid, input_resolution, model_type,
            mlp_ratio, drop, drop_path=dpr[0], layer_i=0)]

        # state-stacking
        for i in range(1, N2-1):
            enc_layers.append(Evo_Block(
                channel_hid, channel_hid, input_resolution, model_type,
                mlp_ratio, drop, drop_path=dpr[i], layer_i=i))

        # up-sampling
        enc_layers.append(Evo_Block(
            channel_hid, channel_in, input_resolution, model_type,
            mlp_ratio, drop, drop_path=drop_path, layer_i=N2-1))
        self.enc = nn.Sequential(*enc_layers)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)

        z = x
        for i in range(self.N2):
            z = self.enc[i](z)

        y = z.reshape(B, T, C, H, W)
        return y

if __name__ == '__main__':
    x = torch.randn([1, 1, 69, 180, 360])
    print("input shape:", x.shape)

    model = Spatio_temporal_evolution(channel_in=69,
                                      channel_hid=256,
                                      N2=4,
                                      input_resolution = [64, 64],
                                      model_type='poolformer',
                                      mlp_ratio=4.,
                                      drop=0.0,
                                      drop_path=0.1)
    print(model)
    output = model(x)
    print("output shape:", output.shape)