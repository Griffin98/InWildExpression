import itertools

import math
import torch
import torch.nn as nn

from networks.stylegan2_concat import ConvLayer, EqualLinear, PixelNorm

"""
Conv + Norm + Activation Block
"""
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=5, padding=1, stride=1, norm="batch", activation="lrelu"):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding)

        if norm == "batch":
            self.norm = nn.BatchNorm2d(out_ch, affine=True)
        elif norm == "instance":
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm == "layer":
            self.norm = nn.LayerNorm(out_ch)
        else:
            self.norm = None

        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.Sigmoid()

    def forward(self, input):
        x = self.conv(input)
        if self.norm is not None:
            x = self.norm(x)
        x = self.activation(x)

        return x


class ExpressionFusionBlock(nn.Module):
    def __init__(self, channel=512, style_dim=512):
        super(ExpressionFusionBlock, self).__init__()

        self.linear_layers = nn.ModuleList()
        if channel == style_dim:
            pass
        else:
            rng = int(math.log(style_dim, 2) - math.log(channel, 2))
            in_ch = style_dim
            for i in range(rng):
                out_ch = in_ch // 2
                linear = EqualLinear(in_ch, out_ch, activation='fused_lrelu')
                in_ch = out_ch
                self.linear_layers.append(linear)

        self.conv1 = ConvBlock(channel * 2, channel, kernel=3, padding=1, stride=1, norm="instance", activation="relu")
        self.conv2 = ConvBlock(channel, channel, kernel=3, padding=1, stride=1, norm="instance", activation="relu")

        self.conv3 = ConvBlock(channel * 2, channel, kernel=3, padding=1, stride=1, norm="instance", activation="relu")
        self.conv4 = ConvBlock(channel, channel, kernel=3, padding=1, stride=1, norm="instance", activation="relu")

    def forward(self, map, style):
        for i in range(len(self.linear_layers)):
            style = self.linear_layers[i](style)
            # print(style.shape)

        style = style.unsqueeze(2).unsqueeze(3).repeat(1, 1, map.shape[2], map.shape[3])
        # print(style.shape)

        x = torch.cat([map, style], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)

        # print(x.shape)
        x = torch.cat([map, x], dim=1)
        x = self.conv3(x)
        x = self.conv4(x)

        return x


class ExpressionStyleFusionBlock(nn.Module):
    def __init__(self, in_ch, style_dim, spatial, expression_dim):
        super(ExpressionStyleFusionBlock, self).__init__()

        self.block_size = int(math.log(spatial, 2))

        blocks = nn.ModuleList()

        out_ch = in_ch
        in_ch = in_ch + expression_dim

        for i in range(self.block_size):
            if out_ch == style_dim:
                pass
            else:
                out_ch = out_ch * 2

            spatial = spatial // 2
            if spatial == 1:
                norm = None
            else:
                norm = "instance"

            conv_block = ConvBlock(in_ch, out_ch, kernel=3, padding=1, stride=2, norm=norm)
            blocks.append(conv_block)

            in_ch = out_ch

        self.blocks = nn.Sequential(*blocks)

        # Expression style
        self.final_linear = EqualLinear(style_dim, style_dim, lr_mul=0.01, activation='fused_lrelu')

    def forward(self, img, exp):

        exp = exp.unsqueeze(2).unsqueeze(3).repeat(1, 1, img.shape[2], img.shape[3])
        x = torch.cat((img, exp), dim=1)
        x = self.blocks(x)

        x = x.view(x.shape[0], -1)

        x = self.final_linear(x)

        return x


class StylizedExpressionEncoder(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            expression_dim,
            concat_indices,
            channel_multiplier=2,
            lr_mlp=0.01,
            narrow=1
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow)
        }

        self.log_size = int(math.log(size, 2))

        self.concat_indices = concat_indices

        # Face Encoder
        conv = [ConvLayer(3, channels[size], 1)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[size]
        self.names = ['ecd%d' % i for i in range(self.log_size - 1)]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True)]
            setattr(self, self.names[self.log_size - i + 1], nn.Sequential(*conv))
            in_channel = out_channel
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, style_dim, activation='fused_lrelu'))

        # Fusion MLP
        layers = []
        for i in range(n_mlp//2):
            if i == 0:
                # Append 1D expression input
                layers.append(
                    EqualLinear(
                        style_dim + expression_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                    )
                )
            else:
                layers.append(
                    EqualLinear(
                        style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                    )
                )
        self.fusion_linear = nn.Sequential(*layers)

        # ExpressionStyleFusion Blocks
        self.sfb_names = ['sfb%d' % i for i in range(self.log_size - 1)]
        for i in range(3, self.log_size + 2):
            spatial = 2 ** (i - 1)
            out_channel = channels[spatial]
            sfb = ExpressionStyleFusionBlock(out_channel, style_dim, spatial, expression_dim=expression_dim)
            setattr(self, self.sfb_names[i - 3], sfb)

        # ExpressionFusion Blocks
        self.fb_names = ['fb%d' % i for i in range(len(self.concat_indices))]
        for i in range(len(self.concat_indices)):
            fb = ExpressionFusionBlock(channel=channels[self.concat_indices[i]], style_dim=style_dim)
            setattr(self, self.fb_names[i], fb)

    def forward(self, inputs, expression):
        noise = []
        for i in range(self.log_size - 1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            noise.append(inputs)
        inputs = inputs.view(inputs.shape[0], -1)
        noise = noise[::-1]

        outs = self.final_linear(inputs)
        inputs = torch.cat((outs, expression), dim=1)
        fusion_style = self.fusion_linear(inputs)

        styles = []
        concat_noise = []
        for i in range(self.log_size - 1):
            sfb = getattr(self, self.sfb_names[i])
            style = sfb(noise[i], expression)
            styles.append(style)

            if noise[i].shape[2] in self.concat_indices:
                index = self.concat_indices.index(noise[i].shape[2])
                fb = getattr(self, self.fb_names[index])
                block = fb(noise[i], fusion_style)
                concat_noise.append(block)
            else:
                concat_noise.append(None)

            # iterate for 2nd time
            concat_noise.append(None)

        styles = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in styles))
        styles = torch.stack(styles, dim=1)
        return outs, concat_noise, styles


if __name__ == "__main__":
    in_ch = 128
    spatial_size = 32
    style_dim = 512
    batch_size = 4
    exp_dim = 50
    output_size = 512
    n_mlp = 8
    concat_indices = [32, 64, 128, 256]

    enc = StylizedExpressionEncoder(style_dim=style_dim, size=output_size, n_mlp=n_mlp, expression_dim=exp_dim,
                                    concat_indices=concat_indices).to("cpu")
    x = torch.randn(batch_size, 3, output_size, output_size).to("cpu")
    exp = torch.randn(batch_size, exp_dim).to("cpu")

    y, noise, styles = enc(x, exp)
    noise = noise[1:]
    print(len(noise))
    for n in noise:
        if n is None:
            print("None")
        else:
            print(n.shape)


