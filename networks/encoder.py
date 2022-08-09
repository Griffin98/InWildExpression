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
    def __init__(self, in_ch, style_dim, spatial, expression_dim):
        super(ExpressionFusionBlock, self).__init__()

        self.block_size = int(math.log(spatial, 2))

        blocks = nn.ModuleList()

        out_ch = in_ch
        in_ch = in_ch + expression_dim

        # Ensure for lower spatial dimension we have atleast 4 conv block
        if self.block_size < 4:
            n_same_block = 4 - self.block_size
            for i in range(n_same_block):
                conv_block = ConvBlock(in_ch, in_ch, kernel=3, padding=1, stride=1)
                blocks.append(conv_block)

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
        layers = []
        for i in range(4):
            if i == 0:
                # Append 1D expression input
                layers.append(
                    EqualLinear(
                        style_dim * 2, style_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )
            else:
                layers.append(
                    EqualLinear(
                        style_dim, style_dim, lr_mul=0.01, activation='fused_lrelu'
                    )
                )

        self.expression_style = nn.Sequential(*layers)

    def forward(self, img, exp, pre_style):

        exp = exp.unsqueeze(2).unsqueeze(3).repeat(1, 1, img.shape[2], img.shape[3])
        x = torch.cat((img, exp), dim=1)
        x = self.blocks(x)

        x = x.view(x.shape[0], -1)
        x = torch.cat((x, pre_style), dim=1)

        x = self.expression_style(x)

        return x


class StylizedExpressionEncoder(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            n_mlp,
            expression_dim,
            channel_multiplier=2,
            lr_mlp=0.01,
            isconcat=True,
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
        layers = [PixelNorm()]
        for i in range(n_mlp):
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

        # Expression Fusion Blocks
        self.fb_names = ['fb%d' % i for i in range(self.log_size - 1)]
        for i in range(3, self.log_size + 2):
            spatial = 2 ** (i - 1)
            out_channel = channels[spatial]
            fb = ExpressionFusionBlock(out_channel, style_dim, spatial, expression_dim=expression_dim)
            setattr(self, self.fb_names[i - 3], fb)


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
        style = self.fusion_linear(inputs)

        styles = []
        for i in range(self.log_size - 1):
            fb = getattr(self, self.fb_names[i])
            style = fb(noise[i], expression, style)
            styles.append(style)

        noise = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noise))
        styles = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in styles))
        styles = torch.stack(styles, dim=1)
        return outs, noise, styles


if __name__ == "__main__":
    enc = StylizedExpressionEncoder(256, 512, 8, expression_dim=50).to("cpu")

    x = torch.randn(3, 3, 256, 256).to("cpu")
    exp = torch.randn(3, 50).to("cpu")

    y, noise, styles = enc(x, exp)
    for n in noise:
        print(n.shape)
    #

    # from stylegan2 import Generator
    # from torchvision.utils import save_image
    # gen = Generator(512, 512, 8, isconcat=True).to("cpu")
    #
    # styles = torch.stack(styles, dim=1)
    # z, _ = gen([styles], noise=noise[1:], input_is_latent=True, randomize_noise=True)
    # save_image(z, "scratch_stack.png")
    #
    # ss = torch.stack(ss).permute(1,0,2)
    # z, _ = gen([ss], noise=noise[1:], input_is_latent=True, randomize_noise=True)
    # save_image(z, "scratch_permute.png")


