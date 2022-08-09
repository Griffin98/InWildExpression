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


class StylizedExpressionEncoder(nn.Module):
    def __init__(
            self,
            size,
            style_dim,
            expression_dim,
            channel_multiplier=2,
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
            EqualLinear(channels[4] * 4 * 4, style_dim - expression_dim, activation='fused_lrelu'))


    def forward(self, inputs):
        temp = []
        for i in range(self.log_size - 1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            temp.append(inputs)
        inputs = inputs.view(inputs.shape[0], -1)
        temp = temp[::-1]

        outs = self.final_linear(inputs)

        noises = []
        for i in range(self.log_size * 2 - 3):
            if i == 4:
                noise = temp[2]
            else:
                noise = None

            noises.append(noise)

        return outs, noises


if __name__ == "__main__":
    enc = StylizedExpressionEncoder(512, 512, expression_dim=50).to("cpu")

    x = torch.randn(3, 3, 512, 512).to("cpu")
    exp = torch.randn(3, 50).to("cpu")

    z, noise = enc(x)

    from stylegan2_inject_concat import Generator

    gan = Generator(512, 512, 8, isconcat=True).to("cpu")

    pretrained_dict = torch.load("weights/ffhq-512-avg-tpurun1.pt")['g_ema']
    from utils.load_stylegan_weights import load_indexed_stylegan_dict

    modified_dict = load_indexed_stylegan_dict(gan, pretrained_dict)
    gan.load_state_dict(modified_dict)

    gan.load_state_dict(pretrained_dict)
    z = torch.cat((z, exp), dim=1)

    y, _ = gan([z], noise=noise)

    print(y.shape)

    from torchvision.utils import save_image
    save_image(y, "modified.png")





