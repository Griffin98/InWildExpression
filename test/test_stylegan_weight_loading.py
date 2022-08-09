import math

import __init_paths
import torch
from networks.stylegan2_concat import Generator

from torchvision.utils import save_image


def load_stylegan_dict(model, pretrained_dict, log_size):

    model = model.state_dict()

    # Input layer
    # Alter conv1.activate.bias
    original_tensor = pretrained_dict["conv1.activate.bias"]
    required_tensor = model["conv1.activate.bias"]
    alter_layer = torch.randn_like(required_tensor)
    alter_layer[:original_tensor.shape[0]] = original_tensor
    pretrained_dict["conv1.activate.bias"] = alter_layer

    # Alter to_rgb1.conv.weight
    original_tensor = pretrained_dict["to_rgb1.conv.weight"]
    required_tensor = model["to_rgb1.conv.weight"]
    alter_layer = torch.randn_like(required_tensor)
    alter_layer[:, :, :original_tensor.shape[2], :, :] = original_tensor
    pretrained_dict["to_rgb1.conv.weight"] = alter_layer

    # Alter  to_rgb1.conv.modulation.weight:
    original_tensor = pretrained_dict["to_rgb1.conv.modulation.weight"]
    required_tensor = model["to_rgb1.conv.modulation.weight"]
    alter_layer = torch.randn_like(required_tensor)
    alter_layer[:original_tensor.shape[0], :] = original_tensor
    pretrained_dict["to_rgb1.conv.modulation.weight"] = alter_layer

    # Alter to_rgb1.conv.modulation.bias
    original_tensor = pretrained_dict["to_rgb1.conv.modulation.bias"]
    required_tensor = model["to_rgb1.conv.modulation.bias"]
    alter_layer = torch.randn_like(required_tensor)
    alter_layer[:original_tensor.shape[0]] = original_tensor
    pretrained_dict["to_rgb1.conv.modulation.bias"] = alter_layer


    # convs.{}.conv.weight [:,:,alter,:,:]
    # convs.{}.conv.modulation.weight [alter,:]
    # convs.{}.conv.modulation.bias [alter]
    # convs.{}.conv.activate.bias [alter]
    for index in range(log_size * 2 - 2):
        print("Altering convs.{}.conv.weight".format(index))
        original_tensor = pretrained_dict["convs.{}.conv.weight".format(index)]
        required_tensor = model["convs.{}.conv.weight".format(index)]
        alter_layer = torch.randn_like(required_tensor)
        alter_layer[:, :, :original_tensor.shape[2], :, :] = original_tensor
        pretrained_dict["convs.{}.conv.weight".format(index)] = alter_layer

        print("Altering convs.{}.conv.modulation.weight".format(index))
        original_tensor = pretrained_dict["convs.{}.conv.modulation.weight".format(index)]
        required_tensor = model["convs.{}.conv.modulation.weight".format(index)]
        alter_layer = torch.randn_like(required_tensor)
        alter_layer[:original_tensor.shape[0], :] = original_tensor
        pretrained_dict["convs.{}.conv.modulation.weight".format(index)] = alter_layer

        print("Altering convs.{}.conv.modulation.bias".format(index))
        original_tensor = pretrained_dict["convs.{}.conv.modulation.bias".format(index)]
        required_tensor = model["convs.{}.conv.modulation.bias".format(index)]
        alter_layer = torch.randn_like(required_tensor)
        alter_layer[:original_tensor.shape[0]] = original_tensor
        pretrained_dict["convs.{}.conv.modulation.bias".format(index)] = alter_layer

        print("Altering convs.{}.activate.bias".format(index))
        original_tensor = pretrained_dict["convs.{}.activate.bias".format(index)]
        required_tensor = model["convs.{}.activate.bias".format(index)]
        alter_layer = torch.randn_like(required_tensor)
        alter_layer[:original_tensor.shape[0]] = original_tensor
        pretrained_dict["convs.{}.activate.bias".format(index)] = alter_layer

    # to_rgbs.{}.conv.weight [:,:,alter,:,:]
    # to_rgbs.{}.conv.modulation.weight [alter,:]
    # to_rgbs.{}.conv.modulation.bias [alter]

    for index in range(log_size - 1):
        print("Altering to_rgbs.{}.conv.weight".format(index))
        original_tensor = pretrained_dict["to_rgbs.{}.conv.weight".format(index)]
        required_tensor = model["to_rgbs.{}.conv.weight".format(index)]
        alter_layer = torch.randn_like(required_tensor)
        alter_layer[:, :, :original_tensor.shape[2], :, :] = original_tensor
        pretrained_dict["to_rgbs.{}.conv.weight".format(index)] = alter_layer

        print("Altering to_rgbs.{}.conv.modulation.weight".format(index))
        original_tensor = pretrained_dict["to_rgbs.{}.conv.modulation.weight".format(index)]
        required_tensor = model["to_rgbs.{}.conv.modulation.weight".format(index)]
        alter_layer = torch.randn_like(required_tensor)
        alter_layer[:original_tensor.shape[0], :] = original_tensor
        pretrained_dict["to_rgbs.{}.conv.modulation.weight".format(index)] = alter_layer

        print("Altering to_rgbs.{}.conv.modulation.bias".format(index))
        original_tensor = pretrained_dict["to_rgbs.{}.conv.modulation.bias".format(index)]
        required_tensor = model["to_rgbs.{}.conv.modulation.bias".format(index)]
        alter_layer = torch.randn_like(required_tensor)
        alter_layer[:original_tensor.shape[0]] = original_tensor
        pretrained_dict["to_rgbs.{}.conv.modulation.bias".format(index)] = alter_layer

    return pretrained_dict



def main():
    gan = Generator(512, 512, 8, device="cuda", isconcat=False).to("cuda")

    pretrained_dict = torch.load("weights/ffhq-512-avg-tpurun1.pt")["g_ema"]

    log_size = int(math.log(512, 2))

    modified_dict = load_stylegan_dict(gan, pretrained_dict, log_size-1)

    gan.load_state_dict(modified_dict)

    gan.load_state_dict(pretrained_dict)
    gan.eval()
    sample_z = torch.randn(4, 512, device="cuda")
    out, _ = gan([sample_z])



if __name__ == "__main__":
    main()
