import math

import torch

"""
Function to load modified stylegan weight by noise concatenation.
"""
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


def old(model, pretrained_dict):

    model = model.state_dict()

    # Input layer
    # Alter conv1.activate.bias
    original_tensor = pretrained_dict["convs.3.activate.bias"]
    required_tensor = model["convs.3.activate.bias"]
    alter_layer = torch.randn_like(required_tensor)
    alter_layer[:original_tensor.shape[0]] = original_tensor
    pretrained_dict["convs.3.activate.bias"] = alter_layer

    index = 4
    original_tensor = pretrained_dict["convs.{}.conv.weight".format(index)]
    required_tensor = model["convs.{}.conv.weight".format(index)]
    alter_layer = torch.randn_like(required_tensor)
    alter_layer[:, :, :original_tensor.shape[2], :, :] = original_tensor
    pretrained_dict["convs.{}.conv.weight".format(index)] = alter_layer

    # print("Altering convs.{}.conv.modulation.weight".format(index))
    original_tensor = pretrained_dict["convs.{}.conv.modulation.weight".format(index)]
    required_tensor = model["convs.{}.conv.modulation.weight".format(index)]
    alter_layer = torch.randn_like(required_tensor)
    alter_layer[:original_tensor.shape[0], :] = original_tensor
    pretrained_dict["convs.{}.conv.modulation.weight".format(index)] = alter_layer

    # print("Altering convs.{}.conv.modulation.bias".format(index))
    original_tensor = pretrained_dict["convs.{}.conv.modulation.bias".format(index)]
    required_tensor = model["convs.{}.conv.modulation.bias".format(index)]
    alter_layer = torch.randn_like(required_tensor)
    alter_layer[:original_tensor.shape[0]] = original_tensor
    pretrained_dict["convs.{}.conv.modulation.bias".format(index)] = alter_layer

    # Alter to_rgb1.conv.weight
    original_tensor = pretrained_dict["to_rgbs.1.conv.weight"]
    required_tensor = model["to_rgbs.1.conv.weight"]
    alter_layer = torch.randn_like(required_tensor)
    alter_layer[:, :, :original_tensor.shape[2], :, :] = original_tensor
    pretrained_dict["to_rgbs.1.conv.weight"] = alter_layer

    # Alter  to_rgb1.conv.modulation.weight:
    original_tensor = pretrained_dict["to_rgbs.1.conv.modulation.weight"]
    required_tensor = model["to_rgbs.1.conv.modulation.weight"]
    alter_layer = torch.randn_like(required_tensor)
    alter_layer[:original_tensor.shape[0], :] = original_tensor
    pretrained_dict["to_rgbs.1.conv.modulation.weight"] = alter_layer

    # Alter to_rgb1.conv.modulation.bias
    original_tensor = pretrained_dict["to_rgbs.1.conv.modulation.bias"]
    required_tensor = model["to_rgbs.1.conv.modulation.bias"]
    alter_layer = torch.randn_like(required_tensor)
    alter_layer[:original_tensor.shape[0]] = original_tensor
    pretrained_dict["to_rgbs.1.conv.modulation.bias"] = alter_layer


    return pretrained_dict


def load_indexed_stylegan_dict(model, pretrained_dict, concat_index):
    if len(concat_index) < 1:
        return pretrained_dict

    model = model.state_dict()

    start = int(math.log(concat_index[0], 2) - 1)
    end = int(math.log(concat_index[-1], 2) * 2 - 4)

    for i in range(start, end):

        if i % 2 == 0:
            # modify noise weight
            original_tensor = pretrained_dict["convs.{}.noise.weight".format(i)]
            required_tensor = model["convs.{}.noise.weight".format(i)]
            if original_tensor.shape != required_tensor.shape:
                print("Modifying weight at: convs.{}.noise.weight".format(i))
                pretrained_dict["convs.{}.noise.weight".format(i)] = torch.randn_like(required_tensor)

            # modify activation bias
            original_tensor = pretrained_dict["convs.{}.activate.bias".format(i)]
            required_tensor = model["convs.{}.activate.bias".format(i)]
            if original_tensor.shape != required_tensor.shape:
                print("Modifying weight at: convs.{}.activate.bias".format(i))
                alter_layer = torch.randn_like(required_tensor)
                alter_layer[:original_tensor.shape[0]] = original_tensor
                pretrained_dict["convs.{}.activate.bias".format(i)] = alter_layer

        else:
            # modify conv weight
            original_tensor = pretrained_dict["convs.{}.conv.weight".format(i)]
            required_tensor = model["convs.{}.conv.weight".format(i)]
            if original_tensor.shape != required_tensor.shape:
                print("Modifying weight at: convs.{}.conv.weight".format(i))
                alter_layer = torch.randn_like(required_tensor)
                alter_layer[:, :, :original_tensor.shape[2], :, :] = original_tensor
                pretrained_dict["convs.{}.conv.weight".format(i)] = alter_layer

            # modify conv modulation weight
            original_tensor = pretrained_dict["convs.{}.conv.modulation.weight".format(i)]
            required_tensor = model["convs.{}.conv.modulation.weight".format(i)]
            if original_tensor.shape != required_tensor.shape:
                print("Modifying weight at: convs.{}.conv.modulation.weight".format(i))
                alter_layer = torch.randn_like(required_tensor)
                alter_layer[:original_tensor.shape[0], :] = original_tensor
                pretrained_dict["convs.{}.conv.modulation.weight".format(i)] = alter_layer

            # # modify conv modulation bias
            original_tensor = pretrained_dict["convs.{}.conv.modulation.bias".format(i)]
            required_tensor = model["convs.{}.conv.modulation.bias".format(i)]
            if original_tensor.shape != required_tensor.shape:
                print("Modifying weight at: convs.{}.conv.modulation.bias".format(i))
                alter_layer = torch.randn_like(required_tensor)
                alter_layer[:original_tensor.shape[0]] = original_tensor
                pretrained_dict["convs.{}.conv.modulation.bias".format(i)] = alter_layer

    return pretrained_dict