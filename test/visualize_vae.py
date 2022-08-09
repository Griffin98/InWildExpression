import torch
import matplotlib.pyplot as plt; plt.rcParams['figure.dpi'] = 200

from torch.utils.data import DataLoader

from networks.VAE import Encoder, Decoder
from dataset.VAEDataset import _ExpressionDataset
from dataset.FFHQDataset import _FFHQDataset
from criteria.deca.model import DECAModel
from criteria.deca.utils.config import cfg as deca_cfg
from criteria.deca.exp_warp_loss import ExpWarpLoss
from criteria.deca.exp_loss import ExpLoss
from criteria.deca.utils.detectors import FAN
from criteria.deca.utils.process_data import ProcessData
from torchvision.utils import save_image, make_grid


device = 'cpu'

# Sweep parameters
hyperparameter_defaults = dict(
    batch_size=32,
    lr=2e-4,
    num_layers=3,
    layer_dim=1024,
    latent_dim=18,
    kld_weight=0.00025,
    activation="lrelu",
    epochs=75
)


def get_state_dict(model, pretrained_dict, type="encoder"):
    state = model.state_dict()

    if type == "decoder":
        state["layers.0.weight"] = pretrained_dict["decoder.layers.0.weight"]
        state["layers.0.bias"] = pretrained_dict["decoder.layers.0.bias"]
        state["layers.1.weight"] = pretrained_dict["decoder.layers.1.weight"]
        state["layers.1.bias"] = pretrained_dict["decoder.layers.1.bias"]
        state["final.weight"] = pretrained_dict["decoder.final.weight"]
        state["final.bias"] = pretrained_dict["decoder.final.bias"]

    if type == "encoder":
        state["layers.0.weight"] = pretrained_dict["encoder.layers.0.weight"]
        state["layers.0.bias"] = pretrained_dict["encoder.layers.0.bias"]
        state["layers.1.weight"] = pretrained_dict["encoder.layers.1.weight"]
        state["layers.1.bias"] = pretrained_dict["encoder.layers.1.bias"]
        state["mu.weight"] = pretrained_dict["encoder.mu.weight"]
        state["mu.bias"] = pretrained_dict["encoder.mu.bias"]
        state["log_var.weight"] = pretrained_dict["encoder.log_var.weight"]
        state["log_var.bias"] = pretrained_dict["encoder.log_var.bias"]

    return state


def main():
    pretrained_dict = torch.load("weights/vae.ckpt")
    pretrained_dict = pretrained_dict["state_dict"]


    encoder = Encoder(50, 1024, 18, 3, 'lrelu')
    encoder.load_state_dict(get_state_dict(encoder, pretrained_dict, "encoder"))
    encoder.to("cuda")
    decoder = Decoder(50, 1024, 18, 3, "lrelu")
    decoder.load_state_dict(get_state_dict(decoder, pretrained_dict, "decoder"))
    decoder.to("cuda")

    detector = FAN()
    pd = ProcessData(detector=detector)
    deca_cfg.model.use_tex = True
    deca_cfg.rasterizer_type = "standard"
    deca = DECAModel(config=deca_cfg, device="cuda")
    expWarp = ExpWarpLoss(model=deca)
    expLoss = ExpLoss(model=deca, type="cosine")
    dataset = _FFHQDataset(data_dir="mini_ffhq/input", image_size=512)
    loader = DataLoader(dataset, shuffle=False, num_workers=2, batch_size=4)

    batch = next(iter(loader))
    batch = batch.to("cuda")
    dict = pd.run(batch)

    torch.manual_seed(401)
    losses = []

    count = 0

    for i in range(10000):
        exp = torch.clamp(torch.randn(4, 18), min=-3, max=3)
        # exp = torch.clamp(torch.randn(4, 50), min=-2, max=2)
        exp = exp.type_as(batch)


        exp = decoder(exp)

        loss = expLoss(dict["images"], exp)
        if loss > 1:
            count += 1
        # print("Loss: ", loss)
        losses.append(loss)

        # fake_image = dict["images"]
        # tforms = dict["tforms"]
        # original_images = dict["original_images"]
        #
        # rendered = expWarp.get_target_expression_image(fake_image, exp, tforms, original_images)
        # save_image(make_grid(rendered), "mini_ffhq/output_vae_exp/tgt_exp_image_{}.png".format(i))

    print("Max Loss value: ", max(losses))
    print("Min Loss value: ", min(losses))
    print("Mean Loss value: ", )
    print("Count: ", count)



if __name__ == "__main__":
    main()