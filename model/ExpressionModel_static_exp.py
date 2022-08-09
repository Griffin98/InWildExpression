import os
import math
import random

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from cv2 import imread

from skimage.io import imread
from torchvision.transforms import Compose, Resize
from torchvision.utils import make_grid
# from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from criteria.deca.exp_loss import ExpLoss
from criteria.deca.exp_warp_loss import ExpWarpLoss
# from criteria.deca.model import DECAModel
from criteria.deca.simple import DECAModel
from criteria.deca.models.encoders import ResnetEncoder
from criteria.deca.models.FLAME import FLAME, FLAMETex
from criteria.deca.utils.config import cfg as deca_config
from criteria.deca.utils.renderer import SRenderY, set_rasterizer
from criteria.deca.utils.detectors import FAN
from criteria.deca.utils.process_data import ProcessData
from criteria.deca.utils.util import copy_state_dict

from criteria.id.id_loss import IDLoss
from criteria.id.model_irse import Backbone
from criteria.stylegan import g_path_regularize, d_r1_loss

from criteria.lpips_loss.loss import LPIPSLoss

from networks.encoder2 import StylizedExpressionEncoder
from networks.stylegan2_inject_concat import Generator, Discriminator
from networks.VAE import Decoder


from utils.load_stylegan_weights import load_indexed_stylegan_dict
from utils.load_vae_weights import load_conditioned_vae_weights


class ExpressionModule(pl.LightningModule):
    def __init__(self, opts):
        super(ExpressionModule, self).__init__()
        self.opts = opts

        print("Creating Expression Model")
        self.vae_decoder = Decoder(self.opts.vae_input_dim, self.opts.vae_layer_dim, self.opts.vae_latent_dim,
                                   self.opts.vae_num_layers, "lrelu")
        pretrained_dict = torch.load(self.opts.vae_weights)
        pretrained_dict = pretrained_dict["state_dict"]
        self.vae_decoder.load_state_dict(load_conditioned_vae_weights(self.vae_decoder, pretrained_dict, "decoder"))
        self.vae_decoder.requires_grad_(False)
        self.vae_decoder.eval()

        print("Creating Expression Model")
        self.net = ExpressionModel(self.opts)
        # self.net_ema = ExpressionModel(self.opts)

        print("Creating Discriminator")
        self.disc = Discriminator(size=self.opts.output_size)

        print("Creating ID loss")
        retinaFace = Backbone(input_size=self.opts.rf_size, num_layers=self.opts.rf_num_layers, drop_ratio=self.opts.rf_drop_ratio,
                              mode=self.opts.rf_model_name).requires_grad_(False).eval()
        self.id_loss = IDLoss(retinaFace)
        self.l2_loss = nn.MSELoss()

        self.setup_deca()

        print("Creating LPIPS loss")
        self.lpips = LPIPSLoss(net_type="alex")

        detector = FAN().requires_grad_(False)
        self.pd = ProcessData(detector=detector)

        self.mean_path_length = 0
        self.save_hyperparameters()

        self.transform = Compose([
            Resize(self.opts.output_size)
        ])

        random.seed(73)

    def setup_deca(self):
        print("Creating Deca Model")
        model_cfg = deca_config.model

        n_param = model_cfg.n_shape + model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light
        n_detail = model_cfg.n_detail

        # encoders
        E_flame = ResnetEncoder(outsize=n_param)
        E_detail = ResnetEncoder(outsize=n_detail)
        # decoders
        flame = FLAME(model_cfg)
        flametex = FLAMETex(model_cfg)

        pretrained_weights = torch.load(self.opts.deca_weights)
        copy_state_dict(E_flame.state_dict(), pretrained_weights['E_flame'])
        copy_state_dict(E_detail.state_dict(), pretrained_weights['E_detail'])

        E_flame.requires_grad_(False).eval()
        E_detail.requires_grad_(False).eval()

        set_rasterizer(deca_config.rasterizer_type)
        render = SRenderY(deca_config.dataset.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size,
                               rasterizer_type=deca_config.rasterizer_type)
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size])

        render_utils = {}
        render_utils["renderer"] = render
        render_utils["uv_face_mask"] = uv_face_eye_mask

        self.deca_model = DECAModel(E_flame=E_flame, E_detail=E_detail, flame=flame,
                                    flametex=flametex, render_utils=render_utils).requires_grad_(False).eval()

        print("Creating Expression and Expression Warp Loss")
        self.exp_loss = ExpLoss(model=self.deca_model, type="cosine")
        self.exp_warp_loss = ExpWarpLoss(model=self.deca_model)



    def g_non_exp_saturating_loss(self, fake_pred, fake_img, real_img, target_expression):
        dict = {}
        loss = 0.0

        if self.opts.lambda_disc > 0:
            gan_loss = F.softplus(-fake_pred).mean()
            dict['gan_loss'] = gan_loss
            loss += self.opts.lambda_disc * gan_loss

        if self.opts.lambda_id > 0:
            loss_id = self.id_loss(fake_img, real_img)
            dict["id_loss"] = loss_id
            loss += self.opts.lambda_id * loss_id

        if self.opts.lambda_l2 > 0:
            loss_l2 = self.l2_loss(fake_img, real_img)
            dict['l2_loss'] = loss_l2
            loss += self.opts.lambda_l2 * loss_l2

        if self.opts.lambda_lpips > 0:
            loss_lpips = self.lpips(fake_img, real_img)
            dict["lpips_loss"] = loss_lpips
            loss += self.opts.lambda_lpips * loss_lpips

        if self.opts.lambda_exp > 0 or self.opts.lambda_exp_warp >0:
            data = self.pd.run(fake_img)

            if self.opts.lambda_exp > 0:
                loss_exp = self.exp_loss(data["images"], target_expression)
                dict['exp_loss'] = loss_exp
                loss += self.opts.lambda_exp * loss_exp

            if self.opts.lambda_exp_warp > 0:
                loss_exp_warp = self.exp_warp_loss(data["images"], target_expression, data["tforms"], data["original_images"])
                dict["exp_warp_loss"] = loss_exp_warp
                loss += self.opts.lambda_exp_warp * loss_exp_warp

        dict["loss"] = loss

        return loss, dict

    def disc_loss(self, real_pred, fake_pred):
        real_loss = F.softplus(-real_pred)
        fake_loss = F.softplus(fake_pred)

        loss = real_loss.mean() + fake_loss.mean()

        dict = {}
        dict["loss"] = loss

        return loss, dict

    def predict(self, batch, expression):
        self.net.requires_grad_(False)
        self.disc.requires_grad_(False)

        fake_pred, _ = self.net(batch, expression)

        return fake_pred

    def train_generator(self, batch, expression):
        # accumulate(self.net_ema, self.net, 0.997784)

        self.net.requires_grad_(True)
        self.disc.requires_grad_(False)

        fake_image, _ = self.net(batch, expression)
        fake_pred = self.disc(fake_image)

        loss, loss_dict = self.g_non_exp_saturating_loss(fake_pred=fake_pred, fake_img=fake_image, real_img=batch,
                                                         target_expression=expression)

        if self.global_step % self.opts.g_reg_every == 0 and self.opts.lambda_path_loss > 0:
            fake_img, latents = self.net(batch, expression)

            path_loss, self.mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, self.mean_path_length
            )
            weighted_path_loss = self.opts.lambda_path_loss * self.opts.g_reg_every * path_loss

            loss += weighted_path_loss

            loss_dict["path_loss"] = weighted_path_loss

        return loss, loss_dict

    def train_discriminator(self, batch, expression):
        self.net.requires_grad_(False)
        self.disc.requires_grad_(True)

        real_pred = self.disc(batch)

        fake_img, _ = self.net(batch, expression)
        fake_pred = self.disc(fake_img)

        loss, loss_dict = self.disc_loss(real_pred=real_pred, fake_pred=fake_pred)

        if self.global_step % self.opts.d_reg_every == 0:
            batch.requires_grad = True

            real_pred = self.disc(batch)
            r1_loss = d_r1_loss(real_pred=real_pred, real_img=batch)

            loss_dict["r1_loss"] = r1_loss

            loss += r1_loss * 5 * self.opts.d_reg_every

        return loss, loss_dict

    def training_step(self, batch, batch_idx, optimizer_idx):

        if self.opts.training_stage == "imitate":
            data = self.pd.run(batch)
            expression = self.exp_loss.get_expression(data["images"])
        else:
            expression = torch.clamp(torch.randn(batch.shape[0], self.opts.vae_latent_dim), min=-3, max=3)
            expression = expression.type_as(batch)
            expression = self.vae_decoder(expression)

        if optimizer_idx == 0:
            loss, loss_dict = self.train_generator(batch, expression)

            self.log_metrics(loss_dict, "generator")

        else:
            loss, loss_dict = self.train_discriminator(batch, expression)

            self.log_metrics(loss_dict, "discriminator")

        if self.global_step % self.opts.image_interval == 0:
            self.log_image(batch, expression)

        return loss


    def configure_optimizers(self):
        g_reg_ratio = self.opts.g_reg_every / (self.opts.g_reg_every + 1)
        d_reg_ratio = self.opts.d_reg_every / (self.opts.d_reg_every + 1)

        g_optim = optim.Adam(params=self.net.parameters(), lr=self.opts.learning_rate * g_reg_ratio,
                             betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio))

        d_optim = optim.AdamW(params=self.disc.parameters(), lr=self.opts.learning_rate * d_reg_ratio,
                              betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio))

        return [g_optim, d_optim]

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():

            if key == "loss":
                self.log(f'{prefix}/{key}', value, on_step=True, prog_bar=True)
            else:
                self.log(f'{prefix}/{key}', value, on_step=True)

    def log_image(self, batch, expression):

        pred = self.predict(batch, expression)
        data = self.pd.run(batch)

        if self.opts.training_stage == "contrastive" or self.opts.training_stage == "combined":
            n_row = 3
            _, _, original_render, trarget_exp_image = self.exp_warp_loss.get_warped_images(data["images"], expression, data["tforms"],
                                                                    data["original_images"])
        else:
            n_row = 2

        grid = []
        for i in range(batch.shape[0]):
            input_image = batch[i]
            grid.append(self.pd.inv_transform(input_image))

            if self.opts.training_stage == "contrastive" or self.opts.training_stage == "combined":
                target_exp_image = trarget_exp_image[i].unsqueeze(0)
                target_exp_image = self.transform(target_exp_image).squeeze(0)
                grid.append(target_exp_image)

            output_image = pred[i]
            grid.append(self.pd.inv_transform(output_image))


        grid = torch.stack(grid, dim=0)
        grid = make_grid(grid, nrow=n_row)
        self.logger.experiment.log({"image": [wandb.Image(grid)]}, step=self.global_step)


class ExpressionModel(nn.Module):
    def __init__(self, opts):
        super(ExpressionModel, self).__init__()

        self.opts = opts

        concat_indices = [32]

        self.encoder = StylizedExpressionEncoder(size=self.opts.output_size, style_dim=self.opts.style_dim,
                                                 n_mlp=self.opts.n_mlp,
                                                 expression_dim=self.opts.expression_dim,
                                                 concat_indices=concat_indices)
        self.decoder = Generator(size=self.opts.output_size, style_dim=self.opts.style_dim, n_mlp=self.opts.n_mlp,
                                 concat_index=concat_indices)

        if opts.load_stylegan_weights and (opts.training_stage == "imitate" or opts.training_stage == "combined"):
            print("Loading stylegan weights")
            pretrained_dict = torch.load(self.opts.stylegan_weights)["g_ema"]
            modified_dict = load_indexed_stylegan_dict(self.decoder, pretrained_dict, concat_indices)
            self.decoder.load_state_dict(modified_dict)

    def forward(self, image, expression, return_latents=True):
        embedding, noise, styles = self.encoder(image, expression)

        y, latent = self.decoder([styles], noise=noise[1:], input_is_latent=True, return_latents=return_latents)

        return y, latent


if __name__ == "__main__":
    from options.train_options import TrainOptions
    opts = TrainOptions().parse()
    expModel = ExpressionModel(opts).to("cuda")

    from torchvision.utils import save_image

    # print(out.shape)
    exp = torch.randn(1, 50).to("cuda")
    for i in range(10):
        img = torch.randn(1, 3, 512, 512).to("cuda")

        out, _ = expModel(img, exp)
        save_image(out, "grid_{}.png".format(i))