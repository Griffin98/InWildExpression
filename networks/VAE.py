"""
TwoStageVAE

Reference: https://github.com/daib13/TwoStageVAE
Paper: https://openreview.net/forum?id=B1e0X3C9tQ

"""

import pytorch_lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


from torchmetrics.functional import accuracy



class TwoStageVAE(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        print("Loading model")

        self.num_layers = hparams.num_layers
        self.latent_dim = hparams.latent_dim
        self.input_dim = 50
        self.layer_dim = hparams.layer_dim
        self.activation = hparams.activation

        self.lr = hparams.lr
        self.kld_weight = hparams.kld_weight

        self.encoder = Encoder(self.input_dim, self.layer_dim, self.latent_dim, self.num_layers, self.activation)
        self.decoder = Decoder(self.input_dim, self.layer_dim, self.latent_dim, self.num_layers, self.activation)

        # self.train_acc = torchmetrics.Accuracy()

    def training_step(self, data, batch_idx):

        mu, log_var = self.encoder(data)
        z = self.reparameterize(mu, log_var)

        y = self.decoder(z)

        loss, loss_dict = self.calculate_train_loss(data, y, mu, log_var)
        self.log_metrics(loss_dict, "train")

        return loss

    def validation_step(self, data, batch_idx):

        mu, log_var = self.encoder(data)
        z = self.reparameterize(mu, log_var)

        y = self.decoder(z)

        loss, loss_dict = self.calculate_val_loss(data, y)
        self.log_metrics(loss_dict, "val")

        return loss

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps * std + mu

    def calculate_train_loss(self, input, y, mu, log_var):
        loss = 0
        loss_dict = {}

        l1_loss = F.l1_loss(input, y)
        loss_dict["l1_loss"] = l1_loss

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss_dict["kld_loss"] = kld_loss

        loss = l1_loss + self.kld_weight * kld_loss
        loss_dict["loss"] = loss

        return loss, loss_dict

    def calculate_val_loss(self, input, y):
        loss_dict = {}

        loss = F.l1_loss(input, y)
        loss_dict["loss"] = loss

        return loss, loss_dict

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():

            if key == "loss":
                self.log(f'{prefix}/{key}', value, on_step=True, prog_bar=True)
            else:
                self.log(f'{prefix}/{key}', value, on_step=True, prog_bar=True)


class Encoder(nn.Module):
    def __init__(self, input_dim, layer_dim, latent_dim, num_layers, activation):
        super(Encoder, self).__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers - 1):

            if i == 0:
                layer = nn.Linear(input_dim, layer_dim)
            else:
                layer = nn.Linear(layer_dim, layer_dim)

            self.layers.append(layer)

        self.mu = nn.Linear(layer_dim, latent_dim)
        self.log_var = nn.Linear(layer_dim, latent_dim)

        if activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()


    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activation(x)

        mu = self.mu(x)
        log_var = self.log_var(x)

        return mu, log_var


class Decoder(nn.Module):
    def __init__(self, input_dim, layer_dim, latent_dim, num_layers, activation):
        super().__init__()

        self.layers = nn.ModuleList()

        for i in range(num_layers - 1):

            if i == 0:
                layer = nn.Linear(latent_dim, layer_dim)
            else:
                layer = nn.Linear(layer_dim, layer_dim)

            self.layers.append(layer)

        self.final = nn.Linear(layer_dim, input_dim)

        if activation == "lrelu":
            self.activation = nn.LeakyReLU(0.2)
        else:
            self.activation = nn.ReLU()

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.activation(x)

        x = self.final(x)

        return x


if __name__ == "__main__":
    import wandb
    # Sweep parameters
    hyperparameter_defaults = dict(
        batch_size=32,
        lr=1e-3,
        num_layers=3,
        layer_dim=256,
        latent_dim=15,
        kld_weight=0.025,
        activation="lrelu",
        epochs=30
    )

    wandb.init(config=hyperparameter_defaults)
    # Config parameters are automatically set by W&B sweep agent
    config = wandb.config

    model = TwoStageVAE(hparams=config)
