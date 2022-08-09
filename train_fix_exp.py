import os

import __init_paths
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from dataset.FFHQDataset import FFHQDataModule
from options.train_options import TrainOptions
from model.ExpressionModel_static_exp import ExpressionModule


def main():
    opts = TrainOptions().parse()

    model = ExpressionModule(opts)
    dm = FFHQDataModule(opts)

    run_name = "Run#3_E1_{}".format(opts.training_stage)

    checkpoint_path = os.path.join(os.path.expanduser("~"), "checkpoints", run_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    wandb_logger = WandbLogger(project="Expression", name=run_name)

    trainer = pl.Trainer(max_epochs=opts.max_epoch, gpus=-1, strategy="ddp", logger=wandb_logger,
                         enable_checkpointing=True, weights_save_path=checkpoint_path,
                         limit_train_batches=0.20)
    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
