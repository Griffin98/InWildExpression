import os

import __init_paths
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from dataset.FFHQDataset import FFHQDataModule
from options.train_options import TrainOptions
from model.ExpressionModel import ExpressionModule


def main():
    opts = TrainOptions().parse()

    model = ExpressionModule(opts)
    dm = FFHQDataModule(opts)

    run_name = "Run#1_E3_{}".format(opts.training_stage)

    checkpoint_path = os.path.join(os.path.expanduser("~"), "checkpoints", run_name)
    os.makedirs(checkpoint_path, exist_ok=True)

    wandb_logger = WandbLogger(project="Expression", name=run_name)

    trainer = pl.Trainer(max_epochs=opts.max_epoch, gpus=-1, strategy="ddp", logger=wandb_logger,
                         enable_checkpointing=True, weights_save_path=checkpoint_path,
                         resume_from_checkpoint="/home/p_dhyeyd/checkpoints/Run#1_E2_imitate/Expression/2cvtf46n/checkpoints/epoch=19-step=35999.ckpt")

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
