import numpy as np
import os, sys
import cv2
import torch

from criteria.deca.model import DECAModel
from criteria.deca.utils.config import cfg as deca_cfg
from criteria.deca.exp_warp_loss import ExpWarpLoss
from criteria.deca.utils.detectors import FAN
from criteria.deca.utils.process_data import ProcessData
from torchvision.utils import save_image, make_grid, draw_segmentation_masks
from torchvision.transforms import Compose, Normalize

from dataset.FFHQDataset import _FFHQDataset
from torch.utils.data import DataLoader

def main():

    # load test images
    detector = FAN()
    pd = ProcessData(detector=detector)


    # # run DECA
    deca_cfg.model.use_tex = True
    #
    deca_cfg.rasterizer_type = "standard"
    deca = DECAModel(config = deca_cfg, device="cuda")
    expWarp = ExpWarpLoss(model=deca)

    dataset = _FFHQDataset(data_dir="mini_ffhq", image_size=512)
    loader = DataLoader(dataset, shuffle=False, num_workers=2, batch_size=4)
    torch.manual_seed(0)
    for (id, data) in enumerate(loader):
        data = data.to("cuda")
        dict = pd.run(data)
        # save_image(make_grid(dict["images"] / 255.), "images_{}.png".format(id))
        exp = torch.clamp(torch.randn(4, 50), min=-2, max=2)
        exp = exp.type_as(data)

        fake_image = dict["images"]
        tforms = dict["tforms"]
        original_images = dict["original_images"]

        rendered = expWarp.get_target_expression_image(fake_image, exp, tforms, original_images)
        save_image(make_grid(rendered), "target_exp_image_{}.png".format(id))


if __name__ == '__main__':
    main()
