import sys

import cv2
import torch
import torchvision

from torchvision.utils import save_image, make_grid
from criteria.deca.utils.config import cfg as deca_cfg
from criteria.deca.model import DECAModel
from criteria.deca.utils.process_data import ProcessData
from criteria.deca.utils.detectors import FAN

from criteria.deca.exp_loss import ExpLoss

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
    exp_loss = ExpLoss(model=deca)

    dataset = _FFHQDataset(data_dir="mini_ffhq", image_size=512)
    loader = DataLoader(dataset, shuffle=False, num_workers=2, batch_size=1)
    torch.manual_seed(0)
    for i in range(1):
        for (id, data) in enumerate(loader):
            data = data.to("cuda")
            dict = pd.run(data)

            # exp_loss = ExpLoss(model=deca)
            original_expression = exp_loss.get_expression(dict["images"])
            loss = exp_loss(dict["images"], original_expression)
            print("Constant exp loss: {}", loss)
            #
            original_expression = torch.randn(2,50)
            original_expression = original_expression.type_as(data)
            loss = exp_loss(dict["images"], original_expression)
            print("Randn exp loss: {}", loss)



if __name__ == "__main__":
    main()