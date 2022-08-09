import sys
import os
import torch

from PIL import Image
from torchvision.utils import save_image
from torchvision.transforms import Compose, ToTensor, Resize


def test_inject_concat():
    from model.ExpressionModel_Concat import ExpressionModule
    model = ExpressionModule.load_from_checkpoint("checkpoints/epoch=23-step=74879.ckpt")
    model.eval()

    compose = Compose([
        Resize(512),
        ToTensor()
    ])

    image = Image.open("mini_ffhq/northrup_000001.png")
    image = compose(image).unsqueeze(0)

    for i in range(10):
        exp = torch.randn(1, 50)

        outs = model.predict(image, exp)
        out = outs[-1]

        save_image(out, "output/image_{}.png".format(i))

        # for j in range(len(outs)):
        #     out = outs[j]
        #
        #     save_image(out, "output/image_{}x{}.png".format(out.shape[2], out.shape[3]))

def test_concat():
    from model.ExpressionModel import ExpressionModule
    model = ExpressionModule.load_from_checkpoint("checkpoints/epoch=19-step=61759.ckpt")
    model.eval()

    compose = Compose([
        Resize(512),
        ToTensor()
    ])

    image = Image.open("mini_ffhq/olia_000001.png")
    image = compose(image).unsqueeze(0)

    for i in range(10):
        exp = torch.randn(1, 50)

        outs = model.predict(image, exp)

        for j in range(len(outs)):
            out = outs[j]

            save_image(out, "output/image_{}x{}.png".format(out.shape[2], out.shape[3]))

def test_adain_concat():
    from model.ExpressionModel import ExpressionModule
    model = ExpressionModule.load_from_checkpoint("checkpoints/epoch=19-step=35999.ckpt")
    model.eval()

    compose = Compose([
        Resize(256),
        ToTensor()
    ])

    image = Image.open("mini_ffhq/contrave_000001.png")
    image = compose(image).unsqueeze(0)

    for i in range(10):
        exp = torch.randn(1, 50)

        outs = model.predict(image, exp)

        for j in range(len(outs)):
            out = outs[j]

            save_image(out, "output/image_{}x{}.png".format(out.shape[2], out.shape[3]))

def test_stylegan():
    from networks.stylegan2_concat import Generator

    gan = Generator(isconcat=False, style_dim=512, size=512, n_mlp=8)
    stat_dict = torch.load("weights/ffhq-512-avg-tpurun1.pt")["g_ema"]
    gan.load_state_dict(stat_dict)


    for j in range(10):
        z = torch.randn(4, 512)
        outs, _ = gan([z])
        out = outs[-1]
        save_image(out, "output/image_{}x{}.png".format(j, j))


def test_static_exp():
    from torch.utils.data import DataLoader
    from torchvision.utils import make_grid, save_image

    from model.ExpressionModel_static_exp import ExpressionModule
    from dataset.FFHQDataset import _FFHQDataset
    from criteria.deca.model import DECAModel
    from criteria.deca.utils.config import cfg as deca_cfg
    from criteria.deca.exp_warp_loss import ExpWarpLoss
    from criteria.deca.utils.detectors import FAN
    from criteria.deca.utils.process_data import ProcessData


    # batch = 2
    #
    # detector = FAN()
    # pd = ProcessData(detector=detector)
    #
    # # # run DECA
    # deca_cfg.model.use_tex = True
    # deca_cfg.rasterizer_type = "standard"
    # deca = DECAModel(config=deca_cfg,device="cuda")
    # expWarp = ExpWarpLoss(model=deca)

    model = ExpressionModule.load_from_checkpoint("~/Templates/epoch=29-step=37439.ckpt")
    dict = model.net.state_dict()
    torch.save(dict, "model.pth")

    compose = Compose([
        Resize(512)
    ])

    # dataset = _FFHQDataset(data_dir="mini_ffhq", image_size=512)
    # loader = DataLoader(dataset, shuffle=False, num_workers=2, batch_size=batch)
    #
    # out_dir = os.path.join(os.getcwd(), "output", "mixing")
    # os.makedirs(out_dir, exist_ok=True)
    #
    # for i in range(10):
    #     run_dir = os.path.join(out_dir, "iter_{}".format(i))
    #     os.makedirs(run_dir, exist_ok=True)
    #
    #     exp = torch.randn(batch, 50)
    #
    #     for (id, data) in enumerate(loader):
    #         exp = exp.type_as(data)
    #
    #         outs = model.predict(data, exp)
    #
    #         data = data.to("cuda")
    #         exp = exp.to("cuda")
    #         dict = pd.run(data)
    #         fake_image = dict["images"]
    #         tforms = dict["tforms"]
    #         original_images = dict["original_images"]
    #         required_image = expWarp.get_target_expression_image(fake_image, exp, tforms, original_images)
    #         required_image = compose(required_image)
    #
    #         outs = torch.concat((pd.inv_transform(data.cpu()), required_image.cpu(), pd.inv_transform(outs.cpu())),
    #                             dim=0)
    #         grid = make_grid(outs, nrow=2)
    #
    #         save_image(grid, os.path.join(run_dir, "grid_{}.png".format(id)))

def test():
    from model.ExpressionModel_static_exp import ExpressionModel
    from options.train_options import TrainOptions
    opts = TrainOptions().parse()

    model = ExpressionModel(opts)
    state_dict = torch.load("model.pth")
    for key in state_dict.keys():
        print(key)



if __name__ == "__main__":
    test()