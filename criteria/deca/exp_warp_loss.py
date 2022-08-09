import cv2

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.transforms import Compose, GaussianBlur

class ExpWarpLoss(nn.Module):
    def __init__(self, model):
        super(ExpWarpLoss, self).__init__()

        self.model = model
        self.model.eval()

        self.transform = Compose([
            GaussianBlur(3, 1)
        ])

    def get_target_expression_image(self, fake_images, expressions, tforms, original_images):

        batch_size = fake_images.shape[0]
        rendered = []

        for i in range(batch_size):
            fake_image = fake_images[i][None, ...]
            expression = expressions[i][None, ...]
            tform = tforms[i][None, ...]
            original_image = original_images[i][None, ...]

            with torch.no_grad():
                id_codedict = self.model.encode(fake_image)
            id_opdict, id_visdict = self.model.decode(id_codedict, tform=tform)


            id_codedict["exp"] = expression
            tform = torch.inverse(tform).transpose(1, 2)
            transfer_opdict, transfer_visdict = self.model.decode(id_codedict, tform=tform,
                                                original_image=original_image, render_orig=True)

            id_visdict['transferred_shape'] = transfer_visdict['shape_detail_images']

            transfer_opdict['uv_texture_gt'] = id_opdict['uv_texture_gt']

            render_transferred = self.model.render(transfer_opdict['verts'], transfer_opdict['trans_verts'],
                                                   id_opdict['uv_texture_gt'], id_codedict['light'])

            trasferred_rendered = render_transferred["images"]

            rendered.append(trasferred_rendered.squeeze(0))

        rendered = torch.stack(rendered, dim=0)

        return rendered

    @staticmethod
    def fix_mask(masks):
        fixed_masks = []

        for i in range(masks.shape[0]):
            mask = masks[i].cpu().detach().permute(1, 2, 0).numpy() * 255
            mask[mask >= 200] = 255
            mask[mask < 200] = 0
            mask = mask.astype(np.uint8)

            cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            for c in cnts:
                cv2.drawContours(mask, [c], 0, (255, 255, 255), -1)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

            mask = torch.Tensor(mask).unsqueeze(2).permute(2, 0, 1) / 255.

            fixed_masks.append(mask)

        fixed_masks = torch.stack(fixed_masks, dim=0)
        fixed_masks = fixed_masks.type_as(masks)

        return fixed_masks

    def get_warped_images(self, fake_images, expressions, tforms, original_images):
        device = fake_images.device

        with torch.no_grad():
            id_codedict = self.model.encode(fake_images)
        id_opdict = self.model.decode(id_codedict, tform=tforms, original_image=original_images)

        original_rendered = id_opdict["rendered_images"]
        original_mask = id_opdict["mask"]

        id_codedict["exp"] = expressions
        tforms = torch.inverse(tforms).transpose(1, 2)
        transfer_opdict = self.model.decode(id_codedict, tform=tforms, original_image=original_images)

        render_transferred = self.model.render(transfer_opdict['verts'], transfer_opdict['trans_verts'],
                                               id_opdict['albedo'], id_codedict['light'])

        trasferred_rendered = render_transferred["images"]
        transferred_mask = transfer_opdict["mask"]

        img_diff = original_rendered - trasferred_rendered
        flow = img_diff[:, 0:2, :, :]
        flow_grid = np.linspace(-1, 1, flow.size(2))
        flow_grid = np.meshgrid(flow_grid, flow_grid)
        flow_grid = np.stack(flow_grid, 2)
        flow_grid = torch.Tensor(flow_grid).unsqueeze(0).repeat(fake_images.size(0), 1, 1, 1)
        flow_grid = torch.autograd.Variable(flow_grid, requires_grad=True)
        flow_sampler = (flow.permute(0, 2, 3, 1).to(device) + flow_grid.to(device)).clamp(min=-1, max=1)
        warped_images = F.grid_sample(fake_images, flow_sampler)

        masks = original_mask + transferred_mask

        masks = self.fix_mask(masks)

        return warped_images, masks, original_rendered, trasferred_rendered

    def forward(self, fake_images, expressions, tforms, original_images):

        warped_images, masks, _, _ = self.get_warped_images(fake_images=fake_images, expressions=expressions, tforms=tforms,
                                                      original_images=original_images)

        warped_images = self.transform(warped_images) * masks
        # warped_images = warped_images * masks
        fake_images = self.transform(fake_images) * masks
        # fake_images = fake_images * masks

        loss = F.mse_loss(fake_images, warped_images) * 100

        return loss