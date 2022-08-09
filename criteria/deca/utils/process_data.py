import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.transform import estimate_transform, warp
from torchvision.transforms import Compose, ToTensor, Normalize

from criteria.deca.utils.detectors import FAN


class ProcessData():
        def __init__(self, detector):
            self.detector = detector
            self.scale = 1.25
            self.resolution_inp = 224

            self.transform = Compose([
                ToTensor()
            ])

            self.inv_transform = Compose([
                Normalize((0., 0., 0.), (2, 2, 2)),
                Normalize((-0.5, -0.5, -0.5), (1., 1., 1.))
            ])

        def bbox2point(self, left, right, top, bottom, type='bbox'):
            ''' bbox from detector and landmarks are different
            '''
            if type == 'kpt68':
                old_size = (right - left + bottom - top) / 2 * 1.1
                center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
            elif type == 'bbox':
                old_size = (right - left + bottom - top) / 2
                center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
            else:
                raise NotImplementedError
            return old_size, center

        def run(self, image_batch):
            """
            :param self:
            :param image_batch: tensor
            """

            image_batch = self.inv_transform(image_batch)

            size = image_batch.shape[0]

            images = []
            original_images = []
            tforms = []

            for i in range(size):
                image = image_batch[i]

                # # Convert to numpy
                image = image.permute(1, 2, 0).detach().to("cuda:0") * 255.
                #
                # if len(image.shape) == 2:
                #     image = image[:, :, None].repeat(1, 1, 3)
                # if len(image.shape) == 3 and image.shape[2] > 3:
                #     image = image[:, :, :3]

                # image = image * 255.

                _, h, w = image.shape

                bbox, bbox_type = self.detector(image)

                if len(bbox) < 4:
                    # print('no face detected! run original image')
                    left = 0
                    right = h - 1
                    top = 0
                    bottom = w - 1
                else:
                    left = bbox[0]
                    right = bbox[2]
                    top = bbox[1]
                    bottom = bbox[3]
                old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
                size = int(old_size * self.scale)
                src_pts = np.array(
                    [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                     [center[0] + size / 2, center[1] - size / 2]])

                DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
                tform = estimate_transform('similarity', src_pts, DST_PTS)

                image = image.cpu().detach().numpy()

                dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))

                dst_image = self.transform(dst_image) / 255.
                dst_image = dst_image.type_as(image_batch)
                image = self.transform(image) / 255.
                image = image.type_as(image_batch)
                tform = torch.tensor(tform.params).float()
                tform = tform.type_as(image_batch)

                images.append(dst_image)
                original_images.append(image)
                tforms.append(tform)

            return {
                'images': torch.stack(images),
                'original_images': torch.stack(original_images),
                'tforms': torch.stack(tforms)
            }




