import cv2
import face_alignment
import numpy as np
import torch
from skimage import io
from skimage.transform import warp, estimate_transform
import time


def bbox2point( left, right, top, bottom, type='bbox'):
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

st = time.time()

model2 = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cuda:0")

img = io.imread("mini_ffhq/biden_000001.png")

image = torch.Tensor(img).to("cpu")


out = model2.get_landmarks_from_image(image)

kpt = out[0].squeeze()
left = np.min(kpt[:,0]); right = np.max(kpt[:,0]);
top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])
bbox = [left,top, right, bottom]

if len(bbox) < 4:
    print('no face detected! run original image')
    left = 0
    right = h - 1
    top = 0
    bottom = w - 1
else:
    left = bbox[0]
    right = bbox[2]
    top = bbox[1]
    bottom = bbox[3]
old_size, center = bbox2point(left, right, top, bottom, type="kpt68")
size = int(old_size * 1.25)
src_pts = np.array(
    [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
     [center[0] + size / 2, center[1] - size / 2]])

DST_PTS = np.array([[0, 0], [0, 224 - 1], [224 - 1, 0]])
tform = estimate_transform('similarity', src_pts, DST_PTS)

image = image.cpu().detach().numpy()

dst_image = warp(img, tform.inverse, output_shape=(244, 224))

cv2.imshow("Image", dst_image)
cv2.waitKey(0)