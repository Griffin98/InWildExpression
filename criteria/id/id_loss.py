import torch
from torch import nn
import numpy as np

class IDLoss(nn.Module):
    def __init__(self, net):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = net
        self.facenet.eval()
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))

    def extract_feats(self, x):
        _, _, h, w = x.shape
        assert h==w
        ss = h//256
        x = x[:, :, 35*ss:-33*ss, 32*ss:-36*ss]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y, y_hat):
        n_samples = y.size(0)
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0.0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += (1 - diff_target)
        return loss / n_samples

    def cal_identity_similarity(self, y, y_hat):
        n_samples = y.size(0)
        y_feats = self.extract_feats(y)
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        similarities = []
        for i in range(n_samples):
            similarities.append(y_hat_feats[i].dot(y_feats[i]).cpu().item())
        return np.mean(similarities)

