# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import torch
import torch.nn.functional as F
import torch.nn as nn

import numpy as np
from skimage.io import imread
from criteria.deca.utils.renderer import SRenderY, set_rasterizer
from criteria.deca.utils import util
from criteria.deca.utils.config import cfg as deca_cfg

class DECAModel(nn.Module):
    def __init__(self, E_flame, E_detail, flame, flametex, render_utils):
        super(DECAModel, self).__init__()

        self.cfg = deca_cfg

        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size

        model_cfg = self.cfg.model

        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}

        self.E_flame = E_flame
        self.E_detail = E_detail
        self.flame = flame
        self.flametex = flametex

        self.E_flame.eval()
        self.E_detail.eval()
        self.flame.eval()
        self.flametex.eval()

        set_rasterizer(self.cfg.rasterizer_type)
        self.render = render_utils["renderer"]
        self.uv_face_eye_mask = render_utils["uv_face_mask"]


    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0
        for key in num_dict:
            end = start+int(num_dict[key])
            code_dict[key] = code[:, start:end]
            start = end
            if key == 'light':
                code_dict[key] = code_dict[key].reshape(code_dict[key].shape[0], 9, 3)
        return code_dict

    # @torch.no_grad()
    def encode(self, images, use_detail=True):

        if use_detail:
            # use_detail is for training detail model, need to set coarse model as eval mode
            with torch.no_grad():
                parameters = self.E_flame(images)
        else:
            parameters = self.E_flame(images)
        codedict = self.decompose_code(parameters, self.param_dict)
        codedict['images'] = images
        if use_detail:
            detailcode = self.E_detail(images)
            codedict['detail'] = detailcode
        return codedict

    def decode(self, codedict, rendering=True, original_image=None, tform=None):
        images = codedict['images']
        device = images.device
        batch_size = images.shape[0]

        ## decode
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'], pose_params=codedict['pose'])
        if self.cfg.model.use_tex:
            albedo = self.flametex(codedict['tex'])
        else:
            albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device)
        landmarks3d_world = landmarks3d.clone()

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:,:,:2]; landmarks2d[:,:,1:] = -landmarks2d[:,:,1:]#; landmarks2d = landmarks2d*self.image_size/2 + self.image_size/2
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']); landmarks3d[:,:,1:] = -landmarks3d[:,:,1:] #; landmarks3d = landmarks3d*self.image_size/2 + self.image_size/2
        trans_verts = util.batch_orth_proj(verts, codedict['cam']); trans_verts[:,:,1:] = -trans_verts[:,:,1:]
        opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'landmarks3d_world': landmarks3d_world,
        }

        ## rendering
        if rendering:
            ops = self.render(verts, trans_verts, albedo, codedict['light'])

            ## output
            opdict['grid'] = ops['grid']
            opdict['rendered_images'] = ops['images']
            opdict['alpha_images'] = ops['alpha_images']
            opdict['normal_images'] = ops['normal_images']

        if self.cfg.model.use_tex:
            opdict['albedo'] = albedo

        mask_face_eye = F.grid_sample(self.uv_face_eye_mask.expand(batch_size, -1, -1, -1).to(device),
                                      opdict['grid'].detach(), align_corners=False)

        opdict["mask_face_eye"] = mask_face_eye
        opdict["mask"] = mask_face_eye * opdict["alpha_images"]

        return opdict