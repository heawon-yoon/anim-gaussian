#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import torch.nn as nn
from utils.loss_utils import l1_loss, ssim


class HumanSceneLoss(nn.Module):
    def __init__(
        self,
        l_ssim_w=0.2,
        l_l1_w=0.8,

    ):
        super(HumanSceneLoss, self).__init__()
        
        self.l_ssim_w = l_ssim_w
        self.l_l1_w = l_l1_w


    def forward(
        self, 
        viewpoint_cam,
        render_pkg,
        smpl_output,
        bg,
        opt,
    ):
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
            "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        mask = viewpoint_cam.mask.unsqueeze(0).cuda()
        gt_image = gt_image * mask + bg[:, None, None] * (1. - mask)
        Ll1 = l1_loss(image, gt_image, mask)
        loss_ssim = 1.0 - ssim(image, gt_image)
        loss_ssim = loss_ssim * (mask.sum() / (image.shape[-1] * image.shape[-2]))
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (loss_ssim)
        
        return loss
    