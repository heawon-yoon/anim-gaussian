#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from scene.cameras import Camera
import shutil
import os
import random
import json
from avatar.avatar_dataset_readers  import read_data_info
from scene.gaussian_model import GaussianModel
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
import torch
from modules.smpl_layer import SMPL
from torch import nn
import numpy as np
from utils.general import (
    inverse_sigmoid,
    create_video,
)
from PIL import Image, ImageDraw
from utils.loss_utils import l1_loss, ssim
from avatar.loss import HumanSceneLoss
from avatar.fit_shape import FitShape
from simple_knn._C import distCUDA2
import cv2
from random import randint

import smplx
from utils.rotations import (
    axis_angle_to_rotation_6d,
    matrix_to_quaternion,
    quaternion_multiply,
    quaternion_to_matrix,
    rotation_6d_to_axis_angle,
    torch_rotation_matrix_from_vectors,
)
from avatar.utils import (
    get_rotating_camera,
    get_smpl_canon_params,
    get_smpl_static_params,
    get_static_camera
)
from tqdm import tqdm
from utils.geometry import transformations
import torchvision
from gaussian_renderer import render, render_deformed
from scene.cameras import MiniCam
from pytorch3d.ops.knn import knn_points
from lpips import LPIPS
from utils.general import save_images
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.sampler import PatchSampler
import trimesh
SCALE_Z = 1e-5

SMPL_PATH = 'data/smpl'

AMASS_SMPLH_TO_SMPL_JOINTS = np.arange(0,156).reshape((-1,3))[[
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
    11, 12, 13, 14, 15, 16, 17, 18,
    19, 20, 21, 22, 37
]].reshape(-1)

class Avatar:

    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        """
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = load_iteration
        self.gaussians = gaussians
        self.device = 'cuda'
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(args.source_path) and args.test == False:
            scene_info = read_data_info(args.source_path, None)

            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras_back.json"), 'w') as file:
                json.dump(json_cams, file)

            if shuffle:
                random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
                random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

            self.cameras_extent = scene_info.nerf_normalization["radius"]

            for resolution_scale in resolution_scales:
                print("Loading Training Cameras")
                self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                print("Loading Test Cameras")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # self.smpl_template = subdivide_smpl_model(smoothing=True, n_iter=2).to(self.device)
        self.smpl_template = SMPL(SMPL_PATH).to(self.device)
        self.smpl = SMPL(SMPL_PATH).to(self.device)
        edges = trimesh.Trimesh(
            vertices=self.smpl_template.v_template.detach().cpu().numpy(),
            faces=self.smpl_template.faces, process=False
        ).edges_unique
        self.edges = torch.from_numpy(edges).to(self.device).long()
        self.lpips = LPIPS(net="vgg", pretrained=True).to('cuda')
        self.patch_sampler = PatchSampler(num_patch=4, patch_size=128, ratio_mask=0.9, dilate=0)
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        else:
            #self.optimize_init()
            self.initialize()

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    @torch.no_grad()
    def get_vitruvian_verts(self):
        vitruvian_pose = get_predefined_pose('t_pose', self.device)[0]
        vitruvian_pose = torch.zeros((1, 69), dtype=torch.float32, device='cuda')
        vitruvian_pose[:, 2] = 0.05
        vitruvian_pose[:, 5] = -0.05
        vitruvian_pose[:, 47] = -0.4
        vitruvian_pose[:, 50] = 0.4
        vitruvian_pose[:, 48] = -0.78
        vitruvian_pose[:, 51] = -0.78

        betas = [-0.5773,  0.1266, -0.5128,  0.1527,  2.3070, -2.3127, -0.6074, -1.1296,
         -0.8285,  0.1158]
        betas = torch.zeros(10)
        self.betas = torch.tensor(betas, dtype=torch.float32, device=self.device)
        smpl_output = self.smpl(body_pose=vitruvian_pose, betas=self.betas[None], disable_posedirs=False)
        vitruvian_verts = smpl_output.vertices[0]
        self.T_t2vitruvian = smpl_output.T[0].detach()
        self.inv_T_t2vitruvian = torch.inverse(self.T_t2vitruvian)
        self.canonical_offsets = smpl_output.shape_offsets + smpl_output.pose_offsets
        self.canonical_offsets = self.canonical_offsets[0]
        self.vitruvian_verts = vitruvian_verts.detach()
        return vitruvian_verts.detach()


    @torch.no_grad()
    def get_vitruvian_verts_template(self):
        vitruvian_pose = torch.zeros((1, 69), dtype=torch.float32, device='cuda')
        vitruvian_pose[:, 2] = 0.05
        vitruvian_pose[:, 5] = -0.05
        vitruvian_pose[:, 47] = -0.4
        vitruvian_pose[:, 50] = 0.4
        vitruvian_pose[:, 48] = -0.78
        vitruvian_pose[:, 51] = -0.78
        smpl_output = self.smpl_template(body_pose=vitruvian_pose, betas=self.betas[None], disable_posedirs=False)
        vitruvian_verts = smpl_output.vertices[0]
        return vitruvian_verts.detach()

    def initialize(self):
        self.get_vitruvian_verts()
        t_pose_verts = self.get_vitruvian_verts_template()
        colors = torch.ones_like(t_pose_verts) * 0.5

        shs = torch.zeros((colors.shape[0], 3, 16)).float().cuda()
        shs[:, :3, 0] = colors
        shs[:, 3:, 1:] = 0.0

        scales = torch.zeros_like(t_pose_verts)
        for v in range(t_pose_verts.shape[0]):
            selected_edges = torch.any(self.edges == v, dim=-1)
            selected_edges_len = torch.norm(
                t_pose_verts[self.edges[selected_edges][0]] - t_pose_verts[self.edges[selected_edges][1]],
                dim=-1
            )
            scales[v, 0] = torch.log(torch.max(selected_edges_len))
            scales[v, 1] = torch.log(torch.max(selected_edges_len))

            scales[v, 2] = torch.log(torch.max(selected_edges_len))


        import trimesh
        mesh = trimesh.Trimesh(vertices=t_pose_verts.detach().cpu().numpy(), faces=self.smpl_template.faces)
        vert_normals = torch.tensor(mesh.vertex_normals).float().cuda()

        gs_normals = torch.zeros_like(vert_normals)
        gs_normals[:, 2] = 1.0

        norm_rotmat = torch_rotation_matrix_from_vectors(gs_normals, vert_normals)

        rotq = matrix_to_quaternion(norm_rotmat)

        self.normals = gs_normals

        opacity = inverse_sigmoid(0.1 * torch.ones((t_pose_verts.shape[0], 1), dtype=torch.float, device="cuda"))

        self.n_gs = t_pose_verts.shape[0]
        self.gaussians._xyz = nn.Parameter(t_pose_verts.requires_grad_(True))
        self.gaussians._features_dc = nn.Parameter(shs[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self.gaussians._features_rest = nn.Parameter(shs[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self.gaussians._scaling = nn.Parameter(scales.requires_grad_(True))
        self.gaussians._rotation = nn.Parameter(rotq.requires_grad_(True))
        self.gaussians._opacity = nn.Parameter(opacity.requires_grad_(True))
        self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")


    def initialize_no_grad(self):
        betas = np.zeros(10)
        self.betas = torch.tensor(betas, dtype=torch.float32, device=self.device)
        t_pose_verts = self.get_vitruvian_verts_template()

        colors = torch.ones_like(t_pose_verts) * 0.5

        shs = torch.zeros((colors.shape[0], 3, 16)).float().cuda()
        shs[:, :3, 0] = colors
        shs[:, 3:, 1:] = 0.0

        scales = torch.zeros_like(t_pose_verts)
        for v in range(t_pose_verts.shape[0]):
            selected_edges = torch.any(self.edges == v, dim=-1)
            selected_edges_len = torch.norm(
                t_pose_verts[self.edges[selected_edges][0]] - t_pose_verts[self.edges[selected_edges][1]],
                dim=-1
            )
            scales[v, 0] = torch.log(torch.max(selected_edges_len))
            scales[v, 1] = torch.log(torch.max(selected_edges_len))

            scales[v, 2] = torch.log(torch.max(selected_edges_len))


        # dist2 = torch.clamp_min(
        #     distCUDA2(t_pose_verts.float().cuda()), 0.0000001)[..., None].repeat([1, 3])
        #
        # scales = torch.log(torch.sqrt(dist2))

        import trimesh
        mesh = trimesh.Trimesh(vertices=t_pose_verts.detach().cpu().numpy(), faces=self.smpl_template.faces)
        vert_normals = torch.tensor(mesh.vertex_normals).float().cuda()

        gs_normals = torch.zeros_like(vert_normals)
        gs_normals[:, 2] = 1.0

        norm_rotmat = torch_rotation_matrix_from_vectors(gs_normals, vert_normals)

        rotq = matrix_to_quaternion(norm_rotmat)

        self.normals = gs_normals

        opacity = inverse_sigmoid(0.1 * torch.ones((t_pose_verts.shape[0], 1), dtype=torch.float, device="cuda"))

        self.n_gs = t_pose_verts.shape[0]
        self.gaussians._xyz = t_pose_verts
        self.gaussians._features_dc = shs[:, :, 0:1].transpose(1, 2)
        self.gaussians._features_rest = shs[:, :, 1:].transpose(1, 2)
        self.gaussians._scaling = scales
        self.gaussians._rotation = rotq
        self.gaussians._opacity = opacity
        self.gaussians.max_radii2D = torch.zeros((self.gaussians.get_xyz.shape[0]), device="cuda")


    @torch.no_grad()
    def render_canonical(self, iter=None, nframes=60, is_train_progress=False, pose_type=None, pipe=None, bg=None):
        iter_s = 'final' if iter is None else f'{iter:06d}'
        iter_s += f'_{pose_type}' if pose_type is not None else ''


        os.makedirs('./output/canon/', exist_ok=True)

        camera_params = get_rotating_camera(
            dist=5.0, img_size=256 if is_train_progress else 512,
            nframes=nframes, device='cuda',
            angle_limit=torch.pi if is_train_progress else 2 * torch.pi,
        )

        betas = self.betas.detach()

        static_smpl_params = get_smpl_static_params(
            betas=betas
        )

        pbar = range(nframes) if is_train_progress else tqdm(range(nframes), desc="Canonical:")

        for idx in pbar:

            cam_p = camera_params[idx]
            fovx = cam_p["fovx"]
            fovy = cam_p["fovy"]
            width = cam_p["image_width"]
            height = cam_p["image_height"]
            world_view_transform = cam_p["world_view_transform"]
            full_proj_transform = cam_p["full_proj_transform"]
            data = dict(static_smpl_params, **cam_p)
            viewpoint_cam = MiniCam(width, height, fovy, fovx, 0.05, 100.0, world_view_transform, full_proj_transform)

            render_pkg = render(viewpoint_cam, self.gaussians, pipe, bg)

            image = render_pkg["render"]

            torchvision.utils.save_image(image, f'./output/canon/{idx:05d}.png')

        video_fname = f'./output/canon/canon_{iter_s}.mp4'
        create_video(f'./output/canon/', video_fname, fps=10)

    @torch.no_grad()
    def animate(self, motion_path=None, iter=None, keep_images=False, pipe=None, bg=None):


        iter_s = 'final' if iter is None else f'{iter:06d}'

        os.makedirs('./output/anim/', exist_ok=True)
        start_idx,end_idx,skip = 0,400,4
        motions = np.load(motion_path)
        poses = torch.tensor(motions['poses'][start_idx:end_idx:skip, AMASS_SMPLH_TO_SMPL_JOINTS], dtype=torch.float32, device='cuda')
        poses[:,61:] = 0
        transl = torch.tensor(motions['trans'][start_idx:end_idx:skip], dtype=torch.float32,
                             device='cuda')

        camera_params = get_rotating_camera(
            dist=5.0, img_size=512,
            nframes=1, device='cuda',
            angle_limit=2 * torch.pi,
        )
        manual_trans = torch.tensor((np.array([0, -1, 0])), dtype=torch.float32,
                     device='cuda')
        manual_rot = np.array([-90.4, 0, 0]) / 180 * np.pi
        manual_rotmat = torch.tensor((transformations.euler_matrix(*manual_rot)[:3, :3]),dtype=torch.float32,
                     device='cuda')
        manual_scale = torch.tensor((1.0), dtype=torch.float32,
                     device='cuda')
        ext_tfs = (manual_trans.unsqueeze(0), manual_rotmat.unsqueeze(0), manual_scale.unsqueeze(0))
        for idx in tqdm(range(poses.shape[0]), desc="Animation"):

            print(idx)
            output = self.forward(
                global_orient=poses[idx-1,:3],
                body_pose=poses[idx-1,3:],
                betas=self.betas,
                transl=transl[idx-1],
                ext_tfs=ext_tfs,
            )

            # poses = get_predefined_pose('t_pose', self.device)[0]
            # output = self.forward(
            #     global_orient=torch.zeros(3, device=poses.device, dtype=poses.dtype),
            #     body_pose=poses,
            #     betas=self.betas,
            #     transl=None,
            #     ext_tfs=None,
            # )

            feats = output['shs']
            means3D = output['xyz']
            opacity = output['opacity']
            scales = output['scales']
            rotations = output['rotq']
            active_sh_degree = output['active_sh_degree']

            cam_p = camera_params[0]
            fovx = cam_p["fovx"]
            fovy = cam_p["fovy"]
            width = cam_p["image_width"]
            height = cam_p["image_height"]
            world_view_transform = cam_p["world_view_transform"]
            full_proj_transform = cam_p["full_proj_transform"]
            viewpoint_cam = MiniCam(width, height, fovy, fovx, 0.05, 100.0, world_view_transform, full_proj_transform)

            render_pkg = render_deformed(viewpoint_cam, feats, means3D, opacity, scales, rotations, active_sh_degree ,pipe, bg)

            image = render_pkg["render"]

            image = render_pkg["render"]

            torchvision.utils.save_image(image, f'./output/anim/{idx:05d}.png')

        video_fname = f'./output/anim/anim_{iter_s}.mp4'
        create_video(f'./output/anim/', video_fname, fps=20)

    def forward(
            self,
            global_orient=None,
            body_pose=None,
            betas=None,
            transl=None,
            smpl_scale=None,
            dataset_idx=-1,
            is_train=False,
            ext_tfs=None,
    ):
        gs_scales = self.gaussians.scaling_activation(self.gaussians._scaling)
        gs_rotq = self.gaussians.rotation_activation(self.gaussians._rotation)
        gs_xyz = self.gaussians._xyz
        gs_opacity = self.gaussians.opacity_activation(self.gaussians._opacity)
        gs_shs = self.gaussians.get_features

        gs_scales_canon = gs_scales.clone()
        gs_rotmat = quaternion_to_matrix(gs_rotq)

        if hasattr(self, 'global_orient') and global_orient is None:
            global_orient = rotation_6d_to_axis_angle(
                self.global_orient[dataset_idx].reshape(-1, 6)).reshape(3)

        if hasattr(self, 'body_pose') and body_pose is None:
            body_pose = rotation_6d_to_axis_angle(
                self.body_pose[dataset_idx].reshape(-1, 6)).reshape(23 * 3)

        if hasattr(self, 'betas') and betas is None:
            betas = self.betas

        if hasattr(self, 'transl') and transl is None:
            transl = self.transl[dataset_idx]

        # vitruvian -> t-pose -> posed
        # remove & reapply the blendshape
        smpl_output = self.smpl(
            betas=betas.unsqueeze(0),
            body_pose=body_pose.unsqueeze(0),
            global_orient=global_orient.unsqueeze(0),
            disable_posedirs=False,
        )
        curr_offsets = (smpl_output.shape_offsets + smpl_output.pose_offsets)[0]
        T_t2pose = smpl_output.T[0]
        T_vitruvian2t = self.inv_T_t2vitruvian.clone()
        T_vitruvian2t[..., :3, 3] = T_vitruvian2t[..., :3, 3] + self.canonical_offsets - curr_offsets
        T_vitruvian2pose = T_t2pose @ T_vitruvian2t

        _, lbs_T = smpl_lbsmap_top_k(
            lbs_weights=self.smpl.lbs_weights,
            verts_transform=T_vitruvian2pose.unsqueeze(0),
            points=gs_xyz.unsqueeze(0),
            template_points=self.vitruvian_verts.unsqueeze(0),
            K=6,
        )
        lbs_T = lbs_T.squeeze(0)

        homogen_coord = torch.ones_like(gs_xyz[..., :1])
        gs_xyz_homo = torch.cat([gs_xyz, homogen_coord], dim=-1)
        deformed_xyz = torch.matmul(lbs_T, gs_xyz_homo.unsqueeze(-1))[..., :3, 0]

        if smpl_scale is not None:
            deformed_xyz = deformed_xyz * smpl_scale.unsqueeze(0)
            gs_scales = gs_scales * smpl_scale.unsqueeze(0)

        if transl is not None:
            deformed_xyz = deformed_xyz + transl.unsqueeze(0)

        deformed_gs_rotmat = lbs_T[:, :3, :3] @ gs_rotmat
        deformed_gs_rotq = matrix_to_quaternion(deformed_gs_rotmat)

        if ext_tfs is not None:
            tr, rotmat, sc = ext_tfs
            deformed_xyz = (tr[..., None] + (sc[None] * (rotmat @ deformed_xyz[..., None]))).squeeze(-1)
            gs_scales = sc * gs_scales

            rotq = matrix_to_quaternion(rotmat)
            deformed_gs_rotq = quaternion_multiply(rotq, deformed_gs_rotq)
            deformed_gs_rotmat = quaternion_to_matrix(deformed_gs_rotq)

        self.gaussians.normals = torch.zeros_like(gs_xyz)
        self.gaussians.normals[:, 2] = 1.0


        deformed_gs_shs = gs_shs.clone()

        return {
            'xyz': deformed_xyz,
            'xyz_canon': gs_xyz,
            'xyz_offsets': torch.zeros_like(gs_xyz),
            'scales': gs_scales,
            'scales_canon': gs_scales_canon,
            'rotq': deformed_gs_rotq,
            'rotq_canon': gs_rotq,
            'rotmat': deformed_gs_rotmat,
            'rotmat_canon': gs_rotmat,
            'shs': deformed_gs_shs,
            'opacity': gs_opacity,
            'active_sh_degree': self.gaussians.active_sh_degree,
        }

    def optimize_init(self):

        parser = ArgumentParser(description="Training script parameters")
        lp = ModelParams(parser)
        op = OptimizationParams(parser)
        pp = PipelineParams(parser)
        args = parser.parse_args()
        opt = op.extract(args)
        self.initialize_no_grad()
        pipe = pp.extract(args)
        viewpoint_stack = None
        viewpoint_cam = self.getTrainCameras().copy()[1]
        self.gaussians.smpl_opt = FitShape()

        self.gaussians.training_setup(opt)
        gt_image = viewpoint_cam.original_image
        key_model = keypointrcnn_resnet50_fpn(pretrained=True)
        key_model.eval().cuda()  # Set the model to inference mode
        gt_predictions = key_model(gt_image.unsqueeze(0))
        gt_keypoints = gt_predictions[0]['keypoints'][0]
        indices = torch.tensor([5,6, 7, 8, 9, 10, 11, 12])
        gt_keypoints = gt_keypoints[indices]
        for iteration in tqdm(range(5001)):
            # Pick a random Camera
            # if not viewpoint_stack:
            #     viewpoint_stack = self.getTrainCameras().copy()
            # idx = randint(0, len(viewpoint_stack) - 1)
            # viewpoint_cam = viewpoint_stack.pop(idx)

            smpl_output = self.gaussians.smpl_opt()
            self.gaussians._xyz = smpl_output.vertices[0]
            t_pose_verts = smpl_output.vertices[0]
            scales = torch.empty((t_pose_verts.shape[0], 3), device='cuda')

            # 将 edges 拆分为两个顶点索引的集合
            edges_v0 = self.edges[:, 0]
            edges_v1 = self.edges[:, 1]

            # 计算所有边的长度
            edge_lengths = torch.norm(t_pose_verts[edges_v0] - t_pose_verts[edges_v1], dim=-1)


            # 接着，我们需要找到每个顶点相连的最大的边长度，这里我们使用掩码而不是 `scatter_add`
            # 创建一个 (6890, 22889) 的掩码张量
            mask = torch.zeros((t_pose_verts.shape[0], self.edges.shape[0]), dtype=torch.bool, device='cuda')
            mask.scatter_(0, self.edges.t(), 1)

            # 利用掩码计算每个顶点连接的最大边的长度
            max_lengths = torch.max(mask * edge_lengths, dim=-1).values

            # 计算缩放比例
            log_max_lengths = torch.log(max_lengths)
            scales[:, 0] = log_max_lengths
            scales[:, 1] = log_max_lengths
            scales[:, 2] = log_max_lengths
            self.gaussians._scaling = scales
            bg = torch.zeros(3, device="cuda")

            render_pkg = render(viewpoint_cam, self.gaussians, pipe, bg,override_color=None)

            #loss = loss_fn(viewpoint_cam, render_pkg, smpl_output, bg, opt)

            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg[
                "viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

            pred_predictions = key_model(image.unsqueeze(0))
            pred_keypoints = pred_predictions[0]['keypoints'][0]
            pred_keypoints = pred_keypoints[indices]
            loss = l1_loss(gt_keypoints/1024, pred_keypoints/1024)
            #gt_image = viewpoint_cam.original_image.cuda()
            # mask = viewpoint_cam.mask.unsqueeze(0).cuda()
            # mask = mask + bg[:, None, None] * (1. - mask)
            # Ll1 = l1_loss(image, mask, mask)
            # loss_ssim = 1.0 - ssim(image, mask)
            # loss_ssim = loss_ssim * (mask.sum() / (image.shape[-1] * image.shape[-2]))
            # loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (loss_ssim)
            loss.backward(retain_graph=True)
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad(set_to_none=True)

            print(loss, self.gaussians.smpl_opt.beta)
            if iteration % 100 == 0:
                os.makedirs('shape/', exist_ok=True)
                log_pred_img = (image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                log_pred_img = Image.fromarray(log_pred_img)
                log_gt_img = (viewpoint_cam.original_image.detach().cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                log_gt_img = Image.fromarray(log_gt_img)
                draw = draw_img(log_gt_img,gt_keypoints)
                draw = draw_img(log_pred_img, pred_keypoints)
                log_img = mask_image(log_gt_img, log_pred_img)
                log_pred_img.save(f"shape/pred_image_{iteration}.png")
                log_gt_img.save(f"shape/mask_image_{iteration}.png")
                log_img.save(f"shape/merged_image_{iteration}.png")
        self.betas = self.gaussians.smpl_opt.beta[0].detach()

def get_predefined_pose(pose_type, device):
    if pose_type == 'da_pose':
        body_pose = torch.zeros((1, 69), dtype=torch.float32, device=device)
        body_pose[:, 2] = 1.0
        body_pose[:, 5] = -1.0
    elif pose_type == 'a_pose':
        body_pose = torch.zeros((1, 69), dtype=torch.float32, device=device)
        body_pose[:, 2] = 0.2
        body_pose[:, 5] = -0.2
        body_pose[:, 47] = -0.8
        body_pose[:, 50] = 0.8
    elif pose_type == 't_pose':
        body_pose = torch.zeros((1, 69), dtype=torch.float32, device=device)

    return body_pose


def batch_index_select(data, inds):
    bs, nv = data.shape[:2]
    device = data.device
    inds = inds + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    data = data.reshape(bs*nv, *data.shape[2:])
    return data[inds.long()]


def smpl_lbsmap_top_k(
        lbs_weights,
        verts_transform,
        points,
        template_points,
        K=6,
        addition_info=None
):
    '''ref: https://github.com/JanaldoChen/Anim-NeRF
    Args:
    '''
    bz, np, _ = points.shape
    with torch.no_grad():
        results = knn_points(points, template_points, K=K)
        dists, idxs = results.dists, results.idx
    neighbs_dist = dists
    neighbs = idxs
    weight_std = 0.1
    weight_std2 = 2. * weight_std ** 2
    xyz_neighbs_lbs_weight = lbs_weights[neighbs]  # (bs, n_rays*K, k_neigh, 24)
    xyz_neighbs_weight_conf = torch.exp(
        -torch.sum(
            torch.abs(xyz_neighbs_lbs_weight - xyz_neighbs_lbs_weight[..., 0:1, :]), dim=-1
        ) / weight_std2)  # (bs, n_rays*K, k_neigh)
    xyz_neighbs_weight_conf = torch.gt(xyz_neighbs_weight_conf, 0.9).float()
    xyz_neighbs_weight = torch.exp(-neighbs_dist)  # (bs, n_rays*K, k_neigh)
    xyz_neighbs_weight *= xyz_neighbs_weight_conf
    xyz_neighbs_weight = xyz_neighbs_weight / xyz_neighbs_weight.sum(-1, keepdim=True)  # (bs, n_rays*K, k_neigh)

    xyz_neighbs_transform = batch_index_select(verts_transform, neighbs)  # (bs, n_rays*K, k_neigh, 4, 4)
    xyz_transform = torch.sum(xyz_neighbs_weight.unsqueeze(-1).unsqueeze(-1) * xyz_neighbs_transform,
                              dim=2)  # (bs, n_rays*K, 4, 4)
    xyz_dist = torch.sum(xyz_neighbs_weight * neighbs_dist, dim=2, keepdim=True)  # (bs, n_rays*K, 1)

    if addition_info is not None:  # [bz, nv, 3]
        xyz_neighbs_info = batch_index_select(addition_info, neighbs)
        xyz_info = torch.sum(xyz_neighbs_weight.unsqueeze(-1) * xyz_neighbs_info, dim=2)
        return xyz_dist, xyz_transform, xyz_info
    else:
        return xyz_dist, xyz_transform


def draw_img(image,keypoints):
    draw = ImageDraw.Draw(image)
    for i, point in enumerate(keypoints):

        if point[2] > 0.9:  # 如果置信度大于0，则绘制该关键点
            x, y = point[:2]
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(255, 0, 0))  # 绘制红色圆点
    return draw

def mask_image(bottom_image, top_image, threshold=127):
    top_image = top_image.convert("RGBA")


    # 将top图片的mask从RGBA转换为L（灰度）
    mask = top_image.convert("L")

    # 将mask的阈值设置为127，以上的像素为100%不透明，以下的像素为100%透明
    mask = mask.point(lambda x: 100 if x > threshold else 0, '1')

    # 使用mask来覆盖bottom图片
    bottom_image.putalpha(mask)

    return bottom_image
