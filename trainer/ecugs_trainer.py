# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
from tqdm import tqdm
from random import randint
import math
import numpy as np
import random
from collections import defaultdict, OrderedDict
import json
import gzip
import torch
import torch.nn.functional as F
from torchvision import io
from PIL import Image
from einops import rearrange
import pickle
import scipy
import imageio
import glob
import cv2
import open3d as o3d

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render
from scene.gaussian_model_cf import ECUGSRender as GS_Render

from utils.graphics_utils import BasicPointCloud, focal2fov, procrustes
from scene.cameras import Camera
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch import lpips
from utils.image_utils import psnr, colorize
from utils.utils_poses.align_traj import align_ate_c2b_use_a2b
from utils.utils_poses.comp_ate import compute_rpe, compute_ATE

from kornia.geometry.depth import depth_to_3d, depth_to_normals
from kornia.geometry.camera import project_points

import pdb

from .trainer import GaussianTrainer
from .losses import Loss, compute_scale_and_shift

# Import geometric constraints
try:
    from .geometric_constraints import (
        ECUGSPhotometricGeometricLoss,
        ECUGSCalibrationRefinementLoss,
        ECUGSScaleRegularizationLoss
    )
    GEOMETRIC_CONSTRAINTS_AVAILABLE = True
except ImportError:
    GEOMETRIC_CONSTRAINTS_AVAILABLE = False
    print("Warning: Geometric constraints not available")

# Import ECU-GS uncertainty-aware modules
try:
    from .ecugs_losses import ECUGSJointLoss
    from utils.ecugs_uncertainty import ECUGSUncertaintyEstimator
    ECUGS_UNCERTAINTY_AVAILABLE = True
except ImportError:
    ECUGS_UNCERTAINTY_AVAILABLE = False
    print("Warning: ECU-GS uncertainty modules not available")

from copy import copy
from utils.vis_utils import interp_poses_bspline, generate_spiral_nerf, plot_pose


def contruct_pose(poses):
    n_trgt = poses.shape[0]
    for i in range(n_trgt-1, 0, -1):
        poses = torch.cat(
            (poses[:i], poses[[i-1]]@poses[i:]), 0)
    return poses


class ECUGSTrainer(GaussianTrainer):
    def __init__(self, data_root, model_cfg, pipe_cfg, optim_cfg):
        super().__init__(data_root, model_cfg, pipe_cfg, optim_cfg)
        self.model_cfg = model_cfg
        self.pipe_cfg = pipe_cfg
        self.optim_cfg = optim_cfg

        # LiDAR相关配置
        self.use_lidar = getattr(model_cfg, 'use_lidar', False)
        self.optimize_lidar_extrinsic = getattr(model_cfg, 'optimize_lidar_extrinsic', False)
        self.use_lidar_odometry = getattr(model_cfg, 'use_lidar_odometry', False)
        self.use_depth_scale_match = getattr(model_cfg, 'use_depth_scale_match', False)
        self.save_global_pcd = getattr(model_cfg, 'save_global_pcd', True)
        
        # 天空高斯配置
        self.use_sky_gaussians = getattr(model_cfg, 'use_sky_gaussians', False)
        self.num_sky_gaussians = getattr(model_cfg, 'num_sky_gaussians', 10000)
        self.sky_radius = getattr(model_cfg, 'sky_radius', 50.0)
        
        # Anchor-Auxiliary系统配置 (TLC-Calib)
        self.use_anchor_auxiliary = getattr(model_cfg, 'use_anchor_auxiliary', True)
        self.num_auxiliary = getattr(model_cfg, 'num_auxiliary', 5)
        self.adaptive_voxel_beta = getattr(model_cfg, 'adaptive_voxel_beta', 5000.0)
        
        # 几何约束配置
        self.use_geometric_loss = getattr(model_cfg, 'use_geometric_loss', True)
        self.lambda_scale_reg = getattr(model_cfg, 'lambda_scale_reg', 0.01)
        self.lambda_aniso_reg = getattr(model_cfg, 'lambda_aniso_reg', 0.001)
        
        # 标定精化配置
        self.use_plane_constraint = getattr(model_cfg, 'use_plane_constraint', True)
        self.use_edge_alignment = getattr(model_cfg, 'use_edge_alignment', True)
        self.calibration_warmup = getattr(model_cfg, 'calibration_warmup', 500)

        # ECU-GS 不确定度感知配置
        self.use_uncertainty_weighting = getattr(model_cfg, 'use_uncertainty_weighting', False)
        self.use_normal_consistency = getattr(model_cfg, 'use_normal_consistency', False)
        self.three_stage_training = getattr(model_cfg, 'three_stage_training', False)
        self.lambda_pose_reg = getattr(model_cfg, 'lambda_pose_reg', 0.1)
        self.gamma_n = getattr(model_cfg, 'gamma_n', 0.75)
        self.sigma_d2 = getattr(model_cfg, 'sigma_d2', 0.01)
        self.beta_t = getattr(model_cfg, 'beta_t', 1.0)
        self.beta_n = getattr(model_cfg, 'beta_n', 0.5)
        self.residual_tau0 = getattr(model_cfg, 'residual_tau0', 0.3)
        self.residual_kappa = getattr(model_cfg, 'residual_kappa', 2.0)
        self.max_spawn_gaussians = getattr(model_cfg, 'max_spawn_gaussians', 1000)
        self.prune_opacity_threshold = getattr(model_cfg, 'prune_opacity_threshold', 0.01)
        self.prune_age_threshold = getattr(model_cfg, 'prune_age_threshold', 500)
        self.frame_recent_weight = getattr(model_cfg, 'frame_recent_weight', 2.0)
        self.frame_history_weight = getattr(model_cfg, 'frame_history_weight', 1.0)
        self.keyframe_expansion_gain = getattr(model_cfg, 'keyframe_expansion_gain', 0.05)
        self.keyframe_min_baseline = getattr(model_cfg, 'keyframe_min_baseline', 1.0)
        self.extrinsic_update_interval = getattr(model_cfg, 'extrinsic_update_interval', 10)

        # ECU-GS 运行时状态
        self._current_stage = 'intermediate'
        self.pose_priors = {}
        self.occupied_bins = set()
        self.keyframes = []
        self.last_keyframe_center = None
        self._frames_since_ext_cov = 0
        self.uncertainty_estimator = None
        self.ecugs_joint_loss = None
        if ECUGS_UNCERTAINTY_AVAILABLE and self.use_uncertainty_weighting:
            self.uncertainty_estimator = ECUGSUncertaintyEstimator(device='cuda')
        
        # 初始化LiDAR外参
        init_lidar2cam = None
        if self.use_lidar:
            from utils.lidar_loader import get_lidar2cam_gt
            init_lidar2cam = get_lidar2cam_gt()

        self.gs_render = GS_Render(white_background=False,
                                   view_dependent=model_cfg.view_dependent,
                                   optimize_lidar_extrinsic=self.optimize_lidar_extrinsic,
                                   init_lidar2cam=init_lidar2cam,
                                   use_sky_gaussians=self.use_sky_gaussians,
                                   num_sky_gaussians=self.num_sky_gaussians,
                                   sky_radius=self.sky_radius)
        self.gs_render_local = GS_Render(white_background=False,
                                         view_dependent=model_cfg.view_dependent,
                                         optimize_lidar_extrinsic=False,
                                         init_lidar2cam=init_lidar2cam,
                                         use_sky_gaussians=False)  # local renderer不需要天空
        self.use_mask = self.pipe_cfg.use_mask
        self.use_mono = self.pipe_cfg.use_mono
        self.near = 0.01
        
        # 初始化LiDAR相关组件
        self.setup_lidar_components()
        self.setup_losses()
    
    def setup_lidar_components(self):
        """初始化LiDAR相关组件"""
        self.lidar_calibrator = None
        self.scale_matcher = None
        self.lidar_odometry = None
        self.lidar_point_clouds = []  # 存储每帧的LiDAR点云
        self.optimized_lidar2cam = None
        
        if self.use_lidar:
            # coming soon: extrinsic calibration / depth scale matching / LiDAR odometry
            try:
                from utils.lidar_calibration import (ECUGSCalibrator, ECUGSDepthScaleMatcher,
                                                       ECUGSLiDAROdometry)
            except ImportError:
                ECUGSCalibrator = ECUGSDepthScaleMatcher = ECUGSLiDAROdometry = None
            
            if self.optimize_lidar_extrinsic:
                # 初始化外参标定器
                from utils.lidar_loader import get_lidar2cam_gt
                init_lidar2cam = get_lidar2cam_gt()
                self.lidar_calibrator = ECUGSCalibrator(
                    init_extrinsic=init_lidar2cam,
                    learnable=True,
                    lr=1e-4
                )
            
            if self.use_depth_scale_match:
                # 初始化深度尺度匹配器
                self.scale_matcher = ECUGSDepthScaleMatcher(method='ransac')
            
            if self.use_lidar_odometry:
                # 初始化LiDAR里程计
                self.lidar_odometry = ECUGSLiDAROdometry(method='point-to-plane')
    
    def init_sky_gaussians(self, scene_center=None, sky_color=None):
        """
        初始化天空高斯
        
        Args:
            scene_center: 场景中心，用于定位天空球
            sky_color: 天空颜色，默认为浅蓝色
        """
        if not self.use_sky_gaussians:
            return
        
        if self.gs_render.sky_gaussians is not None:
            # 如果没有提供场景中心，尝试从场景高斯计算
            if scene_center is None:
                xyz = self.gs_render.gaussians.get_xyz
                scene_center = xyz.mean(dim=0).detach().cpu().numpy()
            
            print(f"Initializing sky gaussians at center: {scene_center}")
            self.gs_render.init_sky_gaussians(
                center=scene_center,
                color=sky_color
            )
            
            # 设置天空高斯的优化器
            if hasattr(self, 'sky_optimizer'):
                del self.sky_optimizer
            
            sky_params = self.gs_render.get_sky_optimizer_params()
            if len(sky_params) > 0:
                self.sky_optimizer = torch.optim.Adam(sky_params, lr=0.0, eps=1e-15)
                print(f"Created sky optimizer with {len(sky_params)} param groups")
        else:
            print("Warning: sky_gaussians is None")
    
    def get_current_lidar2cam(self):
        """获取当前的LiDAR到相机外参"""
        if self.lidar_calibrator is not None:
            return self.lidar_calibrator.get_extrinsic()
        elif self.gs_render.lidar_calibrator is not None:
            return self.gs_render.lidar_calibrator.get_extrinsic()
        else:
            from utils.lidar_loader import get_lidar2cam_gt
            return torch.from_numpy(get_lidar2cam_gt()).cuda()
    
    def save_optimized_extrinsic(self, filepath):
        """保存优化后的外参到txt文件"""
        from utils.lidar_calibration import save_extrinsic_to_txt
        lidar2cam = self.get_current_lidar2cam()
        if isinstance(lidar2cam, torch.Tensor):
            lidar2cam = lidar2cam.detach().cpu().numpy()
        save_extrinsic_to_txt(lidar2cam, filepath)
        print(f"Saved optimized LiDAR-Camera extrinsic to {filepath}")
    
    def save_global_point_cloud(self, filepath, voxel_size=0.05):
        """保存全局点云"""
        from utils.lidar_calibration import save_global_point_cloud, merge_point_clouds
        
        if len(self.lidar_point_clouds) == 0:
            print("No LiDAR point clouds to save")
            return
        
        # 获取所有位姿
        poses = []
        for idx in range(len(self.lidar_point_clouds)):
            pose = self.gs_render.gaussians.get_RT(idx)
            if isinstance(pose, torch.Tensor):
                pose = pose.detach().cpu().numpy()
            poses.append(pose)
        
        # 合并点云
        global_pcd = merge_point_clouds(self.lidar_point_clouds, poses, voxel_size)
        
        # 保存
        save_global_point_cloud(global_pcd, filepath, also_save_mesh=True)
        print(f"Saved global point cloud to {filepath}")
        
        return global_pcd

    def setup_losses(self):
        self.loss_func = Loss(self.optim_cfg)
        
        # 几何约束损失 (用于非结构化场景)
        self.geometric_loss = None
        if GEOMETRIC_CONSTRAINTS_AVAILABLE:
            self.geometric_loss = ECUGSPhotometricGeometricLoss(
                lambda_photo=1.0,
                lambda_scale=0.01,      # 尺度正则化权重
                lambda_depth=0.1,       # 深度一致性权重
                lambda_aniso=0.001,     # 各向异性正则化权重
                use_ssim=True
            )
            
        # 标定精化损失
        self.calibration_loss = None
        if GEOMETRIC_CONSTRAINTS_AVAILABLE and self.optimize_lidar_extrinsic:
            self.calibration_loss = ECUGSCalibrationRefinementLoss(
                use_plane_constraint=True,
                use_edge_alignment=True,
                plane_weight=0.1,
                edge_weight=0.1
            )

        # ECU-GS 不确定度感知联合损失 (Eq.29)
        if ECUGS_UNCERTAINTY_AVAILABLE and self.use_uncertainty_weighting:
            self.ecugs_joint_loss = ECUGSJointLoss(
                lambda_rgb=1.0,
                lambda_ssim=getattr(self.optim_cfg, 'lambda_dssim', 0.2),
                lambda_depth=getattr(self.model_cfg, 'lambda_depth_lidar', 0.1),
                lambda_normal=getattr(self.model_cfg, 'lambda_normal', 0.05),
                lambda_edge=getattr(self.model_cfg, 'lambda_edge', 0.5),
                lambda_pose=self.lambda_pose_reg,
                gamma_n=self.gamma_n,
                sigma_d2=self.sigma_d2,
            )

    def get_training_stage(self, view_idx=None):
        if not self.three_stage_training:
            return 'intermediate'
        global_iter = getattr(self, 'global_iteration', 0)
        if global_iter < self.calibration_warmup:
            return 'coarse'
        if view_idx is not None and hasattr(self, 'seq_len') and \
                view_idx > min(int(self.seq_len * 0.8), self.seq_len - 5):
            return 'final'
        return 'intermediate'

    def check_keyframe(self, fidx, bin_size=2.0):
        # coming soon: keyframe selection via spatial-coverage novelty (Eq.30);
        # every frame is treated as a keyframe in this release
        return True

    def train_step(self,
                   gs_render,
                   viewpoint_cam,
                   iteration,
                   pipe,
                   optim_opt,
                   colors_precomp=None,
                   update_gaussians=True,
                   update_cam=True,
                   update_distort=False,
                   densify=True,
                   prev_gaussians=None,
                   use_reproject=False,
                   use_matcher=False,
                   ref_fidx=None,
                   reset=True,
                   reproj_loss=None,
                   **kwargs,
                   ):
        # Render
        render_pkg = gs_render.render(
            viewpoint_cam,
            compute_cov3D_python=pipe.compute_cov3D_python,
            convert_SHs_python=pipe.convert_SHs_python,
            override_color=colors_precomp)

        if prev_gaussians is not None:
            with torch.no_grad():
                # Render
                render_pkg_prev = prev_gaussians.render(
                    viewpoint_cam,
                    compute_cov3D_python=pipe.compute_cov3D_python,
                    convert_SHs_python=pipe.convert_SHs_python,
                    override_color=colors_precomp)
            mask = (render_pkg["alpha"] > 0.5).float()
            render_pkg["image"] = render_pkg["image"] * \
                mask + render_pkg_prev["image"] * (1 - mask)
            render_pkg["depth"] = render_pkg["depth"] * \
                mask + render_pkg_prev["depth"] * (1 - mask)

        image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["image"],
                                                                   render_pkg["viewspace_points"],
                                                                   render_pkg["visibility_filter"],
                                                                   render_pkg["radii"])
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss_dict = self.compute_loss(render_pkg, viewpoint_cam,
                                      pipe, iteration,
                                      use_reproject, use_matcher,
                                      ref_fidx, **kwargs)

        loss = loss_dict['loss']
        # with torch.no_grad():
        #     s_max, s_min = gs_render.gaussians.get_scaling.max(dim = 1)[0], gs_render.gaussians.get_scaling.min(dim = 1)[0]
        # scale_loss = torch.relu(s_max / s_min - 1.0).mean()
        # loss = loss + 1.0 * scale_loss
        loss.backward()

        with torch.no_grad():
            # Progress bar
            # try:
            #     self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            # except:
            #     pdb.set_trace()
            # mask = visibility_filter.reshape(gt_image.shape[1:])[None]
            psnr_train = psnr(image, gt_image).mean().double()
            self.just_reset = False
            if iteration < optim_opt.densify_until_iter and densify:
                # Keep track of max radii in image-space for pruning
                try:
                    gs_render.gaussians.max_radii2D[visibility_filter] = torch.max(gs_render.gaussians.max_radii2D[visibility_filter],
                                                                                   radii[visibility_filter])
                except:
                    pdb.set_trace()
                gs_render.gaussians.add_densification_stats(
                    viewspace_point_tensor, visibility_filter)

                if iteration > optim_opt.densify_from_iter and iteration % optim_opt.densification_interval == 0:
                    size_threshold = 20 if iteration > optim_opt.opacity_reset_interval else None
                    self.gs_render.gaussians.densify_and_prune(optim_opt.densify_grad_threshold, 0.005,
                                                               gs_render.radius, size_threshold)

                if iteration % optim_opt.opacity_reset_interval == 0 and reset and iteration < optim_opt.reset_until_iter:
                    gs_render.gaussians.reset_opacity()
                    self.just_reset = True

            if update_gaussians:
                gs_render.gaussians.optimizer.step()
                gs_render.gaussians.optimizer.zero_grad(set_to_none=True)
            if getattr(gs_render.gaussians, "camera_optimizer", None) is not None and update_cam:
                current_fidx = gs_render.gaussians.seq_idx
                gs_render.gaussians.camera_optimizer[current_fidx].step()
                gs_render.gaussians.camera_optimizer[current_fidx].zero_grad(
                    set_to_none=True)
            
            # 更新LiDAR外参优化器 (calibration warmup 后启用, final 阶段冻结外参)
            if self.optimize_lidar_extrinsic and self.lidar_calibrator is not None:
                if self.lidar_calibrator.learnable and iteration > self.calibration_warmup \
                        and self._current_stage != 'final':
                    # 使用较小的学习率
                    for param in self.lidar_calibrator.parameters():
                        if param.grad is not None:
                            param.data = param.data - 1e-5 * param.grad
                    # 手动清零梯度
                    for param in self.lidar_calibrator.parameters():
                        if param.grad is not None:
                            param.grad.zero_()
            
            # 更新天空高斯优化器（注意：天空高斯不参与LiDAR外参优化）
            if self.use_sky_gaussians and hasattr(self, 'sky_optimizer') and update_gaussians:
                if self.sky_optimizer is not None:
                    self.sky_optimizer.step()
                    self.sky_optimizer.zero_grad(set_to_none=True)
            
            ### >>> 7. 每 500 it 进化锚
            # 检查优化器是否包含完整的高斯参数（不只是位姿）
            can_evolve = True
            if hasattr(gs_render.gaussians, 'optimizer') and gs_render.gaussians.optimizer is not None:
                param_groups = gs_render.gaussians.optimizer.param_groups
                param_names = [pg.get('name', '') for pg in param_groups]
                # 如果只优化位姿（'R'）和旋转，不启用 evolve_anchors
                if set(param_names) <= {'R', 'rotation'}:
                    can_evolve = False
            
            # 第一帧训练时（前1000次迭代）不启用 evolve_anchors，避免 PSNR 急剧下降
            global_iter = getattr(self, 'global_iteration', 0)
            if global_iter < 1000:
                can_evolve = False
            
            if iteration % 500 == 0 and can_evolve:
                # coming soon: uncertainty-aware densification with residual
                # threshold tau0 + kappa * sigma_ext(u) (Eq.33);
                # a constant residual threshold is used in this release
                gs_render.gaussians.evolve_anchors(
                    render_pkg, viewpoint_cam, iteration, every=500)
                gs_render.gaussians.prune_anchor_dead(
                    min_opacity=self.prune_opacity_threshold,
                    patience=self.prune_age_threshold)

                # Anchor-Auxiliary系统: 过滤浮动锚点
                if hasattr(gs_render.gaussians, 'filter_anchor_floaters'):
                    gs_render.gaussians.filter_anchor_floaters(
                        min_opacity=self.prune_opacity_threshold,
                        patience=self.prune_age_threshold)
                    
                # Anchor-Auxiliary系统: 固定锚点位置
                if hasattr(gs_render.gaussians, 'fix_anchor_positions'):
                    gs_render.gaussians.fix_anchor_positions()


        return loss_dict, render_pkg, psnr_train

    def init_two_view(self, view_idx_1, view_idx_2, pipe, optim_opt):
        # prepare data
        self.loss_func.depth_loss_type = "invariant"
        cam_info, pcd, viewpoint_cam = self.prepare_data(view_idx_1,
                                                         orthogonal=True,
                                                         down_sample=True)
        radius = np.linalg.norm(pcd.points, axis=1).max()
        
        # 打印初始化信息
        print(f"[Init] Viewpoint image shape: {viewpoint_cam.original_image.shape}")
        print(f"[Init] PCD points: {len(pcd.points)}")

        # Initialize gaussians
        self.gs_render.reset_model()
        self.gs_render.init_model(pcd,)
        self.gs_render.gaussians.init_RT_seq(self.seq_len)
        self.gs_render.gaussians.set_seq_idx(view_idx_1)
        self.gs_render.gaussians.rotate_seq = False
        
        # 初始化Anchor-Auxiliary系统 (TLC-Calib)
        if self.use_lidar and hasattr(self, 'lidar_point_clouds') and len(self.lidar_point_clouds) > 0:
            # 合并所有LiDAR点云
            import open3d as o3d
            all_lidar_points = []
            for pcd_lidar in self.lidar_point_clouds:
                points = np.asarray(pcd_lidar.points)
                all_lidar_points.append(points)
            
            if len(all_lidar_points) > 0:
                lidar_points = np.vstack(all_lidar_points)
                
                # 获取位姿
                poses = []
                for idx in range(min(len(self.lidar_point_clouds), self.seq_len)):
                    try:
                        pose = self.gs_render.gaussians.get_RT(idx)
                        if isinstance(pose, torch.Tensor):
                            pose = pose.detach().cpu().numpy()
                        poses.append(pose)
                    except:
                        break
                
                # 初始化Anchor-Auxiliary系统
                self.gs_render.gaussians.init_anchor_auxiliary_system(
                    lidar_points=lidar_points,
                    poses=poses if len(poses) > 0 else [np.eye(4)],
                    num_auxiliary=5,
                    use_auxiliary=True
                )
        
        # 初始化天空高斯（在场景高斯初始化之后）
        if self.use_sky_gaussians:
            scene_center = pcd.points.mean(axis=0)
            self.init_sky_gaussians(scene_center=scene_center)
        
        # Fit relative pose
        print(f"optimizing frame {view_idx_1:03d}")
        optim_opt.iterations = 1000
        # 允许densification来帮助模型逃离局部最优
        optim_opt.densify_from_iter = 200  # 200次迭代后开始densification
        optim_opt.densify_until_iter = 900  # 900次迭代前结束densification
        optim_opt.densify_interval = 100    # 每100次迭代densify一次
        optim_opt.densify_grad_threshold = 0.0001  # 更低的阈值，更容易densify
        progress_bar = tqdm(range(optim_opt.iterations),
                            desc="Training progress")
        # 允许位置学习以逃离局部最优（LiDAR点云初始化后需要微调位置）
        self.gs_render.gaussians.training_setup(optim_opt, fix_pos=False,)
        
        # 优先使用 LiDAR 深度，如果没有则使用单目深度
        depth_supervision = None
        if hasattr(viewpoint_cam, 'depth_gt') and viewpoint_cam.depth_gt is not None:
            depth_supervision = viewpoint_cam.depth_gt
            # 检查是否是 LiDAR 深度（有实际数值范围）还是单目深度
            depth_min = depth_supervision[depth_supervision > 0].min().item() if (depth_supervision > 0).any() else 0
            depth_max = depth_supervision.max().item()
            print(f"Using depth supervision for frame {view_idx_1:03d}: min={depth_min:.2f}, max={depth_max:.2f}")
        elif self.mono_depth is not None and view_idx_1 < len(self.mono_depth):
            depth_supervision = self.mono_depth[view_idx_1]
            print(f"Using mono depth supervision for frame {view_idx_1:03d}")
        
        for iteration in range(1, optim_opt.iterations+1):
            # Update learning rate
            self.gs_render.gaussians.update_learning_rate(iteration)
            
            # 第一帧训练时不使用几何约束，避免 PSNR 下降
            # 使用 legacy 损失（L1 + SSIM）更稳定
            use_geometric = (iteration > 500)  # 500次迭代后再启用几何约束
            
            # 在densify_from_iter之后启用densification
            densify = (iteration >= optim_opt.densify_from_iter)
            
            loss_dict, rend_dict, psnr_train = self.train_step(self.gs_render,
                                                          viewpoint_cam, iteration,
                                                          pipe, optim_opt,
                                                          depth_gt=depth_supervision if use_geometric else None,
                                                          update_gaussians=True,
                                                          update_cam=False,
                                                          densify=densify,
                                                          )
            # 每 100 次迭代打印详细损失信息
            if iteration % 100 == 0 or iteration == 500 or iteration == 700:
                loss_str = f"Iter {iteration}: PSNR={psnr_train:.2f}"
                for k, v in loss_dict.items():
                    if k != 'loss':
                        loss_str += f", {k}={v:.4f}"
                print(f"\n[Frame {view_idx_1:03d}] {loss_str}")
                
            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}",
                                          "edge_loss" : f"{loss_dict.get('edge',0):.4f}",
                                          "Number points": f"{self.gs_render.gaussians.get_xyz.shape[0]}",})
                progress_bar.update(10)
            if iteration == optim_opt.iterations:
                progress_bar.close()

        self.pcd_stack = []
        self.pcd_stack.append(self.gs_render.gaussians.get_xyz.detach())
        model_params = self.gs_render.gaussians.capture()
        return model_params

    def add_view_v2(self, view_idx, view_idx_prev, reverse=False, is_keyframe=True):
        # Initialize gaussians
        self.loss_func.depth_loss_type = "invariant"
        pipe = copy(self.pipe_cfg)
        optim_opt = copy(self.optim_cfg)

        # 三阶段训练调度: coarse -> intermediate -> final
        self._current_stage = self.get_training_stage(view_idx)
        
        # 创建结果路径
        result_path = f"output/{pipe.expname if pipe.expname else 'progressive'}/{self.category}_{self.seq_name}"
        os.makedirs(f"{result_path}/train", exist_ok=True)
        # prepare data
        cam_info, pcd, viewpoint_cam = self.prepare_data(view_idx_prev,
                                                         orthogonal=True,
                                                         down_sample=True)
        radius = np.linalg.norm(pcd.points, axis=1).max()
        self.gs_render_local.reset_model()
        self.gs_render_local.init_model(pcd)
        
        # 为局部渲染器也初始化Anchor-Auxiliary系统（如果可用）
        if self.use_lidar and hasattr(self, 'lidar_point_clouds') and len(self.lidar_point_clouds) > view_idx_prev:
            import open3d as o3d
            pcd_lidar = self.lidar_point_clouds[view_idx_prev]
            if pcd_lidar is not None:
                lidar_points = np.asarray(pcd_lidar.points)
                # 获取当前位姿
                pose = self.gs_render.gaussians.get_RT(view_idx_prev)
                if isinstance(pose, torch.Tensor):
                    pose = pose.detach().cpu().numpy()
                
                self.gs_render_local.gaussians.init_anchor_auxiliary_system(
                    lidar_points=lidar_points,
                    poses=[pose],
                    num_auxiliary=self.num_auxiliary,
                    use_auxiliary=True,
                    beta_t=self.beta_t,
                    beta_n=self.beta_n
                )

        # Fit current gaussian with LiDAR depth supervision
        # 非关键帧减少局部建图迭代 (Eq.30 增量地图扩展)
        optim_opt.iterations = 1500 if is_keyframe else 400
        optim_opt.densify_from_iter = optim_opt.iterations + 1
        progress_bar = tqdm(range(optim_opt.iterations),
                            desc="Training progress (local)")
        self.gs_render_local.gaussians.training_setup(
            optim_opt, fix_pos=True,)
        
        # 加载LiDAR深度用于监督
        lidar_depth = None
        if self.use_lidar and hasattr(viewpoint_cam, 'depth_gt') and viewpoint_cam.depth_gt is not None:
            lidar_depth = viewpoint_cam.depth_gt
        
        for iteration in range(1, optim_opt.iterations+1):
            # Update learning rate
            self.gs_render_local.gaussians.update_learning_rate(iteration)
            
            # 使用LiDAR深度监督训练
            loss_dict, rend_dict, psnr_train = self.train_step(self.gs_render_local,
                                                          viewpoint_cam, iteration,
                                                          pipe, optim_opt,
                                                          depth_gt=lidar_depth,  # 启用LiDAR深度监督
                                                          update_gaussians=True,
                                                          update_cam=False,
                                                          updata_distort=False,
                                                          densify=False,
                                                          )
            # 降低PSNR阈值，让训练更充分
            if psnr_train > 30 and iteration > 800:
                progress_bar.close()
                break

            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}",
                                          "edge_loss" : f"{loss_dict.get('edge',0):.4f}",
                                          "Number points": f"{self.gs_render_local.gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == optim_opt.iterations:
                progress_bar.close()

        print(f"optimizing frame {view_idx:03d} (pose refinement)")
        viewpoint_cam_ref = self.load_viewpoint_cam(view_idx,
                                                    load_depth=True)
        # final 阶段减少位姿更新 (三阶段调度)
        optim_opt.iterations = 250 if self._current_stage == 'final' else 500
        optim_opt.densify_from_iter = optim_opt.iterations + 1
        # 使用前一帧的位姿作为初始化，而不是单位矩阵
        # 这样高斯位置和相机位姿保持同步，渲染结果才正确
        pose_init = self.gs_render.gaussians.get_RT(view_idx_prev).detach()
        self.gs_render_local.gaussians.init_RT(pose=pose_init)
        self.gs_render_local.gaussians.training_setup_fix_position(
            optim_opt, gaussian_rot=False)
        # 关闭 rotate_xyz，高斯位置保持世界坐标系中，通过相机位姿变换
        self.gs_render_local.gaussians.rotate_xyz = False

        # 创建基础相机（不指定位姿，使用默认单位矩阵）
        # 我们将在每次迭代时动态更新位姿
        viewpoint_cam_base = self.load_viewpoint_cam(view_idx, load_depth=True)
        
        # 获取新视角的LiDAR深度
        lidar_depth_ref = None
        if self.use_lidar and hasattr(viewpoint_cam_base, 'depth_gt') and viewpoint_cam_base.depth_gt is not None:
            lidar_depth_ref = viewpoint_cam_base.depth_gt
        
        progress_bar = tqdm(range(optim_opt.iterations),
                            desc="Training progress (pose)")
        for iteration in range(1, optim_opt.iterations+1):
            # Update learning rate
            self.gs_render_local.gaussians.update_learning_rate(iteration)
            
            # 获取当前优化的位姿（保持梯度）
            current_pose_c2w = self.gs_render_local.gaussians.get_RT()  # C2W
            # 转换为W2C用于相机
            current_pose_w2c = torch.inverse(current_pose_c2w)
            
            # 构建 world_view_transform (4x4矩阵) 并保持梯度
            world_view_transform = current_pose_w2c.transpose(0, 1).cuda()
            viewpoint_cam_base.world_view_transform = world_view_transform
            viewpoint_cam_base.full_proj_transform = (
                world_view_transform.unsqueeze(0).bmm(
                    viewpoint_cam_base.projection_matrix.unsqueeze(0)
                )
            ).squeeze(0)
            viewpoint_cam_base.camera_center = current_pose_c2w[:3, 3]  # C2W的平移部分就是相机中心
            
            loss_dict, rend_dict_ref, psnr_train = self.train_step(self.gs_render_local,
                                                              viewpoint_cam_base, iteration,
                                                              pipe, optim_opt,
                                                              depth_gt=lidar_depth_ref,  # 使用预加载的LiDAR深度
                                                              densify=False,
                                                              )
            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}",
                                          "edge_loss" : f"{loss_dict.get('edge',0):.4f}",
                                          "Number points": f"{self.gs_render_local.gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)
            if iteration == optim_opt.iterations:
                progress_bar.close()

        # self.visualize(rend_dict_ref, "vis/render_optim.png",
        #                gt_image=viewpoint_cam_ref.original_image.cuda(),
        #                gt_depth=self.mono_depth[view_idx_prev])
        local_model_params = self.gs_render_local.gaussians.capture()

        # pcd under view_idx_prev frame
        pcd = self.gs_render_local.gaussians._xyz.detach()
        rel_pose = self.gs_render_local.gaussians.get_RT().detach()
        pose = rel_pose @ self.gs_render.gaussians.get_RT(
            view_idx_prev).detach()
        self.gs_render.gaussians.update_RT_seq(pose, view_idx)
        # 记录窗口位姿先验快照 (Eq.19 位姿正则)
        self.pose_priors[view_idx] = pose.detach().clone()

        self.gs_render.gaussians.rotate_seq = False
        pipe.convert_SHs_python = self.gs_render.gaussians.rotate_seq

        if self.just_reset:
            num_iterations = 500
            self.just_reset = False
            for iteration in range(1, num_iterations):
                fidx = randint(0, view_idx_prev)
                self.global_iteration += 1
                self.gs_render.gaussians.update_learning_rate(
                    self.global_iteration)
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=self.gs_render.gaussians.get_RT(
                                                            fidx).detach().cpu(),
                                                        load_depth=True)
                # 使用LiDAR深度监督
                lidar_depth_reset = None
                if self.use_lidar and hasattr(viewpoint_cam, 'depth_gt') and viewpoint_cam.depth_gt is not None:
                    lidar_depth_reset = viewpoint_cam.depth_gt
                    
                loss, rend_dict_ref, psnr_train = self.train_step(self.gs_render,
                                                                  viewpoint_cam,
                                                                  self.global_iteration,
                                                                  pipe, self.optim_cfg,
                                                                  update_gaussians=True,
                                                                  update_cam=False,
                                                                  depth_gt=lidar_depth_reset,  # 启用LiDAR深度监督
                                                                  update_distort=False,
                                                                  )

        num_iterations = self.single_step
        if max(view_idx, view_idx_prev) > min(int(self.seq_len * 0.8), self.seq_len-5):
            num_iterations = 1000
        elif min(view_idx, view_idx_prev) < int(self.single_step // 100):
            num_iterations = 100

        progress_bar = tqdm(range(num_iterations), desc="Training progress")

        for iteration in range(1, num_iterations+1):

            # coming soon: recency-weighted frame sampling (Eq.32);
            # uniform sampling over the window is used in this release
            K = max(view_idx, 1)
            fidx = randint(1, K)

            self.global_iteration += 1
            if self.gs_render.gaussians.rotate_seq:
                self.gs_render.gaussians.set_seq_idx(fidx)
            viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                    pose=self.gs_render.gaussians.get_RT(
                                                        fidx).detach().cpu()
                                                    if not self.gs_render.gaussians.rotate_seq
                                                    else None,
                                                    load_depth=True)
            # Update learning rate
            self.gs_render.gaussians.update_learning_rate(
                self.global_iteration)

            # 使用LiDAR深度监督进行全局优化
            lidar_depth_global = None
            if self.use_lidar and hasattr(viewpoint_cam, 'depth_gt') and viewpoint_cam.depth_gt is not None:
                lidar_depth_global = viewpoint_cam.depth_gt
                
            loss_dict, rend_dict_ref, psnr_train = self.train_step(self.gs_render,
                                                              viewpoint_cam,
                                                              self.global_iteration,
                                                              pipe, self.optim_cfg,
                                                              update_gaussians=True,
                                                              update_cam=False,
                                                              depth_gt=lidar_depth_global,  # 启用LiDAR深度监督
                                                              update_distort=self.pipe_cfg.distortion,
                                                              )

            if self.global_iteration % 1000 == 0:
                self.gs_render.gaussians.oneupSHdegree()

            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}",
                                          "edge_loss" : f"{loss_dict.get('edge',0):.4f}",
                                          "Number points": f"{self.gs_render.gaussians.get_xyz.shape[0]}"})
                progress_bar.update(10)

            if iteration == num_iterations:
                progress_bar.close()

        # 保存局部模型的渲染结果到 train 文件夹
        with torch.no_grad():
            # 使用局部渲染器渲染 view_idx_prev 帧
            viewpoint_cam_local = self.load_viewpoint_cam(view_idx_prev,
                                                         pose=self.gs_render.gaussians.get_RT(view_idx_prev).detach().cpu()
                                                         if not self.gs_render.gaussians.rotate_seq else None)
            render_dict_local = self.gs_render_local.render(viewpoint_cam_local,
                                                           compute_cov3D_python=pipe.compute_cov3D_python,
                                                           convert_SHs_python=pipe.convert_SHs_python)
            gt_image_local = viewpoint_cam_local.original_image.cuda()
            psnr_local = psnr(render_dict_local["image"], gt_image_local).mean().double()
            self.visualize(render_dict_local,
                          f"{result_path}/train/local_{view_idx:03d}_from_{view_idx_prev:03d}_psnr{psnr_local:.2f}.png",
                          gt_image=gt_image_local, save_ply=False)

        return pcd, local_model_params

    def create_pcd_from_render(self, render_dict, viewpoint_cam):
        intrinsics = torch.from_numpy(viewpoint_cam.intrinsics).float().cuda()
        depth = render_dict["depth"].squeeze()
        image = render_dict["image"]
        pts = depth_to_3d(depth[None, None],
                          intrinsics[None],
                          normalize_points=False)
        points = pts.squeeze().permute(1, 2, 0).detach().cpu().reshape(-1, 3).numpy()
        colors = image.permute(1, 2, 0).detach().cpu().reshape(-1, 3).numpy()
        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(colors)
        pcd_data = pcd_data.farthest_point_down_sample(num_samples=30_000)
        colors = np.asarray(pcd_data.colors, dtype=np.float32)
        points = np.asarray(pcd_data.points, dtype=np.float32)
        normals = np.asarray(pcd_data.normals, dtype=np.float32)
        pcd = BasicPointCloud(points, colors, normals)
        return pcd

    def train_from_progressive(self, ):
        pipe = copy(self.pipe_cfg)
        self.single_step = 500 # 300 for faster training; 500 for better results

        num_iterations = self.single_step * (self.seq_len // 10) * 10
        self.optim_cfg.iterations = num_iterations
        self.optim_cfg.position_lr_max_steps = num_iterations
        self.optim_cfg.opacity_reset_interval = num_iterations // 10
        self.optim_cfg.densify_until_iter = num_iterations
        self.optim_cfg.reset_until_iter = int(num_iterations * 0.8)
        self.optim_cfg.densify_from_iter = 1000
        self.optim_cfg.densify_from_iter = self.single_step


        if pipe.expname == "":
            expname = "progressive"
        else:
            expname = pipe.expname
        pipe.convert_SHs_python = True
        optim_opt = copy(self.optim_cfg)
        result_path = f"output/{expname}/{self.category}_{self.seq_name}"
        os.makedirs(result_path, exist_ok=True)

        pose_dict = dict()
        poses_gt = []
        for seq_data in self.data:
            try:
                R = seq_data.R.transpose()
                t = seq_data.T
            except:
                R = np.eye(3)
                t = np.zeros(3)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            poses_gt.append(torch.from_numpy(pose))
        pose_dict["poses_gt"] = torch.stack(poses_gt)
        max_frame = self.seq_len
        start_frame = 1
        end_frame = max_frame



        os.makedirs(f"{result_path}/pose", exist_ok=True)
        os.makedirs(f"{result_path}/mesh", exist_ok=True)

        num_eppch = 1
        reverse = False
        for epoch in range(num_eppch):
            gauss_params = self.init_two_view(
                0, end_frame, pipe, copy(self.optim_cfg))
            
            self.global_iteration = 0
            optim_opt = copy(self.optim_cfg)
            self.gs_render.gaussians.rotate_seq = True
            self.gs_render.gaussians.training_setup(self.optim_cfg,
                                                    fit_pose=True,)
            self.match_results = OrderedDict()
            for fidx in range(start_frame, end_frame):
                # 关键帧选择: 空间新颖度 G_exp (Eq.30)
                is_keyframe = self.check_keyframe(fidx)
                pcd_new, local_gauss_params = self.add_view_v2(
                    fidx, fidx-1, is_keyframe=is_keyframe)
                self.gs_render.gaussians.rotate_seq = False
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=self.gs_render.gaussians.get_RT(
                                                            fidx).detach().cpu(),
                                                        )
                render_dict = self.gs_render.render(viewpoint_cam,
                                                    compute_cov3D_python=pipe.compute_cov3D_python,
                                                    convert_SHs_python=pipe.convert_SHs_python)
                gt_image = viewpoint_cam.original_image.cuda()
                psnr_train = psnr(render_dict["image"],
                                    gt_image).mean().double()
                print(
                    'Frames {:03d}/{:03d}, PSNR : {:.03f}'.format(fidx, self.seq_len-1, psnr_train))
                # 保存全局模型渐进结果到 progress 文件夹
                os.makedirs(f"{result_path}/progress", exist_ok=True)
                self.visualize(render_dict,
                                f"{result_path}/progress/global_{self.global_iteration:06d}_{fidx:03d}.png",
                                gt_image=gt_image, save_ply=False)

            with torch.no_grad():
                psnr_test = 0.0
                pose_dict["poses_pred"] = []
                self.render_depth = OrderedDict()
                self.gs_render.gaussians.rotate_seq = False
                self.gs_render.gaussians.rotate_xyz = False

                for val_idx in range(end_frame):
                    viewpoint_cam = self.load_viewpoint_cam(val_idx,
                                                            pose=self.gs_render.gaussians.get_RT(
                                                                val_idx).detach().cpu(),
                                                            )
                    render_dict = self.gs_render.render(viewpoint_cam,
                                                        compute_cov3D_python=pipe.compute_cov3D_python,
                                                        convert_SHs_python=pipe.convert_SHs_python)
                    self.render_depth[val_idx] = render_dict["depth"]
                    gt_image = viewpoint_cam.original_image.cuda()
                    psnr_test += psnr(render_dict["image"],
                                        gt_image).mean().double()
                    self.visualize(render_dict,
                                    f"{result_path}/eval/ep{epoch:02d}_{self.global_iteration:06d}_{val_idx:03d}.png",
                                    gt_image=gt_image, save_ply=False)
                print('Number of {:03d} to {:03d} frames: PSNR : {:.03f}'.format(
                    start_frame,
                    end_frame,
                    psnr_test / (end_frame)))

                for idx in range(self.seq_len):
                    pose = self.gs_render.gaussians.get_RT(idx)
                    pose_dict["poses_pred"].append(pose.detach().cpu())

            pose_dict["poses_pred"] = torch.stack(pose_dict["poses_pred"])
            pose_dict["poses_gt"] = torch.stack(poses_gt)
            pose_dict["match_results"] = self.match_results
            
            # 保存优化后的LiDAR外参
            if self.use_lidar:
                lidar_calib_path = f"{result_path}/lidar_calibration.txt"
                self.save_optimized_extrinsic(lidar_calib_path)
                
                # 在pose_dict中也保存一份
                lidar2cam = self.get_current_lidar2cam()
                if isinstance(lidar2cam, torch.Tensor):
                    lidar2cam = lidar2cam.detach().cpu().numpy()
                pose_dict["lidar2cam_optimized"] = lidar2cam
                
                # 导入并保存初始外参
                from utils.lidar_loader import get_lidar2cam_gt
                pose_dict["lidar2cam_init"] = get_lidar2cam_gt()
            
            torch.save(
                pose_dict, f"{result_path}/pose/ep{epoch:02d}_init.pth")
            os.makedirs(f"{result_path}/chkpnt", exist_ok=True)
            torch.save(self.gs_render.gaussians.capture(),
                        f"{result_path}/chkpnt/ep{epoch:02d}_init.pth")
            
            # 保存全局点云
            if self.use_lidar and self.save_global_pcd:
                global_pcd_path = f"{result_path}/global_point_cloud.ply"
                self.save_global_point_cloud(global_pcd_path, voxel_size=0.05)



    def eval_nvs(self, ):
        pipe = copy(self.pipe_cfg)
        optim_opt = copy(self.optim_cfg)
        num_epochs = 200
        num_iterations = num_epochs * self.seq_len
        optim_opt.iterations = num_iterations
        optim_opt.position_lr_max_steps = num_iterations
        optim_opt.densify_until_iter = num_iterations // 2
        optim_opt.reset_until_iter = num_iterations // 2
        optim_opt.opacity_reset_interval = num_iterations // 10
        optim_opt.densification_interval = 100
        optim_opt.densify_from_iter = 500
        # self.optim_cfg.densification_interval = 100

        if pipe.expname == "":
            expname = "progressive"
        else:
            expname = pipe.expname
        pipe.convert_SHs_python = True
        optim_opt = copy(self.optim_cfg)
        # result_path = f"vis/{expname}/{self.category}_{self.seq_name}"
        result_path = os.path.dirname(
            self.model_cfg.model_path).replace('chkpnt', 'test')
        os.makedirs(result_path, exist_ok=True)

        pose_dict = dict()
        pose_dict["poses_gt"] = []
        for seq_data in self.data:
            try:
                R = seq_data.R.transpose()
                t = seq_data.T
            except:
                R = np.eye(3)
                t = np.zeros(3)
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = t
            pose_dict["poses_gt"].append(torch.from_numpy(pose))

        max_frame = self.seq_len
        start_frame = 0
        end_frame = max_frame
        if self.model_cfg.model_path != "":
            self.gs_render.gaussians.restore(
                torch.load(self.model_cfg.model_path), self.optim_cfg)
            pose_dict_train = torch.load(
                self.model_cfg.model_path.replace('chkpnt', 'pose'))
            self.gs_render.gaussians.rotate_seq = True

        sample_rate = 2 if "Family" in result_path else 8
        pose_test_init = pose_dict_train['poses_pred'][int(
            sample_rate/2)::sample_rate-1][:max_frame]
        self.gs_render.gaussians.init_RT_seq(
            self.seq_len, pose_test_init.float())
        self.gs_render.gaussians.rotate_seq = True
        self.gs_render.gaussians.training_setup(optim_opt,
                                                fix_pos=True,
                                                fix_feat=True,
                                                fit_pose=True,)
        progress_bar = tqdm(range(num_iterations),
                            desc="Training progress")

        iteration = 0
        for epoch in range(num_epochs):
            for fidx in range(self.seq_len):
                iteration += 1
                self.gs_render.gaussians.rotate_seq = True
                self.gs_render.gaussians.set_seq_idx(fidx)
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=None,
                                                        load_depth=True,
                                                        )
                # self.gs_render.gaussians.update_learning_rate_camera(
                #     fidx, iteration)
                loss_dict, rend_dict, psnr_train = self.train_step(self.gs_render,
                                                                   viewpoint_cam,
                                                                   iteration, pipe, optim_opt,
                                                                   densify=False,
                                                                   depth_gt=None,
                                                                   update_cam=True,
                                                                   update_gaussians=False,
                                                                   reset=False,
                                                                   )
                if iteration % 10 == 0:
                    progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}"})
                    progress_bar.update(10)
                if iteration == optim_opt.iterations:
                    progress_bar.close()

        psnr_test = 0
        ssim_test = 0
        lpips_test = 0
        with torch.no_grad():
            for fidx in range(self.seq_len):
                self.gs_render.gaussians.rotate_seq = False
                viewpoint_cam = self.load_viewpoint_cam(fidx,
                                                        pose=self.gs_render.gaussians.get_RT(
                                                            fidx).detach().cpu(),
                                                        load_depth=True,
                                                        )

                render_dict = self.gs_render.render(viewpoint_cam,
                                                    compute_cov3D_python=False,
                                                    convert_SHs_python=False)
                gt_image = viewpoint_cam.original_image.cuda()
                psnr_test += psnr(render_dict["image"],
                                  gt_image).mean().double()
                ssim_test += ssim(render_dict["image"],
                                  gt_image).mean().double()
                lpips_test += lpips(render_dict["image"],
                                    gt_image, net_type="vgg").mean().double()
                self.visualize(render_dict,
                               f"{result_path}/test/{fidx:04d}.png",
                               gt_image=gt_image, save_ply=False)
        with open(f"{result_path}/test.txt", 'w') as f:
            f.write('PSNR : {:.03f}, SSIM : {:.03f}, LPIPS : {:.03f}'.format(
                    psnr_test / end_frame,
                    ssim_test / end_frame,
                    lpips_test / end_frame))
            f.close()

        print('Number of {:03d} to {:03d} frames: PSNR : {:.03f}, SSIM : {:.03f}, LPIPS : {:.03f}'.format(
            start_frame,
            end_frame,
            psnr_test / end_frame,
            ssim_test / end_frame,
            lpips_test / end_frame))

    def eval_pose(self, ):
        pipe = copy(self.pipe_cfg)
        optim_opt = copy(self.optim_cfg)
        result_path = os.path.dirname(
            self.model_cfg.model_path).replace('chkpnt', 'pose')
        os.makedirs(result_path, exist_ok=True)
        pose_path = os.path.join(result_path, 'ep00_init.pth')
        poses = torch.load(pose_path)
        poses_pred = poses['poses_pred'].inverse().cpu()
        poses_gt_c2w = poses['poses_gt'].inverse().cpu()
        poses_gt = poses_gt_c2w[:len(poses_pred)].clone()
        # align scale first (we do this because scale differennt a lot)
        trans_gt_align, trans_est_align, _ = self.align_pose(poses_gt[:, :3, -1].numpy(),
                                                             poses_pred[:, :3, -1].numpy())
        poses_gt[:, :3, -1] = torch.from_numpy(trans_gt_align)
        poses_pred[:, :3, -1] = torch.from_numpy(trans_est_align)

        c2ws_est_aligned = align_ate_c2b_use_a2b(poses_pred, poses_gt)
        ate = compute_ATE(poses_gt.cpu().numpy(),
                          c2ws_est_aligned.cpu().numpy())
        rpe_trans, rpe_rot = compute_rpe(
            poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
        print("{0:.3f}".format(rpe_trans*100),
              '&' "{0:.3f}".format(rpe_rot * 180 / np.pi),
              '&', "{0:.3f}".format(ate))
        plot_pose(poses_gt.cpu().numpy(), c2ws_est_aligned.cpu().numpy(), pose_path)
        with open(f"{result_path}/pose_eval.txt", 'w') as f:
            f.write("RPE_trans: {:.03f}, RPE_rot: {:.03f}, ATE: {:.03f}".format(
                rpe_trans*100,
                rpe_rot * 180 / np.pi,
                ate))
            f.close()

    def align_pose(self, pose1, pose2):
        mtx1 = np.array(pose1, dtype=np.double, copy=True)
        mtx2 = np.array(pose2, dtype=np.double, copy=True)

        if mtx1.ndim != 2 or mtx2.ndim != 2:
            raise ValueError("Input matrices must be two-dimensional")
        if mtx1.shape != mtx2.shape:
            raise ValueError("Input matrices must be of same shape")
        if mtx1.size == 0:
            raise ValueError("Input matrices must be >0 rows and >0 cols")

        # translate all the data to the origin
        mtx1 -= np.mean(mtx1, 0)
        mtx2 -= np.mean(mtx2, 0)

        norm1 = np.linalg.norm(mtx1)
        norm2 = np.linalg.norm(mtx2)

        if norm1 == 0 or norm2 == 0:
            raise ValueError("Input matrices must contain >1 unique points")

        # change scaling of data (in rows) such that trace(mtx*mtx') = 1
        mtx1 /= norm1
        mtx2 /= norm2

        # transform mtx2 to minimize disparity
        R, s = scipy.linalg.orthogonal_procrustes(mtx1, mtx2)
        mtx2 = mtx2 * s

        return mtx1, mtx2, R

    def render_nvs(self, traj_opt='bspline', N_novel_imgs=120, degree=100):
        result_path = os.path.dirname(
            self.model_cfg.model_path).replace('chkpnt', 'nvs')
        os.makedirs(result_path, exist_ok=True)
        self.gs_render.gaussians.restore(
            torch.load(self.model_cfg.model_path), self.optim_cfg)
        pose_dict_train = torch.load(
            self.model_cfg.model_path.replace('chkpnt', 'pose'))
        poses_pred_w2c_train = pose_dict_train['poses_pred'].cpu()
        if traj_opt == 'bspline':
            i_train = self.i_train
            if "co3d" in self.model_cfg.source_path:
                poses_pred_w2c_train = poses_pred_w2c_train[:100]
                i_train = self.i_train[:100]
            c2ws = interp_poses_bspline(poses_pred_w2c_train.inverse(), N_novel_imgs,
                                        i_train, degree)
            w2cs = c2ws.inverse()

        self.gs_render.gaussians.rotate_seq = False
        render_dir = f"{result_path}/{traj_opt}"
        os.makedirs(render_dir, exist_ok=True)
        for fidx, pose in enumerate(w2cs):
            viewpoint_cam = self.load_viewpoint_cam(10,
                                                    pose=pose,
                                                    )
            render_dict = self.gs_render.render(viewpoint_cam,
                                                compute_cov3D_python=False,
                                                convert_SHs_python=False)
            self.visualize(render_dict,
                           f"{render_dir}/img_out/{fidx:04d}.png",
                           save_ply=False)

        imgs = []
        for img in sorted(glob.glob(os.path.join(render_dir, "img_out", "*.png"))):
            if "depth" in img:
                continue
            rgb = cv2.imread(img)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            depth = cv2.imread(img.replace(".png", "_depth.png"))
            depth = cv2.cvtColor(depth, cv2.COLOR_BGR2RGB)
            rgb = np.hstack([rgb, depth])
            imgs.append(rgb)

        imgs = np.stack(imgs, axis=0)

        video_out_dir = os.path.join(render_dir, 'video_out')
        if not os.path.exists(video_out_dir):
            os.makedirs(video_out_dir)
        imageio.mimwrite(os.path.join(
            video_out_dir, f'{self.category}_{self.seq_name}_ours.mp4'), imgs, fps=30, quality=9)

    def save_model(self, epoch):
        pass

    def compute_loss(self,
                     render_dict,
                     viewpoint_cam,
                     pipe_opt,
                     iteration,
                     use_reproject=False,
                     use_matcher=False,
                     ref_fidx=None,
                     **kwargs):
        import torch.nn.functional as F
        import kornia
        
        # 根据是否传入 depth_gt 决定是否使用几何约束损失
        # 如果没有 depth_gt，使用 legacy 损失（L1 + SSIM）更稳定
        use_geometric = kwargs.get('depth_gt') is not None

        # ECU-GS 不确定度感知联合损失 (Eq.29), 需要稀疏LiDAR深度
        if self.ecugs_joint_loss is not None and \
                getattr(viewpoint_cam, 'lidar_depth_sparse', None) is not None:
            return self._compute_loss_ecugs(render_dict, viewpoint_cam,
                                            pipe_opt, iteration, **kwargs)

        if GEOMETRIC_CONSTRAINTS_AVAILABLE and self.geometric_loss is not None and use_geometric:
            return self._compute_loss_geometric(render_dict, viewpoint_cam,
                                               pipe_opt, iteration, **kwargs)
        else:
            return self._compute_loss_legacy(render_dict, viewpoint_cam,
                                            pipe_opt, iteration, **kwargs)

    def _compute_loss_ecugs(self,
                            render_dict,
                            viewpoint_cam,
                            pipe_opt,
                            iteration,
                            **kwargs):
        image = render_dict.get("image")
        gt_image = viewpoint_cam.original_image.cuda()
        rendered_depth = render_dict.get("depth", None)
        if rendered_depth is not None:
            rendered_depth = rendered_depth.squeeze()

        lidar_depth = viewpoint_cam.lidar_depth_sparse.cuda().squeeze()
        valid_mask = viewpoint_cam.lidar_valid_mask.cuda().squeeze()
        lidar_count = viewpoint_cam.lidar_count.cuda().squeeze()

        # 外参不确定度沿深度传播 (Eq.17/18), 生成逐像素 σ²_ext
        sigma_ext2 = None
        pts_cam = getattr(viewpoint_cam, 'lidar_points_cam', None)
        if self.uncertainty_estimator is not None and pts_cam is not None \
                and len(pts_cam) > 0:
            var_pts = self.uncertainty_estimator.propagate_depth_uncertainty(pts_cam)
            H, W = lidar_depth.shape
            K = torch.from_numpy(viewpoint_cam.intrinsics).float().cuda()
            z = pts_cam[:, 2].clamp_min(1e-6)
            u = torch.round(K[0, 0] * pts_cam[:, 0] / z + K[0, 2]).long()
            v = torch.round(K[1, 1] * pts_cam[:, 1] / z + K[1, 2]).long()
            in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            sigma_ext2 = torch.zeros(H, W, device='cuda')
            cnt = torch.zeros(H, W, device='cuda')
            lin = (v[in_img] * W + u[in_img])
            sigma_ext2.view(-1).index_add_(0, lin, var_pts[in_img].cuda())
            cnt.view(-1).index_add_(0, lin, torch.ones_like(lin, dtype=torch.float))
            sigma_ext2 = sigma_ext2 / cnt.clamp_min(1.0)
            viewpoint_cam.sigma_ext2 = sigma_ext2

        # 定期更新外参协方差 (Eq.17): J 为深度投影雅可比, W 由 MAD 残差尺度加权
        if self.uncertainty_estimator is not None and pts_cam is not None \
                and len(pts_cam) > 0 and rendered_depth is not None:
            self._frames_since_ext_cov += 1
            if self._frames_since_ext_cov >= self.extrinsic_update_interval:
                self._frames_since_ext_cov = 0
                with torch.no_grad():
                    res = (rendered_depth[valid_mask] - lidar_depth[valid_mask]).abs()
                    sigma_hat = self.uncertainty_estimator.mad_scale(res)
                    J = self.uncertainty_estimator.depth_jacobian_wrt_extrinsic(pts_cam)
                    w = torch.full((J.shape[0],),
                                   1.0 / (J.shape[0] * sigma_hat ** 2),
                                   dtype=torch.float64, device=J.device)
                    median_range = self.uncertainty_estimator.median_range(pts_cam)
                    self.uncertainty_estimator.update_extrinsic_covariance(
                        [J], [w], median_range)

        # 位姿-外参正则 (Eq.19): 位姿相对窗口先验, 外参相对初始标定弱先验
        pose_reg_kwargs = None
        fidx = getattr(viewpoint_cam, 'uid', None)
        if fidx is not None and fidx in self.pose_priors:
            cur_pose = self.gs_render.gaussians.get_RT(fidx)
            prior_pose = self.pose_priors[fidx]
            omega = torch.ones(1, 6, device='cuda')
            pose_reg_kwargs = dict(
                cur_poses=cur_pose.reshape(1, 4, 4),
                prior_poses=prior_pose.reshape(1, 4, 4).cuda(),
                omega_poses=omega,
            )
            if self.optimize_lidar_extrinsic and self.lidar_calibrator is not None:
                from utils.lidar_loader import get_lidar2cam_gt
                init_ext = torch.from_numpy(get_lidar2cam_gt()).float().cuda()
                pose_reg_kwargs.update(
                    cur_extrinsic=self.lidar_calibrator.get_extrinsic().float(),
                    prior_extrinsic=init_ext,
                    omega_extrinsic=torch.full((6,), 0.1, device='cuda'),
                )

        intrinsics = None
        if self.use_normal_consistency and hasattr(viewpoint_cam, 'intrinsics'):
            K = viewpoint_cam.intrinsics
            intrinsics = (float(K[0, 0]), float(K[1, 1]),
                          float(K[0, 2]), float(K[1, 2]))

        total_loss, loss_dict = self.ecugs_joint_loss(
            image, gt_image,
            render_depth=rendered_depth,
            lidar_depth=lidar_depth,
            valid_mask=valid_mask,
            sigma_ext2=sigma_ext2,
            lidar_count=lidar_count,
            intrinsics=intrinsics,
            pose_reg_kwargs=pose_reg_kwargs,
        )

        # 三阶段调度: coarse 阶段几何/标定约束主导, final 阶段冻结外参
        if self._current_stage == 'coarse':
            total_loss = loss_dict['loss_rgb'] + 2.0 * (total_loss - loss_dict['loss_rgb'])
            loss_dict['loss'] = total_loss

        # 标定精化损失 (粗阶段主导项之一)
        if self.optimize_lidar_extrinsic and self.calibration_loss is not None \
                and self._current_stage != 'final':
            lidar2cam = self.get_current_lidar2cam()
            if getattr(viewpoint_cam, 'lidar_points', None) is not None:
                lidar_points = torch.from_numpy(viewpoint_cam.lidar_points).cuda()
                intr_mat = torch.from_numpy(viewpoint_cam.intrinsics).cuda()
                calib_loss, calib_dict = self.calibration_loss(
                    lidar_points=lidar_points,
                    rendered_depth=rendered_depth,
                    extrinsic=lidar2cam,
                    intrinsics=intr_mat
                )
                calib_weight = 0.1 if self._current_stage != 'coarse' else 0.2
                total_loss = total_loss + calib_weight * calib_loss
                loss_dict.update({f'calib_{k}': v for k, v in calib_dict.items()})

        loss_dict['loss'] = total_loss
        loss_dict['depth'] = float(loss_dict.get('loss_depth', 0.0))
        loss_dict['edge'] = float(loss_dict.get('loss_edge', 0.0))
        return loss_dict
    
    def _compute_loss_geometric(self,
                                 render_dict,
                                 viewpoint_cam,
                                 pipe_opt,
                                 iteration,
                                 **kwargs):
        """
        使用几何约束的新损失计算 (TLC-Calib + 3DGS-Calib)
        """
        image = render_dict.get("image")
        gt_image = viewpoint_cam.original_image.cuda() if hasattr(viewpoint_cam, 'original_image') else None
        
        rendered_depth = render_dict.get("depth", None)
        if rendered_depth is not None:
            rendered_depth = rendered_depth.squeeze()
            
        # 优先使用传入的 depth_gt，否则使用 viewpoint_cam.depth_gt
        gt_depth = kwargs.get('depth_gt', None)
        if gt_depth is None:
            gt_depth = getattr(viewpoint_cam, 'depth_gt', None)
        depth_mask = None
        if gt_depth is not None:
            depth_mask = (gt_depth > 0) & (gt_depth < 10.0)
            
        # 获取高斯尺度和可见性
        gaussian_scales = None
        visibility_mask = render_dict.get("visibility_filter", None)
        if hasattr(self.gs_render, 'gaussians'):
            gaussian_scales = self.gs_render.gaussians.get_scaling
            
        # 使用ECUGSPhotometricGeometricLoss
        total_loss, loss_dict = self.geometric_loss(
            rendered_image=image,
            gt_image=gt_image,
            rendered_depth=rendered_depth,
            gt_depth=gt_depth,
            depth_mask=depth_mask,
            gaussian_scales=gaussian_scales,
            visibility_mask=visibility_mask
        )
        
        # 添加标定精化损失
        if self.optimize_lidar_extrinsic and self.calibration_loss is not None:
            # 获取当前外参
            lidar2cam = self.get_current_lidar2cam()
            
            # 如果有LiDAR点云数据
            if hasattr(viewpoint_cam, 'lidar_points') and viewpoint_cam.lidar_points is not None:
                lidar_points = torch.from_numpy(viewpoint_cam.lidar_points).cuda()
                intrinsics = torch.from_numpy(viewpoint_cam.intrinsics).cuda()
                
                calib_loss, calib_dict = self.calibration_loss(
                    lidar_points=lidar_points,
                    rendered_depth=rendered_depth,
                    extrinsic=lidar2cam,
                    intrinsics=intrinsics
                )
                
                total_loss = total_loss + 0.1 * calib_loss
                loss_dict.update({f'calib_{k}': v for k, v in calib_dict.items()})
                
        # 更新总损失值
        loss_dict['loss'] = total_loss
        
        return loss_dict
    
    def _compute_loss_legacy(self,
                            render_dict,
                            viewpoint_cam,
                            pipe_opt,
                            iteration,
                            **kwargs):
        """原始损失计算方法 (保持兼容性)"""
        import torch.nn.functional as F
        import kornia
        loss = 0.0
        loss_dict = {}
        if "image" in render_dict:
            image = render_dict["image"]
            gt_image = viewpoint_cam.original_image.cuda()
            loss_dict = self.loss_func(image, gt_image, **kwargs)
            loss += loss_dict['loss']
        # 优先使用传入的 depth_gt
        depth_gt_source = kwargs.get('depth_gt', None)
        if depth_gt_source is None and hasattr(viewpoint_cam, 'depth_gt'):
            depth_gt_source = viewpoint_cam.depth_gt
            
        if "depth" in render_dict and depth_gt_source is not None:
            lidar_depth = depth_gt_source
            ren_depth   = render_dict["depth"].squeeze()
            mask = (lidar_depth > 0) & (lidar_depth < 10.0)
            if mask.sum() > 0:
                # 密度不确定加权
                if hasattr(viewpoint_cam, 'density_map') and viewpoint_cam.density_map is not None:
                    sigma = 1.0 / (viewpoint_cam.density_map + 1e-3)
                    w = torch.exp(-((ren_depth - lidar_depth) ** 2) / (2.0 * sigma ** 2))
                else:
                    w = torch.ones_like(ren_depth)
                loss_depth = (w[mask] * F.l1_loss(ren_depth[mask], lidar_depth[mask], reduction='none')).mean()
                loss = loss + loss_depth
                loss_dict['depth'] = loss_depth.item()

                # 边缘损失
                edge_gt   = kornia.filters.sobel(lidar_depth[None, None])[0, 0]
                edge_pred = kornia.filters.sobel(ren_depth[None, None])[0, 0]
                loss_edge = F.l1_loss(edge_pred[mask], edge_gt[mask]) * 0.5
                loss = loss + loss_edge
                loss_dict['edge'] = loss_edge.item()
            else:
                loss_dict['depth'] = 0.0
                loss_dict['edge'] = 0.0
                
            # LiDAR外参正则化
            if self.optimize_lidar_extrinsic and self.lidar_calibrator is not None:
                current_extrinsic = self.lidar_calibrator.get_extrinsic()
                from utils.lidar_loader import get_lidar2cam_gt
                init_extrinsic = torch.from_numpy(get_lidar2cam_gt()).cuda()
                R_diff = current_extrinsic[:3, :3] - init_extrinsic[:3, :3]
                R_loss = R_diff.pow(2).mean()
                t_diff = current_extrinsic[:3, 3] - init_extrinsic[:3, 3]
                t_loss = t_diff.pow(2).mean()
                lidar_reg_loss = 0.01 * R_loss + 0.1 * t_loss
                loss = loss + lidar_reg_loss
                loss_dict['lidar_reg'] = lidar_reg_loss.item()
        else:
            loss_dict['edge'] = 0.0
            
        loss_dict['loss'] = loss
        return loss_dict

    def visualize(self, render_pkg, filename, gt_image=None, gt_depth=None, save_ply=False):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        if "depth" in render_pkg:
            rend_depth = Image.fromarray(
                colorize(render_pkg["depth"].detach().cpu().numpy(),
                         cmap='magma_r')).convert("RGB")
            if gt_depth is not None:
                gt_depth = Image.fromarray(
                    colorize(gt_depth.detach().cpu().numpy(),
                             cmap='magma_r')).convert("RGB")
                rend_depth = Image.fromarray(np.hstack([np.asarray(gt_depth),
                                                        np.asarray(rend_depth)]))
            rend_depth.save(filename.replace(".png", "_depth.png"))
        if "acc" in render_pkg:
            rend_acc = Image.fromarray(
                colorize(render_pkg["acc"].detach().cpu().numpy(),
                         cmap='magma_r')).convert("RGB")
            rend_acc.save(filename.replace(".png", "_acc.png"))

        rend_img = Image.fromarray(
            np.asarray(render_pkg["image"].detach().cpu().permute(1, 2, 0).numpy()
                       * 255.0, dtype=np.uint8)).convert("RGB")
        if gt_image is not None:
            gt_image = Image.fromarray(
                np.asarray(
                    gt_image.permute(1, 2, 0).cpu().numpy() * 255.0,
                    dtype=np.uint8)).convert("RGB")
            rend_img = Image.fromarray(np.hstack([np.asarray(gt_image),
                                                  np.asarray(rend_img)]))
        rend_img.save(filename)

        if save_ply:
            points = self.gs_render.gaussians._xyz.detach().cpu().numpy()
            pcd_data = o3d.geometry.PointCloud()
            pcd_data.points = o3d.utility.Vector3dVector(points)
            pcd_data.colors = o3d.utility.Vector3dVector(np.ones_like(points))
            o3d.io.write_point_cloud(
                filename.replace('.png', '.ply'), pcd_data)

    
    def construct_point(self, gs_model, poses, iteration, result_path, stop_frame=-1):
        volume = o3d.pipelines.integration.ScalableTSDFVolume(
            voxel_length=0.01,
            sdf_trunc=3 * 0.01,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8
        )
        pipe = self.pipe_cfg
        optim_opt = self.optim_cfg
        pipe.convert_SHs_python = True
        if poses is None:
            poses = torch.stack(
                [gs_model.gaussians.get_RT(idx).detach().cpu()
                 for idx in range(self.seq_len)])

        self.gs_render.gaussians.rotate_seq = False
        stop_frame = len(poses) if stop_frame == -1 else stop_frame
        with torch.no_grad():
            progress_bar = tqdm(range(self.seq_len),
                                desc="Reconstructing point cloud")
            for idx in range(len(poses)):
                if idx > stop_frame:
                    break

                viewpoint_cam = self.load_viewpoint_cam(
                    idx, pose=poses[idx], load_depth=True)
                # if idx not in self.render_depth:
                render_dict = gs_model.render(
                    viewpoint_cam,
                    compute_cov3D_python=pipe.compute_cov3D_python,
                    convert_SHs_python=pipe.convert_SHs_python)
                render_depth = render_dict['depth'].detach().squeeze()

                rgb = viewpoint_cam.original_image.cuda().permute(1, 2, 0).detach().cpu().numpy()
                rgb = (rgb * 255).astype(np.uint8)

                depth = render_depth.detach().cpu().numpy()


                H, W = depth.shape
                intrinsic = viewpoint_cam.intrinsics
                fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]

                rgb = o3d.geometry.Image(rgb)
                depth = o3d.geometry.Image(depth)
                rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                    rgb, depth, depth_scale=1.0, depth_trunc=10.0, convert_rgb_to_intensity=False
                )
                intrinsic = o3d.camera.PinholeCameraIntrinsic(
                    width=W, height=H, fx=fx,  fy=fy, cx=cx, cy=cy)
                # pose = self.gs_render.gaussians.get_RT(idx).detach().cpu().numpy()
                volume.integrate(rgbd, intrinsic, poses[idx])
                progress_bar.update(1)
        progress_bar.close()

        self.gs_render.gaussians.rotate_seq = True
        mesh = volume.extract_triangle_mesh()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(
            f"{result_path}/{self.gs_render.rot_type}_{iteration:06d}.ply", mesh)

        points = np.asarray(mesh.vertices, dtype=np.float32)
        colors = np.asarray(mesh.vertex_colors)
        normals = np.asarray(mesh.vertex_normals)
        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(colors)
        pcd_data.normals = o3d.utility.Vector3dVector(normals)

        pcd_data = pcd_data.voxel_down_sample(voxel_size=0.01)
        o3d.io.write_point_cloud(
            f"{result_path}/{self.gs_render.rot_type}_{iteration:06d}.ply", pcd_data)
        points = np.asarray(pcd_data.points)
        colors = np.asarray(pcd_data.colors)
        normals = np.asarray(pcd_data.normals)
        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        return pcd