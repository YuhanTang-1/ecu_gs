# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE_inria.md file.
#
# For inquiries contact  george.drettakis@inria.fr

import torch
from lietorch import SO3, SE3, Sim3, LieGroupParameter
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import matplotlib.pyplot as plt
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH, SH2RGB
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.sh_utils import eval_sh
from scipy.spatial.transform import Rotation as R
from kornia.geometry.depth import depth_to_3d
import math
from typing import Optional, List, Tuple

# Import Anchor-Auxiliary System
try:
    from .anchor_auxiliary_gaussians import (
        ECUGSAnchorAuxiliarySystem,
        ECUGSAdaptiveVoxelController,
        ECUGSAuxiliaryGaussianMLP,
        ECUGSHybridInitializer
    )
    ANCHOR_AUXILIARY_AVAILABLE = True
except ImportError:
    ANCHOR_AUXILIARY_AVAILABLE = False
    print("Warning: Anchor-Auxiliary system not available")

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from utils.camera_conversion import (matrix_to_quaternion,
                                     )


import pdb


# -------------------- 纯 torch 的 FPS --------------------
def farthest_point_sample(pts, K):
    """
    pts: (N,3) tensor
    K: int
    return: K 个索引
    """
    N, _ = pts.shape
    idx  = torch.zeros(K, dtype=torch.long, device=pts.device)
    dist = torch.ones(N, device=pts.device) * 1e8
    farthest = torch.randint(0, N, (1,)).item()
    for i in range(K):
        idx[i] = farthest
        dist[farthest] = 0
        cur = pts[farthest]
        dist_c = ((pts - cur) ** 2).sum(-1)
        dist = torch.minimum(dist, dist_c)
        farthest = torch.argmax(dist).item()
    return idx
# -------------------------------------------------------

class ECUGSGaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, view_dependent=True):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.rotate_xyz = False
        self.rotate_seq = False
        self.seq_idx = 0
        self.view_dependent = view_dependent
        self.setup_functions()


        ### >>> 1. 锚缓冲区（非 nn.Module，用裸 tensor）
        self.is_anchor   = torch.zeros(0, dtype=torch.bool, device="cuda")      # 动态大小
        self.anchor_feat = torch.empty(0, 32, device="cuda")
        self.anchor_age  = torch.zeros(0, dtype=torch.long, device="cuda")    # 连续低透明计数
        self.spawn_threshold = 0.30        # 残差阈值（可调）
        self.prune_threshold = 0.01        # 透明阈值
        
        ### >>> 2. Anchor-Auxiliary System (TLC-Calib)
        self.anchor_auxiliary_system = None
        self.use_anchor_auxiliary = False
        self.anchor_positions_fixed = None  # 固定的锚点位置 (LiDAR初始化)
        self.auxiliary_gaussian_params = None  # 辅助高斯参数

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.P,
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
         self._xyz,
         self._features_dc,
         self._features_rest,
         self._scaling,
         self._rotation,
         self._opacity,
         self.max_radii2D,
         xyz_gradient_accum,
         denom,
         opt_dict,
         self.spatial_lr_scale,
         self.P) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        xyz = self._xyz.clone()
        if self.rotate_xyz:
            xyz = self.P[0].retr().act(xyz)
            return xyz
        elif self.rotate_seq:
            xyz = self.P[self.seq_idx].retr().act(xyz)
            return xyz
        else:
            return self._xyz

    def get_RT(self, idx=None):
        if getattr(self, "P", None) is None:
            return torch.eye(4, device="cuda")

        if self.rotate_xyz:
            Rt = self.P[0].retr().matrix()
        else:
            if idx is None:
                Rt = self.P[self.seq_idx].retr().matrix()
            else:
                Rt = self.P[idx].retr().matrix()

        return Rt.squeeze()

    def set_seq_idx(self, idx):
        if idx < 0:
            self.rotate_seq = False
            self.rotate_xyz = False
        else:
            self.seq_idx = idx

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_features_noview(self):
        features_dc = self._features_dc.squeeze()
        return features_dc

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    ### >>> Anchor-Auxiliary System Methods (TLC-Calib)
    def init_anchor_auxiliary_system(self,
                                     lidar_points: np.ndarray,
                                     poses: List[np.ndarray],
                                     num_auxiliary: int = 5,
                                     use_auxiliary: bool = True,
                                     beta_t: float = 1.0,
                                     beta_n: float = 0.5):
        """
        初始化锚点-辅助高斯系统
        
        Args:
            lidar_points: (N, 3) LiDAR点云
            poses: 相机位姿列表
            num_auxiliary: 每个锚点对应的辅助高斯数量
            use_auxiliary: 是否使用辅助高斯
        """
        if not ANCHOR_AUXILIARY_AVAILABLE:
            print("Warning: Anchor-Auxiliary system not available, skipping initialization")
            return
        
        # 获取当前高斯数量
        n_current = self._xyz.shape[0]
        
        self.anchor_auxiliary_system = ECUGSAnchorAuxiliarySystem(
            num_auxiliary=num_auxiliary,
            use_auxiliary=use_auxiliary,
            device="cuda"
        )
        
        # 从LiDAR点云初始化锚点
        anchor_points = self.anchor_auxiliary_system.initialize_from_lidar(
            lidar_points, poses)
        
        # 保存锚点位置 (不优化)
        self.anchor_positions_fixed = torch.from_numpy(anchor_points).float().cuda()
        self.anchor_positions_fixed.requires_grad = False
        
        self.use_anchor_auxiliary = True
        
        # 初始化锚点标记 - 长度必须与当前高斯数量一致
        n_anchors = len(anchor_points)
        
        # 限制锚点数量不超过高斯数量
        if n_anchors > n_current:
            print(f"[AnchorAuxiliary] Warning: {n_anchors} anchors > {n_current} gaussians, truncating to {n_current}")
            self.anchor_positions_fixed = self.anchor_positions_fixed[:n_current]
            n_anchors = n_current
        
        # 策略: 使用LiDAR锚点替换前N个高斯的位置
        # 这样锚点就是前n_anchors个高斯
        with torch.no_grad():
            self._xyz[:n_anchors] = self.anchor_positions_fixed.clone()

        # 锚点高斯法向各向异性初始化 (Eq.22): R=[t1,t2,n], s_xy=β_t·d̄, s_z=β_n·d̄
        try:
            hybrid_init = ECUGSHybridInitializer(device="cuda")
            quats, log_scales, _, _ = hybrid_init.compute_anchor_gaussian_params(
                self.anchor_positions_fixed, beta_t=beta_t, beta_n=beta_n)
            with torch.no_grad():
                self._rotation[:n_anchors] = quats[:n_anchors].to(self._rotation.device)
                self._scaling[:n_anchors] = log_scales[:n_anchors].to(self._scaling.device)
        except Exception as e:
            print(f"[AnchorAuxiliary] Normal-based init skipped: {e}")
        
        # 创建is_anchor标记 - 前n_anchors个为True，其余为False
        self.is_anchor = torch.zeros(n_current, dtype=torch.bool, device="cuda")
        self.is_anchor[:n_anchors] = True
        
        # 初始化锚点相关的buffer
        self.anchor_age = torch.zeros(n_current, dtype=torch.long, device="cuda")
        self.anchor_feat = torch.zeros(n_current, 32, device="cuda")
        # 只为真正的锚点计算特征
        self.anchor_feat[:n_anchors] = self.anchor_encoder(self.anchor_positions_fixed)
        
        print(f"[AnchorAuxiliary] Initialized {n_anchors} anchors (out of {n_current} gaussians) "
              f"with {num_auxiliary} auxiliary per anchor")
        
    def update_auxiliary_gaussians(self, view_dir: Optional[torch.Tensor] = None):
        """
        更新辅助高斯参数
        
        Args:
            view_dir: (3,) 视角方向，None则使用默认
        """
        if not self.use_anchor_auxiliary or self.anchor_auxiliary_system is None:
            return
            
        if view_dir is not None:
            self.anchor_auxiliary_system.update_view_direction(view_dir)
            
        # 生成辅助高斯
        aux_pos, aux_scales, aux_colors, aux_opacities = \
            self.anchor_auxiliary_system.generate_auxiliary_gaussians()
            
        if aux_pos is not None:
            self.auxiliary_gaussian_params = {
                'positions': aux_pos,
                'scales': aux_scales,
                'colors': aux_colors,
                'opacities': aux_opacities
            }
            
    def get_combined_gaussians(self) -> Tuple[torch.Tensor, ...]:
        """
        获取合并后的高斯参数 (锚点 + 辅助)
        
        Returns:
            xyz: (N + M, 3) 位置
            features: (N + M, C) 特征
            opacities: (N + M, 1) 不透明度
            scales: (N + M, 3) 尺度
            rotations: (N + M, 4) 旋转
        """
        # 基础高斯参数
        xyz = self._xyz
        opacities = self._opacity
        scales = self._scaling
        rotations = self._rotation
        
        # 如果有辅助高斯，合并
        if self.use_anchor_auxiliary and self.auxiliary_gaussian_params is not None:
            aux = self.auxiliary_gaussian_params
            
            # 合并位置
            xyz = torch.cat([xyz, aux['positions']], dim=0)
            
            # 合并不透明度
            aux_opacities_inv = self.inverse_opacity_activation(aux['opacities'].unsqueeze(-1))
            opacities = torch.cat([opacities, aux_opacities_inv], dim=0)
            
            # 合并尺度
            aux_scales_inv = self.scaling_inverse_activation(aux['scales'])
            scales = torch.cat([scales, aux_scales_inv], dim=0)
            
            # 辅助高斯的旋转设为identity
            aux_rotations = torch.zeros(aux['positions'].shape[0], 4, device="cuda")
            aux_rotations[:, 0] = 1  # identity quaternion
            rotations = torch.cat([rotations, aux_rotations], dim=0)
            
        return xyz, self.get_features, opacities, scales, rotations
        
    def filter_anchor_floaters(self, min_opacity: float = 0.01, patience: int = 500):
        """过滤低透明度的浮动锚点 - 只处理is_anchor=True的高斯"""
        if not self.use_anchor_auxiliary or self.anchor_auxiliary_system is None:
            return
            
        with torch.no_grad():
            n_cur = self._xyz.shape[0]
            
            # 确保buffer长度一致
            if self.anchor_age.shape[0] != n_cur:
                self.anchor_age = self.anchor_age[:n_cur]
            if self.is_anchor.shape[0] != n_cur:
                self.is_anchor = self.is_anchor[:n_cur]
            
            # 只处理真正的锚点
            if self.is_anchor.sum() == 0:
                return
            
            # 获取所有高斯的不透明度，然后选择锚点
            all_opacity = self.get_opacity.squeeze()
            anchor_opacity = all_opacity[self.is_anchor]
            
            # 确保 anchor_age 大小匹配
            if self.anchor_age.shape[0] != n_cur:
                self.anchor_age = torch.zeros(n_cur, dtype=torch.long, device='cuda')
            
            # 获取当前锚点的年龄
            anchor_age_subset = self.anchor_age[self.is_anchor]
            
            # 检测低透明度
            low_opacity = anchor_opacity < min_opacity
            anchor_age_subset = anchor_age_subset + low_opacity.long()
            anchor_age_subset[~low_opacity] = 0
            
            # 更新回主buffer
            self.anchor_age[self.is_anchor] = anchor_age_subset
            
            # 标记死亡锚点（只在锚点中）
            dead_in_anchors = anchor_age_subset > patience
            if dead_in_anchors.sum() > 0:
                print(f"[AnchorAuxiliary] Found {dead_in_anchors.sum().item()} dead anchor points")
                
            # 更新锚点有效性
            if hasattr(self, 'anchor_valid_mask'):
                if self.anchor_valid_mask.shape[0] != n_cur:
                    self.anchor_valid_mask = torch.ones(n_cur, dtype=torch.bool, device='cuda')
                anchor_valid_subset = self.anchor_valid_mask[self.is_anchor]
                anchor_valid_subset &= ~dead_in_anchors
                self.anchor_valid_mask[self.is_anchor] = anchor_valid_subset
                
    def get_adaptive_voxel_size(self, total_iterations: int) -> List[Tuple[int, float]]:
        """获取自适应体素大小调度"""
        if self.anchor_auxiliary_system is not None:
            return self.anchor_auxiliary_system.voxel_controller.get_coarse_to_fine_schedule(total_iterations)
        return [(0, 0.05)]  # 默认5cm
        
    def fix_anchor_positions(self):
        """固定锚点位置 (不优化)"""
        if self.anchor_positions_fixed is not None:
            # 确保锚点位置不参与梯度计算
            self.anchor_positions_fixed.requires_grad = False
            
            # 更新实际高斯位置为固定锚点位置
            n_anchors = min(len(self.anchor_positions_fixed), len(self._xyz))
            with torch.no_grad():
                self._xyz[:n_anchors] = self.anchor_positions_fixed[:n_anchors].clone()
                
    ### >>> End of Anchor-Auxiliary System Methods

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        if self.view_dependent:
            fused_color = RGB2SH(torch.tensor(
                np.asarray(pcd.colors)).float().cuda())
        else:
            fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        features = torch.zeros(
            (fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color[:, :3]
        features[:, 3:, 1:] = 0.0
        # features = torch.cat([features, features], dim=1)
        print("Number of points at initialisation : ",
              fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(
            np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(
            1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(
            features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

                ### >>> 2. 初始锚标记
        n_init = self._xyz.shape[0]
        self.is_anchor  = torch.ones(n_init, dtype=torch.bool, device="cuda")
        self.anchor_feat = self.anchor_encoder(self._xyz)          # 32-d 特征
        self.anchor_age  = torch.zeros(n_init, dtype=torch.long, device="cuda")


    ### >>> 3. 锚编码器（随机权重，冻住）
    def anchor_encoder(self, xyz):
        if not hasattr(self, '_enc'):
            self._enc = nn.Sequential(
                nn.Linear(3,64), nn.ReLU(),
                nn.Linear(64,64), nn.ReLU(),
                nn.Linear(64,32)
            ).to(xyz.device)
            for p in self._enc.parameters(): p.requires_grad = False
        with torch.no_grad():
            return self._enc(xyz)

    ### >>> 4. 锚生长：输入新 3D 坐标，直接克隆属性并标锚
    def densify_and_clone_anchor(self, new_xyz):
        """把 new_xyz 克隆成高斯，同时标记为锚（显存安全版）"""
        n_new = new_xyz.shape[0]
        # 分块最近邻
        chunk = 1000
        idx = torch.zeros(n_new, dtype=torch.long, device='cuda')
        with torch.no_grad():
            for i in range(0, n_new, chunk):
                end = min(i + chunk, n_new)
                dist_cpu = torch.cdist(new_xyz[i:end].cpu(), self._xyz.cpu())
                _, idx[i:end] = dist_cpu.min(dim=1)
                idx[i:end] = idx[i:end].cuda()

        # 扩展高斯属性
        self._xyz          = nn.Parameter(torch.cat([self._xyz,          new_xyz]),          requires_grad=True)
        self._features_dc  = nn.Parameter(torch.cat([self._features_dc,  self._features_dc[idx]]),  requires_grad=True)
        self._features_rest= nn.Parameter(torch.cat([self._features_rest,self._features_rest[idx]]),requires_grad=True)
        self._scaling      = nn.Parameter(torch.cat([self._scaling,      self._scaling[idx]]),      requires_grad=True)
        self._rotation     = nn.Parameter(torch.cat([self._rotation,     self._rotation[idx]]),     requires_grad=True)
        self._opacity      = nn.Parameter(torch.cat([self._opacity,      self._opacity[idx]]),      requires_grad=True)

        # optimizer 同步 - densification_postfix_anchor 会处理锚 buffer
        self.densification_postfix_anchor(
            self._xyz[-n_new:], self._features_dc[-n_new:], self._features_rest[-n_new:],
            self._opacity[-n_new:], self._scaling[-n_new:], self._rotation[-n_new:], n_new)
        ### >>> 5. 锚进化主入口：每 N it 调用一次
    def evolve_anchors(self, render_pkg, viewpoint_cam, iteration, every=500,
                       sigma_ext2=None, tau0=None, kappa=2.0, max_new_points=1000):
        if iteration % every != 0: return

        gt_depth = viewpoint_cam.depth_gt                         # 需要提前挂到 camera
        if gt_depth is None:
            return

        ren_depth = render_pkg['depth'].squeeze()
        residual  = torch.abs(ren_depth - gt_depth)
        # 残差阈值 (Eq.33): |D - D̃| > τ_0 + κ·σ_ext(u)
        if sigma_ext2 is not None and tau0 is not None:
            threshold = tau0 + kappa * sigma_ext2.squeeze().clamp_min(0.0).sqrt()
        else:
            threshold = self.spawn_threshold
        res_mask  = residual > threshold
        if res_mask.sum() == 0: return
        
        # 反投 3D
        K = torch.from_numpy(viewpoint_cam.intrinsics).float().to(gt_depth.device)
        pts_3d = depth_to_3d(gt_depth[None,None], K[None], normalize_points=False)
        pts_3d = pts_3d.squeeze().permute(1,2,0)                  # H×W×3
        candidates = pts_3d[res_mask]                             # N×3
        
        # FPS 采样，限制数量
        num_to_add = min(len(candidates), max_new_points)
        if num_to_add < 10:  # 太少就不添加了
            return
            
        idx_fps = farthest_point_sample(candidates, num_to_add)
        new_xyz = candidates[idx_fps]
        self.densify_and_clone_anchor(new_xyz)


    ### >>> 6. 锚剪枝：每 500 it 调用
    def prune_anchor_dead(self, min_opacity=0.01, patience=500):
        """锚点死亡检测与剪枝 - 只处理真正的锚点（is_anchor=True）"""
        with torch.no_grad():
            n_cur = self._xyz.shape[0]

            # 1. 兜底同步：如果 buffer 长度不一致，直接切片对齐
            if self.anchor_age.shape[0] != n_cur:
                self.anchor_age  = self.anchor_age[:n_cur]
                self.is_anchor   = self.is_anchor[:n_cur]
                self.anchor_feat = self.anchor_feat[:n_cur]

            # 2. 只处理真正的锚点（is_anchor == True）
            if self.is_anchor.sum() == 0:
                return
            
            # 获取所有高斯的不透明度
            all_opacity = self.get_opacity.squeeze()
            
            # 只选择锚点对应的不透明度
            anchor_mask = self.is_anchor
            anchor_opacity = all_opacity[anchor_mask]
            anchor_age_subset = self.anchor_age[anchor_mask]
            
            # 检测低透明度锚点
            low_opacity = anchor_opacity < min_opacity
            
            # 更新锚点年龄（只更新锚点部分）
            anchor_age_subset = anchor_age_subset + low_opacity.long()
            anchor_age_subset[~low_opacity] = 0
            
            # 检测死亡锚点
            dead_in_anchors = anchor_age_subset > patience
            
            if dead_in_anchors.sum() == 0:
                # 没有死亡锚点，更新年龄buffer
                self.anchor_age[anchor_mask] = anchor_age_subset
                return
            
            # 构建完整的高斯死亡掩码（非锚点永远不死）
            dead_mask = torch.zeros(n_cur, dtype=torch.bool, device='cuda')
            anchor_indices = torch.where(anchor_mask)[0]
            dead_indices = anchor_indices[dead_in_anchors]
            dead_mask[dead_indices] = True
            
            print(f"[Anchor] Pruning {dead_mask.sum().item()} dead anchor points")
            
            # 3. 剪枝（高斯 + 锚 buffer 同步）
            self.prune_points_anchor(dead_mask)

    def fix_position(self):
        self._xyz = nn.Parameter(self._xyz.detach().requires_grad_(False))
        self._features_dc = nn.Parameter(
            self._features_dc.detach().requires_grad_(False))
        self._features_rest = nn.Parameter(
            self._features_rest.detach().requires_grad_(False))

        self._scaling = nn.Parameter(
            self._scaling.detach().requires_grad_(False))
        self._rotation = nn.Parameter(
            self._rotation.detach().requires_grad_(False))
        self._opacity = nn.Parameter(
            self._opacity.detach().requires_grad_(False))
        _xyz = self._xyz.detach().clone()

    def freeze(self):
        self._xyz = nn.Parameter(self._xyz.detach().requires_grad_(False))
        self._features_dc = nn.Parameter(
            self._features_dc.detach().requires_grad_(False))
        self._features_rest = nn.Parameter(
            self._features_rest.detach().requires_grad_(False))
        self._scaling = nn.Parameter(
            self._scaling.detach().requires_grad_(False))
        self._rotation = nn.Parameter(
            self._rotation.detach().requires_grad_(False))
        self._opacity = nn.Parameter(
            self._opacity.detach().requires_grad_(False))

    def training_setup(self, training_args, fix_pos=False,
                       fix_feat=False, fit_pose=False):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        l = []

        _xyz_lr = training_args.position_lr_init * \
            self.spatial_lr_scale if not fix_pos else 0.0
        feat_lr_factor = 1.0 if not fix_feat else 0.0

        l += [
            {'params': [self._xyz], 'lr': _xyz_lr, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr *
                feat_lr_factor, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr /
                20.0 * feat_lr_factor, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr *
                feat_lr_factor, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr *
                feat_lr_factor, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr *
                feat_lr_factor, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        if fit_pose:
            rotation_lr_factor = 1.0 if (fix_pos and fix_feat) else 1.0

            if self.rotate_seq:
                self.camera_optimizer = []
                for idx in range(len(self.P)):
                    l_cam = [
                        {'params': [self.P[idx]],
                            'lr': training_args.rotation_lr, "name": "R"},
                    ]
                    self.camera_optimizer.append(
                        torch.optim.Adam(l_cam, lr=0.0, eps=1e-15))
                # l_cam = [
                #         {'params': [self.P], 'lr': training_args.rotation_lr, "name": "R"},
                #     ]
                # self.camera_optimizer = torch.optim.Adam(l_cam, lr=0.0, eps=1e-15)
            else:
                l_cam = [
                    {'params': [self.P],
                        'lr': training_args.rotation_lr, "name": "R"},
                ]
                self.camera_optimizer = [
                    torch.optim.Adam(l_cam, lr=0.0, eps=1e-15)]
            self.camera_scheduler_args = get_expon_lr_func(lr_init=training_args.rotation_lr,
                                                           lr_final=training_args.rotation_lr * 0.1,
                                                           lr_delay_mult=0.1,
                                                           max_steps=training_args.position_lr_max_steps)
        else:
            self.camera_optimizer = None

    def training_setup_fix_position(self, training_args, gaussian_rot=True):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        if gaussian_rot:
            lr_factor = 1.0
        else:
            lr_factor = 0.1

        l = [
            {'params': [self.P[0]],
                'lr': training_args.rotation_lr, "name": "R"},
        ]
        if gaussian_rot:
            l += [
                {'params': [self._rotation],
                    'lr': training_args.rotation_lr, "name": "rotation"}
            ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def init_RT(self, pcd=None, pose=None):
        if pose is None:
            pose_init = torch.as_tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).cuda().requires_grad_(True)
            self.P = [LieGroupParameter(SE3(pose_init[None]))]
        else:
            quat = matrix_to_quaternion(pose[:3, :3])
            # matrix_to_quaternion returns [w, x, y, z], but lietorch expects [x, y, z, w]
            quat = torch.cat([quat[1:], quat[0:1]], dim=-1)
            pose = torch.cat((pose[:3, 3], quat), -
                             1).cuda().requires_grad_(True)
            self.P = [LieGroupParameter(SE3(pose[None]))]

        self.rotate_xyz = True
        self.rotate_seq = False

    def init_RT_seq(self, seq_len, pose=None):
        if pose is None:
            pose_init = torch.as_tensor(
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).cuda().requires_grad_(True)
            self.P = [LieGroupParameter(SE3(pose_init[None]))
                      for _ in range(seq_len)]
            # pose_init = LieGroupParameter(SE3(torch.stack([pose_init for _ in range(seq_len)])))
            # self.P = LieGroupParameter(SE3(pose_init))
        else:
            quat = R.from_matrix(pose[..., :3, :3].numpy()).as_quat()
            pose = torch.cat((pose[..., :3, 3], torch.from_numpy(
                quat).float()), -1).cuda().requires_grad_(True)
            self.P = [LieGroupParameter(SE3(pose[idx][None]))
                      for idx in range(seq_len)]
            # if getattr(self, "P", None) is None:
            #     self.P = [LieGroupParameter(SE3(pose[idx][None]))
            #               for idx in range(seq_len)]
            # else:
            #     for idx in range(seq_len):
            #         self.P[idx].data = SE3(pose[idx][None]).data
            # self.P = LieGroupParameter(SE3(pose))

        self.rotate_seq = True
        self.rotate_xyz = False

    def update_RT_seq(self, pose, idx):
        quat = matrix_to_quaternion(pose[:3, :3])
        quat = quat[..., [1, 2, 3, 0]]
        pose = torch.cat((pose[:3, 3], quat.float()), -
                         1).cuda().requires_grad_(True)
        self.P[idx] = LieGroupParameter(SE3(pose[None]))
        self.P[idx].group = SE3(pose[None])

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def update_learning_rate_camera(self, cam_idx, iteration):
        ''' Learning rate scheduling per step '''
        if isinstance(self.camera_optimizer, list):
            for param_group in self.camera_optimizer[cam_idx].param_groups:
                lr = self.camera_scheduler_args(iteration)
                param_group['lr'] = lr
        else:
            for param_group in self.camera_optimizer.param_groups:
                lr = self.camera_scheduler_args(iteration)
                param_group['lr'] = lr

    def freeze_camera(self):
        for param_group in self.camera_optimizer.param_groups:
            param_group['lr'] = 0.0

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(
            1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(
            1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4')
                      for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def export_gaussian(self, ):
        gassuian = {}
        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        max_radii2D = self.max_radii2D.detach().cpu().numpy()
        gassuian['xyz'] = xyz
        gassuian['normals'] = normals
        gassuian['f_dc'] = f_dc
        gassuian['f_rest'] = f_rest
        gassuian['opacities'] = opacities
        gassuian['scale'] = scale
        gassuian['rotation'] = rotation
        gassuian['max_radii2D'] = max_radii2D

        return gassuian

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(
            torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(
            opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, num_gauss=-1):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(
            extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        if num_gauss == -1:
            num_gauss = xyz.shape[0]
        self._xyz = nn.Parameter(torch.tensor(
            xyz[:num_gauss], dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(
            features_dc[:num_gauss], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(
            features_extra[:num_gauss], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(
            opacities[:num_gauss], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(
            scales[:num_gauss], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(
            rots[:num_gauss], dtype=torch.float, device="cuda").requires_grad_(True))

        R = torch.eye(3, device="cuda")
        T = torch.zeros(1, 3, device="cuda")
        self.R = nn.Parameter(R.requires_grad_(True))
        self.T = nn.Parameter(T.requires_grad_(True))
        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(
                    group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            # if group["name"] not in ["xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"]:
            #     continue
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            # if group["name"] not in tensors_dict:
            #     continue
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        # if "xyz" in optimizable_tensors:
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        # if "rotation" in optimizable_tensors:
        self._rotation = optimizable_tensors["rotation"]
        # self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros(
            (self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    ### >>> 锚-aware 重写
    def densification_postfix_anchor(self, new_xyz, new_features_dc, new_features_rest,
                                    new_opacities, new_scaling, new_rotation,
                                    n_new):
        """先调用原版 postfix，再把锚 buffer 同步拉长 n_new
        注意：新添加的高斯不标记为锚点（is_anchor=False）"""
        self.densification_postfix(new_xyz, new_features_dc, new_features_rest,
                                new_opacities, new_scaling, new_rotation)
        # 同步锚 buffer：新添加的高斯标记为非锚点（False）
        self.is_anchor   = torch.cat([self.is_anchor,   torch.zeros(n_new, dtype=torch.bool, device='cuda')])
        # 对于非锚点，不需要追踪 anchor_feat 和 anchor_age
        # 但为了保持 buffer 长度一致，用零填充
        self.anchor_feat = torch.cat([self.anchor_feat, torch.zeros(n_new, 32, device='cuda')])
        # 确保 anchor_age 存在并同步
        if not hasattr(self, 'anchor_age'):
            n_current = len(self._xyz) - n_new
            self.anchor_age = torch.zeros(n_current, dtype=torch.long, device='cuda')
        self.anchor_age  = torch.cat([self.anchor_age, torch.zeros(n_new, dtype=torch.long, device='cuda')])

    def prune_points_anchor(self, mask):
        """先调用原版 prune，再把锚 buffer 同步裁剪"""
        self.prune_points(mask)          # 先删高斯
        # 再删锚 buffer
        self.is_anchor   = self.is_anchor[~mask]
        self.anchor_feat = self.anchor_feat[~mask]
        if hasattr(self, 'anchor_age'):
            self.anchor_age  = self.anchor_age[~mask]

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(
            padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(
            self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + \
            self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(
            self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(
            N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)

        self.densification_postfix_anchor(
            new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_xyz.shape[0])

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(
            N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points_anchor(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(
            grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix_anchor(
            new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_xyz.shape[0])

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(
                prune_mask, big_points_vs), big_points_ws)
        self.prune_points_anchor(prune_mask)

        torch.cuda.empty_cache()

    def prune(self, max_grad, min_opacity, extent, max_screen_size):
        prune_mask = (self.get_opacity < min_opacity).squeeze()

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(prune_mask, big_points_vs)

            prune_mask = torch.logical_or(torch.logical_or(
                prune_mask, big_points_vs), big_points_ws)
        self.prune_points_anchor(prune_mask)

        torch.cuda.empty_cache()

    def densify(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(
            viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


class ECUGSRender:
    def __init__(self, sh_degree=3, white_background=False,
                 radius=1, view_dependent=False,
                 optimize_lidar_extrinsic=False,
                 init_lidar2cam=None,
                 use_sky_gaussians=False,
                 num_sky_gaussians=10000,
                 sky_radius=50.0):

        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius
        self.view_dependent = view_dependent
        self.optimize_lidar_extrinsic = optimize_lidar_extrinsic
        self.use_sky_gaussians = use_sky_gaussians

        self.gaussians = ECUGSGaussianModel(sh_degree,
                                         view_dependent=self.view_dependent)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
            device="cuda",
        )
        
        # LiDAR 外参标定器
        self.lidar_calibrator = None
        if optimize_lidar_extrinsic:
            from utils.lidar_calibration import ECUGSCalibrator
            self.lidar_calibrator = ECUGSCalibrator(
                init_extrinsic=init_lidar2cam,
                learnable=True,
                lr=1e-4
            )
        
        # 天空高斯
        self.sky_gaussians = None
        if use_sky_gaussians:
            from utils.sky_gaussians import ECUGSSkyGaussians
            self.sky_gaussians = ECUGSSkyGaussians(
                num_sky_gaussians=num_sky_gaussians,
                sky_radius=sky_radius,
                sky_sh_degree=sh_degree,
                view_dependent=view_dependent,
                device="cuda"
            )

    def init_model(self, input=None, num_pts=10000, radius=1.0):

        if input is None:
            # init from random points
            phis = np.random.random((num_pts,)) * 2 * np.pi
            costheta = np.random.random((num_pts,)) * 2 - 1
            thetas = np.arccos(costheta)
            mu = np.random.random((num_pts,))
            radius = radius * np.cbrt(mu)
            x = radius * np.sin(thetas) * np.cos(phis)
            y = radius * np.sin(thetas) * np.sin(phis)
            z = radius * np.cos(thetas)
            xyz = np.stack((x, y, z), axis=1)
            # xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3

            shs = np.random.random((num_pts, 3)) / 255.0
            pcd = BasicPointCloud(
                points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3))
            )
            self.gaussians.create_from_pcd(pcd, 10)
            self.radius = radius.max()

        elif isinstance(input, BasicPointCloud):
            # load from a provided pcd
            radius = np.linalg.norm(input.points, axis=1).max()
            # TODO: check if this is correct with radius
            # self.gaussians.create_from_pcd(input, 1)
            self.gaussians.create_from_pcd(input, radius)
            self.radius = radius
        else:
            # load from saved ply
            self.gaussians.load_ply(input)

    def reset_model(self):
        self.gaussians = ECUGSGaussianModel(
            self.sh_degree, self.view_dependent)
        
    def init_sky_gaussians(self, center=None, color=None):
        """初始化天空高斯"""
        if self.sky_gaussians is not None:
            self.sky_gaussians.initialize_on_sphere(center=center, color=color)
            print(f"Initialized {len(self.sky_gaussians)} sky gaussians")
        else:
            print("Warning: sky_gaussians is None, cannot initialize")
    
    def get_sky_optimizer_params(self, lr_dict=None):
        """获取天空高斯的优化器参数"""
        if self.sky_gaussians is None or not self.sky_gaussians.is_initialized:
            return []
        
        if lr_dict is None:
            lr_dict = {
                'sky_xyz': 0.0001,
                'sky_features': 0.0025,
                'sky_opacity': 0.05,
                'sky_scaling': 0.005,
                'sky_rotation': 0.001,
            }
        
        return self.sky_gaussians.get_optimizer_params(lr_dict)
    
    def get_all_params_with_sky(self):
        """获取所有参数（包括场景高斯和天空高斯）"""
        params = list(self.gaussians.parameters())
        if self.sky_gaussians is not None and self.sky_gaussians.is_initialized:
            sky_params = self.sky_gaussians.get_params()
            params.extend([sky_params[k] for k in sky_params])
        return params

    def render(
        self,
        viewpoint_camera,
        scaling_modifier=1.0,
        invert_bg_color=False,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
    ):
        """
        Render the scene. 

        Background tensor (bg_color) must be on GPU!
        """

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                self.gaussians.get_xyz,
                dtype=self.gaussians.get_xyz.dtype,
                requires_grad=True,
                device="cuda",
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self.bg_color if not invert_bg_color else 1 - self.bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=self.gaussians.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = self.gaussians.get_xyz
        means2D = screenspace_points
        opacity = self.gaussians.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
        else:
            scales = self.gaussians.get_scaling
            rotations = self.gaussians.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                if self.view_dependent:
                    shs_view = self.gaussians.get_features.transpose(1, 2).view(
                        -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                    )
                    fidx = viewpoint_camera.uid
                    camera_center = self.gaussians.get_RT(fidx).inverse()[
                        :3, 3].detach()
                    camera_center = camera_center[None].repeat(
                        self.gaussians.get_features.shape[0], 1)
                    dir_pp = self.gaussians._xyz - camera_center
                    dir_pp_normalized = dir_pp / \
                        dir_pp.norm(dim=1, keepdim=True)
                    sh2rgb = eval_sh(
                        self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                    )
                    colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
                else:
                    colors_precomp = self.gaussians.get_features_noview
            else:
                shs = self.gaussians.get_features
        else:
            colors_precomp = override_color
        
        # ---------- 合并辅助高斯 (Anchor-Auxiliary System) ----------
        # 先生成辅助高斯，再合并到场景高斯中
        if self.gaussians.use_anchor_auxiliary:
            cam_center = viewpoint_camera.camera_center
            if not isinstance(cam_center, torch.Tensor):
                cam_center = torch.tensor(cam_center, device="cuda")
            else:
                cam_center = cam_center.clone().detach().to("cuda")
            self.gaussians.update_auxiliary_gaussians(view_dir=cam_center)
            
        # ---------- 合并天空高斯 ----------
        # 注意：天空高斯不参与LiDAR外参优化，因为它们在远距离球面上
        sky_mask = None
        if self.use_sky_gaussians and self.sky_gaussians is not None and self.sky_gaussians.is_initialized:
            from utils.sky_gaussians import merge_sky_and_scene_gaussians
            
            # 获取场景高斯特征（处理shs）
            scene_features = self.gaussians.get_features
            
            # 合并场景高斯和天空高斯
            means3D, merged_features, opacity, scales, rotations, is_sky = merge_sky_and_scene_gaussians(
                self.sky_gaussians,
                means3D,
                scene_features,
                opacity,
                scales,
                rotations,
            )
            
            # 处理SH或预计算颜色
            if shs is not None:
                # 使用球谐函数
                sky_features = self.sky_gaussians.get_features
                shs = torch.cat([shs, sky_features], dim=0)
            elif colors_precomp is not None:
                # 使用预计算颜色，天空高斯也需要预计算
                sky_colors = self.sky_gaussians.get_features_noview
                colors_precomp = torch.cat([colors_precomp, sky_colors], dim=0)
            
            # 更新screenspace_points（天空高斯也需要梯度）
            sky_screenspace = torch.zeros(
                self.sky_gaussians.get_xyz.shape[0],
                3,  # xyz
                dtype=screenspace_points.dtype,
                device=screenspace_points.device,
                requires_grad=True
            )
            means2D = torch.cat([screenspace_points, sky_screenspace], dim=0)
            try:
                means2D.retain_grad()
            except:
                pass
            
            sky_mask = is_sky

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        out = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        if len(out) == 4:
            rendered_image, radii, rendered_depth, rendered_alpha = out
            rendered_image = rendered_image.clamp(0, 1)

            # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
            # They will be excluded from value updates used in the splitting criteria.
            return {
                "image": rendered_image,
                "depth": rendered_depth,
                "alpha": rendered_alpha,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
            }
        elif len(out) == 3:
            rendered_image, radii, rendered_depth = out

            rendered_image = rendered_image.clamp(0, 1)

            # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
            # They will be excluded from value updates used in the splitting criteria.
            return {
                "image": rendered_image,
                "depth": rendered_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter": radii > 0,
                "radii": radii,
            }
