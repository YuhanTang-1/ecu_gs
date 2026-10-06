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

from argparse import ArgumentParser, Namespace
import sys
import os


class GroupParams:
    pass


class ParamGroup:
    def __init__(self, parser: ArgumentParser, name: str, fill_none=False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument(
                        "--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument(
                        "--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = True
        self.rot_type = "6d"
        self.view_dependent = True
        # self.model_type = 'original'
        self.depth_model_type = "dpt"
        self.mode = "train"
        # self.eval_nvs = False
        # self.eval_pose = False
        # self.vis_mesh = False
        # self.render_nvs = False
        self.traj_opt = "bspline"
        # LiDAR相关参数（非结构化场景重建默认启用）
        self.use_lidar = True
        self.optimize_lidar_extrinsic = False  # coming soon: online extrinsic calibration
        self.use_lidar_odometry = False
        self.use_depth_scale_match = False  # coming soon: LiDAR depth scale matching
        self.save_global_pcd = False
        # 天空高斯参数
        self.use_sky_gaussians = False
        self.num_sky_gaussians = 10000
        self.sky_radius = 50.0
        # Anchor-Auxiliary系统参数 (TLC-Calib)
        self.use_anchor_auxiliary = False      # coming soon: anchor-auxiliary system
        self.num_auxiliary = 5                 # 每个锚点的辅助高斯数量
        self.adaptive_voxel_beta = 5000.0      # 自适应体素控制参数
        # 几何约束参数
        self.use_geometric_loss = False        # coming soon: geometric constraint losses
        self.lambda_scale_reg = 0.01           # 尺度正则化权重
        self.lambda_aniso_reg = 0.001          # 各向异性正则化权重
        # 标定精化参数
        self.use_plane_constraint = True       # 地面平面约束
        self.use_edge_alignment = True         # 边缘对齐约束
        self.calibration_warmup = 500          # 标定预热迭代数
        # ECU-GS 不确定度感知参数
        self.use_uncertainty_weighting = False  # coming soon: uncertainty-weighted supervision (Eq.26)
        self.use_normal_consistency = False    # coming soon: normal consistency loss (Eq.27)
        self.three_stage_training = False      # coming soon: three-stage training schedule
        self.lambda_depth_lidar = 0.1          # 深度一致性权重 λ_d
        self.lambda_normal = 0.05              # 法向一致性权重 λ_n
        self.lambda_edge = 0.5                 # 边缘对齐权重 λ_e
        self.lambda_pose_reg = 0.1             # 位姿-外参正则权重 λ_p
        self.gamma_n = 0.75                    # 稀疏区域增强系数 γ_n
        self.sigma_d2 = 0.01                   # LiDAR 深度测量方差 σ²_d (m²)
        self.beta_t = 1.0                      # 锚点切向尺度系数 β_t (Eq.22)
        self.beta_n = 0.5                      # 锚点法向尺度系数 β_n (Eq.22)
        self.beta_v = 0.1                      # 视觉补全尺度系数 β_v (Eq.23)
        self.residual_tau0 = 0.3               # 残差驱动致密化基础阈值 τ_0 (Eq.33)
        self.residual_kappa = 2.0              # 不确定度容忍系数 κ (Eq.33)
        self.max_spawn_gaussians = 1000        # 单次致密化上限 N_spawn (Eq.34)
        self.prune_opacity_threshold = 0.01    # 锚点剪枝不透明度阈值 τ_α (Eq.35)
        self.prune_age_threshold = 500         # 锚点剪枝最小存活期 τ_age (Eq.35)
        self.frame_recent_weight = 2.0         # 近帧采样权重 ω_r (Eq.32)
        self.frame_history_weight = 1.0        # 历史帧采样权重 ω_h (Eq.32)
        self.keyframe_expansion_gain = 0.05    # 关键帧空间新颖度阈值 G_exp (Eq.30)
        self.keyframe_min_baseline = 1.0       # 关键帧最小视角基线 (米)
        self.extrinsic_update_interval = 10    # 外参不确定度更新间隔 (帧)
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        # self.mode = "color"
        self.use_gt_pcd = False
        self.use_mask = False
        self.use_ref_img = False
        self.init_mode = "rand"
        self.use_mono = True
        self.interval = 15
        self.expname = ""
        self.use_sampon = False
        self.refine = False
        self.distortion = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.feature_lr = 0.0025
        self.opacity_lr = 0.05
        self.scaling_lr = 0.005
        self.rotation_lr = 0.001
        self.percent_dense = 0.01
        self.lambda_dssim = 0.2
        self.lambda_depth = 0.0
        self.lambda_dist_2nd_loss = 0.0
        self.lambda_pc = 0.0
        self.lambda_rgb_s = 0.0
        self.depth_loss_type = "invariant"
        # self.depth_loss_type = "l1"
        self.match_method = "dense"
        self.densification_interval = 100
        self.densify_interval = 500
        self.prune_interval = 2000
        self.opacity_reset_interval = 3000
        self.densify_from_iter = 500
        self.densify_until_iter = 15_000
        self.reset_until_iter = 15_000
        self.densify_grad_threshold = 0.0002
        super().__init__(parser, "Optimization Parameters")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
