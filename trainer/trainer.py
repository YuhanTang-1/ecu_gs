# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import pdb
from kornia.geometry.depth import depth_to_3d, depth_to_normals
from pytorch3d.utils import opencv_from_cameras_projection
from pytorch3d.renderer import PerspectiveCameras
import pytorch3d
from utils.image_utils import psnr, colorize
from utils.loss_utils import l1_loss, ssim
from scene.cameras import Camera
from utils.graphics_utils import BasicPointCloud, focal2fov, procrustes, fov2focal
from scene.gaussian_model import GaussianModel
from scene import Scene
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.dataset_readers import sceneLoadTypeCallbacks, CameraInfo, read_intrinsics_binary
from utils.lidar_loader import (lidar_depth_from_image, get_lidar2cam_gt, 
                                   lidar_pointcloud_from_image, lidar_pointcloud_for_gs_init)
import glob
from copy import copy
import open3d as o3d
from einops import rearrange
from PIL import Image
import os
from tqdm import tqdm
import math
import numpy as np
import cv2
from collections import defaultdict, OrderedDict
import torch
import torch.nn.functional as F
#torch.hub.help("intel-isl/MiDaS", "DPT_BEiT_L_384", force_reload=True)


class GaussianTrainer(object):

    def __init__(self, data_root, model_cfg, pipe_cfg, optim_cfg):
        self.model_cfg = model_cfg
        self.pipe_cfg = pipe_cfg
        self.optim_cfg = optim_cfg
        data_info = data_root.split('/')
        self.seq_name = data_info[-1]
        self.category = data_info[-2]
        self.data_root = data_root.split(self.category)[0]
        self.depth_model_type = model_cfg.depth_model_type
        self.rgb_images = OrderedDict()
        self.render_depth = OrderedDict()
        self.render_image = OrderedDict()
        self.mono_depth = OrderedDict()
        self.setup_dataset()
        self.setup_depth_predictor()

    def setup_depth_predictor(self,):
        # we recommand to use the following depth models:
        # - "midas" for the Tank and Temples dataset
        # - "zoe" for the CO3D dataset
        # - "depth_anything" for the custom dataset
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.depth_model_type == "zoe":
            repo = "isl-org/ZoeDepth"
            model_zoe_n = torch.hub.load(repo, "ZoeD_NK", pretrained=True)
            zoe = model_zoe_n.to(device)
            self.depth_model = zoe
        elif self.depth_model_type == "depth_anything":
            from torchvision.transforms import Compose
            from submodules.DepthAnything.depth_anything.dpt import DepthAnything
            from submodules.DepthAnything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
            encoder = 'vits' # can also be 'vitb' or 'vitl'
            depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder)).eval()
            # depth_anything = DepthAnything.from_pretrained('checkpoints/depth_anything_metric_depth_outdoor', local_files_only=True).eval()

            self.depth_transforms = Compose([
                Resize(
                    width=518,
                    height=518,
                    resize_target=False,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=14,
                    resize_method='lower_bound',
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                PrepareForNet(),
            ])
            self.depth_model = depth_anything
        else:
            midas = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid")
            midas.to(device)
            midas.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.depth_transforms = midas_transforms.dpt_transform
            self.depth_model = midas
        self.mono_depth = OrderedDict()

    def predict_depth(self, img):
        if self.depth_model_type == "zoe":
            depth = self.depth_model.infer_pil(Image.fromarray(img.astype(np.uint8)),
                                               output_type='tensor')
        elif self.depth_model_type == "depth_anything":
            image = self.depth_transforms({'image': img/255.})['image']
            image = torch.from_numpy(image).unsqueeze(0)
            # depth shape: 1xHxW
            prediction = self.depth_model(image)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze().detach()
            # depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            
            # depth = depth.cpu().numpy().astype(np.uint8)
            # depth_color = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)
            # pdb.set_trace()
            scale = 0.0305
            shift = 0.15
            depth = scale * prediction + shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            input_batch = self.depth_transforms(img).to(device)
            with torch.no_grad():
                prediction = self.depth_model(input_batch)

                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            scale = 0.000305
            shift = 0.1378
            depth = scale * prediction + shift
            depth[depth < 1e-8] = 1e-8
            depth = 1.0 / depth
        return depth

    def setup_dataset(self):
        source_path = self.model_cfg.source_path
        cameras_intrinsic_file_bin = os.path.join(source_path, "sparse/0", "cameras.bin")
        cameras_intrinsic_file_txt = os.path.join(source_path, "sparse/0", "cameras.txt")
        max_frames = 300
        # if os.path.exists(cameras_intrinsic_file):
        #     images = sorted(glob.glob(os.path.join(source_path, "images", "*.jpg")))
        #     if len(images)>max_frames:
        #         images = images[-max_frames:]
        #     cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        #     intr = cam_intrinsics[1]
        #     focal_length_x = intr.params[0]
        #     focal_length_y = intr.params[1]
        #     height = intr.height
        #     width = intr.width
        #     intr_mat = np.array(
        #         [[focal_length_x, 0, width/2], [0, focal_length_y, height/2], [0, 0, 1]])
        #     self.intrinsic = intr_mat
        # else:
        images = sorted(glob.glob(os.path.join(source_path, "images/*.jpg")))
        if len(images) == 0:  # 如果没有PNG，尝试JPG作为备选
            images = sorted(glob.glob(os.path.join(source_path, "images", "*.png")))
        if len(images) > max_frames:
            interval = len(images) // max_frames
            images = images[::interval]
        print("Total images: ", len(images))
        width, height = Image.open(images[0]).size
        if os.path.exists(cameras_intrinsic_file_txt):
            # 读取txt格式的相机参数
            with open(cameras_intrinsic_file_txt, 'r') as f:
                lines = f.readlines()
            
            # 跳过注释行，找到第一个相机参数
            camera_params = None
            for line in lines:
                if line.strip() and not line.startswith('#'):
                    parts = line.strip().split()
                    camera_id = int(parts[0])
                    camera_model = parts[1]
                    width_txt = int(parts[2])
                    height_txt = int(parts[3])
                    
                    if camera_model == "SIMPLE_PINHOLE":
                        fx = float(parts[4])
                        cx = float(parts[5])
                        cy = float(parts[6])
                        fy = fx
                    elif camera_model == "PINHOLE":
                        fx = float(parts[4])
                        fy = float(parts[5])
                        cx = float(parts[6])
                        cy = float(parts[7])
                    else:
                        # 其他相机模型，使用默认值
                        continue
                    
                    intr_mat = np.array(
                        [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
                    self.intrinsic = intr_mat
                    print(f"Loaded camera from txt: model={camera_model}, fx={fx}, fy={fy}, cx={cx}, cy={cy}")
                    break
                    
        elif os.path.exists(cameras_intrinsic_file_bin):
            cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file_bin)
            intr = cam_intrinsics[1]
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            height = intr.height
            width = intr.width
            intr_mat = np.array(
                [[focal_length_x, 0, width/2], [0, focal_length_y, height/2], [0, 0, 1]])
            self.intrinsic = intr_mat
        else:
            # use some hardcoded values
            fov = 79.0
            FoVx = fov * math.pi / 180
            intr_mat = np.eye(3)
            intr_mat[0, 0] = fov2focal(FoVx, width)
            intr_mat[1, 1] = fov2focal(FoVx, width)
            intr_mat[0, 2] = width / 2
            intr_mat[1, 2] = height / 2
            self.intrinsic = intr_mat

        if min(width, height) > 1000:
            width = width // 2
            height = height // 2
        
        intr_mat[:2, :] /= 2
        self.intrinsic = intr_mat


        sample_rate = 8
        ids = np.arange(len(images))
        self.i_test = ids[int(sample_rate/2)::sample_rate]
        self.i_train = np.array([i for i in ids if i not in self.i_test])
        if "eval" in self.model_cfg.mode:
            self.data = [images[i] for i in self.i_test]
        else:
            self.data = [images[i] for i in self.i_train]
        self.seq_len = len(self.data)

    def setup_model(self, pcd):
        radius = np.linalg.norm(pcd.points, axis=1).max()
        gaussians = GaussianModel(sh_degree=3)
        gaussians.create_from_pcd(pcd, math.ceil(radius))
        self.model = gaussians
        self.radius = radius

    def setup_optimizer(self, optim_cfg):
        pass

    def prepare_custom_data(self, idx, down_sample=True,
                            orthogonal=True, pose=None,
                            load_depth=True, **kwargs):
        image_name = self.data[idx]
        intrinsics = self.intrinsic
        uid = idx

        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            original_image = Image.open(image_name).convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {image_name}")
        
        width, height = original_image.size
        if min(width, height) > 1000:
            original_image = original_image.resize(
                (width // 2, height // 2), Image.LANCZOS)
            width, height = original_image.size
        image_np = np.asarray(original_image) / 255.0
        color_torch = torch.from_numpy(np.asarray(
            original_image) / 255.0).permute(2, 0, 1).float()
        if orthogonal:
            R = np.eye(3)
            t = np.zeros(3)
        elif pose is not None:
            # get_RT返回C2W，但Camera期望W2C，需要求逆
            pose_np = pose.numpy() if hasattr(pose, 'numpy') else pose
            pose_w2c = np.linalg.inv(pose_np)
            R = pose_w2c[:3, :3]
            t = pose_w2c[:3, 3]
        else:
            R = np.eye(3)
            t = np.zeros(3)
        focal_length_x = self.intrinsic[0, 0]
        focal_length_y = self.intrinsic[1, 1]
        FoVy = focal2fov(focal_length_y, height)
        FoVx = focal2fov(focal_length_x, width)

        
        cam_info = {}
        pose_src = np.eye(4)
        cam_info["gt_pose"] = copy(pose_src)
        cam_info["intrinsics"] = intrinsics

        cam_info["FoVx"] = FoVx
        cam_info["FoVy"] = FoVy
        cam_info["R"] = R
        cam_info["t"] = t


        # if load_depth:
        #     if idx not in self.mono_depth:
        #         depth_tensor = self.predict_depth(np.asarray(original_image))
        #         # depth_tensor = self.depth_model.infer_pil(image_pil, output_type='tensor')
        #         depth_tensor[depth_tensor < self.near] = self.near
        #         self.mono_depth[idx] = depth_tensor.cuda()
        #     else:
        #         depth_tensor = self.mono_depth[idx]
        # else:
        #     w, h = original_image.size
        #     depth_tensor = torch.ones((h, w))
        #     self.mono_depth[idx] = depth_tensor.cuda()

        # intr_mat_tensor = torch.from_numpy(
        #     intrinsics).float().to(depth_tensor.device)
        # pts = depth_to_3d(depth_tensor[None, None],
        #                   intr_mat_tensor[None],
        #                   normalize_points=False)

        # points = pts[0].permute(1, 2, 0).cpu().numpy().reshape(-1, 3)

        # viewpoint_camera = Camera(idx, R, t, FoVx, FoVy, color_torch,
        #                         gt_alpha_mask=None, image_name=image_name,
        #                         intrinsics=self.intrinsic,
        #                         uid=idx, is_co3d=True)

        # pcd_data = o3d.geometry.PointCloud()
        # pcd_data.points = o3d.utility.Vector3dVector(points)
        # pcd_data.colors = o3d.utility.Vector3dVector(image_np.reshape(-1, 3))
        # pcd_data.estimate_normals()
        # if down_sample:
        #     voxel_size = 0.01
        #     while len(pcd_data.points)> 1_000_000:
        #         pcd_data = pcd_data.voxel_down_sample(voxel_size=voxel_size)
        #         voxel_size *= 5

        # colors = np.asarray(pcd_data.colors, dtype=np.float32)
        # points = np.asarray(pcd_data.points, dtype=np.float32)
        # normals = np.asarray(pcd_data.normals, dtype=np.float32)
        # pcd = BasicPointCloud(points, colors, normals)
        # 存储LiDAR点云（用于后续全局点云输出）
        lidar_pcd = None
        lidar_depth_sparse = None
        lidar_points_raw = None
        lidar_points_cam = None

        if load_depth:
            if idx not in self.mono_depth:
                lidar_dir = os.path.join(os.path.dirname(os.path.dirname(self.data[idx])), "lidar")
                if os.path.isdir(lidar_dir):
                    # 获取当前的LiDAR外参（可能是优化后的）
                    if hasattr(self, 'get_current_lidar2cam'):
                        T_lidar2cam = self.get_current_lidar2cam()
                        if isinstance(T_lidar2cam, torch.Tensor):
                            T_lidar2cam = T_lidar2cam.detach().cpu().numpy()
                    else:
                        T_lidar2cam = get_lidar2cam_gt()

                    # 加载LiDAR点云（用于全局点云输出）
                    if hasattr(self, 'save_global_pcd') and self.save_global_pcd:
                        lidar_pcd = lidar_pointcloud_from_image(
                            image_name, lidar_dir, T_lidar2cam, max_range=80.0
                        )

                    # 加载原始LiDAR点并变换到相机系 (不确定度传播/标定精化用)
                    from utils.lidar_loader import read_pcd
                    base_name = os.path.splitext(os.path.basename(image_name))[0]
                    lidar_path = os.path.join(lidar_dir, base_name + '.pcd')
                    if os.path.exists(lidar_path):
                        lidar_points_raw = read_pcd(lidar_path)
                        T_mat = torch.from_numpy(T_lidar2cam).float().cuda()
                        pts_l = torch.from_numpy(lidar_points_raw).float().cuda()
                        pts_c = (T_mat[:3, :3] @ pts_l.T + T_mat[:3, 3:4]).T
                        valid_c = (pts_c[:, 2] > 0) & (pts_c[:, 2] < 80.0)
                        lidar_points_cam = pts_c[valid_c]

                    lidar_depth_sparse = lidar_depth_from_image(
                        image_name, lidar_dir, T_lidar2cam, intrinsics, height, width
                    )
                    
                    # 深度尺度匹配
                    if hasattr(self, 'use_depth_scale_match') and self.use_depth_scale_match:
                        # 先预测单目深度
                        mono_depth_tensor = self.predict_depth(np.asarray(original_image))
                        mono_depth_tensor = mono_depth_tensor.cuda()
                        
                        # 获取雷达深度
                        lidar_depth_tensor = lidar_depth_from_image(
                            image_name, lidar_dir, T_lidar2cam, intrinsics, height, width
                        )
                        
                        # 尺度匹配
                        if hasattr(self, 'scale_matcher') and self.scale_matcher is not None:
                            scale = self.scale_matcher.compute_scale(
                                lidar_depth_tensor, mono_depth_tensor
                            )
                            depth_tensor = mono_depth_tensor * scale
                        else:
                            depth_tensor = lidar_depth_tensor
                    else:
                        depth_tensor = lidar_depth_from_image(
                            image_name, lidar_dir, T_lidar2cam, intrinsics, height, width
                        )
                else:
                    depth_tensor = self.predict_depth(np.asarray(original_image))
                # 统一无效值为 near
                depth_tensor = torch.nan_to_num(depth_tensor, nan=self.near)
                self.mono_depth[idx] = depth_tensor.cuda()
            else:
                depth_tensor = self.mono_depth[idx]
        else:
            depth_tensor = torch.ones((height, width), device="cuda")

        # 生成点云：使用深度图反投影（保证密度）
        # 如果有LiDAR，用LiDAR校正单目深度尺度；否则直接用单目深度
        intr_mat_tensor = torch.from_numpy(intrinsics).float().to(depth_tensor.device)
        pts = depth_to_3d(depth_tensor[None, None], intr_mat_tensor[None], normalize_points=False)
        points = pts[0].permute(1, 2, 0).cpu().numpy().reshape(-1, 3)
        
        # 相机坐标系 → 世界坐标系（orthogonal 时不变）
        if not orthogonal and pose is not None:
            pose_inv = np.linalg.inv(pose)
            points = (pose_inv[:3, :3] @ points.T).T + pose_inv[:3, 3]
        
        colors = image_np.reshape(-1, 3)
        
        # 过滤掉无效点（nan或距离太远）
        valid_mask = np.isfinite(points).all(axis=1) & (np.linalg.norm(points, axis=1) < 100)
        points = points[valid_mask]
        colors = colors[valid_mask]
        
        print(f"[Depth Init] Using {len(points)} valid depth points for initialization")
        
        # 构造 Gaussian 点云
        pcd_data = o3d.geometry.PointCloud()
        pcd_data.points = o3d.utility.Vector3dVector(points)
        pcd_data.colors = o3d.utility.Vector3dVector(colors)
        pcd_data.estimate_normals()

        if down_sample and len(pcd_data.points) > 1_000_000:
            voxel_size = 0.01
            while len(pcd_data.points) > 1_000_000:
                pcd_data = pcd_data.voxel_down_sample(voxel_size=voxel_size)
                voxel_size *= 5

        colors = np.asarray(pcd_data.colors, dtype=np.float32)
        points = np.asarray(pcd_data.points, dtype=np.float32)
        normals = np.asarray(pcd_data.normals, dtype=np.float32)
        pcd = BasicPointCloud(points, colors, normals)

        viewpoint_camera = Camera(idx, R, t, FoVx, FoVy, color_torch,
                                  gt_alpha_mask=None, image_name=image_name,
                                  intrinsics=intrinsics, uid=uid, is_co3d=True)
        
        viewpoint_camera.depth_gt = depth_tensor     # 供 evolve_anchors 使用

        # 稀疏LiDAR深度/采样计数/相机系点 (不确定度加权监督, Eq.18/26)
        if lidar_depth_sparse is not None:
            viewpoint_camera.lidar_depth_sparse = lidar_depth_sparse.cuda()
            valid_l = torch.isfinite(lidar_depth_sparse) & \
                      (lidar_depth_sparse > 0) & (lidar_depth_sparse < 80.0)
            viewpoint_camera.lidar_valid_mask = valid_l.cuda()
            viewpoint_camera.lidar_count = (
                F.avg_pool2d(valid_l.float()[None, None], kernel_size=5,
                             stride=1, padding=2)[0, 0] * 25.0).cuda()
        if lidar_points_raw is not None:
            viewpoint_camera.lidar_points = lidar_points_raw
        if lidar_points_cam is not None:
            viewpoint_camera.lidar_points_cam = lidar_points_cam
        
        # 存储LiDAR点云（用于全局点云输出）
        if lidar_pcd is not None:
            if not hasattr(self, 'lidar_point_clouds'):
                self.lidar_point_clouds = []
            # 确保列表长度足够
            while len(self.lidar_point_clouds) <= idx:
                self.lidar_point_clouds.append(None)
            self.lidar_point_clouds[idx] = lidar_pcd

        return cam_info, pcd, viewpoint_camera


    def prepare_data(self, idx, down_sample=True,
                     orthogonal=True, learn_pose=False,
                     pose=None, load_depth=True,
                     load_gt=False):
        return self.prepare_custom_data(idx, down_sample=down_sample,
                                        orthogonal=orthogonal,
                                        pose=pose,
                                        load_depth=load_depth)

    def load_viewpoint_cam(self, idx, pose=None, load_depth=False):
        density_map = None
        image_name = self.data[idx]
        intrinsics = self.intrinsic.copy()

        if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            original_image = Image.open(image_name).convert("RGB")
        else:
            raise ValueError(f"Unsupported image format: {image_name}")
        width, height = original_image.size
        w, h = original_image.size # original_image
        if min(width, height) > 1000:
            original_image = original_image.resize(
                (width // 2, height // 2), Image.LANCZOS)
            width, height = original_image.size
        color_torch = torch.from_numpy(np.asarray(
            original_image) / 255.0).permute(2, 0, 1).float()
        if pose is None:
            R = np.eye(3)
            t = np.zeros(3)
        else:
            # get_RT返回C2W，但Camera期望W2C，需要求逆
            pose_np = pose.numpy() if hasattr(pose, 'numpy') else pose
            pose_w2c = np.linalg.inv(pose_np)
            R = pose_w2c[:3, :3]
            t = pose_w2c[:3, 3]
        focal_length_x = self.intrinsic[0, 0]
        focal_length_y = self.intrinsic[1, 1]
        FoVy = focal2fov(focal_length_y, height)
        FoVx = focal2fov(focal_length_x, width)
        viewpoint_camera = Camera(idx, R, t, FoVx, FoVy, color_torch,
                                gt_alpha_mask=None, image_name=image_name,
                                intrinsics=self.intrinsic,
                                uid=idx, is_co3d=True)
        if load_depth:
            if idx not in self.mono_depth:
                depth_tensor = self.predict_depth(np.asarray(original_image))
                self.mono_depth[idx] = depth_tensor.cuda()

        if load_depth:
            if idx not in self.mono_depth:
                depth_tensor = self.predict_depth((np.asarray(original_image) * 255).astype(np.uint8))
                depth_tensor = torch.nan_to_num(depth_tensor, nan=self.near)
                self.mono_depth[idx] = depth_tensor.cuda()
            else:
                depth_tensor = self.mono_depth[idx]
            viewpoint_camera.depth_gt = depth_tensor          # 供损失用
        else:
            viewpoint_camera.depth_gt = None

        # ---------- 3. 生成密度图 ----------
        # ---------- 密度图：一次性插值 + 缓存 ----------
        if load_depth and hasattr(viewpoint_camera, 'depth_gt') and viewpoint_camera.depth_gt is not None:
            if not hasattr(viewpoint_camera, 'density_map_cached'):          # 只生成一次
                with torch.no_grad():
                    target_h, target_w = viewpoint_camera.depth_gt.shape
                    density_map_raw = getattr(viewpoint_camera, 'density_map_raw', None)       # 可能是 None 或 numpy
                    if density_map_raw is None:
                        viewpoint_camera.density_map_cached = None
                    else:
                        # 先转 Tensor 再 GPU 插值 → 与深度同尺寸
                        density_tensor = torch.from_numpy(density_map_raw).unsqueeze(0).unsqueeze(0).cuda()
                        density_resized = F.interpolate(density_tensor,
                                                        size=(target_h, target_w),
                                                        mode='bilinear',
                                                        align_corners=False).squeeze()
                        viewpoint_camera.density_map_cached = density_resized   # 缓存
            viewpoint_camera.density_map = viewpoint_camera.density_map_cached   # 挂接
        else:
            viewpoint_camera.density_map = None
                

        return viewpoint_camera

    def train_step(self, viewpoint_cam,
                   iteration, background,
                   pipe, optim_opt, colors_precomp=None):
        # Render
        render_pkg = render(viewpoint_cam, self.model, pipe, background,
                            override_color=colors_precomp)
        image, viewspace_point_tensor, visibility_filter, radii = (render_pkg["render"],
                                                                   render_pkg["viewspace_points"],
                                                                   render_pkg["visibility_filter"],
                                                                   render_pkg["radii"])
        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        # loss = (1.0 - optim_opt.lambda_dssim) * Ll1 + optim_opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss = Ll1
        loss.backward()

        with torch.no_grad():
            # Progress bar
            self.ema_loss_for_log = 0.4 * loss.item() + 0.6 * self.ema_loss_for_log
            psnr_train = psnr(image, gt_image).mean().double()

            if iteration < optim_opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                self.model.max_radii2D[visibility_filter] = torch.max(self.model.max_radii2D[visibility_filter],
                                                                      radii[visibility_filter])
                self.model.add_densification_stats(
                    viewspace_point_tensor, visibility_filter)

                if iteration > optim_opt.densify_from_iter and iteration % optim_opt.densification_interval == 0:
                    size_threshold = 20 if iteration > optim_opt.opacity_reset_interval else None
                    self.model.densify_and_prune(optim_opt.densify_grad_threshold, 0.005,
                                                 self.radius, size_threshold)

                if iteration % optim_opt.opacity_reset_interval == 0:
                    self.model.reset_opacity()
            self.model.optimizer.step()
            self.model.optimizer.zero_grad(set_to_none=True)

        return loss, render_pkg, psnr_train

    def train(self, pipe, optim_opt, viewpoint_cam, colors_precomp=None):
        # fit the following frames
        # _, _, viewpoint_cam = self.prepare_data(idx, orthogonal=True)
        bg_color = [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if colors_precomp is not None:
            background = torch.zeros_like(colors_precomp[0])
        progress_bar = tqdm(range(optim_opt.iterations),
                            desc="Training progress")
        self.ema_loss_for_log = 0.0
        # self.model.training_setup(optim_opt)
        for iteration in range(1, optim_opt.iterations):
            # Update learning rate
            self.model.update_learning_rate(iteration)

            # Every 1000 its we increase the levels of SH up to a maximum degree
            if iteration % 1000 == 0:
                self.model.oneupSHdegree()

            # viewpoint_cam = self.create_viewpoint()
            loss, rend_dict, psnr_train = self.train_step(viewpoint_cam, iteration,
                                                          background, pipe, optim_opt,
                                                          colors_precomp=colors_precomp)

            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_train:.{2}f}"})
                progress_bar.update(10)
            if iteration == optim_opt.iterations:
                progress_bar.close()

    def obtain_center_feat(self,):
        pass

    def visualize(self, render_pkg, filename):
        if "depth" in render_pkg:
            rend_depth = Image.fromarray(
                colorize(render_pkg["depth"].detach().cpu().numpy(),
                         cmap='magma_r')).convert("RGB")
            rend_depth.save(filename.replace(".png", "_depth.png"))
        if "acc" in render_pkg:
            rend_acc = Image.fromarray(
                colorize(render_pkg["acc"].detach().cpu().numpy(),
                         cmap='magma_r')).convert("RGB")
            rend_acc.save(filename.replace(".png", "_acc.png"))

        rend_img = Image.fromarray(
            np.asarray(render_pkg["render"].detach().cpu().permute(1,
                                                                   2, 0).numpy() * 255.0, dtype=np.uint8)).convert("RGB")
        rend_img.save(filename)
