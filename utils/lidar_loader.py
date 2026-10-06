import os
import numpy as np
import torch
import open3d as o3d
from kornia.geometry import project_points


def read_pcd(pcd_path):
    """读取 .pcd 文件，返回 (N, 3) 的 numpy 数组"""
    pcd = o3d.io.read_point_cloud(pcd_path)
    points = np.asarray(pcd.points, dtype=np.float32)
    return points


def lidar_depth_from_image(image_name,
                           lidar_dir,
                           T_lidar2cam,
                           intrinsics,
                           H,
                           W,
                           max_range=80.0):
    """
    根据图像文件名，加载对应 .pcd 文件，投影到图像平面生成深度图

    Args:
        image_name: 图像文件名，如 '00000.jpg'
        lidar_dir: lidar文件夹路径，如 '.../lidar'
        T_lidar2cam: (4, 4) 外参，LiDAR -> Camera
        intrinsics: (3, 3) 内参
        H, W: 图像高宽
        max_range: 最大深度，超过则过滤

    Returns:
        depth_map: (H, W) torch.Tensor，无效区域为 nan
    """
    # 构造 lidar 文件路径
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    lidar_path = os.path.join(lidar_dir, base_name + '.pcd')

    # 读取点云
    points = read_pcd(lidar_path)  # (N, 3)
    points = torch.from_numpy(points).cuda()

    # 转到相机坐标系
    if isinstance(T_lidar2cam, torch.Tensor):
        T = T_lidar2cam.float().cuda()
    else:
        T = torch.from_numpy(T_lidar2cam).float().cuda()
    
    points_cam = (T[:3, :3] @ points.T + T[:3, 3:4]).T  # (N, 3)

    # 过滤范围外点
    z = points_cam[:, 2]
    valid_z = (z > 0) & (z < max_range)
    points_cam = points_cam[valid_z]
    z = z[valid_z]

    # 投影到图像平面
    K = torch.from_numpy(intrinsics).float().cuda()
    uv = project_points(points_cam[None], K[None])[0]  # (N, 2)
    u, v = uv[:, 0], uv[:, 1]
    u_round = torch.round(u).long()
    v_round = torch.round(v).long()

    # 构造深度图
    depth_map = torch.full((H, W), float('nan'), device='cuda')
    valid = (u_round >= 0) & (u_round < W) & (v_round >= 0) & (v_round < H)
    depth_map[v_round[valid], u_round[valid]] = z[valid]

    return depth_map


def lidar_depth_with_scale(image_name,
                           lidar_dir,
                           T_lidar2cam,
                           intrinsics,
                           H,
                           W,
                           mono_depth,
                           max_range=80.0,
                           scale_matcher=None):
    """
    生成雷达深度图，并与单目深度进行尺度匹配
    
    Args:
        image_name: 图像文件名
        lidar_dir: lidar文件夹路径
        T_lidar2cam: lidar2cam 外参 (4,4) tensor或numpy
        intrinsics: (3,3) 内参
        H, W: 图像尺寸
        mono_depth: 单目深度图 (H, W) tensor，用于尺度匹配
        max_range: 最大深度
        scale_matcher: 尺度匹配器，如果为None则返回原始雷达深度
    
    Returns:
        lidar_depth: 雷达深度图 (H, W)
        scale: 估计的尺度因子
        scaled_mono_depth: 尺度校正后的单目深度
    """
    # 获取雷达深度
    lidar_depth = lidar_depth_from_image(
        image_name, lidar_dir, T_lidar2cam, intrinsics, H, W, max_range
    )
    
    scale = 1.0
    
    if scale_matcher is not None and mono_depth is not None:
        # 计算尺度因子
        scale = scale_matcher.compute_scale(lidar_depth, mono_depth)
        
        # 校正单目深度
        scaled_mono_depth = mono_depth * scale
    else:
        scaled_mono_depth = mono_depth
    
    return lidar_depth, scale, scaled_mono_depth


def lidar_pointcloud_from_image(image_name,
                                lidar_dir,
                                T_lidar2cam,
                                max_range=80.0) -> o3d.geometry.PointCloud:
    """
    从图像名加载对应的雷达点云（LiDAR坐标系）
    
    Args:
        image_name: 图像文件名
        lidar_dir: lidar文件夹路径
        T_lidar2cam: lidar2cam 外参（用于验证）
        max_range: 最大深度
    
    Returns:
        pcd: Open3D 点云对象
    """
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    lidar_path = os.path.join(lidar_dir, base_name + '.pcd')
    
    pcd = o3d.io.read_point_cloud(lidar_path)
    
    # 过滤范围外点
    points = np.asarray(pcd.points)
    distances = np.linalg.norm(points, axis=1)
    mask = (distances > 0) & (distances < max_range)
    
    pcd = pcd.select_by_index(np.where(mask)[0])
    
    return pcd


def transform_lidar_to_cam(pcd_lidar: o3d.geometry.PointCloud,
                           T_lidar2cam) -> o3d.geometry.PointCloud:
    """
    将雷达点云从 LiDAR 坐标系变换到相机坐标系
    
    Args:
        pcd_lidar: LiDAR 坐标系下的点云
        T_lidar2cam: lidar2cam 变换 (4,4)
    
    Returns:
        pcd_cam: 相机坐标系下的点云
    """
    pcd_cam = o3d.geometry.PointCloud(pcd_lidar)
    
    if isinstance(T_lidar2cam, torch.Tensor):
        T = T_lidar2cam.cpu().numpy()
    else:
        T = T_lidar2cam
    
    pcd_cam.transform(T)
    return pcd_cam


def get_lidar2cam_gt():
    """lidar2camera 真值"""
    # T = np.array([
    #     [ 0.03415912395, 0.99937821808, -0.00867927265000001, -0.0494428],
    #     [ 0.03370043192,  -0.00983067249999992, -0.99938310544, -0.176327],
    #     [ -0.99884755265,  0.03384505544,  -0.0340147964499999, -0.371055],
    #     [ 0.0,         0.0,         0.0,         1.0       ]
    # ], dtype=np.float32) # rellis3d lidar2cam
    # T = np.array([
    #     [ 0.03486164, -0.99821106, -0.04858227, -0.60816898],
    #     [ -0.0095559,  0.04827666, -0.99878854, -0.09604881],
    #     [ 0.99934662,  0.03528365,  -0.0078558, -0.07391513],
    #     [ 0.0,         0.0,         0.0,         1.0       ]
    # ], dtype=np.float32) # automine lidar2cam
    T = np.array([
        [ 0.00389515, -0.99997991, -0.00500164, -0.00687909],
        [ 0.27269952,  0.00587431, -0.96208132,  0.00118207],
        [ 0.96209137,  0.00238351,  0.27271692, -0.07248875],
        [ 0.0,         0.0,         0.0,         1.0       ]
    ], dtype=np.float32)
    return T #WILDSCENE lidar2cam


def get_default_intrinsics():
    """获取默认相机内参（用于测试）"""
    return np.array([
        [1000, 0, 640],
        [0, 1000, 360],
        [0, 0, 1]
    ], dtype=np.float32)


def lidar_pointcloud_for_gs_init(image_name,
                                  lidar_dir,
                                  T_lidar2cam,
                                  intrinsics,
                                  image_np,
                                  max_range=80.0):
    """
    从LiDAR文件加载点云，投影到图像获取颜色，返回相机坐标系下的点云
    只返回在图像范围内的点
    
    Args:
        image_name: 图像文件名
        lidar_dir: lidar文件夹路径
        T_lidar2cam: lidar2cam 外参
        intrinsics: (3,3) 相机内参
        image_np: (H,W,3) 图像数组，用于提取颜色
        max_range: 最大深度范围
        
    Returns:
        points: (N,3) 相机坐标系下的点（OpenGL坐标系：相机看向-Z）
        colors: (N,3) 颜色值
    """
    import os
    
    # 加载LiDAR点云
    base_name = os.path.splitext(os.path.basename(image_name))[0]
    lidar_path = os.path.join(lidar_dir, base_name + '.pcd')
    
    pcd = o3d.io.read_point_cloud(lidar_path)
    points_lidar = np.asarray(pcd.points, dtype=np.float32)
    
    if len(points_lidar) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    
    # 转换到相机坐标系
    if isinstance(T_lidar2cam, torch.Tensor):
        T = T_lidar2cam.cpu().numpy()
    else:
        T = T_lidar2cam
        
    points_h = np.hstack([points_lidar, np.ones((len(points_lidar), 1))])
    points_cam = (T @ points_h.T).T[:, :3]
    
    # 过滤范围
    distances = np.linalg.norm(points_cam, axis=1)
    valid_mask = (distances > 0.1) & (distances < max_range) & (points_cam[:, 2] > 0.1)
    points_cam = points_cam[valid_mask]
    
    if len(points_cam) == 0:
        return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    
    # 投影到图像
    H, W = image_np.shape[:2]
    fx, fy = intrinsics[0, 0], intrinsics[1, 1]
    cx, cy = intrinsics[0, 2], intrinsics[1, 2]
    
    x, y, z = points_cam[:, 0], points_cam[:, 1], points_cam[:, 2]
    u = (fx * x / z + cx).astype(np.int32)
    v = (fy * y / z + cy).astype(np.int32)
    
    # 只保留在图像范围内的点
    in_image = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    points_cam = points_cam[in_image]
    u = u[in_image]
    v = v[in_image]
    
    # 获取颜色
    colors = image_np[v, u]
    
    # 关键修复：OpenGL相机坐标系中相机看向-Z方向
    # LiDAR的Z正方向（相机前面）需要翻转为-Z
    points_cam_opengl = points_cam.copy()
    points_cam_opengl[:, 2] = -points_cam_opengl[:, 2]
    
    return points_cam_opengl.astype(np.float32), colors.astype(np.float32)
