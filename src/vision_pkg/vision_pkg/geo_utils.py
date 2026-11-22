# -*- coding: utf-8 -*-
"""
geo_utils.py —— 机载视觉几何工具

功能：把相机图像中的 2D 像素坐标 (u, v) 反投影为全局 ENU 坐标系下的 3D 点。

坐标系约定（严格遵守《课题二内部接口规范 V3.0》）：
  - 世界系：ENU（东-北-天，Z 轴向上为正），frame_id 必须为 "world"
  - 本模块严禁做 ENU<->NED 转换（NED 转换由飞控桥接统一负责）
  - 相机内参来自接口K：/drone_<id>/camera/camera_info (sensor_msgs/CameraInfo)
  - 无人机位姿来自 /drone_<id>/state (nav_msgs/Odometry)，为 ENU 世界系位姿

几何原理（针孔相机模型 + 地面平面求交）：
  1. 像素 (u,v) -> 相机系射线 d_cam = [(u-cx)/fx, (v-cy)/fy, 1]
  2. d_cam -> 世界系射线 d_world = R_wb @ (R_bc @ d_cam)
     其中 R_wb 为无人机姿态四元数对应的旋转矩阵，R_bc 为相机相对机体的安装外参
  3. 射线 p = p0 + t*d 与地面平面 z = ground_z 求交解出 t，得到目标世界坐标

默认安装方式：相机垂直朝下安装（Gazebo 下视相机最常见布局）：
  - 相机光轴 (cam z) 指向机体 -Z（即指向地面）
  - 相机 x 轴朝机头方向，y 轴朝机体左侧
  - 对应旋转矩阵 R_bc = [[1,0,0],[0,-1,0],[0,0,-1]]
  如果你的相机安装方式不同（如 45 度前视、带云台），改参数 camera_mount_rpy 即可。
"""

import math
import numpy as np


def quat_to_matrix(x, y, z, w):
    """四元数 (x, y, z, w) -> 3x3 旋转矩阵（标准右手系公式）。"""
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-9:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x * x + z * z),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def euler_to_matrix(roll, pitch, yaw):
    """ZYX 欧拉角（弧度）-> 旋转矩阵 R = Rz(yaw) @ Ry(pitch) @ Rx(roll)。"""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def camera_mount_matrix(roll=0.0, pitch=0.0, yaw=0.0):
    """
    相机安装外参：机体坐标系 -> 相机坐标系的旋转 R_bc。

    默认参数 (0,0,0) 对应"相机垂直朝下、镜头朝地"的标准下视安装：
        R_bc = diag(1,-1,-1) 的等价表达见下。
    ROS 相机光学坐标系约定：x 向右、y 向下、z 沿光轴向前。
    机体 FLU 坐标系：x 向前、y 向左、z 向上。
    下视安装即：cam_x = body_x, cam_y = -body_y, cam_z = -body_z。
    """
    if abs(roll) < 1e-12 and abs(pitch) < 1e-12 and abs(yaw) < 1e-12:
        return np.array([[1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0],
                         [0.0, 0.0, -1.0]])
    base = np.array([[1.0, 0.0, 0.0],
                     [0.0, -1.0, 0.0],
                     [0.0, 0.0, -1.0]])
    return euler_to_matrix(roll, pitch, yaw) @ base


def _unknown_cov(unknown_var=1e6):
    """返回“位置未知、姿态无意义”的 6x6 协方差（行优先，PoseWithCovarianceStamped 顺序）。"""
    cov = np.zeros((6, 6))
    cov[0, 0] = cov[1, 1] = cov[2, 2] = unknown_var
    cov[3, 3] = cov[4, 4] = cov[5, 5] = 1.0
    return cov


def pixel_to_world_cov(u, v, camera_info_k, drone_position, drone_quaternion,
                       R_bc, ground_z=0.0, sigma_px=2.0, unknown_var=1e6):
    """
    把像素坐标反投影到世界 ENU 系，并附带位置不确定度（协方差）。

    与 pixel_to_world 的区别：
      - 定位误差会反映到协方差：目标越远、射线越贴近地面（|dz| 越小），位置误差越大。
      - 当射线朝天/平行地面、无法与地面求交时，不再返回 None，而是返回
        valid=False 的高不确定度结果（位置退化为相机原点，协方差为 unknown_var），
        由下游根据协方差自行取舍。

    参数：
        u, v              : 像素坐标（列、行）
        camera_info_k     : CameraInfo.k，9 元素内参矩阵 [fx,0,cx, 0,fy,cy, 0,0,1]
        drone_position    : (px, py, pz) 无人机在 ENU 世界系的位置
        drone_quaternion  : (qx, qy, qz, qw) 无人机姿态（ENU 世界系）
        R_bc              : 相机安装旋转矩阵（见 camera_mount_matrix）
        ground_z          : 目标所处平面高度（默认地面 0.0）
        sigma_px          : 像素检测不确定度（像素），用于估计角度误差
        unknown_var       : 位置不可得时回填的协方差方差值

    返回：
        (position, covariance, valid)
        position   : (x, y, z) 世界系坐标；valid=False 时退化为相机原点占位
        covariance : 6x6 行优先矩阵（顺序 x,y,z,roll,pitch,yaw）
        valid      : 是否与地面平面成功求交（False 表示距离不可得，仅有方位）
    """
    k = camera_info_k
    fx, fy, cx, cy = k[0], k[4], k[2], k[5]

    # 内参无效：连射线方向都算不出，返回全未知
    if fx <= 1e-9 or fy <= 1e-9:
        return tuple(drone_position), _unknown_cov(unknown_var), False

    # 1) 相机系射线（归一化平面）
    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])

    # 2) 相机系 -> 世界系（单位化，便于后续求距离）
    R_wb = quat_to_matrix(*drone_quaternion)
    d_world = R_wb @ (R_bc @ d_cam)
    d_norm = np.linalg.norm(d_world)
    if d_norm < 1e-12:
        return tuple(drone_position), _unknown_cov(unknown_var), False
    d_world = d_world / d_norm

    p0 = np.array(drone_position, dtype=float)
    dz = d_world[2]

    # 3) 与平面 z = ground_z 求交： p0 + t*d_world 的 z 分量等于 ground_z
    if dz >= -1e-9:
        # 射线朝上或平行地面，距离不可得：仅方位有效，位置退化
        return tuple(p0), _unknown_cov(unknown_var), False
    t = (ground_z - p0[2]) / dz
    if t <= 0.0:
        # 交点在相机身后，无物理意义
        return tuple(p0), _unknown_cov(unknown_var), False

    p = p0 + t * d_world
    if not np.all(np.isfinite(p)):
        return tuple(p0), _unknown_cov(unknown_var), False

    # 4) 位置不确定度估计
    #    角度不确定度 sigma_angle = sigma_px / f；地面点误差约 sigma_angle * t。
    #    距离 t 越大、射线越贴地（|dz| 越小导致 t 越大），误差越大。
    f_avg = 0.5 * (fx + fy)
    sigma_angle = sigma_px / f_avg
    var_xy = (sigma_angle * t) ** 2
    var_xy = min(var_xy, unknown_var)  # 防溢出

    cov = np.zeros((6, 6))
    cov[0, 0] = var_xy
    cov[1, 1] = var_xy
    cov[2, 2] = 0.01  # z 被约束在地面平面上，近似确定
    cov[3, 3] = cov[4, 4] = cov[5, 5] = 1.0  # 点目标姿态无意义

    return (float(p[0]), float(p[1]), float(p[2])), cov, True


def pixel_size_to_world(u, v, w_px, camera_info_k, drone_position, drone_quaternion,
                        R_bc, target_real_width):
    """
    已知目标真实宽度时的单目测距定位（不依赖地面平面假设，空中目标可用）。

    解决规划组反馈的问题：地面投影法在目标不在地面（或射线近水平）时距离算不出来。
    本函数利用针孔模型的相似三角形关系直接解算距离：
        distance = fx * target_real_width / w_px
    然后沿像素射线前进该距离得到目标世界坐标：
        point = camera_pos + distance * d_world

    参数：
        u, v              : 目标中心像素坐标
        w_px              : 目标在图像中的像素宽度（包围盒宽）
        target_real_width : 目标真实宽度（米），需事先标定/约定
        其余参数同 pixel_to_world

    返回 (x, y, z) 世界系坐标；参数无效（w_px<=0、fx 无效、宽度未设置）返回 None。
    """
    if target_real_width <= 0.0 or w_px <= 0.0:
        return None
    k = camera_info_k
    fx, fy, cx, cy = k[0], k[4], k[2], k[5]
    if fx <= 1e-9 or fy <= 1e-9:
        return None

    d_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    R_wb = quat_to_matrix(*drone_quaternion)
    d_world = R_wb @ (R_bc @ d_cam)
    d_norm = np.linalg.norm(d_world)
    if d_norm < 1e-12:
        return None
    d_world = d_world / d_norm

    distance = fx * target_real_width / w_px
    p = np.array(drone_position, dtype=float) + distance * d_world
    if not np.all(np.isfinite(p)):
        return None
    return (float(p[0]), float(p[1]), float(p[2]))


def pixel_to_world(u, v, camera_info_k, drone_position, drone_quaternion,
                   R_bc, ground_z=0.0):
    """
    把像素坐标反投影到世界 ENU 系（与地面 z=ground_z 平面求交）。

    返回 (x, y, z) 世界系坐标；若无法求交（射线朝天、平行于地面、内参无效等）返回 None。
    如需附带位置不确定度，请使用 pixel_to_world_cov。
    """
    pos, _cov, valid = pixel_to_world_cov(
        u, v, camera_info_k, drone_position, drone_quaternion, R_bc,
        ground_z=ground_z)
    return pos if valid else None
