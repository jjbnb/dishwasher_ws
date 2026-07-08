"""
运动轨迹插值——位置线性插值 + 姿态 slerp。

Level 1 简单实现，不做碰撞规避。
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def interpolate_waypoints(
    start_pos: np.ndarray,
    start_quat: np.ndarray,
    end_pos: np.ndarray,
    end_quat: np.ndarray,
    num_steps: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """两点间线性插值（位置线性 + 姿态 slerp）。

    Args:
        start_pos: 起始位置 (3,)
        start_quat: 起始四元数 (4,) (w, x, y, z)
        end_pos: 目标位置 (3,)
        end_quat: 目标四元数 (4,) (w, x, y, z)
        num_steps: 插值步数（包含两端点）

    Returns:
        [(pos, quat), ...] 长度 num_steps 的位姿列表
    """
    if num_steps <= 1:
        return [(end_pos.copy(), end_quat.copy())]

    # 位置：线性插值
    t = np.linspace(0, 1, num_steps)
    positions = start_pos + np.outer(t, end_pos - start_pos)

    # 姿态：slerp
    key_rots = R.from_quat([
        [start_quat[1], start_quat[2], start_quat[3], start_quat[0]],  # xyzw
        [end_quat[1], end_quat[2], end_quat[3], end_quat[0]],
    ])
    slerp = Slerp([0, 1], key_rots)
    quats_xyzw = slerp(t).as_quat()  # (N, 4) xyzw

    # 转回 wxyz
    quats_wxyz = np.zeros_like(quats_xyzw)
    quats_wxyz[:, 0] = quats_xyzw[:, 3]
    quats_wxyz[:, 1] = quats_xyzw[:, 0]
    quats_wxyz[:, 2] = quats_xyzw[:, 1]
    quats_wxyz[:, 3] = quats_xyzw[:, 2]

    return [(positions[i], quats_wxyz[i]) for i in range(num_steps)]


def generate_approach_trajectory(
    current_ee_pos: np.ndarray,
    current_ee_quat: np.ndarray,
    target_pos: np.ndarray,
    target_quat: np.ndarray,
    approach_steps: int = 60,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """生成从当前位置到目标位姿的接近轨迹。

    Args:
        current_ee_pos: 末端当前世界位置 (3,)
        current_ee_quat: 末端当前世界四元数 (4,)
        target_pos: 目标世界位置 (3,)
        target_quat: 目标世界四元数 (4,)
        approach_steps: 接近步数

    Returns:
        插值位姿列表
    """
    return interpolate_waypoints(
        current_ee_pos, current_ee_quat,
        target_pos, target_quat,
        approach_steps,
    )
