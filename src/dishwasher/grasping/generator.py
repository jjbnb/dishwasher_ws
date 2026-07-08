"""
抓取姿态生成——Level 1 规则版本。

基于盘子中心位置，生成固定 offset 的侧面抓取姿态。
坐标系统：默认米制，Z 轴向上。原生 all.usd 使用厘米 stage units，
调用方应传入 unit_scale=100.0。

场景布局（来自 all.usd）：
  - Piper 底座: (0.90, 1.37, 0.144)
  - 水槽底部: Z≈-0.264，盘子静止在 Z≈-0.21~-0.20（浮在碰撞面上方）
  - 洗碗机卡槽 (dishwasher_basin_step): 中心 (1.35, 0.95, 0.02)
"""

from __future__ import annotations

import numpy as np
import torch


# ------------------------------------------------------------------
# 预定义抓取参数
# ------------------------------------------------------------------

# 盘子抓取 offset：夹爪从上方接近盘子
GRASP_OFFSET = {
    "pre_grasp_z_offset": 0.10,   # 预抓取位置在盘面上方 10cm
    "grasp_z_offset": 0.03,       # 实际抓取位置比盘面高 3cm（夹爪厚度）
    "post_grasp_z_offset": 0.15,  # 抓起后抬升 15cm
}

# 放置参数——洗碗机卡槽位置
# 来自 all.usd dishwasher_basin_step 世界坐标 ÷100 + Z_SHIFT(0.75)
# all.usd: (113.555cm, 74.844cm, -10.436cm) → (1.136m, 0.748m, -0.104m) → +Z_SHIFT
PLACE_POSITION = (1.136, 0.748, 0.646)    # (x, y, z) world frame, 卡槽释放位置
NATIVE_PLACE_POSITION = (113.555, 74.844, -10.436)  # all.usd stage units (cm)
PLACE_QUAT = (1.0, 0.0, 0.0, 0.0)        # (w, x, y, z) 夹爪竖直向下
PLACE_PRE_Z_OFFSET = 0.10                  # 放置前在卡槽上方悬停


# ------------------------------------------------------------------
# 抓取姿态生成函数
# ------------------------------------------------------------------

def generate_grasp_pose(
    plate_pos: np.ndarray,
    plate_quat: np.ndarray,
    unit_scale: float = 1.0,
) -> dict:
    """基于盘子中心位姿，生成侧面抓取的末端执行器目标姿态。

    方法：从盘子正上方接近，夹爪竖直向下。

    Args:
        plate_pos: 盘子中心位置 (3,) world frame
        plate_quat: 盘子姿态四元数 (4,) (w, x, y, z) world frame
        unit_scale: 长度缩放。米制场景为 1.0，原生 all.usd 厘米场景为 100.0

    Returns:
        {
            "pre_grasp": (pos (3,), quat (4,)),    # 预抓取位置（上方悬停）
            "grasp":     (pos (3,), quat (4,)),    # 实际抓取位置
            "post_grasp":(pos (3,), quat (4,)),    # 抓起后抬升
        }
    """
    px, py, pz = float(plate_pos[0]), float(plate_pos[1]), float(plate_pos[2])
    pre_offset = GRASP_OFFSET["pre_grasp_z_offset"] * unit_scale
    grasp_offset = GRASP_OFFSET["grasp_z_offset"] * unit_scale
    post_offset = GRASP_OFFSET["post_grasp_z_offset"] * unit_scale

    # 夹爪姿态：竖直向下抓取
    grasp_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    # 实际抓取：盘面 + 3cm
    grasp_pos = np.array(
        [px, py, pz + grasp_offset], dtype=np.float64
    )

    # 预抓取：上方 10cm 悬停
    pre_grasp_pos = np.array(
        [px, py, pz + pre_offset], dtype=np.float64
    )

    # 抓起后抬升 15cm
    post_grasp_pos = np.array(
        [px, py, pz + post_offset], dtype=np.float64
    )

    return {
        "pre_grasp": (pre_grasp_pos, grasp_quat.copy()),
        "grasp": (grasp_pos, grasp_quat.copy()),
        "post_grasp": (post_grasp_pos, grasp_quat.copy()),
    }


def generate_place_pose(
    place_idx: int = 0,
    unit_scale: float = 1.0,
    place_position: tuple[float, float, float] | None = None,
) -> dict:
    """生成放置目标姿态——洗碗机卡槽。

    Args:
        place_idx: 卡槽索引，用于计算 y 方向偏移（多个盘子依次放置）
        unit_scale: 长度缩放。米制场景为 1.0，原生 all.usd 厘米场景为 100.0
        place_position: 覆盖放置位置。None 时使用旧米制 PLACE_POSITION

    Returns:
        {
            "pre_place": (pos, quat),   # 卡槽上方悬停
            "place":     (pos, quat),   # 放置释放位置
        }
    """
    px, py, pz = place_position if place_position is not None else PLACE_POSITION

    # 多个盘子时沿 X 方向错开
    if place_idx > 0:
        px += place_idx * 0.04 * unit_scale  # 每个盘子 X 偏移 4cm

    place_quat = np.array(PLACE_QUAT, dtype=np.float64)
    place_pos = np.array([px, py, pz], dtype=np.float64)
    pre_place_pos = np.array(
        [px, py, pz + PLACE_PRE_Z_OFFSET * unit_scale], dtype=np.float64
    )

    return {
        "pre_place": (pre_place_pos, place_quat.copy()),
        "place": (place_pos, place_quat.copy()),
    }


# ------------------------------------------------------------------
# Torch 转换
# ------------------------------------------------------------------

def to_torch(
    pos: np.ndarray, quat: np.ndarray, device: str = "cuda:0"
) -> tuple[torch.Tensor, torch.Tensor]:
    """将 numpy pos/quat 转为 torch (1,3) / (1,4) 张量。"""
    return (
        torch.tensor(pos, dtype=torch.float32, device=device).unsqueeze(0),
        torch.tensor(quat, dtype=torch.float32, device=device).unsqueeze(0),
    )
