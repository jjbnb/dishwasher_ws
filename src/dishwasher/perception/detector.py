"""
Level 1 感知模块——直接从仿真获取 ground truth 位置。

Level 1 场景简单（盘子整齐排列、无遮挡），不需要真正的相机感知。
直接从 RigidObject 读取物体位姿即可。

后续 Level 2 会替换为基于相机 RGB-D 的真实感知管线。
"""

from __future__ import annotations

import torch


def get_plate_positions(plates: dict[str, any]) -> list[dict]:
    """获取所有盘子在 world frame 中的位姿。

    Args:
        plates: SceneLoader.plates — dict[str, RigidObject]

    Returns:
        盘子信息列表，按名称排序:
        [{name, pos (3,), quat (4,), lin_vel (3,), ang_vel (3,)}, ...]
    """
    result = []
    for name in sorted(plates.keys()):
        plate = plates[name]
        pos = plate.data.body_pos_w[0, 0].cpu().numpy()      # (3,)
        quat = plate.data.body_quat_w[0, 0].cpu().numpy()    # (4,) w,x,y,z
        lin_vel = plate.data.body_lin_vel_w[0, 0].cpu().numpy()
        ang_vel = plate.data.body_ang_vel_w[0, 0].cpu().numpy()
        result.append({
            "name": name,
            "pos": pos,
            "quat": quat,
            "lin_vel": lin_vel,
            "ang_vel": ang_vel,
        })
    return result


def get_plate_centers(plates: dict[str, any]) -> torch.Tensor:
    """获取所有盘子中心位置（world frame）。

    Args:
        plates: SceneLoader.plates

    Returns:
        shape (N, 3) 位置张量
    """
    positions = []
    for name in sorted(plates.keys()):
        pos = plates[name].data.body_pos_w[0, 0]
        positions.append(pos)
    if positions:
        return torch.stack(positions)
    return torch.empty(0, 3)


def get_next_plate(plates: dict[str, any], processed: set[str]) -> dict | None:
    """获取下一个未处理的盘子。

    Args:
        plates: SceneLoader.plates
        processed: 已处理盘子名称集合

    Returns:
        下一个盘子信息 dict，若全部处理完则 None
    """
    for name in sorted(plates.keys()):
        if name not in processed:
            pos = plates[name].data.body_pos_w[0, 0].cpu().numpy()
            quat = plates[name].data.body_quat_w[0, 0].cpu().numpy()
            return {"name": name, "pos": pos, "quat": quat}
    return None


def is_plate_in_basin(plate_pos: torch.Tensor, basin_center=(0.5, 0.0, 0.05),
                      basin_threshold=(0.15, 0.10, 0.10)) -> bool:
    """检查盘子是否已在水槽区域内。

    Args:
        plate_pos: 盘子世界坐标 (3,)
        basin_center: 水槽中心 (x, y, z)
        basin_threshold: 各轴容差 (dx, dy, dz)

    Returns:
        True 如果盘子在水槽内
    """
    dx = abs(float(plate_pos[0]) - basin_center[0])
    dy = abs(float(plate_pos[1]) - basin_center[1])
    dz = abs(float(plate_pos[2]) - basin_center[2])
    return dx < basin_threshold[0] and dy < basin_threshold[1] and dz < basin_threshold[2]
