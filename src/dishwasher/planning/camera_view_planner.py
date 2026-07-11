"""Task-space wrist-camera view planning."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R


USD_FROM_ROS = np.diag([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class CameraViewPlan:
    """Desired camera and link6 pose for one wrist camera."""

    arm: str
    target_pos_w: np.ndarray
    camera_pos_w: np.ndarray
    camera_rotm_w_ros: np.ndarray
    camera_rotm_w_usd: np.ndarray
    link6_pos_w: np.ndarray
    link6_quat_w: np.ndarray


def plan_wrist_camera_view(
    *,
    arm: str,
    target_pos_w: np.ndarray,
    arm_base_pos_w: np.ndarray,
    camera_offset: dict,
    backoff: float = 40.0,
    height: float = 42.0,
    side_offset: float = 0.0,
) -> CameraViewPlan:
    """Plan a wrist-camera view in task space.

    The camera is placed above and behind the target, measured from the target
    toward the acting arm base. It then looks at the target with a ROS optical
    frame convention (+Z forward, +X right, +Y down). The result is converted
    into a link6 target using the calibrated fixed link6->camera transform.
    """

    target = np.asarray(target_pos_w, dtype=np.float64)
    base = np.asarray(arm_base_pos_w, dtype=np.float64)
    back_dir = _unit_xy(base[:2] - target[:2])
    side_dir = np.array([-back_dir[1], back_dir[0]], dtype=np.float64)
    if arm == "right":
        side_dir *= -1.0

    camera_xy = target[:2] + backoff * back_dir + side_offset * side_dir
    camera_pos = np.array(
        [camera_xy[0], camera_xy[1], target[2] + height],
        dtype=np.float64,
    )
    rot_ros = _look_at_ros(camera_pos, target)
    rot_usd = rot_ros @ USD_FROM_ROS

    link_rot = rot_usd @ np.asarray(camera_offset["body_rot_camera_usd"], dtype=np.float64).T
    link_pos = camera_pos - link_rot @ np.asarray(camera_offset["body_pos_camera"], dtype=np.float64)
    link_quat = _rotm_to_quat_wxyz(link_rot)

    return CameraViewPlan(
        arm=arm,
        target_pos_w=target,
        camera_pos_w=camera_pos,
        camera_rotm_w_ros=rot_ros,
        camera_rotm_w_usd=rot_usd,
        link6_pos_w=link_pos,
        link6_quat_w=link_quat,
    )


def _look_at_ros(camera_pos: np.ndarray, target_pos: np.ndarray) -> np.ndarray:
    forward = np.asarray(target_pos, dtype=np.float64) - np.asarray(camera_pos, dtype=np.float64)
    forward = forward / max(1.0e-9, float(np.linalg.norm(forward)))

    up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    right = np.cross(forward, up)
    if float(np.linalg.norm(right)) < 1.0e-6:
        right = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        right = right / float(np.linalg.norm(right))

    down = np.cross(forward, right)
    down = down / max(1.0e-9, float(np.linalg.norm(down)))
    return np.column_stack([right, down, forward])


def _unit_xy(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-6:
        return np.array([0.0, 1.0], dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / norm


def _rotm_to_quat_wxyz(rotm: np.ndarray) -> np.ndarray:
    quat_xyzw = R.from_matrix(rotm).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )
