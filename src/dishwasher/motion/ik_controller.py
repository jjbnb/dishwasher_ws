"""
Piper IK 控制器——封装 Isaac Lab 的 DifferentialIKController。

对 Piper 6-DOF 机械臂做逆运动学求解，输出关节目标位置。
Level 1 使用 "dls"（阻尼最小二乘）方法，比 pinv 在奇异点附近更稳定。

用法:
    ik = PiperIKController(robot, device="cuda:0")
    target_pos = torch.tensor([[x, y, z]], device=...)
    target_quat = torch.tensor([[qw, qx, qy, qz]], device=...)
    joint_pos_des = ik.solve(target_pos, target_quat)
    robot.set_joint_position_target(joint_pos_des)

框架无关——不依赖 ManagerBasedRLEnv，直接操作 Articulation。
"""

from __future__ import annotations

import torch

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.assets import Articulation
from isaaclab.utils.math import subtract_frame_transforms, matrix_from_quat, quat_inv


class PiperIKController:
    """Piper 6-DOF 臂的差分 IK 控制器。

    内部包装了 Isaac Lab 的 DifferentialIKController (DLS method)，
    处理了 Jacobian 从 world frame 到 base frame 的变换，
    以及末端执行器位姿从 world frame 到 base frame 的变换。

    Attributes:
        ik: 底层 DifferentialIKController 实例
        ee_body_name: 末端执行器 body 名称（默认 "link6"）
        ee_body_idx: 末端执行器在 Articulation body 列表中的索引
        ee_jacobi_idx: 末端执行器在 Jacobian 矩阵中对应的行索引
        arm_joint_ids: 臂关节 (joint1-6) 在 joint 列表中的索引
    """

    # 从发现脚本确认的 Piper 结构
    EE_BODY_NAME = "link6"
    ARM_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    IGNORE_FIRST_STEP = True  # reset 后第一步 Jacobian 未更新，需跳过

    def __init__(
        self,
        robot: Articulation,
        device: str = "cuda:0",
        ik_method: str = "dls",
        ik_params: dict | None = None,
        position_scale: float = 1.0,
        command_type: str = "pose",
        delta_gain: float = 1.0,
        rotate_jacobian_to_base: bool = True,
    ):
        """初始化 Piper IK 控制器。

        Args:
            robot: 已经 sim.reset() 后的 Piper Articulation 实例
            device: 计算设备
            ik_method: IK 求逆方法 ("pinv", "svd", "trans", "dls")
            ik_params: 方法参数覆盖，None 则使用默认值
            position_scale: 位置缩放。默认 1.0，表示目标位置、body pose、
                Jacobian 使用同一套 stage 单位；只有确认单位不匹配时才覆盖。
            command_type: "pose" 同时控制位置和姿态；"position" 只控制位置。
            delta_gain: 每步 IK 关节增量缩放，1.0 表示使用原始求解增量。
            rotate_jacobian_to_base: 是否把 Jacobian 从 world frame 旋到 base frame。
        """
        self._robot = robot
        self._device = device
        self._position_scale = float(position_scale)
        self._command_type = command_type
        self._delta_gain = float(delta_gain)
        self._rotate_jacobian_to_base = bool(rotate_jacobian_to_base)
        self._needs_reset_skip = False  # reset 后第一步标记

        # ---- 发现末端执行器 ----
        ee_result = robot.find_bodies(self.EE_BODY_NAME)
        ee_idx = ee_result[0][0]
        self.ee_body_idx: int = ee_idx.item() if hasattr(ee_idx, 'item') else int(ee_idx)
        num_jacobian_bodies = robot.root_physx_view.get_jacobians().shape[1]
        if num_jacobian_bodies == robot.num_bodies:
            self.ee_jacobi_idx = self.ee_body_idx
        elif num_jacobian_bodies == robot.num_bodies - 1:
            self.ee_jacobi_idx = self.ee_body_idx - 1
        else:
            raise RuntimeError(
                "Unexpected Jacobian body dimension: "
                f"{num_jacobian_bodies} for {robot.num_bodies} bodies"
            )
        self.num_jacobian_bodies = int(num_jacobian_bodies)

        # ---- 发现臂关节 ----
        arm_result = robot.find_joints(self.ARM_JOINT_NAMES)
        self.arm_joint_ids = arm_result[0]
        if hasattr(self.arm_joint_ids, 'tolist'):
            self.arm_joint_ids = self.arm_joint_ids.tolist()

        # ---- 构建 IK ----
        cfg = DifferentialIKControllerCfg(
            command_type=command_type,
            use_relative_mode=False,
            ik_method=ik_method,
            ik_params=ik_params,
        )
        self.ik = DifferentialIKController(cfg, num_envs=1, device=self._device)

        # ---- 目标位姿缓冲区 ----
        command_dim = 3 if command_type == "position" else 7
        self._command = torch.zeros(1, command_dim, device=self._device)
        self._target_pos = torch.zeros(1, 3, device=self._device)
        self._target_quat = torch.zeros(1, 4, device=self._device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def solve(self, target_pos: torch.Tensor, target_quat: torch.Tensor) -> torch.Tensor:
        """求解 IK 并返回臂关节目标位置。

        Args:
            target_pos: 目标位置，shape (1, 3)，base frame 坐标
            target_quat: 目标四元数 (w, x, y, z)，shape (1, 4)，base frame 坐标

        Returns:
            joint_pos_des: shape (1, 6)，joint1-6 的目标位置 (rad)
        """
        if self._needs_reset_skip:
            # reset 后 Jacobian 尚未更新到最新状态，返回当前关节位置不施加新指令
            joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
            return joint_pos.clone()

        # 1. 获取世界坐标系下的末端位姿和 base 位姿
        ee_pose_w = self._robot.data.body_pose_w[:, self.ee_body_idx]
        root_pose_w = self._robot.data.root_pose_w

        # 2. 转换到 base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
        )

        # 3. 设置 IK 目标
        target_pos_scaled = target_pos * self._position_scale
        self._target_pos[:] = target_pos_scaled
        self._target_quat[:] = target_quat
        if self._command_type == "position":
            self._command[:] = target_pos_scaled
            self.ik.set_command(self._command, ee_quat=ee_quat_b)
        else:
            self._command[:] = torch.cat([target_pos_scaled, target_quat], dim=-1)
            self.ik.set_command(self._command)

        # 4. 获取 Jacobian 并转到 base frame
        jacobian = self._robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobi_idx, :, self.arm_joint_ids
        ]
        if self._rotate_jacobian_to_base:
            base_rot = root_pose_w[:, 3:7]
            base_rot_matrix = matrix_from_quat(quat_inv(base_rot))
            jacobian[:, :3, :] = torch.bmm(base_rot_matrix, jacobian[:, :3, :])
            jacobian[:, 3:, :] = torch.bmm(base_rot_matrix, jacobian[:, 3:, :])

        # 5. 求解
        joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]
        ee_pos_scaled = ee_pos_b * self._position_scale
        joint_pos_des = self.ik.compute(ee_pos_scaled, ee_quat_b, jacobian, joint_pos)
        if self._delta_gain != 1.0:
            joint_pos_des = joint_pos + self._delta_gain * (joint_pos_des - joint_pos)

        return joint_pos_des

    def set_target(self, target_pos: torch.Tensor, target_quat: torch.Tensor):
        """只设置目标不求解（用于预加载目标位姿）。

        Args:
            target_pos: 目标位置，shape (1, 3)，base frame
            target_quat: 目标四元数 (w, x, y, z)，shape (1, 4)，base frame
        """
        self._target_pos[:] = target_pos * self._position_scale
        self._target_quat[:] = target_quat
        if self._command_type == "position":
            self._command[:] = self._target_pos
        else:
            self._command[:] = torch.cat([self._target_pos, target_quat], dim=-1)

    def notify_reset(self):
        """通知控制器机器人已被 reset，下一步将跳过 IK 求解。"""
        self._needs_reset_skip = True
        self.ik.reset()

    def clear_reset_skip(self):
        """清除 reset 跳过标记（在跳过第一步之后调用）。"""
        self._needs_reset_skip = False

    def get_current_ee_pose(self) -> tuple[torch.Tensor, torch.Tensor]:
        """获取末端执行器当前在世界坐标系下的位姿。

        Returns:
            (pos, quat): 位置 (1, 3) 和四元数 (1, 4)，world frame
        """
        ee_pose_w = self._robot.data.body_pose_w[:, self.ee_body_idx]
        return ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]

    def get_ik_error(self) -> tuple[float, float]:
        """获取 IK 的当前位置误差 (m) 和旋转误差 (rad)。

        Returns:
            (pos_error_norm, rot_error_norm): 标量
        """
        from isaaclab.utils.math import compute_pose_error

        ee_pose_w = self._robot.data.body_pose_w[:, self.ee_body_idx]
        root_pose_w = self._robot.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3], ee_pose_w[:, 3:7],
        )

        ee_pos_scaled = ee_pos_b * self._position_scale
        pos_error, rot_error = compute_pose_error(
            ee_pos_scaled, ee_quat_b,
            self._target_pos, self._target_quat,
        )
        return float(torch.norm(pos_error)), float(torch.norm(rot_error))
