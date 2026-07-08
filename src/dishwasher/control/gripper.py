"""
Piper 夹爪控制——简单的 open/close 封装。

夹爪结构（from discovery）:
    joint7: prismatic, 左指
    joint8: prismatic, 右指

Level 1 使用纯位置控制（PD），不需要力控。
"""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation


class PiperGripper:
    """Piper 2 指夹爪的简单位置控制。

    Attributes:
        OPEN_WIDTH: 全开时每个指位移 (m)，正负对称
        CLOSE_WIDTH: 闭合时每个指位移，0 表示闭合
        GRIPPER_JOINT_IDS: 夹爪关节在 joint 列表中的索引
    """

    OPEN_WIDTH = (0.035, -0.035)   # (joint7, joint8)
    CLOSE_WIDTH = (0.0, 0.0)

    GRIPPER_JOINT_IDS = [6, 7]  # joint7, joint8

    def __init__(self, robot: Articulation):
        """初始化夹爪控制器。

        Args:
            robot: 已经 sim.reset() 后的 Piper Articulation
        """
        self._robot = robot
        self._device = robot.device if hasattr(robot, 'device') else 'cuda:0'
        self._gripper_ids = torch.tensor(self.GRIPPER_JOINT_IDS, device=self._device)

    def open(self) -> torch.Tensor:
        """生成打开夹爪的关节目标位置。

        Returns:
            joint_pos_des: shape (1, 2)，joint7-8 开位置
        """
        return torch.tensor([self.OPEN_WIDTH], device=self._device)

    def close(self) -> torch.Tensor:
        """生成闭合夹爪的关节目标位置。

        Returns:
            joint_pos_des: shape (1, 2)，joint7-8 闭位置
        """
        return torch.tensor([self.CLOSE_WIDTH], device=self._device)

    def set_target(self, joint7_pos: float, joint8_pos: float):
        """直接设置夹爪关节位置目标。

        Args:
            joint7_pos: 左指位置 (m)
            joint8_pos: 右指位置 (m)
        """
        target = self._robot.data.joint_pos.clone()
        target[0, 6] = joint7_pos
        target[0, 7] = joint8_pos
        self._robot.set_joint_position_target(target)

    def open_cmd(self):
        """发送打开指令到机器人。"""
        self.set_target(*self.OPEN_WIDTH)

    def close_cmd(self):
        """发送闭合指令到机器人。"""
        self.set_target(*self.CLOSE_WIDTH)

    def is_open(self) -> bool:
        """检查夹爪是否全开。"""
        j7 = float(self._robot.data.joint_pos[0, 6])
        j8 = float(self._robot.data.joint_pos[0, 7])
        return abs(j7 - self.OPEN_WIDTH[0]) < 0.005 and abs(j8 - self.OPEN_WIDTH[1]) < 0.005

    def is_closed(self) -> bool:
        """检查夹爪是否闭合。"""
        j7 = float(self._robot.data.joint_pos[0, 6])
        j8 = float(self._robot.data.joint_pos[0, 7])
        return abs(j7) < 0.005 and abs(j8) < 0.005

    def get_width(self) -> float:
        """获取当前夹爪开口宽度（两指间距）。"""
        j7 = float(self._robot.data.joint_pos[0, 6])
        return abs(j7 * 2)  # 对称开合
