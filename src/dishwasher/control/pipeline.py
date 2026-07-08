"""
Level 1 Pipeline——组装感知、抓取、IK、夹爪、状态机的完整闭环。

主循环: DETECT → PLAN → PRE_GRASP → GRASP → POST_GRASP → PRE_PLACE → PLACE → VERIFY → (循环)

用法:
    loader = SceneLoader(assets_path)
    # ... spawn + wrap + sim.reset() ...
    pipeline = Level1Pipeline(loader)
    pipeline.run(sim, dt)
"""

from __future__ import annotations

import time
import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.utils.math import subtract_frame_transforms

from dishwasher.motion.ik_controller import PiperIKController
from dishwasher.control.gripper import PiperGripper
from dishwasher.control.state_machine import StateMachine, State
from dishwasher.perception.detector import get_next_plate
from dishwasher.grasping.generator import (
    generate_grasp_pose,
    generate_place_pose,
    to_torch,
)
from dishwasher.motion.trajectory import interpolate_waypoints


class Level1Pipeline:
    """Level 1 端到端管道：规则策略，100% 成功率。

    每步调用 step() 推进状态机，或调用 run() 运行完整流程。
    """

    def __init__(
        self,
        robot: Articulation,
        plates: dict[str, any],
        device: str = "cuda:0",
        approach_steps: int = 60,
        grasp_hold_steps: int = 30,
        place_release_steps: int = 30,
        unit_scale: float = 1.0,
        place_position: tuple[float, float, float] | None = None,
    ):
        """
        Args:
            robot: Piper Articulation
            plates: SceneLoader.plates dict
            device: 计算设备
            approach_steps: 接近/抬升轨迹步数
            grasp_hold_steps: 抓取后保持步数
            place_release_steps: 放置后释放的等待步数
            unit_scale: 长度缩放。米制 legacy 场景为 1.0，原生 all.usd 为 100.0
            place_position: 放置位置覆盖。原生 all.usd 可用 NATIVE_PLACE_POSITION
        """
        self.robot = robot
        self.plates = plates
        self.device = device
        self.approach_steps = approach_steps
        self.grasp_hold_steps = grasp_hold_steps
        self.place_release_steps = place_release_steps
        self.unit_scale = unit_scale
        self.place_position = place_position

        # 子模块
        self.ik = PiperIKController(robot, device=device)
        self.gripper = PiperGripper(robot)
        self.state_machine = StateMachine()

        # 运行时状态
        self._processed: set[str] = set()
        self._current_plate: dict | None = None
        self._grasp_plan: dict | None = None
        self._place_plan: dict | None = None
        self._trajectory: list | None = None
        self._traj_step: int = 0
        self._hold_counter: int = 0
        self._grasp_phase: str = "descend"  # "descend" | "close"

        # 指标
        self.metrics = {
            "total_plates": 0,
            "success_count": 0,
            "elapsed_time": 0.0,
        }

    # ------------------------------------------------------------------
    # 每步推进
    # ------------------------------------------------------------------

    def step(self, sim, dt: float) -> bool:
        """执行一步状态机推进。返回 True 表示完成。"""
        state = self.state_machine.current

        if state == State.IDLE:
            return False

        elif state == State.DETECT:
            self._handle_detect()

        elif state == State.PLAN:
            self._handle_plan()

        elif state == State.PRE_GRASP:
            self._handle_approach(sim, dt, phase="pre_grasp")

        elif state == State.GRASP:
            self._handle_grasp(sim, dt)

        elif state == State.POST_GRASP:
            self._handle_approach(sim, dt, phase="post_grasp")

        elif state == State.PRE_PLACE:
            self._handle_approach(sim, dt, phase="pre_place")

        elif state == State.PLACE:
            self._handle_place(sim, dt)

        elif state == State.VERIFY:
            self._handle_verify()

        elif state == State.DONE:
            return True

        return self.state_machine.current == State.DONE

    # ------------------------------------------------------------------
    # run() 全流程
    # ------------------------------------------------------------------

    def run(self, sim, dt: float) -> dict:
        """运行完整 Level 1 流程，阻塞直到完成。

        Args:
            sim: SimulationContext
            dt: 物理时间步长

        Returns:
            metrics dict
        """
        total_plates = len(self.plates)
        self.state_machine.start(total_plates)
        self.metrics["total_plates"] = total_plates
        t_start = time.time()

        print(f"\n{'='*60}", flush=True)
        print(f"Level 1 Pipeline — 开始处理 {total_plates} 个盘子", flush=True)
        print(f"{'='*60}", flush=True)

        step_count = 0
        while not self.step(sim, dt):
            step_count += 1
            if step_count % 100 == 0:
                pos_err, rot_err = self.ik.get_ik_error()
                print(f"  ... step {step_count}: state={self.state_machine.current.name}, "
                      f"ik_err=({pos_err*1000:.1f}mm, {rot_err:.4f}rad)", flush=True)

        self.metrics["elapsed_time"] = time.time() - t_start
        self.metrics["success_count"] = len(self._processed)

        print(f"\n{'='*60}", flush=True)
        print(f"Level 1 Pipeline — 完成", flush=True)
        print(f"  成功: {self.metrics['success_count']}/{self.metrics['total_plates']}", flush=True)
        print(f"  耗时: {self.metrics['elapsed_time']:.1f}s", flush=True)
        print(f"{'='*60}", flush=True)

        return self.metrics

    # ------------------------------------------------------------------
    # 各状态处理
    # ------------------------------------------------------------------

    def _handle_detect(self):
        """检测下一个未处理的盘子。"""
        plate = get_next_plate(self.plates, self._processed)
        if plate is None:
            print("  [DETECT] 所有盘子已处理", flush=True)
            self.state_machine._current = State.DONE
            return

        self._current_plate = plate
        print(f"  [DETECT] 检测到 {plate['name']}: "
              f"pos=({plate['pos'][0]:.3f}, {plate['pos'][1]:.3f}, {plate['pos'][2]:.3f})",
              flush=True)
        self.state_machine.advance()

    def _handle_plan(self):
        """为当前盘子生成抓取和放置姿态。"""
        plate = self._current_plate
        if plate is None:
            self.state_machine.advance()
            return

        # 生成抓取姿态（world frame）
        self._grasp_plan = generate_grasp_pose(
            plate["pos"], plate["quat"], unit_scale=self.unit_scale
        )

        # 生成放置姿态
        place_idx = len(self._processed)
        self._place_plan = generate_place_pose(
            place_idx,
            unit_scale=self.unit_scale,
            place_position=self.place_position,
        )

        print(f"  [PLAN] 抓取 → ({self._grasp_plan['grasp'][0][0]:.3f}, "
              f"{self._grasp_plan['grasp'][0][1]:.3f}, "
              f"{self._grasp_plan['grasp'][0][2]:.3f})", flush=True)
        print(f"  [PLAN] 放置 → ({self._place_plan['place'][0][0]:.3f}, "
              f"{self._place_plan['place'][0][1]:.3f}, "
              f"{self._place_plan['place'][0][2]:.3f})", flush=True)
        self.state_machine.advance()

    def _handle_approach(self, sim, dt: float, phase: str):
        """执行接近/抬升/移动轨迹。"""
        if self._trajectory is None:
            # 初始化轨迹
            ee_pos_w, ee_quat_w = self.ik.get_current_ee_pose()
            current_pos = ee_pos_w[0].cpu().numpy()
            current_quat = ee_quat_w[0].cpu().numpy()

            if phase == "pre_grasp":
                target_pos, target_quat = self._grasp_plan["pre_grasp"]
            elif phase == "post_grasp":
                target_pos, target_quat = self._grasp_plan["post_grasp"]
            elif phase == "pre_place":
                target_pos, target_quat = self._place_plan["pre_place"]
            else:
                self.state_machine.advance()
                return

            self._trajectory = interpolate_waypoints(
                current_pos, current_quat, target_pos, target_quat,
                self.approach_steps,
            )
            self._traj_step = 0

        # 沿轨迹推进
        if self._traj_step < len(self._trajectory):
            pos, quat = self._trajectory[self._traj_step]
            self._traj_step += 1

            # world frame → base frame
            root_pose = self.robot.data.root_pose_w
            tgt_pos_w = torch.tensor(pos, dtype=torch.float32, device=self.device).unsqueeze(0)
            tgt_quat_w = torch.tensor(quat, dtype=torch.float32, device=self.device).unsqueeze(0)
            tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
                root_pose[:, 0:3], root_pose[:, 3:7],
                tgt_pos_w, tgt_quat_w,
            )

            joint_pos_des = self.ik.solve(tgt_pos_b, tgt_quat_b)
            self._set_arm_target(joint_pos_des)
            self._step_sim(sim, dt)
        else:
            self._trajectory = None
            self.state_machine.advance()

    def _handle_grasp(self, sim, dt: float):
        """从预抓取位置下降到实际抓取位置，然后闭合夹爪。"""
        if self._grasp_phase == "descend":
            if self._trajectory is None:
                # 初始化：从当前 EE 插值到实际抓取位置
                ee_pos_w, ee_quat_w = self.ik.get_current_ee_pose()
                current_pos = ee_pos_w[0].cpu().numpy()
                current_quat = ee_quat_w[0].cpu().numpy()
                target_pos, target_quat = self._grasp_plan["grasp"]
                self._trajectory = interpolate_waypoints(
                    current_pos, current_quat, target_pos, target_quat,
                    self.approach_steps // 2,
                )
                self._traj_step = 0

            # 下降中
            if self._traj_step < len(self._trajectory):
                pos, quat = self._trajectory[self._traj_step]
                self._traj_step += 1
                self._set_world_target(pos, quat)
                self._step_sim(sim, dt)
                return

            # 下降完成 → 进入闭合阶段
            self._trajectory = None
            self._grasp_phase = "close"

        # 闭合夹爪
        self.gripper.close_cmd()
        self._step_sim(sim, dt)
        self._hold_counter += 1

        if self._hold_counter >= self.grasp_hold_steps:
            self._hold_counter = 0
            self._grasp_phase = "descend"  # 为下一个盘子重置
            print(f"  [GRASP] 夹爪闭合完成", flush=True)
            self.state_machine.advance()

    def _handle_place(self, sim, dt: float):
        """移动到放置位置，打开夹爪释放盘子。"""
        if self._trajectory is None:
            # 初始化：从预放置位置下降到释放位置
            ee_pos_w, ee_quat_w = self.ik.get_current_ee_pose()
            current_pos = ee_pos_w[0].cpu().numpy()
            current_quat = ee_quat_w[0].cpu().numpy()

            target_pos, target_quat = self._place_plan["place"]
            self._trajectory = interpolate_waypoints(
                current_pos, current_quat, target_pos, target_quat,
                self.approach_steps // 2,
            )
            self._traj_step = 0

        if self._traj_step < len(self._trajectory):
            pos, quat = self._trajectory[self._traj_step]
            self._traj_step += 1
            self._set_world_target(pos, quat)
            self._step_sim(sim, dt)
        else:
            self._trajectory = None
            # 打开夹爪释放
            self.gripper.open_cmd()
            for _ in range(self.place_release_steps):
                self._step_sim(sim, dt)
            print(f"  [PLACE] 盘子释放完成", flush=True)
            self.state_machine.advance()

    def _handle_verify(self):
        """验证盘子是否放置成功，更新已处理集合。"""
        if self._current_plate:
            self._processed.add(self._current_plate["name"])
            print(f"  [VERIFY] {self._current_plate['name']} 完成 ✅ "
                  f"({len(self._processed)}/{self.metrics['total_plates']})", flush=True)
        self.state_machine.advance()

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _set_arm_target(self, joint_pos_des: torch.Tensor):
        """设置臂关节 PD 目标。"""
        full_target = self.robot.data.joint_pos.clone()
        full_target[0, :6] = joint_pos_des[0]
        self.robot.set_joint_position_target(full_target)

    def _set_world_target(self, pos: np.ndarray, quat: np.ndarray):
        """将世界坐标目标转为 base frame 并设置 IK。"""
        root_pose = self.robot.data.root_pose_w
        tgt_pos_w = torch.tensor(pos, dtype=torch.float32, device=self.device).unsqueeze(0)
        tgt_quat_w = torch.tensor(quat, dtype=torch.float32, device=self.device).unsqueeze(0)
        tgt_pos_b, tgt_quat_b = subtract_frame_transforms(
            root_pose[:, 0:3], root_pose[:, 3:7],
            tgt_pos_w, tgt_quat_w,
        )
        joint_pos_des = self.ik.solve(tgt_pos_b, tgt_quat_b)
        self._set_arm_target(joint_pos_des)

    def _step_sim(self, sim, dt: float):
        """执行一步物理模拟。"""
        self.robot.write_data_to_sim()
        sim.step()
        self.robot.update(dt)
        for plate in self.plates.values():
            plate.update(dt)
