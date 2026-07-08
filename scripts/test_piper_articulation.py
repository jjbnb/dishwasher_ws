#!/usr/bin/env python3
"""Piper articulation smoke test on native all.usd."""

from __future__ import annotations

import argparse
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Native all.usd Piper control test")
parser.add_argument(
    "--assets",
    default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher"),
)
parser.add_argument(
    "--keep-runtime-helpers",
    action="store_true",
    help="Keep nested PhysicsScene and ROS2 ActionGraph active.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

from dishwasher.scene.native_loader import NativeSceneLoader


def step_control(sim, robot, dt: float, steps: int):
    for _ in range(steps):
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)


print("=" * 72)
print("Native all.usd Piper articulation test")
print("=" * 72)

loader = NativeSceneLoader(args_cli.assets)
report = loader.validate()
if not report.ok:
    print("[FAIL] Native all.usd validation failed.")
    simulation_app.close()
    raise SystemExit(1)

stage = loader.open_stage()
if not args_cli.keep_runtime_helpers:
    changed = loader.prepare_for_direct_isaaclab(
        stage,
        deactivate_nested_physics_scenes=True,
        deactivate_action_graphs=True,
    )
    print(f"Runtime disabled nested PhysicsScene: {changed['nested_physics_scenes']}")
    print(f"Runtime disabled ActionGraph: {changed['action_graphs']}")

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
loader.wrap_assets(wrap_left=True, wrap_right=False)
robot = loader.piper

sim.reset()
dt = sim.get_physics_dt()

print(f"joints: {robot.num_joints} {robot.joint_names}")
print(f"bodies: {robot.num_bodies} {robot.body_names}")

ok = True

target = robot.data.default_joint_pos.clone()
robot.set_joint_position_target(target)
step_control(sim, robot, dt, 60)

for joint_idx, target_value in [(0, 0.3), (1, 0.5)]:
    target = robot.data.joint_pos.clone()
    target[0, joint_idx] = target_value
    robot.set_joint_position_target(target)
    step_control(sim, robot, dt, 180)
    actual = float(robot.data.joint_pos[0, joint_idx])
    passed = abs(actual - target_value) < 0.12
    ok = ok and passed
    print(
        f"joint{joint_idx + 1}: target={target_value:.3f}, "
        f"actual={actual:.3f}, {'PASS' if passed else 'FAIL'}",
        flush=True,
    )

simulation_app.close()
raise SystemExit(0 if ok else 1)
