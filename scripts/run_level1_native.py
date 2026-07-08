#!/usr/bin/env python3
"""Level 1 native all.usd entry point.

Default mode performs a safe smoke test on the native scene:
open all.usd, wrap native assets, detect plates from native rigid bodies,
generate grasp/place poses in stage units, and solve one IK target.

Use --execute to run the current experimental state-machine pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Level 1 native all.usd smoke runner")
parser.add_argument(
    "--assets",
    default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher"),
)
parser.add_argument("--num_plates", type=int, default=1)
parser.add_argument("--execute", action="store_true", help="Run experimental pipeline")
parser.add_argument(
    "--keep-runtime-helpers",
    action="store_true",
    help="Keep nested PhysicsScene and ROS2 ActionGraph active.",
)
parser.add_argument("--approach_steps", type=int, default=80)
parser.add_argument("--grasp_hold_steps", type=int, default=30)
parser.add_argument("--place_release_steps", type=int, default=30)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils.math import subtract_frame_transforms

from dishwasher.control.pipeline import Level1Pipeline
from dishwasher.grasping.generator import (
    NATIVE_PLACE_POSITION,
    generate_grasp_pose,
    generate_place_pose,
)
from dishwasher.motion.ik_controller import PiperIKController
from dishwasher.perception.detector import get_next_plate, get_plate_positions
from dishwasher.scene.native_loader import NativeSceneLoader


def stage_to_meters(pos):
    arr = np.asarray(pos, dtype=np.float64)
    return arr * 0.01


print("=" * 72)
print("Level 1 native all.usd runner")
print("=" * 72)

loader = NativeSceneLoader(args_cli.assets)
report = loader.validate()
if not report.ok:
    print("[FAIL] Native all.usd validation failed.", flush=True)
    simulation_app.close()
    raise SystemExit(1)

stage = loader.open_stage()
if not args_cli.keep_runtime_helpers:
    changed = loader.prepare_for_direct_isaaclab(
        stage,
        deactivate_nested_physics_scenes=True,
        deactivate_action_graphs=True,
    )
    print(f"runtime disabled nested PhysicsScene: {changed['nested_physics_scenes']}", flush=True)
    print(f"runtime disabled ActionGraph: {changed['action_graphs']}", flush=True)

print("[init] creating SimulationContext", flush=True)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
print("[init] wrapping native assets", flush=True)
loader.wrap_assets(wrap_left=True, wrap_right=False)
print("[init] sim.reset()", flush=True)
sim.reset()
dt = sim.get_physics_dt()
print(f"[init] sim dt={dt:.6f}", flush=True)

robot = loader.piper
robot.set_joint_position_target(robot.data.default_joint_pos.clone())
print("[init] stepping initial settle", flush=True)
for _ in range(10):
    robot.write_data_to_sim()
    sim.step()
    robot.update(dt)
    for plate in loader.plates.values():
        plate.update(dt)
print("[init] initial settle done", flush=True)

all_plates = dict(sorted(loader.plates.items()))
num_plates = max(1, min(args_cli.num_plates, len(all_plates)))
plates = dict(list(all_plates.items())[:num_plates])

print(f"\n[1] Native assets wrapped: left_piper joints={robot.num_joints}")
print(f"    plates selected: {list(plates)}")

print("\n[2] Native plate detection from RigidObject tensors")
for item in get_plate_positions(plates):
    meters = stage_to_meters(item["pos"])
    print(
        f"    {item['name']}: stage=({item['pos'][0]:.3f}, {item['pos'][1]:.3f}, {item['pos'][2]:.3f}) "
        f"meters=({meters[0]:.3f}, {meters[1]:.3f}, {meters[2]:.3f})"
    )

plate = get_next_plate(plates, processed=set())
if plate is None:
    print("[FAIL] No plate detected")
    simulation_app.close()
    raise SystemExit(1)

print("\n[3] Rule-based grasp/place planning in native stage units")
grasp_plan = generate_grasp_pose(plate["pos"], plate["quat"], unit_scale=100.0)
place_plan = generate_place_pose(
    0,
    unit_scale=100.0,
    place_position=NATIVE_PLACE_POSITION,
)
for name, (pos, _quat) in grasp_plan.items():
    meters = stage_to_meters(pos)
    print(
        f"    {name}: stage=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
        f"meters=({meters[0]:.3f}, {meters[1]:.3f}, {meters[2]:.3f})"
    )
for name, (pos, _quat) in place_plan.items():
    meters = stage_to_meters(pos)
    print(
        f"    {name}: stage=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
        f"meters=({meters[0]:.3f}, {meters[1]:.3f}, {meters[2]:.3f})"
    )

print("\n[4] IK smoke solve for first pre_grasp target")
ik = PiperIKController(robot, device=args_cli.device)
target_pos, target_quat = grasp_plan["pre_grasp"]
target_pos_w = torch.tensor(target_pos, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
target_quat_w = torch.tensor(target_quat, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
root_pose = robot.data.root_pose_w
target_pos_b, target_quat_b = subtract_frame_transforms(
    root_pose[:, 0:3],
    root_pose[:, 3:7],
    target_pos_w,
    target_quat_w,
)
joint_pos_des = ik.solve(target_pos_b, target_quat_b)
finite = bool(torch.isfinite(joint_pos_des).all())
print(f"    IK finite: {finite}")
print(
    "    joint target: "
    + ", ".join(f"{float(v):.3f}" for v in joint_pos_des[0].detach().cpu())
)

if not finite:
    simulation_app.close()
    raise SystemExit(1)

if args_cli.execute:
    print("\n[5] Executing experimental Level1Pipeline")
    pipeline = Level1Pipeline(
        robot=robot,
        plates=plates,
        device=args_cli.device,
        approach_steps=args_cli.approach_steps,
        grasp_hold_steps=args_cli.grasp_hold_steps,
        place_release_steps=args_cli.place_release_steps,
        unit_scale=100.0,
        place_position=NATIVE_PLACE_POSITION,
    )
    metrics = pipeline.run(sim, dt)
    print(f"    metrics: {metrics}")
else:
    print("\n[5] Smoke complete. Use --execute for experimental full state machine.")

simulation_app.close()
