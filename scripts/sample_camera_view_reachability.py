#!/usr/bin/env python3
"""Sample native Piper wrist-camera poses for reachable observation views."""

from __future__ import annotations

import argparse
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sample reachable wrist-camera views")
parser.add_argument(
    "--assets",
    default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher"),
)
parser.add_argument("--samples", type=int, default=1200)
parser.add_argument("--seed", type=int, default=11)
parser.add_argument("--top-k", type=int, default=12)
parser.add_argument("--target-stage", type=float, nargs=3, default=[88.0, 96.0, 0.0])
parser.add_argument(
    "--commanded",
    action="store_true",
    help="Command joint targets through the articulation controller and score the actual settled pose.",
)
parser.add_argument("--move-steps", type=int, default=90)
parser.add_argument("--settle-steps", type=int, default=20)
parser.add_argument("--skip-close", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils

from dishwasher.motion.ik_controller import PiperIKController
from dishwasher.perception.native_camera import (
    calibrate_camera_offset_from_body,
    camera_path,
    robot_camera_pose_variant,
)
from dishwasher.scene.native_loader import NativeSceneLoader


def close_app():
    if not args_cli.skip_close:
        simulation_app.close()


def exit_with(code: int):
    close_app()
    if args_cli.skip_close:
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
    raise SystemExit(code)


def tensor_to_np(tensor):
    return tensor.detach().cpu().numpy()


def score_pose(camera_pos, forward, target):
    to_target = target - camera_pos
    distance = float(np.linalg.norm(to_target))
    xy_distance = float(np.linalg.norm(to_target[:2]))
    elevation = float(np.degrees(np.arctan2(camera_pos[2] - target[2], max(1.0, xy_distance))))
    to_target_unit = to_target / max(1.0e-9, distance)
    forward = forward / max(1.0e-9, float(np.linalg.norm(forward)))
    look_error = float(np.degrees(np.arccos(np.clip(float(np.dot(forward, to_target_unit)), -1.0, 1.0))))
    downward = max(0.0, float(-forward[2]))

    # This is a reachability probe, so do not reject low poses outright.
    z_score = min(1.0, max(0.0, (camera_pos[2] - 22.0) / 18.0))
    dist_score = max(0.0, 1.0 - abs(distance - 46.0) / 34.0)
    xy_score = max(0.0, 1.0 - abs(xy_distance - 24.0) / 24.0)
    elev_score = min(1.0, max(0.0, (elevation - 34.0) / 24.0))
    look_score = max(0.0, 1.0 - look_error / 55.0)
    down_score = min(1.0, max(0.0, (downward - 0.35) / 0.45))
    score = (
        0.18 * z_score
        + 0.14 * dist_score
        + 0.12 * xy_score
        + 0.16 * elev_score
        + 0.24 * look_score
        + 0.16 * down_score
    )
    metrics = {
        "score": score,
        "x": float(camera_pos[0]),
        "y": float(camera_pos[1]),
        "z": float(camera_pos[2]),
        "distance": distance,
        "xy_distance": xy_distance,
        "elevation": elevation,
        "look_error": look_error,
        "downward": downward,
    }
    return score, metrics


loader = NativeSceneLoader(args_cli.assets)
report = loader.validate()
if not report.ok:
    print("[FAIL] Native all.usd validation failed.", flush=True)
    exit_with(1)

stage = loader.open_stage()
loader.prepare_for_direct_isaaclab(
    stage,
    deactivate_nested_physics_scenes=True,
    deactivate_action_graphs=True,
)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
loader.wrap_assets(wrap_left=True, wrap_right=True)
robots = {"left": loader.piper_l, "right": loader.piper_r}
sim.reset()
dt = sim.get_physics_dt()
for _ in range(10):
    for robot in robots.values():
        robot.write_data_to_sim()
    sim.step()
    for robot in robots.values():
        robot.update(dt)

camera_offsets = {}
for arm, robot in robots.items():
    camera_name = f"{arm}_wrist"
    prim = stage.GetPrimAtPath(camera_path(loader, camera_name))
    camera_offsets[camera_name] = calibrate_camera_offset_from_body(prim, robot)

target = np.asarray(args_cli.target_stage, dtype=np.float64)
rng = torch.Generator(device=args_cli.device)
rng.manual_seed(args_cli.seed)

print("=" * 72)
print("Reachable wrist-camera view samples")
print("=" * 72)
print(f"target=({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f})")
print(f"samples per arm={args_cli.samples}, seed={args_cli.seed}")

for arm, robot in robots.items():
    ik_probe = PiperIKController(robot, device=args_cli.device)
    arm_joint_ids = ik_probe.arm_joint_ids
    limits = robot.data.soft_joint_pos_limits[0, arm_joint_ids]
    lower = limits[:, 0]
    upper = limits[:, 1]
    default_q = robot.data.joint_pos.clone()
    zero_vel = torch.zeros_like(default_q)
    best = []

    for sample_idx in range(args_cli.samples):
        q = default_q.clone()
        if sample_idx == 0:
            sample = default_q[0, arm_joint_ids]
        else:
            rand = torch.rand((len(arm_joint_ids),), generator=rng, device=args_cli.device)
            sample = lower + rand * (upper - lower)
        q[0, arm_joint_ids] = sample
        if args_cli.commanded:
            robot.write_joint_state_to_sim(default_q, zero_vel)
            robot.set_joint_position_target(default_q)
            robot.write_data_to_sim()
            sim.step()
            robot.update(dt)
            for step_idx in range(args_cli.move_steps):
                alpha = (step_idx + 1) / max(1, args_cli.move_steps)
                command = default_q + alpha * (q - default_q)
                robot.set_joint_position_target(command)
                robot.write_data_to_sim()
                sim.step()
                robot.update(dt)
            robot.set_joint_position_target(q)
            for _ in range(args_cli.settle_steps):
                robot.write_data_to_sim()
                sim.step()
                robot.update(dt)
            actual_sample = robot.data.joint_pos[0, arm_joint_ids]
        else:
            robot.write_joint_state_to_sim(q, zero_vel)
            sim.step()
            robot.update(dt)
            actual_sample = sample
        pose = robot_camera_pose_variant(
            robot,
            camera_offsets[f"{arm}_wrist"],
            device=args_cli.device,
        )
        camera_pos = tensor_to_np(pose.pos_w).astype(np.float64)
        forward = np.asarray(pose.rotm_w_ros, dtype=np.float64)[:, 2]
        score, metrics = score_pose(camera_pos, forward, target)
        best.append((score, metrics, tensor_to_np(sample).astype(np.float64), tensor_to_np(actual_sample).astype(np.float64)))

    best.sort(key=lambda item: item[0], reverse=True)
    print(f"\n[{arm}] top reachable camera poses")
    for rank, (score, metrics, joints, actual) in enumerate(best[: args_cli.top_k]):
        print(
            f"  {rank:02d}: score={score:.3f}, "
            f"pos=({metrics['x']:.2f}, {metrics['y']:.2f}, {metrics['z']:.2f}), "
            f"dist={metrics['distance']:.2f}, "
            f"xy={metrics['xy_distance']:.2f}, elev={metrics['elevation']:.1f}, "
            f"look_err={metrics['look_error']:.1f}, down={metrics['downward']:.2f}, "
            "target_q=[" + ", ".join(f"{float(v):.3f}" for v in joints) + "], "
            "actual_q=[" + ", ".join(f"{float(v):.3f}" for v in actual) + "]",
            flush=True,
        )

exit_with(0)
