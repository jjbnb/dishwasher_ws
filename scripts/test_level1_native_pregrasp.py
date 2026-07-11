#!/usr/bin/env python3
"""Execute only the native Level 1 pre-grasp motion.

This is intentionally narrower than the full Level 1 state machine: it opens
the competition-provided all.usd, wraps the native left Piper and plates,
detects one plate from RigidObject tensors, then commands the arm toward the
safe pre-grasp pose above that plate. It does not close the gripper, descend to
grasp, move plates, or save the source USD.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Native all.usd Level 1 pre-grasp test")
parser.add_argument(
    "--assets",
    default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher"),
)
parser.add_argument(
    "--keep-runtime-helpers",
    action="store_true",
    help="Keep nested PhysicsScene and ROS2 ActionGraph active.",
)
parser.add_argument("--settle-steps", type=int, default=10)
parser.add_argument("--steps", type=int, default=240)
parser.add_argument("--log-every", type=int, default=40)
parser.add_argument(
    "--direct-target",
    action="store_true",
    help="Command the final pre-grasp target every step instead of interpolating.",
)
parser.add_argument(
    "--ik-position-scale",
    type=float,
    default=1.0,
    help="Scale target positions before IK. Native all.usd pre-grasp currently works best at 1.0.",
)
parser.add_argument(
    "--ik-delta-gain",
    type=float,
    default=0.25,
    help="Scale each IK joint delta before sending it as a joint target.",
)
parser.add_argument(
    "--ik-method",
    choices=("pinv", "svd", "trans", "dls"),
    default="dls",
)
parser.add_argument(
    "--ik-k-val",
    type=float,
    default=0.5,
    help="Gain for pinv, svd, or transpose IK methods.",
)
parser.add_argument(
    "--ik-lambda",
    type=float,
    default=0.05,
    help="Damping lambda for DLS IK.",
)
parser.add_argument(
    "--no-rotate-jacobian",
    action="store_true",
    help="Use PhysX Jacobian rows as-is instead of rotating them into root frame.",
)
parser.add_argument(
    "--ee-z-offset",
    type=float,
    default=0.0,
    help="Extra Z offset in native stage units added to the link6 target.",
)
parser.add_argument(
    "--orientation",
    choices=("planned", "current"),
    default="current",
    help="Use the planned grasp quaternion or hold the current EE orientation.",
)
parser.add_argument(
    "--command-type",
    choices=("pose", "position"),
    default="pose",
    help="Differential IK command type.",
)
parser.add_argument(
    "--observe-command-type",
    choices=("pose", "position"),
    default="pose",
    help="Differential IK command type used only for task-space camera observation.",
)
parser.add_argument(
    "--no-clamp-joints",
    action="store_true",
    help="Send raw IK joint targets instead of clamping to soft joint limits.",
)
parser.add_argument(
    "--perception",
    choices=("rigid", "camera"),
    default="rigid",
    help="Use legacy rigid-body plate poses or wrist-camera RGB-D detections.",
)
parser.add_argument(
    "--arm",
    choices=("auto", "left", "right"),
    default="auto",
    help="Arm used for execution. Auto follows the selected wrist camera.",
)
parser.add_argument(
    "--camera",
    choices=("scene", "left_wrist", "right_wrist", "both_wrist"),
    default="both_wrist",
    help="Camera used when --perception camera.",
)
parser.add_argument("--camera-width", type=int, default=640)
parser.add_argument("--camera-height", type=int, default=480)
parser.add_argument(
    "--camera-observe-joints",
    type=float,
    nargs=6,
    default=[0.160, 0.225, -0.304, 0.207, 0.849, -0.609],
    help="Six arm joints for the left/default wrist top-down observation pose.",
)
parser.add_argument(
    "--right-camera-observe-joints",
    type=float,
    nargs=6,
    default=[-0.205, 0.076, -0.001, -0.996, 0.811, -0.254],
    help="Optional six arm joints for the right wrist observation pose.",
)
parser.add_argument(
    "--observe-mode",
    choices=("active", "view", "joint"),
    default="active",
    help="Actively search for a good wrist-camera view, plan one task-space view, or use fixed joint targets.",
)
parser.add_argument(
    "--view-target-stage",
    type=float,
    nargs=3,
    default=[88.0, 96.0, 0.0],
    help="Task-space look-at target for wrist camera observation in native stage units.",
)
parser.add_argument(
    "--view-backoff",
    type=float,
    default=68.0,
    help="Camera XY distance from look-at target back toward the arm base.",
)
parser.add_argument(
    "--view-height",
    type=float,
    default=76.0,
    help="Camera height above the look-at target in native stage units.",
)
parser.add_argument(
    "--view-side-offset",
    type=float,
    default=0.0,
    help="Optional sideways offset for the task-space camera view.",
)
parser.add_argument(
    "--view-min-camera-distance",
    type=float,
    default=30.0,
    help="Minimum camera-to-look-at distance for safe observation.",
)
parser.add_argument(
    "--view-max-camera-distance",
    type=float,
    default=124.0,
    help="Maximum camera-to-look-at distance for useful observation.",
)
parser.add_argument(
    "--view-min-camera-xy-distance",
    type=float,
    default=18.0,
    help="Minimum horizontal camera-to-look-at distance to avoid face-pressed views.",
)
parser.add_argument(
    "--view-min-camera-z",
    type=float,
    default=24.0,
    help="Minimum wrist camera Z height for observation.",
)
parser.add_argument(
    "--view-min-link6-z",
    type=float,
    default=24.0,
    help="Minimum link6 Z height for observation targets.",
)
parser.add_argument(
    "--view-safe-lift-z",
    type=float,
    default=62.0,
    help="Intermediate link6 Z used before moving sideways to an observation target.",
)
parser.add_argument(
    "--view-min-elevation-deg",
    type=float,
    default=24.0,
    help="Minimum camera elevation angle above the look-at target.",
)
parser.add_argument(
    "--view-max-look-error-deg",
    type=float,
    default=65.0,
    help="Maximum allowed angle between camera optical axis and the look-at target.",
)
parser.add_argument(
    "--view-min-downward-z",
    type=float,
    default=0.45,
    help="Minimum downward component of the camera optical axis.",
)
parser.add_argument(
    "--view-robot-side-y-min",
    type=float,
    default=114.0,
    help="Prefer observation cameras on the robot side of the workspace.",
)
parser.add_argument(
    "--view-workspace-x",
    type=float,
    nargs=2,
    default=[58.0, 128.0],
    help="Approximate workspace X bounds used for view safety scoring.",
)
parser.add_argument(
    "--view-workspace-y",
    type=float,
    nargs=2,
    default=[68.0, 126.0],
    help="Approximate workspace Y bounds used for view safety scoring.",
)
parser.add_argument("--camera-observe-steps", type=int, default=240)
parser.add_argument("--camera-settle-steps", type=int, default=50)
parser.add_argument("--camera-warmup-steps", type=int, default=20)
parser.add_argument("--camera-min-area", type=int, default=40)
parser.add_argument(
    "--camera-quality-threshold",
    type=float,
    default=0.20,
    help="Warn when a camera frame is below this visual-observation quality score.",
)
parser.add_argument(
    "--active-min-quality",
    type=float,
    default=0.28,
    help="Minimum combined camera quality accepted by active observation.",
)
parser.add_argument(
    "--active-require-both-cameras",
    action="store_true",
    help="Require both wrist cameras to detect at least one plate before accepting active observation.",
)
parser.add_argument(
    "--allow-low-quality-observe",
    action="store_true",
    help="Continue with the best active-observation attempt even if it is below --active-min-quality.",
)
parser.add_argument(
    "--active-stop-on-first-pass",
    action="store_true",
    help="Stop active observation as soon as one candidate passes; default evaluates all candidates.",
)
parser.add_argument(
    "--active-max-view-attempts",
    type=int,
    default=0,
    help="Maximum generated sphere-view attempts tried after the joint baseline.",
)
parser.add_argument(
    "--active-max-joint-attempts",
    type=int,
    default=5,
    help="Maximum generated joint-pair observation attempts tried after the joint baseline.",
)
parser.add_argument(
    "--camera-detection-index",
    type=int,
    default=0,
    help="Which sorted camera detection to use when --camera-selection index.",
)
parser.add_argument(
    "--camera-selection",
    choices=("planner", "arm_side", "leftmost", "rightmost", "largest", "index"),
    default="planner",
    help="How to choose the camera detection used as the pre-grasp target.",
)
parser.add_argument(
    "--grasp-strategy",
    choices=("auto", "rim", "center"),
    default="auto",
    help="Generate ranked grasp candidates. Auto uses rim candidates with center hover fallback.",
)
parser.add_argument(
    "--planning-merge-distance",
    type=float,
    default=8.0,
    help="3D distance in native stage units for fusing detections from multiple cameras.",
)
parser.add_argument(
    "--candidate-log-count",
    type=int,
    default=6,
    help="Number of ranked grasp candidates to print before execution.",
)
parser.add_argument(
    "--camera-debug-dir",
    default=None,
    help="Optional output directory for RGB and plate-mask snapshots from each camera.",
)
parser.add_argument(
    "--camera-focal-length",
    type=float,
    default=6.0,
    help="Transient focalLength override for camera perception tests.",
)
parser.add_argument(
    "--camera-mount-rpy-deg",
    type=float,
    nargs=3,
    default=[0.0, 0.0, 0.0],
    help="Transient local camera optical-frame correction roll pitch yaw in degrees.",
)
parser.add_argument(
    "--left-camera-mount-rpy-deg",
    type=float,
    nargs=3,
    default=None,
    help="Optional left wrist camera optical-frame correction overriding --camera-mount-rpy-deg.",
)
parser.add_argument(
    "--right-camera-mount-rpy-deg",
    type=float,
    nargs=3,
    default=[0.0, -25.0, 0.0],
    help="Optional right wrist camera optical-frame correction overriding --camera-mount-rpy-deg.",
)
parser.add_argument(
    "--semantic-filter",
    default="class:*",
    help="Replicator semantic filter for camera perception.",
)
parser.add_argument(
    "--demo-view",
    choices=("camera", "scene", "camera_then_scene"),
    default="camera_then_scene",
    help="GUI viewport behavior for camera-perception demos.",
)
parser.add_argument(
    "--pos-tolerance",
    type=float,
    default=8.0,
    help="Final position tolerance in native stage units (centimeters).",
)
parser.add_argument(
    "--hold-seconds",
    type=float,
    default=0.0,
    help="Keep the GUI open after the final readback.",
)
parser.add_argument(
    "--skip-close",
    action="store_true",
    help="Skip simulation_app.close(); useful when Kit hangs during shutdown.",
)
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.utils.math import subtract_frame_transforms
from scipy.spatial.transform import Rotation as R

from dishwasher.grasping.generator import generate_grasp_pose, generate_rim_grasp_pose
from dishwasher.motion.ik_controller import PiperIKController
from dishwasher.motion.trajectory import interpolate_waypoints
from dishwasher.perception.detector import get_next_plate, get_plate_positions
from dishwasher.perception.native_camera import (
    add_runtime_semantics,
    calibrate_camera_offset_from_body,
    camera_path,
    configure_camera_model,
    create_camera_sensor,
    detect_camera_plates,
    robot_camera_pose_variant,
    usd_camera_pose_variant,
)
from dishwasher.perception.camera_plate_detector import semantic_output_to_mask
from dishwasher.planning.grasp_planner import (
    CameraPlateObservation,
    choose_best_candidate,
    fuse_camera_observations,
    generate_bimanual_grasp_candidates,
    with_score_adjustment,
)
from dishwasher.planning.active_view_planner import generate_active_view_attempts
from dishwasher.planning.camera_view_planner import plan_wrist_camera_view
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


def stage_to_meters(pos):
    return np.asarray(pos, dtype=np.float64) * 0.01


def tensor_to_np(tensor):
    return tensor.detach().cpu().numpy()


def active_camera_names() -> list[str]:
    if args_cli.perception != "camera":
        return []
    if args_cli.camera == "both_wrist":
        return ["left_wrist", "right_wrist"]
    return [args_cli.camera]


def camera_arm_name(camera_name: str) -> str | None:
    if camera_name == "left_wrist":
        return "left"
    if camera_name == "right_wrist":
        return "right"
    return None


def camera_mount_rpy_deg(camera_name: str) -> list[float]:
    if camera_name == "left_wrist" and args_cli.left_camera_mount_rpy_deg is not None:
        return args_cli.left_camera_mount_rpy_deg
    if camera_name == "right_wrist" and args_cli.right_camera_mount_rpy_deg is not None:
        return args_cli.right_camera_mount_rpy_deg
    return args_cli.camera_mount_rpy_deg


def selected_arm_name() -> str:
    if args_cli.arm != "auto":
        return args_cli.arm
    if args_cli.perception == "camera" and args_cli.camera == "right_wrist":
        return "right"
    return "left"


def arm_base_stage(arm_name: str) -> np.ndarray:
    if arm_name == "right":
        return np.array([135.0, 137.0, 14.45], dtype=np.float64)
    return np.array([90.0, 137.0, 14.45], dtype=np.float64)


def selected_arm_base_stage() -> np.ndarray:
    return arm_base_stage(selected_arm_name())


def required_arm_names() -> set[str]:
    arms: set[str] = set()
    if args_cli.perception == "camera":
        for camera_name in active_camera_names():
            arm = camera_arm_name(camera_name)
            if arm is not None:
                arms.add(arm)
    if args_cli.arm != "auto" or args_cli.perception != "camera":
        arms.add(selected_arm_name())
    if not arms:
        arms.add("left")
    return arms


def as_list(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [item for item in value if item is not None]
    if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        return [item for item in value if item is not None]
    return [value]


def set_arm_target(robot, ik, joint_pos_des):
    full_target = robot.data.joint_pos.clone()
    full_target[:, ik.arm_joint_ids] = joint_pos_des
    robot.set_joint_position_target(full_target)


def clamp_to_arm_limits(robot, ik, joint_pos_des):
    limits = robot.data.soft_joint_pos_limits[:, ik.arm_joint_ids]
    lower = limits[..., 0]
    upper = limits[..., 1]
    clamped = torch.max(torch.min(joint_pos_des, upper), lower)
    max_delta = float(torch.max(torch.abs(clamped - joint_pos_des)))
    return clamped, max_delta


def step_sim(sim, robots, plates, dt, camera_sensors=None):
    robots_list = as_list(robots)
    camera_sensor_list = as_list(camera_sensors)
    for robot in robots_list:
        robot.write_data_to_sim()
    sim.step()
    for robot in robots_list:
        robot.update(dt)
    for plate in plates.values():
        plate.update(dt)
    for camera_sensor in camera_sensor_list:
        camera_sensor.update(dt)
    simulation_app.update()


def set_viewport_camera(camera_prim_path: str):
    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_prim_path
            print(f"viewport camera: {camera_prim_path}", flush=True)
    except Exception as exc:
        print(f"[WARN] Could not set active viewport camera: {exc}", flush=True)


def apply_camera_mount_correction(camera_prim, rpy_deg: list[float]):
    if max(abs(float(value)) for value in rpy_deg) <= 1.0e-6:
        return
    from pxr import Gf, UsdGeom

    xf = UsdGeom.Xformable(camera_prim)
    orient_op = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpName() == "xformOp:orient":
            orient_op = op
            break
    if orient_op is None:
        orient_op = xf.AddOrientOp()
    current = orient_op.Get()
    if current is None:
        base_rot = R.identity()
    else:
        base_rot = R.from_quat(
            [
                float(current.GetImaginary()[0]),
                float(current.GetImaginary()[1]),
                float(current.GetImaginary()[2]),
                float(current.GetReal()),
            ]
        )
    extra_rot = R.from_euler("xyz", [float(v) for v in rpy_deg], degrees=True)
    new_quat = (base_rot * extra_rot).as_quat()
    orient_op.Set(
        Gf.Quatd(
            float(new_quat[3]),
            Gf.Vec3d(float(new_quat[0]), float(new_quat[1]), float(new_quat[2])),
        )
    )
    print(
        "camera mount correction applied to "
        f"{camera_prim.GetPath().pathString}: "
        f"rpy_deg=({rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f})",
        flush=True,
    )


def arm_joint_ids(robot):
    return [robot.joint_names.index(f"joint{idx}") for idx in range(1, 7)]


def hold_gui(seconds: float):
    if seconds <= 0.0:
        return
    import time

    print(f"Keeping GUI open for {seconds:.1f}s.", flush=True)
    deadline = time.time() + seconds
    while simulation_app.is_running() and time.time() < deadline:
        simulation_app.update()
        time.sleep(0.01)


def observe_joint_target(arm_name: str) -> list[float]:
    if arm_name == "right" and args_cli.right_camera_observe_joints is not None:
        return args_cli.right_camera_observe_joints
    return args_cli.camera_observe_joints


def move_arms_to_observe_pose(sim, robots_by_arm, plates, camera_sensors, dt):
    moving = {
        arm: robot
        for arm, robot in robots_by_arm.items()
        if robot is not None and arm in {camera_arm_name(name) for name in active_camera_names()}
    }
    if not moving:
        return

    joint_ids = {arm: arm_joint_ids(robot) for arm, robot in moving.items()}
    starts = {
        arm: robot.data.joint_pos[:, joint_ids[arm]].clone()
        for arm, robot in moving.items()
    }
    targets = {
        arm: torch.tensor(
            observe_joint_target(arm),
            dtype=torch.float32,
            device=args_cli.device,
        ).unsqueeze(0)
        for arm in moving
    }

    for step_idx in range(args_cli.camera_observe_steps):
        alpha = (step_idx + 1) / max(1, args_cli.camera_observe_steps)
        for arm, robot in moving.items():
            ids = joint_ids[arm]
            full_target = robot.data.joint_pos.clone()
            full_target[:, ids] = starts[arm] + alpha * (targets[arm] - starts[arm])
            robot.set_joint_position_target(full_target)
        step_sim(sim, moving.values(), plates, dt, camera_sensors.values())

    for arm, robot in moving.items():
        ids = joint_ids[arm]
        full_target = robot.data.joint_pos.clone()
        full_target[:, ids] = targets[arm]
        robot.set_joint_position_target(full_target)
    for _ in range(args_cli.camera_settle_steps):
        step_sim(sim, moving.values(), plates, dt, camera_sensors.values())

    for arm, robot in moving.items():
        ids = joint_ids[arm]
        actual = robot.data.joint_pos[0, ids].detach().cpu().numpy()
        print(
            f"    {arm} camera observe target: "
            + ", ".join(f"{v:.3f}" for v in observe_joint_target(arm)),
            flush=True,
        )
        print(
            f"    {arm} camera observe actual: "
            + ", ".join(f"{float(v):.3f}" for v in actual),
            flush=True,
        )


def move_arms_to_joint_observe_targets(
    sim,
    robots_by_arm,
    plates,
    camera_sensors,
    dt,
    targets_by_arm: dict[str, list[float]],
):
    moving = {
        arm: robot
        for arm, robot in robots_by_arm.items()
        if robot is not None and arm in targets_by_arm
    }
    if not moving:
        return

    joint_ids = {arm: arm_joint_ids(robot) for arm, robot in moving.items()}
    starts = {
        arm: robot.data.joint_pos[:, joint_ids[arm]].clone()
        for arm, robot in moving.items()
    }
    targets = {
        arm: torch.tensor(
            targets_by_arm[arm],
            dtype=torch.float32,
            device=args_cli.device,
        ).unsqueeze(0)
        for arm in moving
    }

    for step_idx in range(args_cli.camera_observe_steps):
        alpha = (step_idx + 1) / max(1, args_cli.camera_observe_steps)
        for arm, robot in moving.items():
            ids = joint_ids[arm]
            full_target = robot.data.joint_pos.clone()
            full_target[:, ids] = starts[arm] + alpha * (targets[arm] - starts[arm])
            robot.set_joint_position_target(full_target)
        step_sim(sim, moving.values(), plates, dt, camera_sensors.values())

    for arm, robot in moving.items():
        ids = joint_ids[arm]
        full_target = robot.data.joint_pos.clone()
        full_target[:, ids] = targets[arm]
        robot.set_joint_position_target(full_target)
    for _ in range(args_cli.camera_settle_steps):
        step_sim(sim, moving.values(), plates, dt, camera_sensors.values())

    for arm, robot in moving.items():
        ids = joint_ids[arm]
        actual = robot.data.joint_pos[0, ids].detach().cpu().numpy()
        print(
            f"    {arm} joint-pair observe target: "
            + ", ".join(f"{v:.3f}" for v in targets_by_arm[arm]),
            flush=True,
        )
        print(
            f"    {arm} joint-pair observe actual: "
            + ", ".join(f"{float(v):.3f}" for v in actual),
            flush=True,
        )


def _view_plan_candidates(
    arm: str,
    target: np.ndarray,
    camera_offset: dict,
    seed: dict[str, float] | None = None,
):
    seed = seed or {}
    seed_backoff = float(seed.get("view_backoff", args_cli.view_backoff))
    seed_height = float(seed.get("view_height", args_cli.view_height))
    seed_side_offset = float(seed.get("view_side_offset", args_cli.view_side_offset))
    backoffs = _unique_floats([
        seed_backoff,
        seed_backoff - 12.0,
        seed_backoff + 12.0,
        seed_backoff + 24.0,
        48.0,
        64.0,
        76.0,
        88.0,
    ])
    heights = _unique_floats([
        seed_height,
        seed_height - 12.0,
        seed_height + 12.0,
        seed_height + 24.0,
        64.0,
        72.0,
        84.0,
        96.0,
    ])
    side_offsets = _unique_floats([
        seed_side_offset,
        seed_side_offset - 18.0,
        seed_side_offset + 18.0,
        seed_side_offset - 8.0,
        seed_side_offset + 8.0,
    ])
    plans = []
    for backoff in backoffs:
        if backoff <= 5.0:
            continue
        for height in heights:
            if height < args_cli.view_min_camera_z - float(target[2]):
                continue
            for side_offset in side_offsets:
                plans.append(
                    plan_wrist_camera_view(
                        arm=arm,
                        target_pos_w=target,
                        arm_base_pos_w=arm_base_stage(arm),
                        camera_offset=camera_offset,
                        backoff=backoff,
                        height=height,
                        side_offset=side_offset,
                    )
                )
    return plans


def _view_safety_terms(plan) -> dict[str, float]:
    target = np.asarray(args_cli.view_target_stage, dtype=np.float64)
    camera = np.asarray(plan.camera_pos_w, dtype=np.float64)
    link6 = np.asarray(plan.link6_pos_w, dtype=np.float64)
    camera_dist = float(np.linalg.norm(camera - target))
    camera_xy_dist = float(np.linalg.norm(camera[:2] - target[:2]))
    link6_xy_dist = float(np.linalg.norm(link6[:2] - target[:2]))
    elevation_deg = float(
        np.degrees(
            np.arctan2(camera[2] - target[2], max(1.0, camera_xy_dist))
        )
    )
    workspace_x0, workspace_x1 = args_cli.view_workspace_x
    workspace_y0, workspace_y1 = args_cli.view_workspace_y
    link6_over_workspace = (
        workspace_x0 <= link6[0] <= workspace_x1
        and workspace_y0 <= link6[1] <= workspace_y1
    )

    terms = {
        "camera_too_low": max(0.0, args_cli.view_min_camera_z - float(camera[2])),
        "link6_too_low": max(0.0, args_cli.view_min_link6_z - float(link6[2])),
        "too_close": max(0.0, args_cli.view_min_camera_distance - camera_dist),
        "too_far": max(0.0, camera_dist - args_cli.view_max_camera_distance),
        "xy_too_close": max(0.0, args_cli.view_min_camera_xy_distance - camera_xy_dist),
        "robot_side": max(0.0, args_cli.view_robot_side_y_min - float(camera[1])),
        "workspace_intrusion": 0.0,
        "link6_near_target": max(0.0, 42.0 - link6_xy_dist),
        "low_elevation": max(0.0, args_cli.view_min_elevation_deg - elevation_deg),
    }
    if link6_over_workspace:
        terms["workspace_intrusion"] = max(0.0, args_cli.view_min_link6_z + 18.0 - float(link6[2]))
    return terms


def _score_view_plan(plan, robot, ik) -> float:
    target_pos_w = torch.tensor(
        plan.link6_pos_w, dtype=torch.float32, device=args_cli.device
    ).unsqueeze(0)
    target_quat_w = torch.tensor(
        plan.link6_quat_w, dtype=torch.float32, device=args_cli.device
    ).unsqueeze(0)
    root_pose = robot.data.root_pose_w
    target_pos_b, target_quat_b = subtract_frame_transforms(
        root_pose[:, 0:3],
        root_pose[:, 3:7],
        target_pos_w,
        target_quat_w,
    )
    joint_pos_des = ik.solve(target_pos_b, target_quat_b)
    if not torch.isfinite(joint_pos_des).all():
        return 1.0e6
    _clamped, clamp_delta = clamp_to_arm_limits(robot, ik, joint_pos_des)
    ee_pos_w, _ee_quat_w = ik.get_current_ee_pose()
    motion = float(torch.norm(ee_pos_w - target_pos_w, dim=-1)[0])
    target = np.asarray(args_cli.view_target_stage, dtype=np.float64)
    camera_dist = float(np.linalg.norm(plan.camera_pos_w - target))
    preferred_dist = 88.0
    safety = _view_safety_terms(plan)
    trajectory_clamp = _view_trajectory_max_clamp(plan, robot, ik)
    hard_violation = (
        safety["camera_too_low"]
        + safety["link6_too_low"]
        + safety["too_close"]
        + safety["xy_too_close"]
        + safety["low_elevation"]
        + safety["workspace_intrusion"]
    )
    return (
        500.0 * hard_violation
        + 900.0 * trajectory_clamp
        + 70.0 * safety["robot_side"]
        + 30.0 * safety["link6_near_target"]
        + 12.0 * safety["too_far"]
        + 35.0 * clamp_delta
        + 0.025 * motion
        + 0.018 * abs(camera_dist - preferred_dist)
    )


def _unique_floats(values):
    result = []
    for value in values:
        value = round(float(value), 3)
        if value not in result:
            result.append(value)
    return result


def _safe_view_trajectory(start_pos, start_quat, end_pos, end_quat, num_steps):
    if num_steps <= 2:
        return interpolate_waypoints(start_pos, start_quat, end_pos, end_quat, num_steps)

    lift_pos = np.asarray(start_pos, dtype=np.float64).copy()
    lift_pos[2] = max(
        float(lift_pos[2]),
        args_cli.view_safe_lift_z,
        args_cli.view_min_link6_z + 8.0,
        float(end_pos[2]) + 6.0,
    )
    lift_steps = max(2, int(round(num_steps * 0.35)))
    view_steps = num_steps - lift_steps + 1
    lift_segment = interpolate_waypoints(
        np.asarray(start_pos, dtype=np.float64),
        np.asarray(start_quat, dtype=np.float64),
        lift_pos,
        np.asarray(start_quat, dtype=np.float64),
        lift_steps,
    )
    view_segment = interpolate_waypoints(
        lift_pos,
        np.asarray(start_quat, dtype=np.float64),
        np.asarray(end_pos, dtype=np.float64),
        np.asarray(end_quat, dtype=np.float64),
        view_steps,
    )
    return lift_segment[:-1] + view_segment


def _view_trajectory_max_clamp(plan, robot, ik, sample_steps: int = 24) -> float:
    ee_pos_w, ee_quat_w = ik.get_current_ee_pose()
    trajectory = _safe_view_trajectory(
        tensor_to_np(ee_pos_w[0]),
        tensor_to_np(ee_quat_w[0]),
        plan.link6_pos_w,
        plan.link6_quat_w,
        sample_steps,
    )
    max_clamp = 0.0
    root_pose = robot.data.root_pose_w
    for pos, quat in trajectory:
        target_pos_w = torch.tensor(pos, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
        target_quat_w = torch.tensor(quat, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pose[:, 0:3],
            root_pose[:, 3:7],
            target_pos_w,
            target_quat_w,
        )
        joint_pos_des = ik.solve(target_pos_b, target_quat_b)
        if not torch.isfinite(joint_pos_des).all():
            return 1.0e3
        _clamped, clamp_delta = clamp_to_arm_limits(robot, ik, joint_pos_des)
        max_clamp = max(max_clamp, clamp_delta)
    return max_clamp


def move_arms_to_view_pose(sim, robots_by_arm, ik_by_arm, plates, camera_sensors, camera_offsets, dt):
    moving = {
        arm: robot
        for arm, robot in robots_by_arm.items()
        if robot is not None and arm in {camera_arm_name(name) for name in active_camera_names()}
    }
    if not moving:
        return

    target = np.asarray(args_cli.view_target_stage, dtype=np.float64)
    plans = {}
    trajectories = {}
    for arm, robot in moving.items():
        camera_name = f"{arm}_wrist"
        candidate_plans = _view_plan_candidates(arm, target, camera_offsets[camera_name])
        ranked = [
            (_score_view_plan(candidate, robot, ik_by_arm[arm]), candidate)
            for candidate in candidate_plans
        ]
        ranked.sort(key=lambda item: item[0])
        plan = ranked[0][1]
        score = ranked[0][0]
        safety = _view_safety_terms(plan)
        print(
            f"    [{arm}] selected view candidate: "
            f"score={score:.3f}, tried={len(ranked)}",
            flush=True,
        )
        print(
            f"    [{arm}] selected view safety: "
            + ", ".join(f"{name}={value:.2f}" for name, value in safety.items()),
            flush=True,
        )
        print(
            f"    [{arm}] selected view trajectory clamp: "
            f"{_view_trajectory_max_clamp(plan, robot, ik_by_arm[arm]):.3f}",
            flush=True,
        )
        for rank, (candidate_score, candidate) in enumerate(ranked[:3]):
            candidate_safety = _view_safety_terms(candidate)
            print(
                f"      view rank {rank}: score={candidate_score:.3f}, "
                f"camera=({candidate.camera_pos_w[0]:.1f}, {candidate.camera_pos_w[1]:.1f}, {candidate.camera_pos_w[2]:.1f}), "
                f"link6=({candidate.link6_pos_w[0]:.1f}, {candidate.link6_pos_w[1]:.1f}, {candidate.link6_pos_w[2]:.1f}), "
                f"traj_clamp={_view_trajectory_max_clamp(candidate, robot, ik_by_arm[arm]):.3f}, "
                f"low={candidate_safety['camera_too_low'] + candidate_safety['link6_too_low']:.1f}, "
                f"close={candidate_safety['too_close']:.1f}, "
                f"xy_close={candidate_safety['xy_too_close']:.1f}, "
                f"low_elev={candidate_safety['low_elevation']:.1f}, "
                f"workspace={candidate_safety['workspace_intrusion']:.1f}",
                flush=True,
            )
        plans[arm] = plan
        ik = ik_by_arm[arm]
        ee_pos_w, ee_quat_w = ik.get_current_ee_pose()
        trajectories[arm] = _safe_view_trajectory(
            tensor_to_np(ee_pos_w[0]),
            tensor_to_np(ee_quat_w[0]),
            plan.link6_pos_w,
            plan.link6_quat_w,
            args_cli.camera_observe_steps,
        )
        print(
            f"    [{arm}] task-space camera target: "
            f"look_at=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f}), "
            f"camera=({plan.camera_pos_w[0]:.2f}, {plan.camera_pos_w[1]:.2f}, {plan.camera_pos_w[2]:.2f}), "
            f"link6=({plan.link6_pos_w[0]:.2f}, {plan.link6_pos_w[1]:.2f}, {plan.link6_pos_w[2]:.2f})",
            flush=True,
        )

    max_clamp_delta = {arm: 0.0 for arm in moving}
    for step_idx in range(args_cli.camera_observe_steps):
        for arm, robot in moving.items():
            ik = ik_by_arm[arm]
            pos, quat = trajectories[arm][step_idx]
            target_pos_w = torch.tensor(pos, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
            target_quat_w = torch.tensor(quat, dtype=torch.float32, device=args_cli.device).unsqueeze(0)
            root_pose = robot.data.root_pose_w
            target_pos_b, target_quat_b = subtract_frame_transforms(
                root_pose[:, 0:3],
                root_pose[:, 3:7],
                target_pos_w,
                target_quat_w,
            )
            joint_pos_des = ik.solve(target_pos_b, target_quat_b)
            if not torch.isfinite(joint_pos_des).all():
                print(f"    [WARN] non-finite view IK for {arm} at step {step_idx}", flush=True)
                continue
            if not args_cli.no_clamp_joints:
                joint_pos_des, clamp_delta = clamp_to_arm_limits(robot, ik, joint_pos_des)
                max_clamp_delta[arm] = max(max_clamp_delta[arm], clamp_delta)
            set_arm_target(robot, ik, joint_pos_des)
        step_sim(sim, moving.values(), plates, dt, camera_sensors.values())

    for _ in range(args_cli.camera_settle_steps):
        step_sim(sim, moving.values(), plates, dt, camera_sensors.values())

    for arm, robot in moving.items():
        actual = robot.data.joint_pos[0, arm_joint_ids(robot)].detach().cpu().numpy()
        print(
            f"    {arm} view-planned joint actual: "
            + ", ".join(f"{float(v):.3f}" for v in actual),
            flush=True,
        )
        camera_name = f"{arm}_wrist"
        pose = robot_camera_pose_variant(
            robot,
            camera_offsets[camera_name],
            device=args_cli.device,
        )
        pos = tensor_to_np(pose.pos_w)
        print(
            f"    {arm} view-planned camera actual: "
            f"({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}), "
            f"max_clamp_delta={max_clamp_delta[arm]:.3f}",
            flush=True,
        )
        if pos[2] < args_cli.view_min_camera_z or max_clamp_delta[arm] > 0.35:
            print(
                f"    [WARN] {arm} observe execution is not safe enough: "
                f"camera_z={pos[2]:.2f} min={args_cli.view_min_camera_z:.2f}, "
                f"max_clamp_delta={max_clamp_delta[arm]:.3f}",
                flush=True,
            )


def print_pose(label, pos):
    meters = stage_to_meters(pos)
    print(
        f"{label}: stage=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
        f"meters=({meters[0]:.3f}, {meters[1]:.3f}, {meters[2]:.3f})",
        flush=True,
    )


def save_camera_debug_frame(camera_name: str, frame, *, attempt_label: str | None = None):
    if args_cli.camera_debug_dir is None:
        return
    from pathlib import Path
    from PIL import Image

    out_dir = Path(args_cli.camera_debug_dir)
    if attempt_label:
        out_dir = out_dir / attempt_label
    out_dir = out_dir / camera_name
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb = frame.rgb.detach().cpu().numpy()
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    Image.fromarray(rgb[..., :3]).save(out_dir / "rgb.png")
    Image.fromarray((frame.plate_mask.astype(np.uint8) * 255)).save(
        out_dir / "plate_mask.png"
    )
    print(f"    [{camera_name}] saved debug images: {out_dir.resolve()}", flush=True)


def camera_frame_quality(frame) -> tuple[float, dict[str, float]]:
    mask = np.asarray(frame.plate_mask, dtype=bool)
    height, width = mask.shape
    total_px = max(1, height * width)
    area_ratio = float(mask.sum()) / float(total_px)
    detections = list(frame.detections)
    edge_penalty = 0.0
    if detections:
        largest = max(detections, key=lambda item: item.area_px)
        center_u, center_v = largest.centroid_uv
        center_error = np.sqrt(
            ((center_u - width * 0.5) / max(1.0, width * 0.5)) ** 2
            + ((center_v - height * 0.5) / max(1.0, height * 0.5)) ** 2
        )
        x_min, y_min, x_max, y_max = largest.bbox_xyxy
        edge_margin = max(4, int(round(min(width, height) * 0.03)))
        touches_edge = (
            x_min <= edge_margin
            or y_min <= edge_margin
            or x_max >= width - edge_margin
            or y_max >= height - edge_margin
        )
        edge_penalty = 0.28 if touches_edge else 0.0
    else:
        center_error = 1.0

    size_score = min(1.0, area_ratio / 0.045)
    if area_ratio > 0.16:
        size_score *= max(0.0, 1.0 - (area_ratio - 0.16) / 0.16)
    center_score = max(0.0, 1.0 - float(center_error))
    count_score = min(1.0, len(detections) / 3.0)

    table_mask = semantic_output_to_mask(frame.semantic, frame.semantic_info, target_label="table")
    sink_mask = semantic_output_to_mask(frame.semantic, frame.semantic_info, target_label="sink")
    rack_mask = semantic_output_to_mask(frame.semantic, frame.semantic_info, target_label="rack")
    top_cut = max(1, int(round(height * 0.35)))
    top_table_ratio = float(table_mask[:top_cut, :].mean())
    context_ratio = float((sink_mask | rack_mask).mean())
    context_score = min(1.0, context_ratio / 0.22)
    occlusion_penalty = min(0.55, top_table_ratio * 0.75)

    score = (
        0.28 * size_score
        + 0.34 * center_score
        + 0.20 * context_score
        + 0.18 * count_score
        - occlusion_penalty
        - edge_penalty
    )
    terms = {
        "area_ratio": area_ratio,
        "center_error": float(center_error),
        "detections": float(len(detections)),
        "context_ratio": context_ratio,
        "top_table_ratio": top_table_ratio,
        "occlusion_penalty": occlusion_penalty,
        "edge_penalty": edge_penalty,
    }
    return max(0.0, min(1.0, float(score))), terms


def _ramp_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 1.0 if value >= low else 0.0
    return max(0.0, min(1.0, (float(value) - low) / (high - low)))


def camera_pose_quality(camera_name: str, pose, robot=None, camera_offset: dict | None = None) -> tuple[float, dict[str, float]]:
    target = np.asarray(args_cli.view_target_stage, dtype=np.float64)
    camera_pos = tensor_to_np(pose.pos_w).astype(np.float64)
    forward = np.asarray(pose.rotm_w_ros, dtype=np.float64)[:, 2]
    forward = forward / max(1.0e-9, float(np.linalg.norm(forward)))
    to_target = target - camera_pos
    distance = float(np.linalg.norm(to_target))
    xy_distance = float(np.linalg.norm(to_target[:2]))
    to_target_unit = to_target / max(1.0e-9, distance)
    look_cos = max(-1.0, min(1.0, float(np.dot(forward, to_target_unit))))
    look_error_deg = float(np.degrees(np.arccos(look_cos)))
    elevation_deg = float(
        np.degrees(
            np.arctan2(camera_pos[2] - target[2], max(1.0, xy_distance))
        )
    )
    downward_z = max(0.0, float(-forward[2]))

    link6_pos = camera_pos
    if robot is not None and camera_offset is not None:
        try:
            body_idx = robot.body_names.index(camera_offset["body_name"])
            link6_pos = tensor_to_np(robot.data.body_pos_w[0, body_idx]).astype(np.float64)
        except Exception:
            link6_pos = camera_pos
    link6_xy_distance = float(np.linalg.norm(link6_pos[:2] - target[:2]))

    camera_z_short = max(0.0, args_cli.view_min_camera_z - float(camera_pos[2]))
    distance_short = max(0.0, args_cli.view_min_camera_distance - distance)
    distance_long = max(0.0, distance - args_cli.view_max_camera_distance)
    xy_short = max(0.0, args_cli.view_min_camera_xy_distance - xy_distance)
    low_elevation = max(0.0, args_cli.view_min_elevation_deg - elevation_deg)
    look_error_excess = max(0.0, look_error_deg - args_cli.view_max_look_error_deg)
    downward_short = max(0.0, args_cli.view_min_downward_z - downward_z)
    link6_z_short = max(0.0, args_cli.view_min_link6_z - float(link6_pos[2]))
    robot_side_short = max(0.0, args_cli.view_robot_side_y_min - float(camera_pos[1]))
    link6_near_target = max(0.0, 42.0 - link6_xy_distance)

    z_score = _ramp_score(float(camera_pos[2]), args_cli.view_min_camera_z, args_cli.view_min_camera_z + 24.0)
    distance_score = _ramp_score(distance, args_cli.view_min_camera_distance, args_cli.view_min_camera_distance + 24.0)
    if distance > args_cli.view_max_camera_distance:
        distance_score *= max(0.0, 1.0 - distance_long / 36.0)
    xy_score = _ramp_score(xy_distance, args_cli.view_min_camera_xy_distance, args_cli.view_min_camera_xy_distance + 22.0)
    elevation_score = _ramp_score(elevation_deg, args_cli.view_min_elevation_deg, args_cli.view_min_elevation_deg + 18.0)
    look_score = max(0.0, 1.0 - look_error_deg / max(1.0, args_cli.view_max_look_error_deg))
    downward_score = _ramp_score(downward_z, args_cli.view_min_downward_z, min(1.0, args_cli.view_min_downward_z + 0.32))
    link6_z_score = _ramp_score(float(link6_pos[2]), args_cli.view_min_link6_z, args_cli.view_min_link6_z + 22.0)
    link6_clear_score = _ramp_score(link6_xy_distance, 42.0, 62.0)
    robot_side_score = max(0.0, 1.0 - robot_side_short / 28.0)

    score = (
        0.16 * z_score
        + 0.14 * distance_score
        + 0.12 * xy_score
        + 0.15 * elevation_score
        + 0.18 * look_score
        + 0.11 * downward_score
        + 0.08 * link6_z_score
        + 0.04 * link6_clear_score
        + 0.02 * robot_side_score
    )
    hard_failure = any(
        value > 0.0
        for value in (
            camera_z_short,
            distance_short,
            xy_short,
            low_elevation,
            look_error_excess,
            downward_short,
            link6_z_short,
        )
    )
    if hard_failure:
        score = min(score, 0.35)
    if link6_near_target > 0.0 and link6_z_short > 0.0:
        score = min(score, 0.20)

    terms = {
        "camera_x": float(camera_pos[0]),
        "camera_y": float(camera_pos[1]),
        "camera_z": float(camera_pos[2]),
        "distance": distance,
        "xy_distance": xy_distance,
        "elevation_deg": elevation_deg,
        "look_error_deg": look_error_deg,
        "downward_z": downward_z,
        "link6_z": float(link6_pos[2]),
        "link6_xy_distance": link6_xy_distance,
        "camera_z_short": camera_z_short,
        "distance_short": distance_short,
        "xy_short": xy_short,
        "low_elevation": low_elevation,
        "look_error_excess": look_error_excess,
        "downward_short": downward_short,
        "link6_z_short": link6_z_short,
        "link6_near_target": link6_near_target,
        "hard_pose_failure": 1.0 if hard_failure else 0.0,
    }
    return max(0.0, min(1.0, float(score))), terms


def capture_camera_observations(
    robots_by_arm,
    camera_offsets,
    camera_prims,
    camera_sensors,
    *,
    attempt_label: str | None = None,
):
    observations = []
    qualities: dict[str, float] = {}
    detection_counts: dict[str, int] = {}
    for camera_name in active_camera_names():
        arm_for_camera = camera_arm_name(camera_name)
        if arm_for_camera is None:
            camera_pose = usd_camera_pose_variant(
                camera_prims[camera_name],
                device=args_cli.device,
            )
        else:
            camera_pose = robot_camera_pose_variant(
                robots_by_arm[arm_for_camera],
                camera_offsets[camera_name],
                device=args_cli.device,
            )
        frame = detect_camera_plates(
            camera_sensors[camera_name],
            camera_pose,
            device=args_cli.device,
            min_area_px=args_cli.camera_min_area,
        )
        semantic_labels = {}
        if frame.semantic_info:
            for key in ("idToLabels", "idToSemantics", "idToSemantic"):
                mapping = frame.semantic_info.get(key)
                if isinstance(mapping, dict) and mapping:
                    semantic_labels = mapping
                    break
        prefix = f"    [{camera_name}]"
        if attempt_label:
            prefix = f"    [{attempt_label}:{camera_name}]"
        print(f"{prefix} semantic filter: {args_cli.semantic_filter}", flush=True)
        print(f"{prefix} semantic labels: {semantic_labels}", flush=True)
        print(f"{prefix} plate mask pixels: {int(frame.plate_mask.sum())}", flush=True)
        print(f"{prefix} detections: {len(frame.detections)}", flush=True)
        frame_quality_score, frame_quality_terms = camera_frame_quality(frame)
        pose_quality_score, pose_quality_terms = camera_pose_quality(
            camera_name,
            camera_pose,
            robot=robots_by_arm.get(arm_for_camera) if arm_for_camera is not None else None,
            camera_offset=camera_offsets.get(camera_name),
        )
        quality_score = frame_quality_score * pose_quality_score
        qualities[camera_name] = quality_score
        detection_counts[camera_name] = len(frame.detections)
        print(
            f"{prefix} frame quality: score={frame_quality_score:.3f}, "
            + ", ".join(f"{name}={value:.3f}" for name, value in frame_quality_terms.items()),
            flush=True,
        )
        print(
            f"{prefix} pose quality: score={pose_quality_score:.3f}, "
            + ", ".join(f"{name}={value:.3f}" for name, value in pose_quality_terms.items()),
            flush=True,
        )
        print(
            f"{prefix} view quality: score={quality_score:.3f}",
            flush=True,
        )
        if quality_score < args_cli.camera_quality_threshold:
            print(
                f"    [WARN] {camera_name} view quality below threshold "
                f"{args_cli.camera_quality_threshold:.2f}",
                flush=True,
            )
        save_camera_debug_frame(camera_name, frame, attempt_label=attempt_label)
        for idx, detection in enumerate(frame.detections):
            print(
                f"{prefix} detection {idx}: area={detection.area_px}, "
                f"uv=({detection.centroid_uv[0]:.1f}, {detection.centroid_uv[1]:.1f}), "
                f"bbox={detection.bbox_xyxy}",
                flush=True,
            )
            print_pose("      camera estimate", detection.pos_w)
            observations.append(
                CameraPlateObservation(
                    camera=camera_name,
                    arm=arm_for_camera,
                    detection_index=idx,
                    label=detection.label,
                    area_px=detection.area_px,
                    centroid_uv=detection.centroid_uv,
                    bbox_xyxy=detection.bbox_xyxy,
                    pos_w=np.asarray(detection.pos_w, dtype=np.float64),
                )
            )
    return observations, qualities, detection_counts


def combined_observe_quality(qualities: dict[str, float], detection_counts: dict[str, int]) -> float:
    if not qualities:
        return 0.0
    values = sorted(float(value) for value in qualities.values())
    best = values[-1]
    weakest = values[0]
    dual_bonus = 0.08 if sum(1 for count in detection_counts.values() if count > 0) >= 2 else 0.0
    return max(0.0, min(1.0, 0.70 * best + 0.30 * weakest + dual_bonus))


def move_arms_to_default_pose(sim, robots_by_arm, plates, camera_sensors, dt, steps: int = 120):
    if not robots_by_arm:
        return
    starts = {arm: robot.data.joint_pos.clone() for arm, robot in robots_by_arm.items()}
    targets = {arm: robot.data.default_joint_pos.clone() for arm, robot in robots_by_arm.items()}
    for step_idx in range(steps):
        alpha = (step_idx + 1) / max(1, steps)
        for arm, robot in robots_by_arm.items():
            robot.set_joint_position_target(starts[arm] + alpha * (targets[arm] - starts[arm]))
        step_sim(sim, robots_by_arm.values(), plates, dt, camera_sensors.values())


def active_observation_attempts():
    attempts = [
        {
            "name": "joint_baseline",
            "mode": "joint",
        },
    ]
    attempts.extend(active_joint_pair_attempts(args_cli.active_max_joint_attempts))
    attempts.extend(
        generate_active_view_attempts(
            max_attempts=args_cli.active_max_view_attempts,
            ik_mode="pose",
        )
    )
    if args_cli.active_max_view_attempts > 0:
        attempts.append(
            {
                "name": "view_pose_check",
                "mode": "view",
                "ik": "pose",
                "view_backoff": 64.0,
                "view_height": 76.0,
                "view_side_offset": 0.0,
            }
        )
    return attempts


def active_joint_pair_attempts(max_attempts: int) -> list[dict[str, object]]:
    if max_attempts <= 0:
        return []

    # These are commanded, settled joint targets from
    # scripts/sample_camera_view_reachability.py.  They replace the previous
    # "seed + arbitrary delta" probing with profiles that the position
    # controller can actually hold while both wrist cameras keep plate/context
    # detections in frame.
    profiles = [
        {
            "name": "joint_pair_context_wider",
            "left": [-0.020, 0.345, -0.404, 0.127, 0.989, -0.729],
            "right": [-0.025, 0.196, -0.101, -1.076, 0.951, -0.134],
            "prior": 0.88,
            "intent": "increase scene context while keeping both cameras on plate",
        },
        {
            "name": "joint_pair_balanced_lift",
            "left": [0.160, 0.405, -0.424, 0.107, 1.029, -0.709],
            "right": [-0.205, 0.256, -0.121, -1.096, 0.991, -0.154],
            "prior": 0.82,
            "intent": "slightly lift both wrists without pushing into unreachable overhead IK",
        },
        {
            "name": "joint_pair_left_centered",
            "left": [0.166, 0.225, -0.373, -0.183, 1.085, -1.026],
            "right": [-0.205, 0.076, -0.001, -0.996, 0.811, -0.254],
            "prior": 0.76,
            "intent": "favor left camera centering and keep right camera as confirmation",
        },
        {
            "name": "joint_pair_left_open_context",
            "left": [0.284, 0.206, -0.378, -0.721, 1.208, -1.046],
            "right": [-0.337, 0.027, -0.034, -0.463, 0.569, -1.051],
            "prior": 0.70,
            "intent": "open left shoulder for wider sink/rack context",
        },
        {
            "name": "joint_pair_right_assist",
            "left": [-0.147, 0.234, -0.274, -0.304, 0.724, 0.678],
            "right": [-0.389, 0.021, -0.106, -0.297, 0.766, -1.476],
            "prior": 0.64,
            "intent": "test right-side assist view when left context is acceptable",
        },
    ]
    attempts = []
    for profile in profiles[:max_attempts]:
        attempts.append(
            {
                "name": profile["name"],
                "mode": "joint_pair",
                "left_joints": _clip_observe_joints(np.asarray(profile["left"])).tolist(),
                "right_joints": _clip_observe_joints(np.asarray(profile["right"])).tolist(),
                "prior_score": float(profile["prior"]),
                "intent": profile["intent"],
                "source": "commanded_reachability_profile",
            }
        )
    return attempts


def _clip_observe_joints(joints: np.ndarray) -> np.ndarray:
    lower = np.array([-2.618, 0.0, -2.697, -1.832, -1.220, -3.140])
    upper = np.array([2.618, 3.140, 0.0, 1.832, 1.220, 3.140])
    return np.clip(np.asarray(joints, dtype=np.float64), lower, upper)


def run_with_view_overrides(attempt, fn):
    fields = ("view_backoff", "view_height", "view_side_offset")
    old_values = {field: getattr(args_cli, field) for field in fields}
    try:
        for field in fields:
            if field in attempt:
                setattr(args_cli, field, float(attempt[field]))
        return fn()
    finally:
        for field, value in old_values.items():
            setattr(args_cli, field, value)


def run_active_observation(
    sim,
    robots_by_arm,
    ik_by_arm,
    observe_ik_by_arm,
    plates,
    camera_sensors,
    camera_offsets,
    camera_prims,
    dt,
):
    best = None
    for attempt_idx, attempt in enumerate(active_observation_attempts(), start=1):
        name = str(attempt["name"])
        print(f"    [active] attempt {attempt_idx}: {name}", flush=True)
        if "prior_score" in attempt:
            print(
                f"      candidate prior={float(attempt['prior_score']):.3f}",
                flush=True,
            )
        if "source" in attempt or "intent" in attempt:
            print(
                f"      source={attempt.get('source', 'unknown')}, "
                f"intent={attempt.get('intent', '')}",
                flush=True,
            )
        if "radius" in attempt:
            print(
                f"      sampled view: radius={float(attempt['radius']):.1f}, "
                f"elevation={float(attempt['elevation_deg']):.1f}deg, "
                f"bearing={float(attempt['bearing_deg']):+.1f}deg",
                flush=True,
            )
        move_arms_to_default_pose(sim, robots_by_arm, plates, camera_sensors, dt)
        if attempt["mode"] == "joint":
            move_arms_to_observe_pose(sim, robots_by_arm, plates, camera_sensors, dt)
        elif attempt["mode"] == "joint_pair":
            move_arms_to_joint_observe_targets(
                sim,
                robots_by_arm,
                plates,
                camera_sensors,
                dt,
                {
                    "left": list(attempt["left_joints"]),
                    "right": list(attempt["right_joints"]),
                },
            )
        else:
            attempt_ik = ik_by_arm if attempt.get("ik") == "pose" else observe_ik_by_arm
            run_with_view_overrides(
                attempt,
                lambda: move_arms_to_view_pose(
                    sim,
                    robots_by_arm,
                    attempt_ik,
                    plates,
                    camera_sensors,
                    camera_offsets,
                    dt,
                ),
            )
        for _ in range(args_cli.camera_warmup_steps):
            step_sim(sim, robots_by_arm.values(), plates, dt, camera_sensors.values())
        observations, qualities, detection_counts = capture_camera_observations(
            robots_by_arm,
            camera_offsets,
            camera_prims,
            camera_sensors,
            attempt_label=name,
        )
        quality = combined_observe_quality(qualities, detection_counts)
        both_ok = all(
            detection_counts.get(camera_name, 0) > 0
            for camera_name in active_camera_names()
            if camera_arm_name(camera_name) is not None
        )
        accepted = quality >= args_cli.active_min_quality
        if args_cli.active_require_both_cameras:
            accepted = accepted and both_ok
        print(
            f"    [active] attempt {name}: combined_quality={quality:.3f}, "
            f"both_cameras_detected={both_ok}, accepted={accepted}",
            flush=True,
        )
        if best is None or quality > best["quality"]:
            best = {
                "name": name,
                "observations": observations,
                "qualities": qualities,
                "detection_counts": detection_counts,
                "quality": quality,
                "both_ok": both_ok,
                "accepted": accepted,
            }
        if accepted:
            print(f"    [active] selected observation attempt: {name}", flush=True)
            if args_cli.active_stop_on_first_pass:
                return observations

    if best is None or not best["observations"]:
        print("[FAIL] Active observation found no camera detections", flush=True)
        exit_with(1)
    if best["accepted"]:
        print(
            f"    [active] selected best accepted attempt after full search: "
            f"{best['name']} quality={best['quality']:.3f}",
            flush=True,
        )
        return best["observations"]
    print(
        f"    [active] best attempt was {best['name']} with quality={best['quality']:.3f}; "
        "no attempt met the acceptance threshold.",
        flush=True,
    )
    if not args_cli.allow_low_quality_observe:
        print(
            "[FAIL] Active observation quality is too low; refusing to continue to grasp planning.",
            flush=True,
        )
        exit_with(1)
    print("    [active] continuing with best low-quality observation by request.", flush=True)
    return best["observations"]


def score_candidate_with_ik(candidate, robot, ik):
    target_pos, target_quat = candidate.grasp_plan["pre_grasp"]
    target_pos_w = torch.tensor(
        target_pos, dtype=torch.float32, device=args_cli.device
    ).unsqueeze(0)
    target_quat_w = torch.tensor(
        target_quat, dtype=torch.float32, device=args_cli.device
    ).unsqueeze(0)
    root_pose = robot.data.root_pose_w
    target_pos_b, target_quat_b = subtract_frame_transforms(
        root_pose[:, 0:3],
        root_pose[:, 3:7],
        target_pos_w,
        target_quat_w,
    )
    joint_pos_des = ik.solve(target_pos_b, target_quat_b)
    finite = bool(torch.isfinite(joint_pos_des).all())
    if not finite:
        return with_score_adjustment(candidate, terms={"ik_finite": -100.0})

    _clamped, clamp_delta = clamp_to_arm_limits(robot, ik, joint_pos_des)
    current_ee_pos, _current_ee_quat = ik.get_current_ee_pose()
    target_distance = float(torch.norm(current_ee_pos - target_pos_w, dim=-1)[0])
    current_ee_np = tensor_to_np(current_ee_pos[0])
    base = arm_base_stage(candidate.arm)
    inward_y = max(0.0, float(base[1] - target_pos[1]))
    y_shift = abs(float(current_ee_np[1] - target_pos[1]))
    x_shift = abs(float(current_ee_np[0] - target_pos[0]))
    if 8.0 <= inward_y <= 25.0:
        approach_band = 0.65
    elif inward_y > 25.0:
        approach_band = -0.09 * (inward_y - 25.0)
    else:
        approach_band = -0.04 * (8.0 - inward_y)
    terms = {
        "ik_finite": 0.8,
        "joint_limit_margin": -min(3.0, 4.0 * clamp_delta),
        "motion_distance": -0.004 * target_distance,
        "approach_band": approach_band,
        "ee_y_shift": -0.07 * max(0.0, y_shift - 18.0),
        "ee_x_shift": -0.08 * max(0.0, x_shift - 15.0),
    }
    return with_score_adjustment(candidate, terms=terms)


print("=" * 72)
print("Native all.usd Level 1 pre-grasp execution test")
print("=" * 72)

loader = NativeSceneLoader(args_cli.assets)
report = loader.validate()
if not report.ok:
    print("[FAIL] Native all.usd validation failed.", flush=True)
    exit_with(1)

stage = loader.open_stage()
if args_cli.keep_runtime_helpers:
    print("Keeping nested PhysicsScene and ROS2 ActionGraph active.", flush=True)
else:
    changed = loader.prepare_for_direct_isaaclab(
        stage,
        deactivate_nested_physics_scenes=True,
        deactivate_action_graphs=True,
    )
    print(
        f"runtime disabled nested PhysicsScene: "
        f"{changed['nested_physics_scenes']}",
        flush=True,
    )
    print(f"runtime disabled ActionGraph: {changed['action_graphs']}", flush=True)

camera_sensors = {}
camera_prims = {}
camera_paths = {}
camera_offsets = {}
if args_cli.perception == "camera":
    labeled = add_runtime_semantics(stage, loader, include_scene=True)
    print(
        "runtime semantic labels: "
        + ", ".join(f"{name}={len(paths)}" for name, paths in labeled.items()),
        flush=True,
    )
    for camera_name in active_camera_names():
        cam_path = camera_path(loader, camera_name)
        camera_prim = stage.GetPrimAtPath(cam_path)
        if not camera_prim or not camera_prim.IsValid():
            print(f"[FAIL] Camera prim not found: {cam_path}", flush=True)
            exit_with(1)
        sim_utils.standardize_xform_ops(camera_prim)
        apply_camera_mount_correction(camera_prim, camera_mount_rpy_deg(camera_name))
        camera_model = configure_camera_model(
            camera_prim,
            focal_length=args_cli.camera_focal_length,
        )
        camera_prims[camera_name] = camera_prim
        camera_paths[camera_name] = cam_path
        print(
            f"{camera_name} model: "
            f"focalLength={camera_model['focal_length']:.3f}, "
            f"hfov={camera_model['horizontal_fov_deg']:.1f}deg, "
            f"vfov={camera_model['vertical_fov_deg']:.1f}deg",
            flush=True,
        )
    if active_camera_names():
        set_viewport_camera(camera_paths[active_camera_names()[0]])
    if args_cli.demo_view == "scene":
        set_viewport_camera(loader.paths.scene_camera)

print("[init] creating SimulationContext", flush=True)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
print("[init] wrapping native assets", flush=True)
arms_to_wrap = required_arm_names()
loader.wrap_assets(wrap_left="left" in arms_to_wrap, wrap_right="right" in arms_to_wrap)
robots_by_arm = {
    "left": loader.piper_l,
    "right": loader.piper_r,
}
robots_by_arm = {arm: robot for arm, robot in robots_by_arm.items() if robot is not None}
print(f"[init] wrapped arms: {sorted(robots_by_arm)}", flush=True)
if args_cli.perception == "camera":
    for camera_name, cam_path in camera_paths.items():
        camera_sensors[camera_name] = create_camera_sensor(
            prim_path=cam_path,
            width=args_cli.camera_width,
            height=args_cli.camera_height,
            semantic_filter=args_cli.semantic_filter,
        )

print("[init] sim.reset()", flush=True)
sim.reset()
dt = sim.get_physics_dt()
print(f"[init] sim dt={dt:.6f}", flush=True)

for robot in robots_by_arm.values():
    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
print(f"[init] settling for {args_cli.settle_steps} steps", flush=True)
for _ in range(args_cli.settle_steps):
    step_sim(sim, robots_by_arm.values(), loader.plates, dt, camera_sensors.values())

if args_cli.ik_method == "dls":
    ik_params = {"lambda_val": args_cli.ik_lambda}
elif args_cli.ik_method == "svd":
    ik_params = {
        "k_val": args_cli.ik_k_val,
        "min_singular_value": 1.0e-5,
    }
else:
    ik_params = {"k_val": args_cli.ik_k_val}

ik_by_arm = {
    arm: PiperIKController(
        robot,
        device=args_cli.device,
        ik_method=args_cli.ik_method,
        ik_params=ik_params,
        position_scale=args_cli.ik_position_scale,
        command_type=args_cli.command_type,
        delta_gain=args_cli.ik_delta_gain,
        rotate_jacobian_to_base=not args_cli.no_rotate_jacobian,
    )
    for arm, robot in robots_by_arm.items()
}
observe_ik_by_arm = ik_by_arm
if args_cli.observe_command_type != args_cli.command_type:
    observe_ik_by_arm = {
        arm: PiperIKController(
            robot,
            device=args_cli.device,
            ik_method=args_cli.ik_method,
            ik_params=ik_params,
            position_scale=args_cli.ik_position_scale,
            command_type=args_cli.observe_command_type,
            delta_gain=args_cli.ik_delta_gain,
            rotate_jacobian_to_base=not args_cli.no_rotate_jacobian,
        )
        for arm, robot in robots_by_arm.items()
    }

plates = dict(sorted(loader.plates.items()))
print(f"\n[1] Native assets wrapped: arms={sorted(robots_by_arm)}")
print(f"    plates available: {list(plates)}", flush=True)
print(f"    IK position scale: {args_cli.ik_position_scale}")
print(f"    IK command type: {args_cli.command_type}")
print(f"    observe IK command type: {args_cli.observe_command_type}")
print(f"    IK delta gain: {args_cli.ik_delta_gain}")
print(f"    IK method: {args_cli.ik_method} {ik_params}")
print(f"    rotate Jacobian to base: {not args_cli.no_rotate_jacobian}")
for arm, robot in robots_by_arm.items():
    ik = ik_by_arm[arm]
    print(f"    [{arm}] joints={robot.num_joints}")
    print(f"      ee body: {ik.EE_BODY_NAME} idx={ik.ee_body_idx}")
    print(f"      ee jacobian idx: {ik.ee_jacobi_idx} of {ik.num_jacobian_bodies}")
    print(f"      body names: {robot.body_names}")
    print(f"      arm joint ids: {ik.arm_joint_ids}")
    arm_limits = robot.data.soft_joint_pos_limits[0, ik.arm_joint_ids]
    for local_idx, joint_idx in enumerate(ik.arm_joint_ids):
        name = robot.joint_names[joint_idx]
        lo = float(arm_limits[local_idx, 0])
        hi = float(arm_limits[local_idx, 1])
        val = float(robot.data.joint_pos[0, joint_idx])
        print(f"      {name}: current={val:.3f}, soft_limit=[{lo:.3f}, {hi:.3f}]")

print("\n[2] Plate perception and planning inputs", flush=True)
selected_candidate = None
grasp_plan = None
if args_cli.perception == "rigid":
    arm_name = selected_arm_name()
    robot = robots_by_arm[arm_name]
    ik = ik_by_arm[arm_name]
    for item in get_plate_positions(plates):
        print_pose(f"    {item['name']}", item["pos"])

    plate = get_next_plate(plates, processed=set())
    if plate is None:
        print("[FAIL] No plate detected", flush=True)
        exit_with(1)
else:
    print("    source: dual wrist camera RGB-D + segmentation", flush=True)
    for camera_name in active_camera_names():
        arm_for_camera = camera_arm_name(camera_name)
        if arm_for_camera is None:
            continue
        camera_offsets[camera_name] = calibrate_camera_offset_from_body(
            camera_prims[camera_name],
            robots_by_arm[arm_for_camera],
        )
        offset = camera_offsets[camera_name]["body_pos_camera"]
        print(
            f"    {camera_name} offset from "
            f"{camera_offsets[camera_name]['body_path']}: "
            f"({offset[0]:.3f}, {offset[1]:.3f}, {offset[2]:.3f})",
            flush=True,
        )

    if args_cli.observe_mode == "active":
        observations = run_active_observation(
            sim,
            robots_by_arm,
            ik_by_arm,
            observe_ik_by_arm,
            plates,
            camera_sensors,
            camera_offsets,
            camera_prims,
            dt,
        )
    elif args_cli.observe_mode == "view":
        move_arms_to_view_pose(
            sim,
            robots_by_arm,
            ik_by_arm,
            plates,
            camera_sensors,
            camera_offsets,
            dt,
        )
        for _ in range(args_cli.camera_warmup_steps):
            step_sim(sim, robots_by_arm.values(), plates, dt, camera_sensors.values())
        observations, _qualities, _detection_counts = capture_camera_observations(
            robots_by_arm,
            camera_offsets,
            camera_prims,
            camera_sensors,
        )
    else:
        move_arms_to_observe_pose(sim, robots_by_arm, plates, camera_sensors, dt)
        for _ in range(args_cli.camera_warmup_steps):
            step_sim(sim, robots_by_arm.values(), plates, dt, camera_sensors.values())
        observations, _qualities, _detection_counts = capture_camera_observations(
            robots_by_arm,
            camera_offsets,
            camera_prims,
            camera_sensors,
        )

    if not observations:
        print("[FAIL] No plate detected from any camera", flush=True)
        exit_with(1)

    fused_plates = fuse_camera_observations(
        observations,
        merge_distance=args_cli.planning_merge_distance,
    )
    print(f"    fused plate hypotheses: {len(fused_plates)}", flush=True)
    for idx, fused in enumerate(fused_plates):
        print(
            f"    fused {idx}: name={fused.name}, "
            f"confidence={fused.confidence:.3f}, area={fused.total_area_px}, "
            f"cameras={list(fused.cameras)}, arms={list(fused.arms)}",
            flush=True,
        )
        print_pose("      fused estimate", fused.pos_w)

    candidate_arm_names = sorted(robots_by_arm) if args_cli.arm == "auto" else [args_cli.arm]
    arm_base_positions = {
        arm: arm_base_stage(arm)
        for arm in candidate_arm_names
        if arm in robots_by_arm
    }
    candidates = generate_bimanual_grasp_candidates(
        fused_plates,
        arm_base_positions=arm_base_positions,
        unit_scale=100.0,
        strategy=args_cli.grasp_strategy,
    )
    candidates = [
        score_candidate_with_ik(candidate, robots_by_arm[candidate.arm], ik_by_arm[candidate.arm])
        for candidate in candidates
    ]
    candidates.sort(key=lambda item: item.score, reverse=True)
    selected_candidate = choose_best_candidate(candidates)
    if selected_candidate is None:
        print("[FAIL] Planner did not produce any grasp candidate", flush=True)
        exit_with(1)

    print("    ranked grasp candidates:", flush=True)
    for idx, candidate in enumerate(candidates[: max(1, args_cli.candidate_log_count)]):
        pre = candidate.pre_grasp_pos
        terms = ", ".join(
            f"{name}={value:+.2f}"
            for name, value in sorted(candidate.score_terms.items())
        )
        print(
            f"      {idx}: arm={candidate.arm}, plate={candidate.plate.name}, "
            f"candidate={candidate.candidate_name}, score={candidate.score:+.3f}, "
            f"pre=({pre[0]:.2f}, {pre[1]:.2f}, {pre[2]:.2f})",
            flush=True,
        )
        print(f"         terms: {terms}", flush=True)

    arm_name = selected_candidate.arm
    robot = robots_by_arm[arm_name]
    ik = ik_by_arm[arm_name]
    plate = {
        "name": selected_candidate.plate.name,
        "pos": selected_candidate.plate.pos_w,
        "quat": selected_candidate.plate.quat_w,
        "source": "camera_fused",
        "area_px": selected_candidate.plate.total_area_px,
    }
    grasp_plan = selected_candidate.grasp_plan
    if args_cli.demo_view == "camera_then_scene":
        set_viewport_camera(loader.paths.scene_camera)

print(f"\n[3] Selected plate: {plate['name']}", flush=True)
print(f"    selected execution arm: {arm_name}", flush=True)
if selected_candidate is not None:
    metadata = grasp_plan.get("metadata", {})
    print(
        "    planner selection: "
        f"candidate={selected_candidate.candidate_name}, "
        f"score={selected_candidate.score:+.3f}, "
        f"cameras={list(selected_candidate.plate.cameras)}",
        flush=True,
    )
    radial = metadata.get("radial_xy")
    rim_xy = metadata.get("rim_xy")
    if radial is not None and rim_xy is not None:
        print(
            "    grasp strategy: "
            f"{metadata.get('strategy', selected_candidate.candidate_name)} "
            f"radial=({radial[0]:.3f}, {radial[1]:.3f}) "
            f"rim_xy=({rim_xy[0]:.3f}, {rim_xy[1]:.3f})",
            flush=True,
        )
elif args_cli.grasp_strategy in ("auto", "rim"):
    grasp_plan = generate_rim_grasp_pose(
        plate["pos"],
        plate["quat"],
        arm_base_pos=selected_arm_base_stage(),
        unit_scale=100.0,
    )
    metadata = grasp_plan.get("metadata", {})
    if metadata:
        radial = metadata.get("radial_xy")
        rim_xy = metadata.get("rim_xy")
        print(
            "    grasp strategy: rim "
            f"radial=({radial[0]:.3f}, {radial[1]:.3f}) "
            f"rim_xy=({rim_xy[0]:.3f}, {rim_xy[1]:.3f})",
            flush=True,
        )
else:
    grasp_plan = generate_grasp_pose(plate["pos"], plate["quat"], unit_scale=100.0)
    print("    grasp strategy: center", flush=True)
target_pos, target_quat = grasp_plan["pre_grasp"]
target_pos = target_pos.copy()
target_pos[2] += args_cli.ee_z_offset
current_ee_pos_w, current_ee_quat_w = ik.get_current_ee_pose()
if args_cli.orientation == "current":
    target_quat = tensor_to_np(current_ee_quat_w[0])
print_pose("    current EE", tensor_to_np(current_ee_pos_w[0]))
print_pose("    pre_grasp target", target_pos)
print(f"    extra link6 Z offset: {args_cli.ee_z_offset:.3f}")
print(f"    orientation mode: {args_cli.orientation}")

target_pos_w = torch.tensor(
    target_pos, dtype=torch.float32, device=args_cli.device
).unsqueeze(0)
target_quat_w = torch.tensor(
    target_quat, dtype=torch.float32, device=args_cli.device
).unsqueeze(0)
target_pos_b, target_quat_b = subtract_frame_transforms(
    robot.data.root_pose_w[:, 0:3],
    robot.data.root_pose_w[:, 3:7],
    target_pos_w,
    target_quat_w,
)
print_pose("    target in root frame", tensor_to_np(target_pos_b[0]))
current_pos_b, _current_quat_b = subtract_frame_transforms(
    robot.data.root_pose_w[:, 0:3],
    robot.data.root_pose_w[:, 3:7],
    current_ee_pos_w,
    current_ee_quat_w,
)
print_pose("    current in root frame", tensor_to_np(current_pos_b[0]))

trajectory = None
if not args_cli.direct_target:
    trajectory = interpolate_waypoints(
        tensor_to_np(current_ee_pos_w[0]),
        tensor_to_np(current_ee_quat_w[0]),
        target_pos,
        target_quat,
        args_cli.steps,
    )

print(f"\n[4] Commanding pre_grasp for {args_cli.steps} steps", flush=True)
ok = True
joint_pos_des = None
max_clamp_delta = 0.0
for step_idx in range(args_cli.steps):
    if trajectory is None:
        command_pos_w = target_pos_w
        command_quat_w = target_quat_w
    else:
        waypoint_pos, waypoint_quat = trajectory[step_idx]
        command_pos_w = torch.tensor(
            waypoint_pos, dtype=torch.float32, device=args_cli.device
        ).unsqueeze(0)
        command_quat_w = torch.tensor(
            waypoint_quat, dtype=torch.float32, device=args_cli.device
        ).unsqueeze(0)

    root_pose = robot.data.root_pose_w
    command_pos_b, command_quat_b = subtract_frame_transforms(
        root_pose[:, 0:3],
        root_pose[:, 3:7],
        command_pos_w,
        command_quat_w,
    )
    joint_pos_des = ik.solve(command_pos_b, command_quat_b)
    finite = bool(torch.isfinite(joint_pos_des).all())
    ok = ok and finite
    if not finite:
        print(f"    [FAIL] non-finite IK target at step {step_idx}", flush=True)
        break

    if not args_cli.no_clamp_joints:
        joint_pos_des, clamp_delta = clamp_to_arm_limits(robot, ik, joint_pos_des)
        max_clamp_delta = max(max_clamp_delta, clamp_delta)

    set_arm_target(robot, ik, joint_pos_des)
    step_sim(sim, robots_by_arm.values(), plates, dt, camera_sensors.values())

    if args_cli.log_every > 0 and (
        step_idx == 0 or (step_idx + 1) % args_cli.log_every == 0
    ):
        ee_pos_w, _ee_quat_w = ik.get_current_ee_pose()
        final_pos_error = torch.norm(ee_pos_w - target_pos_w, dim=-1)
        command_pos_error = torch.norm(ee_pos_w - command_pos_w, dim=-1)
        print(
            f"    step {step_idx + 1:04d}: "
            f"final_error={float(final_pos_error[0]):.3f}, "
            f"command_error={float(command_pos_error[0]):.3f}, "
            f"max_clamp_delta={max_clamp_delta:.3f}",
            flush=True,
        )

ee_pos_w, _ee_quat_w = ik.get_current_ee_pose()
ee_pos = tensor_to_np(ee_pos_w[0])
error_stage = float(torch.norm(ee_pos_w - target_pos_w, dim=-1)[0])
error_m = error_stage * 0.01

print("\n[5] Final pre-grasp readback", flush=True)
print_pose("    current EE", ee_pos)
print_pose("    target EE ", target_pos)
print(f"    position error: {error_stage:.3f} stage units ({error_m:.4f} m)", flush=True)
if joint_pos_des is not None:
    print(
        "    final joint target: "
        + ", ".join(f"{float(v):.3f}" for v in joint_pos_des[0].detach().cpu()),
        flush=True,
    )
print(
    "    final joint pos: "
    + ", ".join(
        f"{float(robot.data.joint_pos[0, idx]):.3f}" for idx in ik.arm_joint_ids
    ),
    flush=True,
)
if not args_cli.no_clamp_joints:
    print(f"    max clamp delta: {max_clamp_delta:.3f}", flush=True)

passed = ok and error_stage <= args_cli.pos_tolerance
print(
    f"\n{'PASS' if passed else 'FAIL'}: "
    f"pre_grasp error {error_stage:.3f} <= tolerance {args_cli.pos_tolerance:.3f}",
    flush=True,
)

hold_gui(args_cli.hold_seconds)
exit_with(0 if passed else 1)
