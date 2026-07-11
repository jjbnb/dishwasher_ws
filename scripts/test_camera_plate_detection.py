#!/usr/bin/env python3
"""Detect native plates from camera segmentation + depth.

This script does not use rigid-body poses for detection. It uses ground truth
only at the end to print an evaluation error against camera-estimated centers.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Camera-based native plate detector")
parser.add_argument(
    "--assets",
    default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher"),
)
parser.add_argument(
    "--camera",
    choices=("scene", "left_wrist", "right_wrist"),
    default="scene",
)
parser.add_argument("--width", type=int, default=640)
parser.add_argument("--height", type=int, default=480)
parser.add_argument("--warmup-steps", type=int, default=30)
parser.add_argument("--settle-steps", type=int, default=20)
parser.add_argument("--move-steps", type=int, default=180)
parser.add_argument("--min-area", type=int, default=40)
parser.add_argument(
    "--arm-joint-target",
    type=float,
    nargs=6,
    default=None,
    help="Optional six-joint arm target to move a wrist camera before capture.",
)
parser.add_argument(
    "--pose-variant",
    choices=("transpose", "transpose_inv", "raw", "raw_inv", "robot"),
    default="transpose",
    help="USD matrix orientation convention used for camera extrinsics.",
)
parser.add_argument(
    "--semantic-filter",
    default="class:plate",
    help="Replicator semantic filter. Use 'class:*' to include scene labels.",
)
parser.add_argument(
    "--camera-focal-length",
    type=float,
    default=None,
    help="Optional transient USD Camera focalLength override for GUI/perception tests.",
)
parser.add_argument(
    "--out-dir",
    default="outputs/camera_plate_detection",
)
parser.add_argument(
    "--hold-seconds",
    type=float,
    default=0.0,
    help="Keep the GUI open after detection; useful when running without --headless.",
)
parser.add_argument("--skip-close", action="store_true")
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(enable_cameras=True)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.sensors.camera import Camera, CameraCfg
from PIL import Image
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation as R

from dishwasher.perception.camera_plate_detector import (
    detect_plates_from_camera,
    semantic_output_to_mask,
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


def stage_to_meters(pos):
    return np.asarray(pos, dtype=np.float64) * 0.01


def print_pose(label, pos):
    meters = stage_to_meters(pos)
    print(
        f"{label}: stage=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
        f"meters=({meters[0]:.3f}, {meters[1]:.3f}, {meters[2]:.3f})",
        flush=True,
    )


def camera_path(loader: NativeSceneLoader) -> str:
    if args_cli.camera == "scene":
        return loader.paths.scene_camera
    if args_cli.camera == "left_wrist":
        return loader.paths.left_piper_camera
    return loader.paths.right_piper_camera


def set_viewport_camera(camera_prim_path: str):
    """Point the active GUI viewport at the same native camera used by detection."""

    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_prim_path
            print(f"viewport camera: {camera_prim_path}", flush=True)
    except Exception as exc:
        print(f"[WARN] Could not set active viewport camera: {exc}", flush=True)


def configure_camera_model(camera_prim):
    camera_schema = UsdGeom.Camera(camera_prim)
    if args_cli.camera_focal_length is not None:
        camera_schema.GetFocalLengthAttr().Set(float(args_cli.camera_focal_length))

    focal = float(camera_schema.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera_schema.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera_schema.GetVerticalApertureAttr().Get())
    hfov = np.degrees(2.0 * np.arctan(horizontal_aperture / (2.0 * focal)))
    vfov = np.degrees(2.0 * np.arctan(vertical_aperture / (2.0 * focal)))
    print(
        f"camera model: focalLength={focal:.3f}, "
        f"hfov={hfov:.1f}deg, vfov={vfov:.1f}deg",
        flush=True,
    )


def hold_gui(seconds: float):
    if seconds <= 0.0:
        return
    print(f"Keeping GUI open for {seconds:.1f}s.", flush=True)
    deadline = time.time() + seconds
    while simulation_app.is_running() and time.time() < deadline:
        simulation_app.update()
        time.sleep(0.01)


def arm_joint_ids(robot):
    return [robot.joint_names.index(f"joint{idx}") for idx in range(1, 7)]


def step_sim(sim, camera, dt, robot=None):
    if robot is not None:
        robot.write_data_to_sim()
    sim.step()
    if robot is not None:
        robot.update(dt)
    camera.update(dt)
    simulation_app.update()


def move_robot_to_observe_target(sim, camera, robot, dt):
    if args_cli.arm_joint_target is None or robot is None:
        return None

    joint_ids = arm_joint_ids(robot)
    target = torch.tensor(
        args_cli.arm_joint_target,
        dtype=torch.float32,
        device=args_cli.device,
    ).unsqueeze(0)

    robot.set_joint_position_target(robot.data.default_joint_pos.clone())
    for _ in range(args_cli.settle_steps):
        step_sim(sim, camera, dt, robot)
    camera_offset = calibrate_camera_offset_from_body(camera_prim, robot)

    start = robot.data.joint_pos[:, joint_ids].clone()
    for step_idx in range(args_cli.move_steps):
        alpha = (step_idx + 1) / max(1, args_cli.move_steps)
        full_target = robot.data.joint_pos.clone()
        full_target[:, joint_ids] = start + alpha * (target - start)
        robot.set_joint_position_target(full_target)
        step_sim(sim, camera, dt, robot)

    full_target = robot.data.joint_pos.clone()
    full_target[:, joint_ids] = target
    robot.set_joint_position_target(full_target)
    for _ in range(args_cli.settle_steps):
        step_sim(sim, camera, dt, robot)

    actual = robot.data.joint_pos[0, joint_ids].detach().cpu().numpy()
    print("arm joint target: " + ", ".join(f"{v:.3f}" for v in args_cli.arm_joint_target))
    print("arm joint actual: " + ", ".join(f"{float(v):.3f}" for v in actual))
    return camera_offset


def _quat_wxyz_to_rotm(quat: np.ndarray) -> np.ndarray:
    return R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def _rotm_to_quat_wxyz(rotm: np.ndarray, device: str):
    quat_xyzw = R.from_matrix(rotm).as_quat()
    return torch.tensor(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=torch.float32,
        device=device,
    )


def _usd_camera_pose_opengl(camera_prim):
    transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    matrix = np.asarray(transform).T
    pos = np.asarray(transform.ExtractTranslation(), dtype=np.float64)
    rot = _orthonormalized_rot(matrix[:3, :3])
    return pos, rot


def calibrate_camera_offset_from_body(camera_prim, robot, body_name: str = "link6"):
    """Calibrate fixed body-to-camera offset from the native USD local chain."""

    camera_path = camera_prim.GetPath().pathString
    marker = f"/{body_name}/"
    if marker not in camera_path:
        raise RuntimeError(f"Cannot derive {body_name} path from camera path: {camera_path}")
    body_path = camera_path.split(marker, 1)[0] + f"/{body_name}"
    body_prim = camera_prim.GetStage().GetPrimAtPath(body_path)
    if not body_prim or not body_prim.IsValid():
        raise RuntimeError(f"Cannot find camera parent body prim: {body_path}")

    body_pos, body_rot = _usd_camera_pose_opengl(body_prim)
    camera_pos, camera_rot_usd = _usd_camera_pose_opengl(camera_prim)
    body_rot_camera_usd = body_rot.T @ camera_rot_usd
    body_pos_camera = body_rot.T @ (camera_pos - body_pos)
    print(
        "calibrated wrist camera offset from "
        f"{body_path}: pos=({body_pos_camera[0]:.3f}, "
        f"{body_pos_camera[1]:.3f}, {body_pos_camera[2]:.3f})",
        flush=True,
    )
    return {
        "body_name": body_name,
        "body_rot_camera_usd": body_rot_camera_usd,
        "body_pos_camera": body_pos_camera,
    }


def robot_camera_pose_variant(robot, camera_offset):
    body_idx = robot.body_names.index(camera_offset["body_name"])
    body_pos = robot.data.body_pos_w[0, body_idx].detach().cpu().numpy().astype(np.float64)
    body_quat = robot.data.body_quat_w[0, body_idx].detach().cpu().numpy().astype(np.float64)
    body_rot = _quat_wxyz_to_rotm(body_quat)

    camera_rot_usd = body_rot @ camera_offset["body_rot_camera_usd"]
    camera_pos = body_pos + body_rot @ camera_offset["body_pos_camera"]
    camera_rot_ros = _usd_rot_to_ros(camera_rot_usd)
    camera_pos_t = torch.tensor(camera_pos, dtype=torch.float32, device=args_cli.device)
    camera_quat_ros = _rotm_to_quat_wxyz(camera_rot_ros, args_cli.device)
    return ("robot", camera_pos_t, camera_quat_ros, camera_rot_ros)


def add_plate_semantics(stage, loader: NativeSceneLoader):
    from isaaclab.sim.utils import add_labels

    labeled = []
    for name, root_path in loader.paths.plate_roots.items():
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Gprim):
                add_labels(prim, ["plate"], instance_name="class", overwrite=True)
                add_labels(prim, [name], instance_name="instance", overwrite=True)
                labeled.append(prim.GetPath().pathString)
    return labeled


def add_scene_semantics(stage, loader: NativeSceneLoader):
    from isaaclab.sim.utils import add_labels

    labels = {
        "table": "/World/dishwasher_desk_1_",
        "sink": loader.paths.sink_mesh,
        "rack": loader.paths.rack_mesh,
    }
    labeled: dict[str, list[str]] = {name: [] for name in labels}
    for label, root_path in labels.items():
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Gprim):
                add_labels(prim, [label], instance_name="class", overwrite=True)
                add_labels(prim, [label], instance_name="instance", overwrite=True)
                labeled[label].append(prim.GetPath().pathString)
    return labeled


def get_plate_root_positions(stage, loader: NativeSceneLoader):
    """Read native plate root transforms for offline evaluation only."""

    result = []
    for name, root_path in sorted(loader.paths.plate_roots.items()):
        prim = stage.GetPrimAtPath(root_path)
        if not prim or not prim.IsValid():
            continue
        transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        )
        result.append((name, np.asarray(transform.ExtractTranslation(), dtype=np.float64)))
    return result


def semantic_id_to_labels(info):
    if not info:
        return {}
    for key in ("idToLabels", "idToSemantics", "idToSemantic"):
        mapping = info.get(key)
        if isinstance(mapping, dict) and mapping:
            return mapping
    return {}


def semantic_label_text(labels) -> str:
    if isinstance(labels, dict):
        return " ".join(str(value) for value in labels.values())
    if isinstance(labels, (list, tuple, set)):
        return " ".join(str(value) for value in labels)
    return str(labels)


def semantic_label_color(label_text: str) -> tuple[int, int, int]:
    text = label_text.lower()
    if "plate" in text:
        return (255, 64, 64)
    if "sink" in text:
        return (64, 160, 255)
    if "rack" in text:
        return (64, 220, 128)
    if "table" in text or "desk" in text:
        return (196, 128, 64)
    return (192, 96, 255)


def save_debug_images(out_dir: Path, rgb, depth, semantic, mask, instance=None, semantic_info=None):
    out_dir.mkdir(parents=True, exist_ok=True)

    rgb_np = rgb.detach().cpu().numpy()
    if rgb_np.dtype != np.uint8:
        rgb_np = np.clip(rgb_np, 0, 255).astype(np.uint8)
    Image.fromarray(rgb_np[..., :3]).save(out_dir / "rgb.png")

    depth_np = depth.detach().cpu().numpy()
    if depth_np.ndim == 3:
        depth_np = depth_np[..., 0]
    finite = np.isfinite(depth_np)
    depth_vis = np.zeros_like(depth_np, dtype=np.uint8)
    if finite.any():
        lo, hi = np.percentile(depth_np[finite], [2.0, 98.0])
        denom = max(float(hi - lo), 1.0e-6)
        depth_vis = np.clip((depth_np - lo) / denom * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(depth_vis).save(out_dir / "depth.png")

    sem_np = semantic.detach().cpu().numpy()
    if sem_np.ndim == 3 and sem_np.shape[-1] == 1:
        sem_np = sem_np[..., 0]
    if sem_np.ndim == 2:
        sem_vis = np.zeros((*sem_np.shape, 3), dtype=np.uint8)
        sem_vis[sem_np != 0] = (255, 64, 64)
    else:
        sem_vis = sem_np[..., :3].astype(np.uint8)
    Image.fromarray(sem_vis).save(out_dir / "semantic.png")

    label_map = semantic_id_to_labels(semantic_info)
    if sem_np.ndim == 2 and label_map:
        scene_vis = np.zeros((*sem_np.shape, 3), dtype=np.uint8)
        for raw_id, labels in label_map.items():
            try:
                semantic_id = int(raw_id)
            except ValueError:
                continue
            if semantic_id == 0:
                continue
            scene_vis[sem_np == semantic_id] = semantic_label_color(semantic_label_text(labels))
        Image.fromarray(scene_vis).save(out_dir / "semantic_scene.png")

    Image.fromarray((mask.astype(np.uint8) * 255)).save(out_dir / "plate_mask.png")

    if instance is not None:
        inst_np = instance.detach().cpu().numpy()
        if inst_np.ndim == 3 and inst_np.shape[-1] == 1:
            inst_np = inst_np[..., 0]
        if inst_np.ndim == 2:
            inst_vis = np.zeros((*inst_np.shape, 3), dtype=np.uint8)
            ids = [item for item in np.unique(inst_np) if int(item) != 0]
            palette = np.array(
                [
                    (255, 64, 64),
                    (64, 160, 255),
                    (64, 220, 128),
                    (255, 196, 64),
                    (192, 96, 255),
                ],
                dtype=np.uint8,
            )
            for color_idx, instance_id in enumerate(ids):
                inst_vis[inst_np == instance_id] = palette[color_idx % len(palette)]
        else:
            inst_vis = inst_np[..., :3].astype(np.uint8)
        Image.fromarray(inst_vis).save(out_dir / "instance.png")


def get_usd_camera_pose_ros(camera_prim, device: str):
    """Return camera world pose from USD as position, ROS quaternion, and matrix."""

    transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    matrix = np.asarray(transform).T
    pos_np = np.asarray(transform.ExtractTranslation(), dtype=np.float64)
    rot_ros = _usd_rot_to_ros(_orthonormalized_rot(matrix[:3, :3]))
    quat_xyzw = R.from_matrix(rot_ros).as_quat()
    pos_w = torch.tensor(pos_np, dtype=torch.float32, device=device)
    quat_ros = torch.tensor(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=torch.float32,
        device=device,
    )
    return pos_w, quat_ros, rot_ros


def get_usd_camera_pose_ros_variants(camera_prim, device: str):
    """Return row/column-major orientation variants for diagnostics."""

    transform = UsdGeom.Xformable(camera_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    )
    arr = np.asarray(transform)
    pos_np = np.asarray(transform.ExtractTranslation(), dtype=np.float64)
    variants = []
    for name, matrix in (("raw", arr), ("transpose", arr.T)):
        rot_usd = _orthonormalized_rot(matrix[:3, :3])
        for suffix, rot_candidate in (
            ("", rot_usd),
            ("_inv", rot_usd.T),
        ):
            rot_ros = _usd_rot_to_ros(rot_candidate)
            quat_xyzw = R.from_matrix(rot_ros).as_quat()
            quat_ros = torch.tensor(
                [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
                dtype=torch.float32,
                device=device,
            )
            variants.append(
                (
                    f"{name}{suffix}",
                    torch.tensor(pos_np, dtype=torch.float32, device=device),
                    quat_ros,
                    rot_ros,
                )
            )
    return variants


def _orthonormalized_rot(rot: np.ndarray) -> np.ndarray:
    """Remove small scale/shear terms from a USD xform rotation block."""

    u, _, vh = np.linalg.svd(np.asarray(rot, dtype=np.float64))
    result = u @ vh
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1.0
        result = u @ vh
    return result


def _usd_rot_to_ros(rot_usd: np.ndarray) -> np.ndarray:
    """Convert camera-to-world USD/OpenGL rotation to ROS optical frame."""

    usd_from_ros = np.diag([1.0, -1.0, -1.0])
    return np.asarray(rot_usd, dtype=np.float64) @ usd_from_ros


def run_detection_for_pose(depth, semantic, instance, semantic_info, instance_info, pose_variant):
    _, camera_pos_w, camera_quat_w_ros, camera_rotm_w_ros = pose_variant
    return detect_plates_from_camera(
        depth=depth,
        semantic=semantic,
        intrinsic_matrix=camera.data.intrinsic_matrices[0],
        camera_pos_w=camera_pos_w,
        camera_quat_w_ros=camera_quat_w_ros,
        camera_rotm_w_ros=camera_rotm_w_ros,
        instance=instance,
        semantic_info=semantic_info,
        instance_info=instance_info,
        min_area_px=args_cli.min_area,
        target_label="plate",
        device=args_cli.device,
    )


def nearest_truth_error(detection, truth_positions):
    if not truth_positions:
        return None
    errors = [
        (name, float(np.linalg.norm(detection.pos_w - np.asarray(pos))))
        for name, pos in truth_positions
    ]
    return min(errors, key=lambda item: item[1])


print("=" * 72)
print("Native camera plate detection")
print("=" * 72)

loader = NativeSceneLoader(args_cli.assets)
report = loader.validate()
if not report.ok:
    print("[FAIL] Native all.usd validation failed.", flush=True)
    exit_with(1)

stage = loader.open_stage()
changed = loader.prepare_for_direct_isaaclab(
    stage,
    deactivate_nested_physics_scenes=True,
    deactivate_action_graphs=True,
)
print(f"runtime disabled nested PhysicsScene: {changed['nested_physics_scenes']}")
print(f"runtime disabled ActionGraph: {changed['action_graphs']}")

labeled = add_plate_semantics(stage, loader)
print(f"runtime plate semantic labels: {len(labeled)} prims")
scene_labeled = add_scene_semantics(stage, loader)
print(
    "runtime scene semantic labels: "
    + ", ".join(f"{name}={len(paths)}" for name, paths in scene_labeled.items())
)

cam_path = camera_path(loader)
print(f"camera: {args_cli.camera} {cam_path}")
camera_prim = stage.GetPrimAtPath(cam_path)
if not camera_prim or not camera_prim.IsValid():
    print(f"[FAIL] Camera prim not found: {cam_path}", flush=True)
    exit_with(1)
sim_utils.standardize_xform_ops(camera_prim)
configure_camera_model(camera_prim)
set_viewport_camera(cam_path)

sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
camera = Camera(
    CameraCfg(
        prim_path=cam_path,
        update_period=0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=[
            "rgb",
            "distance_to_image_plane",
            "semantic_segmentation",
            "instance_segmentation_fast",
        ],
        colorize_semantic_segmentation=False,
        colorize_instance_segmentation=False,
        semantic_filter=args_cli.semantic_filter,
        spawn=None,
        update_latest_camera_pose=False,
    )
)
robot = None
if args_cli.arm_joint_target is not None:
    if args_cli.camera == "scene":
        print("[WARN] --arm-joint-target was provided for scene camera; ignoring arm motion.")
    else:
        loader.wrap_assets(
            wrap_left=args_cli.camera == "left_wrist",
            wrap_right=args_cli.camera == "right_wrist",
        )
        robot = loader.piper_l if args_cli.camera == "left_wrist" else loader.piper_r
sim.reset()
dt = sim.get_physics_dt()

robot_camera_offset = move_robot_to_observe_target(sim, camera, robot, dt)

for _ in range(args_cli.warmup_steps):
    step_sim(sim, camera, dt, robot)

rgb = camera.data.output["rgb"][0]
depth = camera.data.output["distance_to_image_plane"][0]
semantic = camera.data.output["semantic_segmentation"][0]
instance = camera.data.output["instance_segmentation_fast"][0]
semantic_info = camera.data.info[0].get("semantic_segmentation")
instance_info = camera.data.info[0].get("instance_segmentation_fast")
mask = semantic_output_to_mask(semantic, semantic_info, target_label="plate")
depth_np = depth.detach().cpu().numpy()
if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
    depth_np = depth_np[..., 0]
depth_in_mask = depth_np[mask]
finite_depth = np.isfinite(depth_in_mask)
print(f"semantic output shape: {tuple(semantic.shape)}")
unique_sem = np.unique(semantic.detach().cpu().numpy())
print(f"semantic unique ids/colors sample: {unique_sem[:12].tolist()} total={len(unique_sem)}")
print(f"semantic filter: {args_cli.semantic_filter}")
print(f"semantic labels: {semantic_id_to_labels(semantic_info)}")
unique_inst = np.unique(instance.detach().cpu().numpy())
print(f"instance unique ids/colors sample: {unique_inst[:12].tolist()} total={len(unique_inst)}")
if depth_in_mask.size:
    finite_count = int(finite_depth.sum())
    print(f"depth under mask: finite={finite_count}/{depth_in_mask.size}")
    if finite_count:
        print(
            f"depth under mask range: "
            f"{float(depth_in_mask[finite_depth].min()):.3f} .. "
            f"{float(depth_in_mask[finite_depth].max()):.3f}"
        )
pose_variants = {
    name: (name, pos, quat, rotm)
    for name, pos, quat, rotm in get_usd_camera_pose_ros_variants(camera_prim, args_cli.device)
}
if robot is not None and robot_camera_offset is not None:
    robot_pose_variant = robot_camera_pose_variant(robot, robot_camera_offset)
    pose_variants[robot_pose_variant[0]] = robot_pose_variant

selected_pose_name = args_cli.pose_variant
if selected_pose_name == "robot" and "robot" not in pose_variants:
    print("[FAIL] pose variant 'robot' requires --arm-joint-target on a wrist camera.", flush=True)
    hold_gui(args_cli.hold_seconds)
    exit_with(1)
selected_pose = pose_variants[selected_pose_name]
_, camera_pos_w, camera_quat_w_ros, camera_rotm_w_ros = selected_pose
print(f"camera pos_w: {camera_pos_w.detach().cpu().numpy().tolist()}")
print(f"camera quat_w_ros: {camera_quat_w_ros.detach().cpu().numpy().tolist()}")
print(f"camera rotm_w_ros: {camera_rotm_w_ros.tolist()}")
print(f"camera intrinsic K: {camera.data.intrinsic_matrices[0].detach().cpu().numpy().tolist()}")
print(f"camera pose variant: {selected_pose_name}")
detections = run_detection_for_pose(
    depth,
    semantic,
    instance,
    semantic_info,
    instance_info,
    selected_pose,
)

out_dir = Path(args_cli.out_dir)
save_debug_images(out_dir, rgb, depth, semantic, mask, instance, semantic_info)
print(f"saved debug images: {out_dir.resolve()}")
print(f"mask pixels: {int(mask.sum())}")
print(f"detections: {len(detections)}")

truth_positions = get_plate_root_positions(stage, loader)
if truth_positions:
    print("\nplate root references for offline evaluation:")
    for name, pos in truth_positions:
        print_pose(f"  {name}", pos)
if truth_positions:
    print("\npose variant diagnostics:")
    for variant in pose_variants.values():
        variant_detections = run_detection_for_pose(
            depth,
            semantic,
            instance,
            semantic_info,
            instance_info,
            variant,
        )
        if not variant_detections:
            print(f"  {variant[0]}: no detections")
            continue
        nearest = nearest_truth_error(variant_detections[0], truth_positions)
        if nearest is None:
            print(f"  {variant[0]}: detections={len(variant_detections)}")
            continue
        name, err = nearest
        estimate = variant_detections[0].pos_w
        print(
            f"  {variant[0]}: nearest={name}, "
            f"error={err:.3f} stage units ({err * 0.01:.4f} m), "
            f"estimate=({estimate[0]:.3f}, {estimate[1]:.3f}, {estimate[2]:.3f})"
        )

for idx, det in enumerate(detections):
    print(
        f"\nDetection {idx}: area={det.area_px}, "
        f"uv=({det.centroid_uv[0]:.1f}, {det.centroid_uv[1]:.1f}), "
        f"bbox={det.bbox_xyxy}",
        flush=True,
    )
    print_pose("  camera estimate", det.pos_w)
    nearest = nearest_truth_error(det, truth_positions)
    if nearest is not None:
        name, err = nearest
        print(f"  nearest root reference: {name}, error={err:.3f} stage units ({err * 0.01:.4f} m)")

if not detections:
    print("[FAIL] No plate pixels were detected from camera output.", flush=True)
    hold_gui(args_cli.hold_seconds)
    exit_with(1)

hold_gui(args_cli.hold_seconds)
exit_with(0)
