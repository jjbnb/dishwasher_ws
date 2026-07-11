"""Native all.usd camera perception helpers.

The functions in this module read Isaac Sim camera outputs. They do not read
plate rigid-body poses for detection. Rigid-body poses may still be used by
callers for offline evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from pxr import Usd, UsdGeom
from scipy.spatial.transform import Rotation as R

from dishwasher.perception.camera_plate_detector import (
    PlateDetection,
    detect_plates_from_camera,
    semantic_output_to_mask,
)
from dishwasher.scene.native_loader import NativeSceneLoader


@dataclass(frozen=True)
class CameraPoseVariant:
    """Camera pose in world frame using ROS optical convention."""

    name: str
    pos_w: torch.Tensor
    quat_w_ros: torch.Tensor
    rotm_w_ros: np.ndarray


@dataclass(frozen=True)
class CameraFrameDetections:
    """A camera frame and plate detections derived from it."""

    rgb: torch.Tensor
    depth: torch.Tensor
    semantic: torch.Tensor
    instance: torch.Tensor
    semantic_info: dict[str, Any] | None
    instance_info: dict[str, Any] | None
    plate_mask: np.ndarray
    detections: list[PlateDetection]


def camera_path(loader: NativeSceneLoader, camera: str) -> str:
    """Return a stable native camera prim path."""

    if camera == "scene":
        return loader.paths.scene_camera
    if camera == "left_wrist":
        return loader.paths.left_piper_camera
    if camera == "right_wrist":
        return loader.paths.right_piper_camera
    raise ValueError(f"Unknown camera: {camera}")


def add_runtime_semantics(stage, loader: NativeSceneLoader, *, include_scene: bool = True):
    """Add transient class/instance labels for camera segmentation."""

    from isaaclab.sim.utils import add_labels

    labeled: dict[str, list[str]] = {"plate": []}
    for name, root_path in loader.paths.plate_roots.items():
        root = stage.GetPrimAtPath(root_path)
        if not root or not root.IsValid():
            continue
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Gprim):
                add_labels(prim, ["plate"], instance_name="class", overwrite=True)
                add_labels(prim, [name], instance_name="instance", overwrite=True)
                labeled["plate"].append(prim.GetPath().pathString)

    if include_scene:
        scene_labels = {
            "table": "/World/dishwasher_desk_1_",
            "sink": loader.paths.sink_mesh,
            "rack": loader.paths.rack_mesh,
        }
        for label, root_path in scene_labels.items():
            labeled[label] = []
            root = stage.GetPrimAtPath(root_path)
            if not root or not root.IsValid():
                continue
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdGeom.Gprim):
                    add_labels(prim, [label], instance_name="class", overwrite=True)
                    add_labels(prim, [label], instance_name="instance", overwrite=True)
                    labeled[label].append(prim.GetPath().pathString)

    return labeled


def configure_camera_model(camera_prim, *, focal_length: float | None = None):
    """Optionally override focal length and report the resulting field of view."""

    camera_schema = UsdGeom.Camera(camera_prim)
    if focal_length is not None:
        camera_schema.GetFocalLengthAttr().Set(float(focal_length))

    focal = float(camera_schema.GetFocalLengthAttr().Get())
    horizontal_aperture = float(camera_schema.GetHorizontalApertureAttr().Get())
    vertical_aperture = float(camera_schema.GetVerticalApertureAttr().Get())
    hfov = np.degrees(2.0 * np.arctan(horizontal_aperture / (2.0 * focal)))
    vfov = np.degrees(2.0 * np.arctan(vertical_aperture / (2.0 * focal)))
    return {
        "focal_length": focal,
        "horizontal_fov_deg": float(hfov),
        "vertical_fov_deg": float(vfov),
    }


def create_camera_sensor(
    *,
    prim_path: str,
    width: int,
    height: int,
    semantic_filter: str = "class:*",
):
    """Create an Isaac Lab camera sensor for RGB-D and segmentation."""

    from isaaclab.sensors.camera import Camera, CameraCfg

    return Camera(
        CameraCfg(
            prim_path=prim_path,
            update_period=0,
            height=height,
            width=width,
            data_types=[
                "rgb",
                "distance_to_image_plane",
                "semantic_segmentation",
                "instance_segmentation_fast",
            ],
            colorize_semantic_segmentation=False,
            colorize_instance_segmentation=False,
            semantic_filter=semantic_filter,
            spawn=None,
            update_latest_camera_pose=False,
        )
    )


def calibrate_camera_offset_from_body(camera_prim, robot, body_name: str = "link6"):
    """Calibrate fixed body-to-camera offset from the USD local chain."""

    camera_prim_path = camera_prim.GetPath().pathString
    marker = f"/{body_name}/"
    if marker not in camera_prim_path:
        raise RuntimeError(f"Cannot derive {body_name} path from camera path: {camera_prim_path}")

    body_path = camera_prim_path.split(marker, 1)[0] + f"/{body_name}"
    body_prim = camera_prim.GetStage().GetPrimAtPath(body_path)
    if not body_prim or not body_prim.IsValid():
        raise RuntimeError(f"Cannot find camera parent body prim: {body_path}")

    body_pos, body_rot = _usd_camera_pose_opengl(body_prim)
    camera_pos, camera_rot_usd = _usd_camera_pose_opengl(camera_prim)
    return {
        "body_name": body_name,
        "body_path": body_path,
        "body_rot_camera_usd": body_rot.T @ camera_rot_usd,
        "body_pos_camera": body_rot.T @ (camera_pos - body_pos),
    }


def robot_camera_pose_variant(robot, camera_offset, *, device: str) -> CameraPoseVariant:
    """Compute wrist camera pose from live robot body pose."""

    body_idx = robot.body_names.index(camera_offset["body_name"])
    body_pos = robot.data.body_pos_w[0, body_idx].detach().cpu().numpy().astype(np.float64)
    body_quat = robot.data.body_quat_w[0, body_idx].detach().cpu().numpy().astype(np.float64)
    body_rot = _quat_wxyz_to_rotm(body_quat)

    camera_rot_usd = body_rot @ camera_offset["body_rot_camera_usd"]
    camera_pos = body_pos + body_rot @ camera_offset["body_pos_camera"]
    camera_rot_ros = _usd_rot_to_ros(camera_rot_usd)
    camera_pos_t = torch.tensor(camera_pos, dtype=torch.float32, device=device)
    camera_quat_ros = _rotm_to_quat_wxyz(camera_rot_ros, device)
    return CameraPoseVariant("robot", camera_pos_t, camera_quat_ros, camera_rot_ros)


def usd_camera_pose_variant(camera_prim, *, device: str) -> CameraPoseVariant:
    """Compute a static USD camera pose in ROS optical convention."""

    camera_pos, camera_rot_usd = _usd_camera_pose_opengl(camera_prim)
    camera_rot_ros = _usd_rot_to_ros(camera_rot_usd)
    camera_pos_t = torch.tensor(camera_pos, dtype=torch.float32, device=device)
    camera_quat_ros = _rotm_to_quat_wxyz(camera_rot_ros, device)
    return CameraPoseVariant("usd", camera_pos_t, camera_quat_ros, camera_rot_ros)


def detect_camera_plates(
    camera_sensor,
    pose: CameraPoseVariant,
    *,
    device: str,
    min_area_px: int = 40,
    target_label: str = "plate",
) -> CameraFrameDetections:
    """Run plate detection on the latest camera frame."""

    rgb = camera_sensor.data.output["rgb"][0]
    depth = camera_sensor.data.output["distance_to_image_plane"][0]
    semantic = camera_sensor.data.output["semantic_segmentation"][0]
    instance = camera_sensor.data.output["instance_segmentation_fast"][0]
    semantic_info = camera_sensor.data.info[0].get("semantic_segmentation")
    instance_info = camera_sensor.data.info[0].get("instance_segmentation_fast")
    mask = semantic_output_to_mask(semantic, semantic_info, target_label=target_label)
    detections = detect_plates_from_camera(
        depth=depth,
        semantic=semantic,
        intrinsic_matrix=camera_sensor.data.intrinsic_matrices[0],
        camera_pos_w=pose.pos_w,
        camera_quat_w_ros=pose.quat_w_ros,
        camera_rotm_w_ros=pose.rotm_w_ros,
        instance=instance,
        semantic_info=semantic_info,
        instance_info=instance_info,
        min_area_px=min_area_px,
        target_label=target_label,
        device=device,
    )
    return CameraFrameDetections(
        rgb=rgb,
        depth=depth,
        semantic=semantic,
        instance=instance,
        semantic_info=semantic_info,
        instance_info=instance_info,
        plate_mask=mask,
        detections=detections,
    )


def detection_to_plate_record(detection: PlateDetection, *, name: str | None = None) -> dict:
    """Convert a camera detection to the legacy plate-record shape."""

    label = name if name is not None else detection.label
    return {
        "name": label,
        "pos": np.asarray(detection.pos_w, dtype=np.float64),
        "quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        "source": "camera",
        "area_px": detection.area_px,
        "centroid_uv": detection.centroid_uv,
        "bbox_xyxy": detection.bbox_xyxy,
    }


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


def _orthonormalized_rot(rot: np.ndarray) -> np.ndarray:
    u, _, vh = np.linalg.svd(np.asarray(rot, dtype=np.float64))
    result = u @ vh
    if np.linalg.det(result) < 0:
        u[:, -1] *= -1.0
        result = u @ vh
    return result


def _usd_rot_to_ros(rot_usd: np.ndarray) -> np.ndarray:
    usd_from_ros = np.diag([1.0, -1.0, -1.0])
    return np.asarray(rot_usd, dtype=np.float64) @ usd_from_ros
