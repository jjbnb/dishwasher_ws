"""Camera-based plate detection helpers.

This module intentionally consumes camera outputs (segmentation + depth) rather
than Isaac rigid-body poses. Ground-truth poses may still be used by scripts for
offline evaluation, but not by the detector itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from scipy import ndimage


@dataclass
class PlateDetection:
    """A single plate candidate estimated from camera pixels."""

    label: str
    area_px: int
    centroid_uv: tuple[float, float]
    pos_w: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]


def semantic_output_to_mask(
    semantic: torch.Tensor | np.ndarray,
    info: dict[str, Any] | None = None,
    *,
    target_label: str = "plate",
) -> np.ndarray:
    """Convert a semantic segmentation output into a binary mask.

    If the camera was configured with ``semantic_filter="class:plate"`` then
    every non-zero semantic id should correspond to plate pixels. When an id to
    label mapping is available, this function uses it to select only labels that
    contain ``target_label``.
    """

    if isinstance(semantic, torch.Tensor):
        sem = semantic.detach().cpu().numpy()
    else:
        sem = np.asarray(semantic)

    if sem.ndim == 3 and sem.shape[-1] == 1:
        sem = sem[..., 0]
    if sem.ndim == 3 and sem.shape[-1] in (3, 4):
        # Colorized segmentation fallback: background is usually black/alpha 0.
        return np.any(sem[..., :3] != 0, axis=-1)

    label_ids = _semantic_label_ids(info, target_label)
    if label_ids:
        return np.isin(sem, list(label_ids))
    if _has_semantic_mapping(info):
        return np.zeros(sem.shape, dtype=bool)
    return sem != 0


def detect_plates_from_camera(
    *,
    depth: torch.Tensor,
    semantic: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w_ros: torch.Tensor,
    camera_rotm_w_ros: np.ndarray | torch.Tensor | None = None,
    instance: torch.Tensor | np.ndarray | None = None,
    semantic_info: dict[str, Any] | None = None,
    instance_info: dict[str, Any] | None = None,
    min_area_px: int = 40,
    target_label: str = "plate",
    device: str | torch.device | None = None,
) -> list[PlateDetection]:
    """Estimate plate centers from semantic mask and depth.

    Args:
        depth: ``(H, W)`` or ``(H, W, 1)`` distance-to-image-plane image.
        semantic: semantic segmentation image for the same camera.
        intrinsic_matrix: camera intrinsics ``K``.
        camera_pos_w: camera world position.
        camera_quat_w_ros: camera world orientation in ROS convention. Kept as a
            fallback for callers that do not have a rotation matrix.
        camera_rotm_w_ros: camera-to-world rotation matrix in ROS optical frame
            convention (``+Z`` forward, ``+X`` right, ``+Y`` down).
        instance: optional instance segmentation image. When present, plate
            pixels are split by instance id before 3D localization.
        semantic_info: optional mapping returned by Replicator.
        instance_info: optional instance mapping returned by Replicator.
        min_area_px: minimum connected-component area.
        target_label: semantic label substring to select.
        device: torch device for point-cloud projection.

    Returns:
        Plate candidates sorted by descending mask area.
    """

    mask = semantic_output_to_mask(
        semantic,
        semantic_info,
        target_label=target_label,
    )
    components = _plate_components(
        mask,
        instance=instance,
        instance_info=instance_info,
        min_area_px=min_area_px,
        target_label=target_label,
    )
    if not components:
        return []

    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth_image = depth[..., 0]
    else:
        depth_image = depth

    detections: list[PlateDetection] = []
    for component_label, component_mask in components:
        area = int(component_mask.sum())
        ys, xs = np.nonzero(component_mask)
        pts = _pixels_to_world(
            xs=xs,
            ys=ys,
            depth=depth_image,
            intrinsic_matrix=intrinsic_matrix,
            camera_pos_w=camera_pos_w,
            camera_quat_w_ros=camera_quat_w_ros,
            camera_rotm_w_ros=camera_rotm_w_ros,
            device=device,
        )
        valid = np.all(np.isfinite(pts), axis=-1)
        pts = pts[valid]
        if pts.size == 0:
            continue

        center = np.median(pts, axis=0)
        centroid_u = float(xs.mean())
        centroid_v = float(ys.mean())
        x_min = int(xs.min())
        x_max = int(xs.max()) + 1
        y_min = int(ys.min())
        y_max = int(ys.max()) + 1
        detections.append(
            PlateDetection(
                label=component_label,
                area_px=area,
                centroid_uv=(centroid_u, centroid_v),
                pos_w=center,
                bbox_xyxy=(
                    x_min,
                    y_min,
                    x_max,
                    y_max,
                ),
            )
        )

    detections.sort(key=lambda item: item.area_px, reverse=True)
    return detections


def _plate_components(
    mask: np.ndarray,
    *,
    instance: torch.Tensor | np.ndarray | None,
    instance_info: dict[str, Any] | None,
    min_area_px: int,
    target_label: str,
) -> list[tuple[str, np.ndarray]]:
    """Split plate pixels into instance masks when possible."""

    if instance is not None:
        instance_ids = _segmentation_ids(instance)
        instance_ids = np.where(mask, instance_ids, 0)
        components: list[tuple[str, np.ndarray]] = []
        for instance_id in sorted(int(item) for item in np.unique(instance_ids) if int(item) != 0):
            component = instance_ids == instance_id
            if int(component.sum()) < min_area_px:
                continue
            label = _instance_label(instance_id, instance_info) or f"{target_label}:{instance_id}"
            components.append((label, component))
        if len(components) > 1:
            return components

    return [
        (f"{target_label}:{idx}", component)
        for idx, component in enumerate(_connected_components(mask, min_area_px), start=1)
    ]


def _connected_components(mask: np.ndarray, min_area_px: int) -> list[np.ndarray]:
    """Return binary connected components sorted by descending area."""

    labels, num_labels = ndimage.label(mask)
    if num_labels == 0:
        return []

    components: list[np.ndarray] = []
    for label_idx in range(1, num_labels + 1):
        component = labels == label_idx
        if int(component.sum()) >= min_area_px:
            components.append(component)
    components.sort(key=lambda item: int(item.sum()), reverse=True)
    return components


def _segmentation_ids(segmentation: torch.Tensor | np.ndarray) -> np.ndarray:
    """Convert integer or colorized segmentation output to a 2D id image."""

    if isinstance(segmentation, torch.Tensor):
        ids = segmentation.detach().cpu().numpy()
    else:
        ids = np.asarray(segmentation)

    if ids.ndim == 3 and ids.shape[-1] == 1:
        return ids[..., 0]
    if ids.ndim == 3 and ids.shape[-1] in (3, 4):
        rgb = ids[..., :3].astype(np.uint32)
        return (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    return ids


def _instance_label(instance_id: int, info: dict[str, Any] | None) -> str | None:
    """Best-effort lookup for an instance id label from Replicator info."""

    if not info:
        return None
    for key in ("idToLabels", "idToSemantics", "idToSemantic"):
        mapping = info.get(key)
        if not isinstance(mapping, dict):
            continue
        labels = mapping.get(str(instance_id), mapping.get(instance_id))
        if labels is None:
            continue
        if isinstance(labels, dict):
            return "/".join(str(value) for value in labels.values())
        if isinstance(labels, (list, tuple, set)):
            return "/".join(str(value) for value in labels)
        return str(labels)
    return None


def _pixels_to_world(
    *,
    xs: np.ndarray,
    ys: np.ndarray,
    depth: torch.Tensor,
    intrinsic_matrix: torch.Tensor,
    camera_pos_w: torch.Tensor,
    camera_quat_w_ros: torch.Tensor,
    camera_rotm_w_ros: np.ndarray | torch.Tensor | None,
    device: str | torch.device | None,
) -> np.ndarray:
    """Backproject selected pixels into world coordinates."""

    depth_np = depth.detach().cpu().numpy()
    if depth_np.ndim == 3 and depth_np.shape[-1] == 1:
        depth_np = depth_np[..., 0]
    k_np = intrinsic_matrix.detach().cpu().numpy()
    z = depth_np[ys, xs]
    x = (xs.astype(np.float64) - float(k_np[0, 2])) / float(k_np[0, 0]) * z
    y = (ys.astype(np.float64) - float(k_np[1, 2])) / float(k_np[1, 1]) * z

    pos = camera_pos_w.detach().cpu().numpy().astype(np.float64)
    points_ros = np.stack([x, y, z], axis=-1)
    if camera_rotm_w_ros is not None:
        rot = np.asarray(camera_rotm_w_ros, dtype=np.float64)
        return points_ros @ rot.T + pos

    # Fallback path for already-validated ROS-frame quaternions.
    from scipy.spatial.transform import Rotation as R

    quat = camera_quat_w_ros.detach().cpu().numpy()
    rot = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return points_ros @ rot.T + pos


def _semantic_label_ids(info: dict[str, Any] | None, target_label: str) -> set[int]:
    """Extract semantic ids whose labels contain ``target_label``."""

    if not info:
        return set()

    target = target_label.lower()
    ids: set[int] = set()
    for key in ("idToLabels", "idToSemantics", "idToSemantic"):
        mapping = info.get(key)
        if not isinstance(mapping, dict):
            continue
        for raw_id, labels in mapping.items():
            if isinstance(labels, dict):
                text = " ".join(str(v) for v in labels.values())
            elif isinstance(labels, (list, tuple, set)):
                text = " ".join(str(v) for v in labels)
            else:
                text = str(labels)
            if target in text.lower():
                try:
                    ids.add(int(raw_id))
                except ValueError:
                    pass
    return ids


def _has_semantic_mapping(info: dict[str, Any] | None) -> bool:
    """Return whether Replicator supplied any semantic id mapping."""

    if not info:
        return False
    for key in ("idToLabels", "idToSemantics", "idToSemantic"):
        mapping = info.get(key)
        if isinstance(mapping, dict) and mapping:
            return True
    return False
