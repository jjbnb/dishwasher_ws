"""Camera-fused bimanual grasp planning.

This module follows the shape used by common robot grasping stacks:

1. collect observations from every available camera,
2. fuse detections into object hypotheses,
3. generate several grasp candidates per object and arm,
4. rank candidates before execution.

It intentionally consumes camera-derived detections only. Simulation rigid-body
poses may be useful for evaluation, but they should not enter this planner.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import cos, radians, sin
from typing import Iterable

import numpy as np

from dishwasher.grasping.generator import (
    generate_grasp_pose,
    generate_rim_grasp_pose,
)


@dataclass(frozen=True)
class CameraPlateObservation:
    """One plate detection estimated from one camera frame."""

    camera: str
    arm: str | None
    detection_index: int
    label: str
    area_px: int
    centroid_uv: tuple[float, float]
    bbox_xyxy: tuple[int, int, int, int]
    pos_w: np.ndarray


@dataclass(frozen=True)
class FusedPlate:
    """A plate hypothesis fused from one or more camera observations."""

    name: str
    pos_w: np.ndarray
    quat_w: np.ndarray
    observations: tuple[CameraPlateObservation, ...]
    total_area_px: int
    confidence: float

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(sorted({obs.camera for obs in self.observations}))

    @property
    def arms(self) -> tuple[str, ...]:
        return tuple(sorted({obs.arm for obs in self.observations if obs.arm}))


@dataclass(frozen=True)
class ArmGraspCandidate:
    """A ranked grasp candidate for one arm and one fused plate."""

    arm: str
    plate: FusedPlate
    grasp_plan: dict
    candidate_name: str
    score: float
    score_terms: dict[str, float]

    @property
    def pre_grasp_pos(self) -> np.ndarray:
        return np.asarray(self.grasp_plan["pre_grasp"][0], dtype=np.float64)


def fuse_camera_observations(
    observations: Iterable[CameraPlateObservation],
    *,
    merge_distance: float = 8.0,
) -> list[FusedPlate]:
    """Fuse per-camera detections by nearby 3D position.

    Args:
        observations: Camera-derived detections in world/stage coordinates.
        merge_distance: Max distance in stage units for detections to be treated
            as the same physical plate.
    """

    sorted_obs = sorted(observations, key=lambda item: item.area_px, reverse=True)
    clusters: list[list[CameraPlateObservation]] = []

    for obs in sorted_obs:
        pos = np.asarray(obs.pos_w, dtype=np.float64)
        best_idx = None
        best_dist = float("inf")
        for idx, cluster in enumerate(clusters):
            center = _weighted_center(cluster)
            dist = float(np.linalg.norm(pos - center))
            if dist < best_dist:
                best_dist = dist
                best_idx = idx
        if best_idx is None or best_dist > merge_distance:
            clusters.append([obs])
        else:
            clusters[best_idx].append(obs)

    fused: list[FusedPlate] = []
    for idx, cluster in enumerate(clusters):
        center = _weighted_center(cluster)
        total_area = int(sum(max(0, obs.area_px) for obs in cluster))
        camera_count = len({obs.camera for obs in cluster})
        confidence = float(np.log1p(total_area) + 0.65 * max(0, camera_count - 1))
        fused.append(
            FusedPlate(
                name=f"fused_plate_{idx}",
                pos_w=center,
                quat_w=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                observations=tuple(cluster),
                total_area_px=total_area,
                confidence=confidence,
            )
        )

    fused.sort(key=lambda item: item.confidence, reverse=True)
    return fused


def generate_bimanual_grasp_candidates(
    plates: Iterable[FusedPlate],
    *,
    arm_base_positions: dict[str, np.ndarray],
    unit_scale: float = 1.0,
    strategy: str = "auto",
    rim_angle_offsets_deg: tuple[float, ...] = (0.0, -25.0, 25.0, -50.0, 50.0),
) -> list[ArmGraspCandidate]:
    """Generate grasp candidates for every visible plate and available arm."""

    candidates: list[ArmGraspCandidate] = []
    midpoint_x = _arm_midpoint_x(arm_base_positions)

    for plate in plates:
        for arm, arm_base in arm_base_positions.items():
            if strategy in ("auto", "rim"):
                primary = _unit_xy(np.asarray(arm_base[:2]) - plate.pos_w[:2])
                for angle in rim_angle_offsets_deg:
                    radial = _rotate_xy(primary, angle)
                    plan = generate_rim_grasp_pose(
                        plate.pos_w,
                        plate.quat_w,
                        arm_base_pos=arm_base,
                        unit_scale=unit_scale,
                        radial_xy=radial,
                        candidate_name=f"rim_{angle:+.0f}deg",
                    )
                    candidates.append(
                        _score_candidate(
                            arm=arm,
                            plate=plate,
                            grasp_plan=plan,
                            candidate_name=f"rim_{angle:+.0f}deg",
                            arm_base=arm_base,
                            midpoint_x=midpoint_x,
                            is_rim=True,
                        )
                    )

            if strategy in ("auto", "center"):
                plan = generate_grasp_pose(
                    plate.pos_w,
                    plate.quat_w,
                    unit_scale=unit_scale,
                )
                metadata = plan.setdefault("metadata", {})
                metadata["strategy"] = "center_hover"
                candidates.append(
                    _score_candidate(
                        arm=arm,
                        plate=plate,
                        grasp_plan=plan,
                        candidate_name="center_hover",
                        arm_base=arm_base,
                        midpoint_x=midpoint_x,
                        is_rim=False,
                    )
                )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def choose_best_candidate(candidates: Iterable[ArmGraspCandidate]) -> ArmGraspCandidate | None:
    """Return the highest-scoring candidate, or None when planning failed."""

    sorted_candidates = sorted(candidates, key=lambda item: item.score, reverse=True)
    return sorted_candidates[0] if sorted_candidates else None


def with_score_adjustment(
    candidate: ArmGraspCandidate,
    *,
    terms: dict[str, float],
) -> ArmGraspCandidate:
    """Return a candidate with extra score terms applied."""

    merged_terms = dict(candidate.score_terms)
    merged_terms.update(terms)
    return replace(
        candidate,
        score=float(sum(merged_terms.values())),
        score_terms=merged_terms,
    )


def _score_candidate(
    *,
    arm: str,
    plate: FusedPlate,
    grasp_plan: dict,
    candidate_name: str,
    arm_base: np.ndarray,
    midpoint_x: float,
    is_rim: bool,
) -> ArmGraspCandidate:
    pre_grasp = np.asarray(grasp_plan["pre_grasp"][0], dtype=np.float64)
    xy_dist = float(np.linalg.norm(pre_grasp[:2] - np.asarray(arm_base[:2], dtype=np.float64)))
    same_arm_views = sum(1 for obs in plate.observations if obs.arm == arm)
    camera_count = len(plate.cameras)
    side_match = (arm == "left" and plate.pos_w[0] <= midpoint_x) or (
        arm == "right" and plate.pos_w[0] >= midpoint_x
    )
    reach_window = _soft_reach_score(xy_dist)

    terms = {
        "vision_confidence": min(2.5, plate.confidence / 3.5),
        "multi_camera": 0.75 * max(0, camera_count - 1),
        "same_arm_view": 0.35 * same_arm_views,
        "arm_side": 0.45 if side_match else -0.25,
        "reach_window": reach_window,
        "shorter_motion": -0.006 * xy_dist,
        "strategy": 0.95 if is_rim else -0.55,
    }

    return ArmGraspCandidate(
        arm=arm,
        plate=plate,
        grasp_plan=grasp_plan,
        candidate_name=candidate_name,
        score=float(sum(terms.values())),
        score_terms=terms,
    )


def _weighted_center(observations: list[CameraPlateObservation]) -> np.ndarray:
    weights = np.asarray([max(1.0, np.sqrt(obs.area_px)) for obs in observations])
    points = np.asarray([obs.pos_w for obs in observations], dtype=np.float64)
    return np.average(points, axis=0, weights=weights)


def _arm_midpoint_x(arm_base_positions: dict[str, np.ndarray]) -> float:
    xs = [float(pos[0]) for pos in arm_base_positions.values()]
    return float(np.mean(xs)) if xs else 0.0


def _unit_xy(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1.0e-6:
        return np.array([1.0, 0.0], dtype=np.float64)
    return np.asarray(vec, dtype=np.float64) / norm


def _rotate_xy(vec: np.ndarray, angle_deg: float) -> np.ndarray:
    theta = radians(angle_deg)
    rot = np.array(
        [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]],
        dtype=np.float64,
    )
    return _unit_xy(rot @ np.asarray(vec, dtype=np.float64))


def _soft_reach_score(distance_stage_units: float) -> float:
    """Heuristic Piper reach score in native all.usd centimeters."""

    if 25.0 <= distance_stage_units <= 85.0:
        return 0.65
    if distance_stage_units < 25.0:
        return -0.03 * (25.0 - distance_stage_units)
    return -0.018 * (distance_stage_units - 85.0)
