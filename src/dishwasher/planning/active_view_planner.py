"""Active wrist-camera view candidate generation.

This module ports the useful part of active-grasp style next-best-view planning:
sample candidate camera views around a task ROI. It intentionally does not
depend on ROS, MoveIt, VGN, or TSDF code; execution and image scoring stay in
the Isaac script.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin


@dataclass(frozen=True)
class ActiveViewAttempt:
    """One observation attempt generated from a view sphere sample."""

    name: str
    mode: str
    ik: str
    view_backoff: float
    view_height: float
    view_side_offset: float
    elevation_deg: float
    bearing_deg: float
    radius: float
    prior_score: float

    def as_attempt_dict(self) -> dict[str, float | str]:
        return {
            "name": self.name,
            "mode": self.mode,
            "ik": self.ik,
            "view_backoff": self.view_backoff,
            "view_height": self.view_height,
            "view_side_offset": self.view_side_offset,
            "elevation_deg": self.elevation_deg,
            "bearing_deg": self.bearing_deg,
            "radius": self.radius,
            "prior_score": self.prior_score,
        }


def generate_active_view_attempts(
    *,
    radii: tuple[float, ...] = (72.0, 84.0, 96.0, 108.0),
    elevation_degs: tuple[float, ...] = (48.0, 56.0, 64.0, 72.0),
    bearing_degs: tuple[float, ...] = (-35.0, -20.0, 0.0, 20.0, 35.0),
    max_attempts: int = 6,
    ik_mode: str = "position",
) -> list[dict[str, float | str]]:
    """Generate active observation attempts from a robot-side view sphere.

    ``view_backoff`` and ``view_side_offset`` are the parameterization already
    used by :func:`plan_wrist_camera_view`: backoff points from the task ROI
    toward the robot base, side offset moves laterally around that axis.
    """

    attempts: list[ActiveViewAttempt] = []
    for radius in radii:
        for elevation_deg in elevation_degs:
            elevation = radians(elevation_deg)
            planar = radius * cos(elevation)
            height = radius * sin(elevation)
            for bearing_deg in bearing_degs:
                bearing = radians(bearing_deg)
                backoff = planar * cos(bearing)
                side_offset = planar * sin(bearing)
                if backoff < 36.0:
                    continue
                prior = _view_prior_score(
                    radius=radius,
                    elevation_deg=elevation_deg,
                    bearing_deg=bearing_deg,
                    backoff=backoff,
                    height=height,
                    side_offset=side_offset,
                )
                attempts.append(
                    ActiveViewAttempt(
                        name=(
                            f"sphere_r{radius:.0f}_e{elevation_deg:.0f}_"
                            f"b{bearing_deg:+.0f}"
                        ),
                        mode="view",
                        ik=ik_mode,
                        view_backoff=backoff,
                        view_height=height,
                        view_side_offset=side_offset,
                        elevation_deg=elevation_deg,
                        bearing_deg=bearing_deg,
                        radius=radius,
                        prior_score=prior,
                    )
                )

    attempts.sort(key=lambda item: item.prior_score, reverse=True)
    return [attempt.as_attempt_dict() for attempt in attempts[:max(0, max_attempts)]]


def _view_prior_score(
    *,
    radius: float,
    elevation_deg: float,
    bearing_deg: float,
    backoff: float,
    height: float,
    side_offset: float,
) -> float:
    """Heuristic prior before executing and scoring real camera frames."""

    # Prefer a real inspection standoff: elevated enough for scene context,
    # but not so vertical that the wrist pose becomes unreachable.
    radius_score = 1.0 - min(1.0, abs(radius - 90.0) / 36.0)
    elevation_score = 1.0 - min(1.0, abs(elevation_deg - 60.0) / 28.0)
    centerline_score = 1.0 - min(1.0, abs(bearing_deg) / 55.0)
    backoff_score = 1.0 - min(1.0, abs(backoff - 56.0) / 42.0)
    height_score = 1.0 - min(1.0, abs(height - 76.0) / 34.0)
    side_score = 1.0 - min(1.0, abs(side_offset) / 42.0)
    return (
        0.22 * radius_score
        + 0.20 * elevation_score
        + 0.18 * centerline_score
        + 0.18 * backoff_score
        + 0.14 * height_score
        + 0.08 * side_score
    )
