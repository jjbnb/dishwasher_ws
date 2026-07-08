"""Native all.usd scene contract and loader.

This module treats the competition-provided ``all.usd`` file as the source of
truth.  It does not respawn table, plates, arms, lights, or collision helpers.
The only optional stage edits are transient runtime disables for ROS2/extra
physics helpers when a direct Isaac Lab control script explicitly asks for it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import os
from collections import Counter
from typing import Any


DEFAULT_ASSETS_ROOT = "~/dishwasher_ws/assets/isaac_dishwisher"

LEFT_PIPER_ROOT = "/World/piper_ros2_"
LEFT_PIPER_ARTICULATION = "/World/piper_ros2_/piper_camera"
LEFT_PIPER_CAMERA = (
    "/World/piper_ros2_/piper_camera/link6/d435_camera_link/Camera"
)

RIGHT_PIPER_ROOT = "/World/piper_ros2__04"
RIGHT_PIPER_ARTICULATION = "/World/piper_ros2__04/piper_camera"
RIGHT_PIPER_CAMERA = (
    "/World/piper_ros2__04/piper_camera/link6/d435_camera_link/Camera"
)

SCENE_CAMERA = "/World/Camera"
DESK_ROOT = "/World/dishwasher_desk_1_"
RACK_MESH = "/World/dishwasher_desk_1_/dishwasher_basin_step/dishwasher_basin_step"
SINK_MESH = "/World/dishwasher_desk_1_/洗碗机水槽/洗碗机水槽"

PLATE_ROOT_PATHS = {
    "plate_1": "/World/plate_1",
    "plate_02": "/World/plate_02",
    "plate_03": "/World/plate_03",
}

PLATE_BODY_PATHS = {
    "plate_1": "/World/plate_1/plate/plate",
    "plate_02": "/World/plate_02/plate/plate",
    "plate_03": "/World/plate_03/plate/plate",
}

REQUIRED_PRIMS = {
    "world": "/World",
    "physics_scene": "/World/physicsScene",
    "room": "/World/roomScene",
    "room_floor_collision": "/World/roomScene/colliders/floor",
    "room_walls_collision": "/World/roomScene/colliders/walls",
    "desk": DESK_ROOT,
    "sink": SINK_MESH,
    "rack": RACK_MESH,
    "left_piper": LEFT_PIPER_ROOT,
    "left_piper_articulation": LEFT_PIPER_ARTICULATION,
    "left_piper_camera": LEFT_PIPER_CAMERA,
    "right_piper": RIGHT_PIPER_ROOT,
    "right_piper_articulation": RIGHT_PIPER_ARTICULATION,
    "right_piper_camera": RIGHT_PIPER_CAMERA,
    "scene_camera": SCENE_CAMERA,
    **{f"{name}_root": path for name, path in PLATE_ROOT_PATHS.items()},
    **{f"{name}_body": path for name, path in PLATE_BODY_PATHS.items()},
}


@dataclass(frozen=True)
class NativeScenePaths:
    """Stable prim paths in the competition-provided scene."""

    all_usd: str
    left_piper_root: str = LEFT_PIPER_ROOT
    left_piper_articulation: str = LEFT_PIPER_ARTICULATION
    right_piper_root: str = RIGHT_PIPER_ROOT
    right_piper_articulation: str = RIGHT_PIPER_ARTICULATION
    plate_bodies: dict[str, str] = field(default_factory=lambda: dict(PLATE_BODY_PATHS))
    plate_roots: dict[str, str] = field(default_factory=lambda: dict(PLATE_ROOT_PATHS))
    scene_camera: str = SCENE_CAMERA
    left_piper_camera: str = LEFT_PIPER_CAMERA
    right_piper_camera: str = RIGHT_PIPER_CAMERA
    sink_mesh: str = SINK_MESH
    rack_mesh: str = RACK_MESH


@dataclass
class NativeSceneValidation:
    """Validation result for the native all.usd file."""

    path: str
    exists: bool
    meters_per_unit: float | None = None
    up_axis: str | None = None
    default_prim: str | None = None
    total_prims: int = 0
    missing_prims: list[str] = field(default_factory=list)
    cameras: list[str] = field(default_factory=list)
    physics_scenes: list[str] = field(default_factory=list)
    articulation_roots: list[str] = field(default_factory=list)
    rigid_bodies: list[str] = field(default_factory=list)
    collisions: list[str] = field(default_factory=list)
    type_counts: dict[str, int] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return (
            self.exists
            and not self.missing_prims
            and self.meters_per_unit == 0.01
            and self.up_axis == "Z"
            and len(self.articulation_roots) >= 2
            and len(self.cameras) >= 3
            and len(self.rigid_bodies) >= 6
            and len(self.collisions) >= 6
        )


class NativeSceneLoader:
    """Load and inspect the competition-provided ``all.usd`` scene."""

    def __init__(self, assets_root: str = DEFAULT_ASSETS_ROOT):
        self.assets_root = os.path.expanduser(assets_root)
        self.all_usd_path = os.path.join(self.assets_root, "all.usd")
        self.paths = NativeScenePaths(all_usd=self.all_usd_path)
        self.piper_l = None
        self.piper_r = None
        self.plates: dict[str, Any] = {}

    def validate(self) -> NativeSceneValidation:
        """Validate native scene structure without launching Isaac Sim."""

        result = NativeSceneValidation(
            path=self.all_usd_path,
            exists=os.path.exists(self.all_usd_path),
        )
        if not result.exists:
            result.missing_prims.extend(REQUIRED_PRIMS.values())
            return result

        from pxr import Usd, UsdPhysics

        stage = Usd.Stage.Open(self.all_usd_path)
        if stage is None:
            result.warnings.append("Usd.Stage.Open returned None")
            return result

        result.meters_per_unit = stage.GetMetadata("metersPerUnit")
        result.up_axis = stage.GetMetadata("upAxis")
        default_prim = stage.GetDefaultPrim()
        result.default_prim = (
            default_prim.GetPath().pathString if default_prim else None
        )

        counts: Counter[str] = Counter()
        for prim in stage.Traverse():
            type_name = prim.GetTypeName() or "typeless"
            counts[type_name] += 1
            path = prim.GetPath().pathString
            if type_name == "Camera":
                result.cameras.append(path)
            if type_name == "PhysicsScene":
                result.physics_scenes.append(path)
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                result.articulation_roots.append(path)
            if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                result.rigid_bodies.append(path)
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                result.collisions.append(path)

        result.total_prims = sum(counts.values())
        result.type_counts = dict(counts)

        for path in REQUIRED_PRIMS.values():
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                result.missing_prims.append(path)

        if result.meters_per_unit != 0.01:
            result.warnings.append(
                f"Expected metersPerUnit=0.01, got {result.meters_per_unit}"
            )
        if result.up_axis != "Z":
            result.warnings.append(f"Expected upAxis=Z, got {result.up_axis}")
        if len(result.physics_scenes) > 1:
            result.warnings.append(
                "Native file contains nested PhysicsScene prims; direct "
                "Isaac Lab control may need transient deactivation."
            )

        return result

    def open_stage(self):
        """Open ``all.usd`` as the active Kit stage and return it."""

        import omni.usd

        if not os.path.exists(self.all_usd_path):
            raise FileNotFoundError(self.all_usd_path)

        ctx = omni.usd.get_context()
        ctx.open_stage(self.all_usd_path)
        stage = ctx.get_stage()
        if stage is None:
            raise RuntimeError(f"Failed to open stage: {self.all_usd_path}")
        return stage

    def prepare_for_direct_isaaclab(
        self,
        stage=None,
        *,
        deactivate_nested_physics_scenes: bool = False,
        deactivate_action_graphs: bool = False,
    ) -> dict[str, list[str]]:
        """Optionally disable runtime helpers without changing source USD.

        This is intentionally opt-in.  It does not rewrite coordinates,
        respawn assets, add collision helpers, or save the source file.
        """

        if stage is None:
            import omni.usd

            stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("No active USD stage")

        changed = {
            "nested_physics_scenes": [],
            "action_graphs": [],
        }

        for prim in stage.Traverse():
            path = prim.GetPath().pathString
            type_name = prim.GetTypeName()
            if (
                deactivate_nested_physics_scenes
                and type_name == "PhysicsScene"
                and path != "/World/physicsScene"
            ):
                prim.SetActive(False)
                changed["nested_physics_scenes"].append(path)
            if deactivate_action_graphs and type_name == "OmniGraph":
                prim.SetActive(False)
                changed["action_graphs"].append(path)

        return changed

    def wrap_assets(self, *, wrap_left: bool = True, wrap_right: bool = False):
        """Wrap existing native prims with Isaac Lab assets.

        The native USD must already be open as the active stage.
        """

        from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg

        from .piper_cfg import PIPER_CFG

        if wrap_left:
            cfg = PIPER_CFG.copy()
            cfg.prim_path = self.paths.left_piper_articulation
            cfg.spawn = None
            # The native all.usd stage uses centimeters (metersPerUnit=0.01).
            cfg.init_state.pos = (90.0, 137.0, 14.45)
            self.piper_l = Articulation(cfg=cfg)

        if wrap_right:
            cfg = PIPER_CFG.copy()
            cfg.prim_path = self.paths.right_piper_articulation
            cfg.spawn = None
            cfg.init_state.pos = (135.0, 137.0, 14.45)
            self.piper_r = Articulation(cfg=cfg)

        self.plates = {}
        for name, prim_path in self.paths.plate_bodies.items():
            self.plates[name] = RigidObject(
                cfg=RigidObjectCfg(
                    prim_path=prim_path,
                    init_state=RigidObjectCfg.InitialStateCfg(),
                )
            )

    @property
    def piper(self):
        """Primary Piper arm used by Level 1/M0 scripts."""

        return self.piper_l
