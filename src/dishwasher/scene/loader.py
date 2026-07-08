"""Legacy component-assembly scene loader.

DEPRECATED for M0: use :mod:`dishwasher.scene.native_loader` to open the
competition-provided ``all.usd`` directly.  This legacy loader converts
coordinates from the native centimeter stage to meters, applies a Z shift, and
respawns scene components.  It is kept only so older Level 1 experiments do not
break while the pipeline is migrated to the native scene baseline.

场景加载器 — 严格按 all.usd 世界空间坐标组装。

坐标系统：米制 (metersPerUnit=1.0)，Z 轴向上。
所有位置来自 all.usd (cm) ÷100 转为米制，加 Z_SHIFT 使场景在地面上方。

all.usd 精确世界坐标（m，Z_SHIFT 前）：
  - Desk:            (0.000, 1.500, -0.750)
  - Sink center:     (0.624, 1.199, -0.264)
  - Rack center:     (1.136, 0.748, -0.104)
  - Piper LEFT:      (0.900, 1.370,  0.145)
  - Piper RIGHT:     (1.350, 1.370,  0.145)
  - Plate 1:         (0.802, 1.022, -0.057)
  - Plate 2:         (0.864, 0.890,  0.001)
  - Plate 3:         (0.922, 1.042,  0.051)
"""

import os
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

from .piper_cfg import PIPER_CFG

# ======================================================================
# Z_SHIFT — 抬升整个场景使桌子底与地面 Z=0 齐平
# Desk world Z=-0.75 → Z_SHIFT=0.75 → desk at Z=0
# ======================================================================
Z_SHIFT = 0.75

# ======================================================================
# 场景布局常量（米制，已加 Z_SHIFT）
# 坐标来源：all.usd 世界空间坐标 ÷100 → 米 → +Z_SHIFT
# ======================================================================

# 桌子
DESK_X, DESK_Y, DESK_Z = 0.000, 1.500, -0.750 + Z_SHIFT  # → (0.0, 1.5, 0.0)

# 水槽（洗碗机水槽 mesh 的世界中心）
SINK_X, SINK_Y, SINK_Z = 0.624, 1.199, -0.264 + Z_SHIFT  # → (0.624, 1.199, 0.486)
# 水槽内腔近似范围（盘子落在此区域）
SINK_INNER_X_MIN, SINK_INNER_X_MAX = 0.62, 1.10
SINK_INNER_Y_MIN, SINK_INNER_Y_MAX = 0.70, 1.20
SINK_BOTTOM_Z = -0.264 + Z_SHIFT  # 碰撞面在水槽中心高度 → 0.486

# 洗碗机卡槽（dishwasher_basin_step mesh 的世界中心）
RACK_X, RACK_Y, RACK_Z = 1.136, 0.748, -0.104 + Z_SHIFT  # → (1.136, 0.748, 0.646)

# Piper 左臂（piper_ros2_）
PIPER_L_X, PIPER_L_Y, PIPER_L_Z = 0.900, 1.370, 0.145 + Z_SHIFT  # → (0.900, 1.370, 0.895)

# Piper 右臂（piper_ros2__04）
PIPER_R_X, PIPER_R_Y, PIPER_R_Z = 1.350, 1.370, 0.145 + Z_SHIFT  # → (1.350, 1.370, 0.895)

# 盘子初始位置（all.usd plate_1, plate_02, plate_03 世界坐标）
PLATE_SPAWN_POSITIONS = [
    (0.802, 1.022, -0.057 + Z_SHIFT),  # → (0.802, 1.022, 0.693)
    (0.864, 0.890,  0.001 + Z_SHIFT),  # → (0.864, 0.890, 0.751)
    (0.922, 1.042,  0.051 + Z_SHIFT),  # → (0.922, 1.042, 0.801)
]

# ======================================================================
# 工具
# ======================================================================


def _clean_physics_from_children(prim):
    """递归清除 prim 子树中所有物理 API（RigidBody/Collision/Mass/ArticulationRoot）。"""
    from pxr import UsdPhysics

    for child in prim.GetChildren():
        for api_cls in [
            UsdPhysics.RigidBodyAPI,
            UsdPhysics.CollisionAPI,
            UsdPhysics.MassAPI,
            UsdPhysics.ArticulationRootAPI,
        ]:
            try:
                if child.HasAPI(api_cls):
                    child.RemoveAPI(api_cls)
            except Exception:
                pass
        _clean_physics_from_children(child)


def _spawn_collision_cuboid(prim_path: str, center: tuple, size: tuple):
    """生成一个运动学碰撞立方体（透明，不可见，用于支撑盘子）。"""
    sim_utils.CuboidCfg(
        size=size,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.3, 0.3, 0.3), opacity=0.0,
        ),
    ).func(
        prim_path,
        sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.3), opacity=0.0,
            ),
        ),
        translation=center,
    )


# ======================================================================
# SceneLoader
# ======================================================================


class SceneLoader:
    """场景加载器 — 严格按 all.usd 世界空间坐标组装。"""

    def __init__(self, assets_root: str = "~/dishwasher_ws/assets/isaac_dishwisher"):
        self.assets_root = os.path.expanduser(assets_root)
        self._piper_l_cfg: Optional[ArticulationCfg] = None
        self._piper_r_cfg: Optional[ArticulationCfg] = None
        self._plate_prims: list[tuple[str, str, float]] = []

        # 运行时可用的资产引用
        self.piper_l: Optional[Articulation] = None  # 左臂
        self.piper_r: Optional[Articulation] = None  # 右臂
        self.plates: dict[str, RigidObject] = {}

    # ---- Static objects -------------------------------------------------

    def spawn_static_objects(self) -> None:
        """生成地面、灯光、桌子 + 水槽底碰撞 + 卡槽底碰撞。"""
        # 地面
        sim_utils.GroundPlaneCfg().func(
            "/World/defaultGroundPlane", sim_utils.GroundPlaneCfg()
        )

        # 灯光
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
            "/World/Light",
            sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)),
        )

        # 桌子 — 引用 dishwasher_desk.usda
        desk_usd = f"{self.assets_root}/dishwasher_desk.usda"
        if os.path.exists(desk_usd):
            sim_utils.UsdFileCfg(usd_path=desk_usd).func(
                "/World/DishwasherDesk",
                sim_utils.UsdFileCfg(usd_path=desk_usd),
                translation=(DESK_X, DESK_Y, DESK_Z),
            )

        # 清除桌子 mesh 上残留的物理 API
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        desk_root = stage.GetPrimAtPath("/World/DishwasherDesk")
        if desk_root:
            _clean_physics_from_children(desk_root)
            print("[Init] 桌子物理 API 已清理", flush=True)

        # 水槽底碰撞面 — 盘子落入水槽后停在此处
        sink_cx = (SINK_INNER_X_MIN + SINK_INNER_X_MAX) / 2
        sink_cy = (SINK_INNER_Y_MIN + SINK_INNER_Y_MAX) / 2
        sink_sx = SINK_INNER_X_MAX - SINK_INNER_X_MIN
        sink_sy = SINK_INNER_Y_MAX - SINK_INNER_Y_MIN
        _spawn_collision_cuboid(
            "/World/CollisionSinkBottom",
            (sink_cx, sink_cy, SINK_BOTTOM_Z),
            (sink_sx, sink_sy, 0.005),
        )

        # 卡槽底碰撞面 — 盘子放置后停在此处
        _spawn_collision_cuboid(
            "/World/CollisionRack",
            (RACK_X, RACK_Y, RACK_Z),
            (0.30, 0.42, 0.005),
        )

        print("[Init] 场景已加载 (all.usd 精确坐标)", flush=True)

    # ---- Piper arms -----------------------------------------------------

    def _spawn_one_piper(
        self, prim_path: str, translation: tuple
    ) -> ArticulationCfg:
        """生成一个 Piper 机械臂并返回 ArticulationCfg。"""
        piper_usd = (
            f"{self.assets_root}/piper/"
            "piper_description_v100_realsense_camera_v2.usd"
        )
        if not os.path.exists(piper_usd):
            raise FileNotFoundError(f"Piper USD not found: {piper_usd}")

        sim_utils.UsdFileCfg(usd_path=piper_usd).func(
            prim_path,
            sim_utils.UsdFileCfg(usd_path=piper_usd),
            translation=translation,
        )

        cfg = PIPER_CFG.copy()
        cfg.prim_path = prim_path
        cfg.init_state.pos = translation
        return cfg

    def spawn_piper_left(
        self, translation: tuple = (PIPER_L_X, PIPER_L_Y, PIPER_L_Z)
    ) -> ArticulationCfg:
        """生成 Piper 左臂。"""
        self._piper_l_cfg = self._spawn_one_piper("/World/Piper_L", translation)
        return self._piper_l_cfg

    def spawn_piper_right(
        self, translation: tuple = (PIPER_R_X, PIPER_R_Y, PIPER_R_Z)
    ) -> ArticulationCfg:
        """生成 Piper 右臂。"""
        self._piper_r_cfg = self._spawn_one_piper("/World/Piper_R", translation)
        return self._piper_r_cfg

    def spawn_piper(
        self, translation: tuple = (PIPER_L_X, PIPER_L_Y, PIPER_L_Z)
    ) -> ArticulationCfg:
        """生成 Piper 左臂（兼容旧接口）。"""
        return self.spawn_piper_left(translation)

    # ---- Plates ---------------------------------------------------------

    def spawn_plates(
        self,
        positions: list[tuple[float, float, float]] | None = None,
    ) -> list[str]:
        """生成盘子 — 位置来自 all.usd 世界坐标。"""

        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()

        if positions is None:
            positions = list(PLATE_SPAWN_POSITIONS)

        plate_urls = [
            f"{self.assets_root}/plate.usdc",
            f"{self.assets_root}/plate_1.usdc",
        ]

        prim_paths = []
        for i, pos in enumerate(positions):
            usd_path = plate_urls[i % len(plate_urls)]
            if not os.path.exists(usd_path):
                continue

            prim_path = f"/World/Objects/Plate_{i}"
            label = f"plate_{i}"

            sim_utils.UsdFileCfg(usd_path=usd_path).func(
                prim_path,
                sim_utils.UsdFileCfg(usd_path=usd_path),
                translation=pos,
            )

            root = stage.GetPrimAtPath(prim_path)
            _clean_physics_from_children(root)

            UsdPhysics.RigidBodyAPI.Apply(root)
            UsdPhysics.CollisionAPI.Apply(root)
            mass_api = UsdPhysics.MassAPI.Apply(root)
            mass = 0.35 if "plate_1" in usd_path else 1.0
            mass_api.GetMassAttr().Set(mass)

            self._plate_prims.append((prim_path, label, mass))
            prim_paths.append(prim_path)

        return prim_paths

    # ---- Wrap assets ----------------------------------------------------

    def wrap_assets(self) -> None:
        """创建 Isaac Lab Articulation / RigidObject 包装。"""
        if self._piper_l_cfg is not None:
            self.piper_l = Articulation(cfg=self._piper_l_cfg)
        if self._piper_r_cfg is not None:
            self.piper_r = Articulation(cfg=self._piper_r_cfg)

        for prim_path, label, _mass in self._plate_prims:
            try:
                obj = RigidObject(
                    cfg=RigidObjectCfg(
                        prim_path=prim_path,
                        init_state=RigidObjectCfg.InitialStateCfg(
                            pos=(0.0, 0.0, 0.0)
                        ),
                    )
                )
                self.plates[label] = obj
            except RuntimeError as e:
                print(f"[SceneLoader] Failed to wrap {label}: {e}")

    # ---- Convenience ----------------------------------------------------

    @property
    def piper(self) -> Optional[Articulation]:
        """主机械臂（Level 1 使用左臂）。"""
        return self.piper_l
