"""场景加载器 — 方案 A（组件化组装）。

将赛方独立 USD 组件逐个 spawn 到默认 stage，施加物理属性后
用 Articulation / RigidObject 包装，供后续感知/规划/控制模块使用。

用法：
    from dishwasher.scene.loader import SceneLoader
    loader = SceneLoader(assets_root="~/dishwasher_ws/assets/isaac_dishwisher")
    loader.spawn_static_objects()       # 地面 + 灯光 + 桌子
    loader.spawn_piper()                # Piper 机械臂
    loader.spawn_plates(n=3)            # 盘子
    # 必须在 sim.reset() 后调用：
    loader.wrap_assets()                # Articulation + RigidObject 包装
"""

import os
from typing import Optional

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg

from .piper_cfg import PIPER_CFG


def _cleanup_physics_apis(prim):
    """递归移除 prim 子树中所有子 prim 上的物理 API。

    原因：赛方 USD 中部分 Mesh prim 自带 RigidBodyAPI/CollisionAPI/MassAPI，
    与根 prim 的 API 冲突导致 PhysX 报 "multiple rigid bodies in hierarchy"。
    """
    from pxr import UsdPhysics

    for child in prim.GetChildren():
        for api_cls in [UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI,
                         UsdPhysics.MassAPI, UsdPhysics.ArticulationRootAPI]:
            if child.HasAPI(api_cls):
                child.RemoveAPI(api_cls)
        _cleanup_physics_apis(child)


class SceneLoader:
    """赛方场景组件化加载器。

    Attributes:
        piper: Piper 机械臂 Articulation 实例 (sim.reset() 后可用)
        plates: dict[str, RigidObject]，key 为 plate name
        assets_root: 赛方素材根目录
    """

    def __init__(self, assets_root: str = "~/dishwasher_ws/assets/isaac_dishwisher"):
        self.assets_root = os.path.expanduser(assets_root)
        self._piper_cfg: Optional[ArticulationCfg] = None
        self._plate_prims: list[tuple[str, str, float]] = []  # (prim_path, label, mass)
        self.piper: Optional[Articulation] = None
        self.plates: dict[str, RigidObject] = {}

    # ---- Static objects ----

    def spawn_static_objects(self) -> None:
        """Spawn 地面、灯光、洗碗机桌子（纯静态几何，无需物理包装）。"""
        # 地面
        sim_utils.GroundPlaneCfg().func(
            "/World/defaultGroundPlane", sim_utils.GroundPlaneCfg()
        )
        # 灯光
        sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func(
            "/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        )
        # 洗碗机桌子
        desk_usd = f"{self.assets_root}/dishwasher_desk.usda"
        if os.path.exists(desk_usd):
            sim_utils.UsdFileCfg(usd_path=desk_usd).func(
                "/World/DishwasherDesk",
                sim_utils.UsdFileCfg(usd_path=desk_usd),
                translation=(0.0, 0.0, 0.0),
            )

    # ---- Piper ----

    def spawn_piper(self, translation: tuple = (0.0, 0.0, 0.0)) -> ArticulationCfg:
        """Spawn Piper 机械臂 USD 模型。

        Returns:
            PIPER_CFG 的副本，可在 sim.reset() 前修改属性。
        """
        piper_usd = f"{self.assets_root}/piper/piper_description_v100_realsense_camera_v2.usd"
        if not os.path.exists(piper_usd):
            raise FileNotFoundError(f"Piper USD not found: {piper_usd}")

        sim_utils.UsdFileCfg(usd_path=piper_usd).func(
            "/World/Piper",
            sim_utils.UsdFileCfg(usd_path=piper_usd),
            translation=translation,
        )

        self._piper_cfg = PIPER_CFG.copy()
        self._piper_cfg.init_state.pos = translation
        return self._piper_cfg

    # ---- Plates ----

    def spawn_plates(
        self,
        positions: list[tuple[float, float, float]] | None = None,
    ) -> list[str]:
        """Spawn 盘子 USD mesh 并施加物理属性。

        盘子 USD 文件是纯 mesh（无 RigidBodyAPI），手动施加：
        1. RigidBodyAPI + CollisionAPI + MassAPI 到根 prim
        2. 递归清除子 prim 上可能残留的物理 API

        Args:
            positions: 每个盘子的初始位置，默认单个盘子 (0.0, 0.0, 0.5)

        Returns:
            prim_path 列表，供 wrap_assets() 使用。
        """
        import omni.usd
        from pxr import UsdPhysics

        stage = omni.usd.get_context().get_stage()

        if positions is None:
            positions = [(0.3, 0.0, 0.8)]

        plate_specs = [
            (f"{self.assets_root}/plate.usdc", 1.0),    # bowl_plate
            (f"{self.assets_root}/plate_1.usdc", 0.35),  # flat plate
        ]

        prim_paths = []
        for i, pos in enumerate(positions):
            # 交替使用两种盘子
            usd_path, mass = plate_specs[i % len(plate_specs)]
            if not os.path.exists(usd_path):
                continue

            prim_path = f"/World/Objects/Plate_{i}"
            label = f"plate_{i}"

            # Spawn mesh
            sim_utils.UsdFileCfg(usd_path=usd_path).func(
                prim_path,
                sim_utils.UsdFileCfg(usd_path=usd_path),
                translation=pos,
            )

            # 清除子 prim 残留物理 API 并施加 API 到根 prim
            root = stage.GetPrimAtPath(prim_path)
            _cleanup_physics_apis(root)

            UsdPhysics.RigidBodyAPI.Apply(root)
            UsdPhysics.CollisionAPI.Apply(root)
            mass_api = UsdPhysics.MassAPI.Apply(root)
            mass_api.GetMassAttr().Set(mass)

            self._plate_prims.append((prim_path, label, mass))
            prim_paths.append(prim_path)

        return prim_paths

    # ---- Wrap ----

    def wrap_assets(self) -> None:
        """在 sim.reset() 之前调用，创建 Articulation / RigidObject 包装。

        必须在 spawn_* 之后、sim.reset() 之前调用。
        """
        # Piper
        if self._piper_cfg is not None:
            self.piper = Articulation(cfg=self._piper_cfg)

        # Plates
        for prim_path, label, mass in self._plate_prims:
            try:
                obj = RigidObject(cfg=RigidObjectCfg(
                    prim_path=prim_path,
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.0),  # pos 由 spawn 时 translation 决定
                    ),
                ))
                self.plates[label] = obj
            except RuntimeError as e:
                print(f"[SceneLoader] Failed to wrap {label}: {e}")
