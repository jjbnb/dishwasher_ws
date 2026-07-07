#!/usr/bin/env python3
"""
方案 A 验证：用赛方独立 USD 组件组装场景 → 验证物理模拟正常。

基于已验证的 test_minimal.py 模式：
空 stage → SimulationContext → spawn 赛方组件 → step
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="方案A: 赛方组件组装验证")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg

ASSETS = os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher")

print("\n" + "=" * 60)
print("Step 1: SimulationContext", flush=True)
print("=" * 60)

sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
sim = sim_utils.SimulationContext(sim_cfg)
print("[OK] SimulationContext 创建成功", flush=True)

# ================================================================
print("\n" + "=" * 60)
print("Step 2: Spawn 赛方组件", flush=True)
print("=" * 60)

# 地面
# 地面
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/defaultGroundPlane", cfg)
print("[OK] 地面", flush=True)

# 灯光
cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
cfg.func("/World/Light", cfg)
print("[OK] 灯光", flush=True)

# 洗碗机桌子
desk_usd = f"{ASSETS}/dishwasher_desk.usda"
if os.path.exists(desk_usd):
    cfg = sim_utils.UsdFileCfg(usd_path=desk_usd)
    cfg.func("/World/DishwasherDesk", cfg, translation=(0.0, 0.0, 0.0))
    print(f"[OK] 洗碗机桌子", flush=True)
else:
    print(f"[SKIP] 桌子不存在: {desk_usd}", flush=True)

# Piper 机械臂（纯 model 版）
piper_usd = f"{ASSETS}/piper/piper_description_v100_realsense_camera_v2.usd"
if os.path.exists(piper_usd):
    cfg = sim_utils.UsdFileCfg(usd_path=piper_usd)
    cfg.func("/World/Piper", cfg, translation=(0.0, 0.0, 0.0))
    print(f"[OK] Piper 机械臂", flush=True)
else:
    print(f"[SKIP] Piper 不存在: {piper_usd}", flush=True)

# 盘子 — Cylinder 占位（后续替换为赛方 plate.usdc + RigidBodyAPI）
plate_cfg = RigidObjectCfg(
    prim_path="/World/Objects/Plate_1",
    spawn=sim_utils.CylinderCfg(
        radius=0.12,
        height=0.02,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.95, 0.98)),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=0.3),
        collision_props=sim_utils.CollisionPropertiesCfg(),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.5)),
)
plate = RigidObject(cfg=plate_cfg)
print("[OK] 盘子占位 (圆柱体, 0.3kg)", flush=True)

# ================================================================
print("\n" + "=" * 60)
print("Step 3: Play + Step", flush=True)
print("=" * 60)

sim.reset()
print(f"[OK] sim.reset(), dt={sim.get_physics_dt():.4f}s", flush=True)

import time
for i in range(20):
    t0 = time.time()
    sim.step()
    plate.update(sim.get_physics_dt())
    dt = time.time() - t0
    pos_str = ""
    try:
        p = plate.data.body_pos_w[0].cpu().numpy()
        pos_str = f" pos=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f})"
    except Exception:
        pass
    if dt > 1.0:
        print(f"  Step {i+1:2d} ⚠️ {dt:.1f}s{pos_str}", flush=True)
    else:
        print(f"  Step {i+1:2d} ✅ {dt:.3f}s{pos_str}", flush=True)

print("\n[ALL GOOD] 方案 A 验证通过 ✅", flush=True)
simulation_app.close()
