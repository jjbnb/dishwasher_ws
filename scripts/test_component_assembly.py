#!/usr/bin/env python3
"""
M0 验证：方案 A（组件化组装）— 不碰 stage，直接 init SimulationContext → 加载组件。

与 dishwasher_test.py 的唯一区别：把 Franka + 占位物体 替换为赛方独立 USD。
Usage:
    conda activate env_isaaclab
    OMNI_KIT_ACCEPT_EULA=YES python scripts/test_component_assembly.py --headless --enable_cameras
"""

import argparse
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="M0: 组件化组装方案")
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from pxr import UsdPhysics

# ============================================================
# Step 1: 初始化 SimulationContext（不碰 stage！）
# ============================================================
print("\n" + "=" * 60)
print("Step 1: 初始化 SimulationContext（使用默认 stage）")
print("=" * 60)

sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
sim = sim_utils.SimulationContext(sim_cfg)
print("[OK] SimulationContext 初始化成功！")

# ============================================================
# Step 2: 加载赛方独立 USD 组件（spawn）
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 逐个 spawn 赛方组件")
print("=" * 60)

assets_root = os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher")
sim.set_camera_view([1.5, 1.5, 1.8], [0.0, 0.0, 0.8])

# ---- 地面 ----
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/defaultGroundPlane", cfg)
print("  [OK] 地面")

# ---- 灯光 ----
cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
cfg.func("/World/Light", cfg)
print("  [OK] 灯光")

# ---- 洗碗机桌子（赛方 USD 文件）----
desk_path = f"{assets_root}/dishwasher_desk.usda"
if os.path.exists(desk_path):
    cfg = sim_utils.UsdFileCfg(usd_path=desk_path)
    cfg.func("/World/DishwasherDesk", cfg, translation=(0.0, 0.0, 0.0))
    print(f"  [OK] 洗碗机桌子: {desk_path}")
else:
    print(f"  [SKIP] 洗碗机桌子不存在: {desk_path}")

# ---- Piper 机械臂（纯 model 版，无 ROS2）----
piper_path = f"{assets_root}/piper/piper_description_v100_realsense_camera_v2.usd"
if os.path.exists(piper_path):
    cfg = sim_utils.UsdFileCfg(usd_path=piper_path)
    cfg.func("/World/Piper", cfg, translation=(0.0, 0.0, 0.0))
    print(f"  [OK] Piper 机械臂: {piper_path}")
else:
    print(f"  [SKIP] Piper 不存在: {piper_path}")

# ---- 盘子 ----
plate_paths = [
    (f"{assets_root}/plate.usdc",     "Plate_Bowl"),
    (f"{assets_root}/plate_1.usdc",   "Plate_Flat"),
]
for ppath, pname in plate_paths:
    if os.path.exists(ppath):
        cfg = sim_utils.UsdFileCfg(usd_path=ppath)
        cfg.func(f"/World/Objects/{pname}", cfg, translation=(0.3, 0.0, 1.0))
        print(f"  [OK] {pname}: {ppath}")
    else:
        print(f"  [SKIP] {pname} 不存在: {ppath}")

# ============================================================
# Step 3: 验证场景
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 场景结构")
print("=" * 60)

import omni.usd
stage = omni.usd.get_context().get_stage()

# PhysicsScene 数量
phys_scenes = []
for prim in stage.Traverse():
    if prim.GetTypeName() == "PhysicsScene":
        phys_scenes.append(prim.GetPath().pathString)
print(f"  PhysicsScene 数量: {len(phys_scenes)}")

# 顶层 prim
for prim in stage.GetPrimAtPath("/World").GetChildren():
    p = prim.GetPath().pathString
    t = prim.GetTypeName()
    cn = len(prim.GetChildren())
    print(f"  [{t}] {p}  children={cn}")

# ============================================================
# Step 4: Play + Step
# ============================================================
print("\n" + "=" * 60)
print("Step 4: Play + Step 测试 (20 步)")
print("=" * 60)

import time
import torch

sim.reset()
sim_dt = sim.get_physics_dt()
print(f"  物理时间步长: {sim_dt:.4f}s")

NUM_STEPS = 20
passed = 0
for i in range(NUM_STEPS):
    t0 = time.time()
    try:
        sim.step()
        elapsed = time.time() - t0
        if elapsed > 1.0:
            print(f"  Step {i+1:2d}/{NUM_STEPS}  ⚠️ {elapsed:.1f}s")
        else:
            print(f"  Step {i+1:2d}/{NUM_STEPS}  ✅ {elapsed:.3f}s")
        passed += 1
    except Exception as e:
        print(f"  Step {i+1:2d}/{NUM_STEPS}  ❌ 异常: {e}")
        break

print(f"\n  通过: {passed}/{NUM_STEPS}")

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
if passed == NUM_STEPS:
    print("方案 A 可行 ✅ — SimulationContext + 逐个 spawn 赛方独立 USD")
else:
    print(f"方案 A 部分通过：{passed}/{NUM_STEPS} 步成功")
print()

simulation_app.close()
