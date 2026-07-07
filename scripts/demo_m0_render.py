#!/usr/bin/env python3
"""
M0 验证演示 — headless 物理模拟 + USD 场景导出。
每个关键时刻导出完整场景到 USD 文件，在 Isaac Sim GUI 中打开即可验证。

用法:
    python -u scripts/demo_m0_render.py

输出:
    ~/dishwasher_ws/results/m0_demo/  包含多个 .usd 场景文件
    用 Isaac Sim GUI 逐一打开验证，或在 GUI 中手动截图
"""
import argparse, os, sys
import numpy as np

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args(["--headless"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("[INFO] Isaac Sim headless 已启动", flush=True)

import isaaclab.sim as sim_utils
from dishwasher.scene.loader import SceneLoader
import torch
import omni.usd

ASSETS = os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher")
OUTPUT_DIR = os.path.expanduser("~/dishwasher_ws/results/m0_demo")
os.makedirs(OUTPUT_DIR, exist_ok=True)

_step_idx = [0]


def export_scene(label: str):
    """导出当前场景到 USD 文件。"""
    idx = _step_idx[0]
    _step_idx[0] += 1
    fname = f"{idx:02d}_{label}.usd"
    fpath = os.path.join(OUTPUT_DIR, fname)

    stage = omni.usd.get_context().get_stage()
    # 强制刷新所有 prim 的变换
    stage.Save()
    stage.Export(fpath, False)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  📁 {fname} ({size_kb:.0f} KB)", flush=True)


# ================================================================
print("=" * 60, flush=True)
print("M0 场景导出演示 (可在 GUI 中打开 USD 验证)", flush=True)
print("=" * 60, flush=True)

# ---- 1. SimulationContext ----
print("\n[1] SimulationContext...", flush=True)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
print("   ✅", flush=True)

# ---- 2. 场景加载 ----
print("\n[2] 场景加载...", flush=True)
loader = SceneLoader(ASSETS)
loader.spawn_static_objects()
loader.spawn_piper(translation=(0.0, 0.0, 0.0))
loader.spawn_plates(positions=[
    (0.35,  0.15, 0.65),
    (0.35, -0.15, 0.65),
    (0.50,  0.00, 0.65),
])
loader.wrap_assets()
print("   ✅ 地面+灯光+桌子+Piper(8joints)+3盘 就绪", flush=True)

# ---- 3. sim.reset() ----
print("\n[3] sim.reset()...", flush=True)
sim.reset()
dt = sim.get_physics_dt()
print(f"   ✅ Physics dt={dt:.4f}s", flush=True)

p = loader.piper


def step_n(n: int):
    for _ in range(n):
        sim.step()
        p.update(dt)
        for plate in loader.plates.values():
            plate.update(dt)


# 📁 1: 初始场景 — 盘子还在空中
print("\n[Export] 初始场景 (盘子悬浮)", flush=True)
step_n(10)
export_scene("init_scene")

# ---- 4. 盘子自由落体 ----
print("\n[4] 盘子自由落体...", flush=True)
step_n(40)
export_scene("plates_mid_fall")

step_n(80)
export_scene("plates_landed")

for label, plate in loader.plates.items():
    z = float(plate.data.body_pos_w[0, 0, 2])
    print(f"   {label}: z={z:.4f} {'✅落地' if z < 0.3 else '⚠️未落'}", flush=True)

# ---- 5. Piper PD 控制 ----
print("\n[5] Piper PD 关节控制...", flush=True)

# Settle default
p.set_joint_position_target(p.data.default_joint_pos.clone())
for _ in range(60):
    p.write_data_to_sim()
    sim.step()
    p.update(dt)
export_scene("piper_default")

# 逐一移动关节
moves = [
    ("joint1→0.5",  0, 0.5),
    ("joint3→-1.0", 2, -1.0),
    ("joint2→0.8",  1, 0.8),
    ("joint4→1.0",  3, 1.0),
]

for desc, jidx, target_val in moves:
    print(f"   {desc}...", flush=True)
    target = p.data.joint_pos.clone()
    target[0, jidx] = target_val
    p.set_joint_position_target(target)
    for _ in range(100):
        p.write_data_to_sim()
        sim.step()
        p.update(dt)
    actual = float(p.data.joint_pos[0, jidx])
    ok = abs(actual - target_val) < 0.05
    print(f"     实际={actual:.3f} {'✅' if ok else '⚠️'}", flush=True)

export_scene("piper_joints_moved")

# ---- 6. 回到默认 ----
print("\n[6] 回到默认位姿...", flush=True)
p.set_joint_position_target(p.data.default_joint_pos.clone())
for _ in range(120):
    p.write_data_to_sim()
    sim.step()
    p.update(dt)
export_scene("piper_restored")

# ---- Summary ----
print("\n" + "=" * 60, flush=True)
print(f"✅ 场景文件保存在: {OUTPUT_DIR}", flush=True)
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"   {f} ({size_kb:.0f} KB)", flush=True)

print(f"\n共 {len(os.listdir(OUTPUT_DIR))} 个 USD 文件", flush=True)
print("\n逐次展示:", flush=True)
print("  01_init_scene.usd      — 初始场景：桌子 + Piper + 3悬浮盘子", flush=True)
print("  02_plates_mid_fall.usd — 盘子下落中 (重力)", flush=True)
print("  03_plates_landed.usd   — 盘子着地 (碰撞检测 ✓)", flush=True)
print("  04_piper_default.usd   — Piper 默认位姿", flush=True)
print("  05_piper_joints_moved.usd — PD 控制移动4个关节后的姿态", flush=True)
print("  06_piper_restored.usd  — 回到默认位姿", flush=True)
print("\n💡 在 Isaac Sim GUI 中打开任意 .usd 文件即可查看：", flush=True)
print("   File → Open → 选择上述文件", flush=True)
print("=" * 60, flush=True)

simulation_app.close()
print("[DONE]", flush=True)
