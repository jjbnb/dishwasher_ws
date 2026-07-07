#!/usr/bin/env python3
"""
M0 GUI 演示：在 Isaac Sim 窗口中展示所有已验证的功能。
- 洗碗机桌子场景
- Piper 机械臂（PD 关节控制演示）
- 盘子物理（重力下落 + 碰撞）
- 画面保持 15 秒供观察

用法（GUI 模式，不加 --headless）:
    python -u scripts/demo_m0_gui.py
"""
import argparse, os, sys, time
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
# 关键：不传 --headless → 默认打开 GUI 窗口
# 使用 CPU 物理引擎避免 CUDA OOM（GUI 渲染仍在 GPU）
args_cli = parser.parse_args(["--device", "cpu"])
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

print("[GUI] Isaac Sim 窗口已启动", flush=True)

import isaaclab.sim as sim_utils
from dishwasher.scene.loader import SceneLoader
import torch

ASSETS = os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher")

# ================================================================
print("=" * 60, flush=True)
print("M0 GUI 演示", flush=True)
print("=" * 60, flush=True)

# ---- 1. SimulationContext ----
print("\n[1] 创建 SimulationContext...", flush=True)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
print("   ✅ SimulationContext 就绪", flush=True)

# ---- 2. 加载全部场景组件 ----
print("\n[2] 加载场景组件...", flush=True)
loader = SceneLoader(ASSETS)
loader.spawn_static_objects()
print("   ✅ 地面 + 灯光 + 洗碗机桌子", flush=True)

loader.spawn_piper(translation=(0.0, 0.0, 0.0))
print("   ✅ Piper 机械臂 (8 joints)", flush=True)

loader.spawn_plates(positions=[
    (0.35,  0.15, 0.7),
    (0.35, -0.15, 0.7),
    (0.50,  0.00, 0.7),
])
print("   ✅ 3 个盘子 (含物理修复 RigidBody+Collision+Mass)", flush=True)

loader.wrap_assets()
print("   ✅ Articulation + RigidObject 包装完成", flush=True)

# ---- 3. sim.reset() ----
print("\n[3] sim.reset()...", flush=True)
sim.reset()
dt = sim.get_physics_dt()
print(f"   ✅ 物理世界已初始化 (dt={dt:.4f}s)", flush=True)

# ---- 4. 让盘子先掉落（演示重力 + 碰撞） ----
print("\n[4] 盘子自由落体（演示重力 + 碰撞）...", flush=True)
for i in range(120):
    sim.step()
    for label, plate in loader.plates.items():
        plate.update(dt)
    if i % 30 == 0:
        for label, plate in loader.plates.items():
            z = float(plate.data.body_pos_w[0, 0, 2])
            print(f"   Step {i:3d}  {label}: z={z:.4f}", flush=True)

for label, plate in loader.plates.items():
    z = float(plate.data.body_pos_w[0, 0, 2])
    status = "✅ 已落地" if z < 0.3 else "⚠️ 未落地"
    print(f"   {label}: z={z:.4f}  {status}", flush=True)

# ---- 5. Piper PD 控制演示 ----
print("\n[5] Piper 关节运动演示...", flush=True)
p = loader.piper

# 先稳定在默认位姿
print("   稳定到默认位姿...", flush=True)
p.set_joint_position_target(p.data.default_joint_pos.clone())
for _ in range(60):
    p.write_data_to_sim()
    sim.step()
    p.update(dt)

# 连续移动多个关节展示
moves = [
    ("joint1 → 0.5",    0, 0.5),
    ("joint2 → 0.8",    1, 0.8),
    ("joint3 → -1.0",   2, -1.0),
    ("joint4 → 1.0",    3, 1.0),
    ("joint1 → 0.0",    0, 0.0),
    ("joint2 → 0.0",    1, 0.0),
    ("joint3 → 0.0",    2, 0.0),
    ("joint4 → 0.0",    3, 0.0),
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
    print(f"     实际={actual:.3f} (目标={target_val:.3f}) {'✅' if abs(actual-target_val) < 0.05 else '⚠️'}", flush=True)

# ---- 6. 保持窗口打开 ----
print("\n" + "=" * 60, flush=True)
print("🎉 M0 演示完成！窗口将保持 20 秒供观察。", flush=True)
print("   检查清单：", flush=True)
print("   - 洗碗机桌子是否可见", flush=True)
print("   - Piper 机械臂是否站立在桌子上", flush=True)
print("   - 3 个盘子是否落在桌面/地面上", flush=True)
print("   - 机械臂各个关节是否依次运动", flush=True)
print("=" * 60, flush=True)

# 持续渲染保持窗口
for i in range(400):
    sim.step()
    if loader.piper is not None:
        loader.piper.update(dt)
    for label, plate in loader.plates.items():
        plate.update(dt)

print("\n[INFO] 正在关闭 Isaac Sim...", flush=True)
simulation_app.close()
print("[DONE] 演示结束", flush=True)
