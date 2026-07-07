#!/usr/bin/env python3
"""
M0 验证：PhysicsScene 数量、碰撞、重力、关节控制全面检查。

用法:
    python -u scripts/test_m0_verify.py --headless
"""

import argparse, os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from dishwasher.scene.loader import SceneLoader
import omni.usd
import torch

ASSETS = os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher")

results = {}  # check_name → passed

def check(name: str, passed: bool, detail: str = ""):
    results[name] = passed
    marker = "✅" if passed else "❌"
    print(f"  {marker} {name}: {detail}", flush=True)

# ================================================================
print("=" * 60)
print("M0 验证", flush=True)
print("=" * 60)

# ---- 1. SimulationContext ----
print("\n[1] SimulationContext", flush=True)
try:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    check("SimulationContext", True)
except Exception as e:
    check("SimulationContext", False, str(e))
    simulation_app.close()
    sys.exit(1)

# ---- 2. Scene loading ----
print("\n[2] Scene loading (方案A组件组装)", flush=True)
loader = SceneLoader(ASSETS)

try:
    loader.spawn_static_objects()
    check("地面+灯光", True)
except Exception as e:
    check("地面+灯光", False, str(e))

try:
    desk_path = f"{ASSETS}/dishwasher_desk.usda"
    check("洗碗机桌子 USD 存在", os.path.exists(desk_path), desk_path)
except Exception as e:
    check("洗碗机桌子", False, str(e))

try:
    piper_usd = f"{ASSETS}/piper/piper_description_v100_realsense_camera_v2.usd"
    check("Piper USD 存在", os.path.exists(piper_usd), piper_usd)
    loader.spawn_piper()
    check("Piper spawn", True)
except Exception as e:
    check("Piper spawn", False, str(e))

try:
    loader.spawn_plates(positions=[(0.0, 0.0, 0.8)])
    check("盘子 spawn (1个)", True)
except Exception as e:
    check("盘子 spawn", False, str(e))

try:
    loader.wrap_assets()
    check("wrap_assets (Articulation+RigidObject)", True)
except Exception as e:
    check("wrap_assets", False, str(e))

# ---- 3. PhysicsScene count ----
print("\n[3] PhysicsScene 数量检查", flush=True)
stage = omni.usd.get_context().get_stage()
ps_count = 0
from pxr import UsdPhysics
for prim in stage.Traverse():
    if prim.GetTypeName() == "PhysicsScene":
        ps_count += 1
        print(f"    PhysicsScene: {prim.GetPath().pathString}", flush=True)
check("PhysicsScene 数量 = 1", ps_count == 1, f"实际 = {ps_count}")

# ---- 4. sim.reset() ----
print("\n[4] sim.reset()", flush=True)
try:
    sim.reset()
    check("sim.reset()", True)
except Exception as e:
    check("sim.reset()", False, str(e))
    simulation_app.close()
    sys.exit(1)

dt = sim.get_physics_dt()

# ---- 5. Piper joints ----
print("\n[5] Piper 关节检查", flush=True)
if loader.piper is not None:
    p = loader.piper
    check("num_joints = 8", p.num_joints == 8, f"实际 = {p.num_joints}")
    check("joint1-6 + joint7-8", p.joint_names == [f"joint{i}" for i in range(1, 9)],
          f"实际 = {p.joint_names}")
    check("num_bodies = 10", p.num_bodies == 10, f"实际 = {p.num_bodies}")
else:
    check("Piper Articulation", False, "piper is None")

# ---- 6. Piper PD control ----
print("\n[6] Piper PD 控制", flush=True)
if loader.piper is not None:
    p = loader.piper

    # Settle at default
    p.set_joint_position_target(p.data.default_joint_pos.clone())
    for _ in range(60):
        p.write_data_to_sim()
        sim.step()
        p.update(dt)

    # Move joint1
    target = p.data.default_joint_pos.clone()
    target[0, 0] = 0.5
    p.set_joint_position_target(target)
    for _ in range(150):
        p.write_data_to_sim()
        sim.step()
        p.update(dt)

    j1 = float(p.data.joint_pos[0, 0])
    check("joint1 → 0.5", abs(j1 - 0.5) < 0.03, f"实际 = {j1:.4f}")

    # Move joint2
    target = p.data.joint_pos.clone()
    target[0, 1] = 0.5
    p.set_joint_position_target(target)
    for _ in range(150):
        p.write_data_to_sim()
        sim.step()
        p.update(dt)

    j2 = float(p.data.joint_pos[0, 1])
    check("joint2 → 0.5", abs(j2 - 0.5) < 0.03, f"实际 = {j2:.4f}")
else:
    check("Piper PD control", False, "piper is None")

# ---- 7. Plate gravity + collision ----
print("\n[7] 盘子物理 (重力+碰撞)", flush=True)
if loader.plates:
    # Reset sim and let plate drop
    # Actually plate was already spawned; just observe
    for label, plate in loader.plates.items():
        plate.update(dt)
        z = float(plate.data.body_pos_w[0, 0, 2])
        speed = float(torch.norm(plate.data.body_lin_vel_w[0, 0]))

        # Plate should be at ground level after settling (z ≈ 0)
        check(f"{label} 碰撞地面 (z<0.3)", z < 0.3, f"z={z:.4f}")
else:
    check("盘子", False, "无盘子")

# ---- 8. Root position stability ----
print("\n[8] Piper 底座稳定性", flush=True)
if loader.piper is not None:
    p = loader.piper
    root_pos = p.data.root_pos_w[0].cpu().numpy()
    root_drift = float(torch.norm(p.data.root_pos_w[0] - p.data.default_root_state[0, :3]))
    check("底座位置稳定 (drift<0.01)", root_drift < 0.01,
          f"pos=({root_pos[0]:.3f},{root_pos[1]:.3f},{root_pos[2]:.3f}), drift={root_drift:.4f}")
else:
    check("底座稳定", False, "piper is None")

# ---- Summary ----
print("\n" + "=" * 60)
passed = sum(results.values())
total = len(results)
print(f"M0 验证: {passed}/{total} 通过", flush=True)
for name, ok in results.items():
    print(f"  {'✅' if ok else '❌'} {name}", flush=True)
print(f"\n{'[ALL GOOD]' if passed == total else '[ISSUES FOUND]'}", flush=True)

simulation_app.close()
