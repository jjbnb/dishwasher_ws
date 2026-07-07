#!/usr/bin/env python3
"""盘子物理修复 — 两个盘子都用 手动 RigidBodyAPI（根 prim 加物理，子 prim 清理干净）"""
import argparse, os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObject, RigidObjectCfg
import omni.usd
from pxr import UsdPhysics
import torch

ASSETS = os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher")
stage = omni.usd.get_context().get_stage()

print("Step 1: SimulationContext", flush=True)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))

print("Step 2: Ground+Light", flush=True)
sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))

def clean_all_physics_apis(prim, root):
    """递归移除所有子 prim 上的一切物理 API（RigidBody, Collision, ArticulationRoot, Mass）"""
    if prim == root:
        # 清理根 prim 上的旧 API（重新施加前）
        for api in [UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MassAPI,
                     UsdPhysics.ArticulationRootAPI]:
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
    else:
        for api in [UsdPhysics.RigidBodyAPI, UsdPhysics.CollisionAPI, UsdPhysics.MassAPI,
                     UsdPhysics.ArticulationRootAPI]:
            if prim.HasAPI(api):
                prim.RemoveAPI(api)
                print(f"    [CLEAN] {api.__name__} from {prim.GetPath().pathString}", flush=True)
    for child in prim.GetChildren():
        clean_all_physics_apis(child, root)


def spawn_and_wrap_plate(usd_rel_path, prim_path, mass, init_pos):
    """Spawn USD mesh, apply physics, wrap with RigidObject."""
    usd_path = f"{ASSETS}/{usd_rel_path}"
    name = prim_path.split("/")[-1]
    print(f"\n--- {name} ({usd_rel_path}) ---", flush=True)

    # Spawn
    sim_utils.UsdFileCfg(usd_path=usd_path).func(prim_path, sim_utils.UsdFileCfg(usd_path=usd_path),
                                                    translation=init_pos)
    root = stage.GetPrimAtPath(prim_path)
    print(f"  Spawned: {prim_path}", flush=True)

    # 清理子 prim 的物理 API
    clean_all_physics_apis(root, root)

    # 根 prim: RigidBodyAPI + CollisionAPI + MassAPI
    UsdPhysics.RigidBodyAPI.Apply(root)
    UsdPhysics.CollisionAPI.Apply(root)
    mass_api = UsdPhysics.MassAPI.Apply(root)
    mass_api.GetMassAttr().Set(mass)
    print(f"  Physics: RigidBodyAPI + CollisionAPI + MassAPI({mass}kg)", flush=True)

    # 检查不会有多个 RigidBodyAPI
    rb_paths = []
    def find_rb(p):
        if p.HasAPI(UsdPhysics.RigidBodyAPI):
            rb_paths.append(p.GetPath().pathString)
        for c in p.GetChildren():
            find_rb(c)
    find_rb(root)
    print(f"  RigidBodyAPI prims: {rb_paths}", flush=True)
    if len(rb_paths) > 1:
        print(f"  ⚠️ WARNING: Multiple RigidBodyAPI prims found!", flush=True)

    # RigidObject
    try:
        obj = RigidObject(cfg=RigidObjectCfg(
            prim_path=prim_path,
            init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos),
        ))
        print(f"  [OK] RigidObject", flush=True)
        return obj
    except Exception as e:
        print(f"  [FAIL] RigidObject: {e}", flush=True)
        return None

# ==== 测试两个盘子 ====
plates = {}
for usd_rel, prim_path, mass, pos in [
    ("plate.usdc",   "/World/Objects/Plate_Bowl", 1.0,  (-0.3, 0.0, 0.8)),
    ("plate_1.usdc", "/World/Objects/Plate_Flat",  0.35, (0.3, 0.0, 0.8)),
]:
    obj = spawn_and_wrap_plate(usd_rel, prim_path, mass, pos)
    if obj:
        plates[prim_path] = obj

if not plates:
    print("\n[FATAL] 无盘子", flush=True)
    simulation_app.close()
    sys.exit(1)

# ==== 物理模拟 ====
print("\n" + "=" * 50)
print("Simulation", flush=True)
sim.reset()
dt = sim.get_physics_dt()
print(f"dt={dt:.4f}s", flush=True)

prev_z = {k: 0.8 for k in plates}
hit = {k: False for k in plates}

for i in range(120):
    sim.step()
    for prim_path, obj in plates.items():
        obj.update(dt)
        name = prim_path.split("/")[-1]
        raw = obj.data.body_pos_w
        z = float(raw[0, 0, 2])
        speed = float(torch.norm(obj.data.body_lin_vel_w[0, 0]))

        if speed < 0.03 and prev_z[prim_path] > 0.1 and not hit[prim_path]:
            hit[prim_path] = True
            print(f"  [{name}] Step {i:3d}: z={z:.4f}  ← 碰撞！", flush=True)
        elif i % 20 == 0 or i < 3:
            print(f"  [{name}] Step {i:3d}: z={z:.4f}  |v|={speed:.4f} {'←' if hit[prim_path] else '↓'}", flush=True)
        prev_z[prim_path] = z

print("\n" + "=" * 50)
print("Final check", flush=True)
all_ok = True
for prim_path, obj in plates.items():
    name = prim_path.split("/")[-1]
    obj.update(dt)
    z = float(obj.data.body_pos_w[0, 0, 2])
    if z < 0.3:
        print(f"  ✅ {name}: z={z:.4f}", flush=True)
    else:
        print(f"  ❌ {name}: z={z:.4f} (未下落)", flush=True)
        all_ok = False

print("\n[ALL GOOD] ✅" if all_ok else "\n[FAIL] ❌", flush=True)
simulation_app.close()
