#!/usr/bin/env python3
"""Native plate physics readback test for all.usd."""

from __future__ import annotations

import argparse
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Native all.usd plate physics test")
parser.add_argument(
    "--assets",
    default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher"),
)
parser.add_argument(
    "--keep-runtime-helpers",
    action="store_true",
    help="Keep nested PhysicsScene and ROS2 ActionGraph active.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch
from pxr import UsdPhysics

from dishwasher.scene.native_loader import NativeSceneLoader, PLATE_BODY_PATHS


print("=" * 72)
print("Native all.usd plate physics test")
print("=" * 72)

loader = NativeSceneLoader(args_cli.assets)
report = loader.validate()
if not report.ok:
    print("[FAIL] Native all.usd validation failed.")
    simulation_app.close()
    raise SystemExit(1)

stage = loader.open_stage()
if not args_cli.keep_runtime_helpers:
    loader.prepare_for_direct_isaaclab(
        stage,
        deactivate_nested_physics_scenes=True,
        deactivate_action_graphs=True,
    )

print("\n[1] Native plate USD physics APIs")
ok = True
for name, path in PLATE_BODY_PATHS.items():
    prim = stage.GetPrimAtPath(path)
    has_rb = prim.HasAPI(UsdPhysics.RigidBodyAPI)
    has_col = prim.HasAPI(UsdPhysics.CollisionAPI)
    has_mass = prim.HasAPI(UsdPhysics.MassAPI)
    mass = UsdPhysics.MassAPI(prim).GetMassAttr().Get() if has_mass else None
    passed = has_rb and has_col and has_mass
    ok = ok and passed
    print(
        f"  {name}: {path} "
        f"RigidBody={has_rb} Collision={has_col} Mass={mass} "
        f"{'PASS' if passed else 'FAIL'}",
        flush=True,
    )

print("\n[2] Isaac Lab rigid object readback")
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
loader.wrap_assets(wrap_left=False, wrap_right=False)

sim.reset()
dt = sim.get_physics_dt()
for _ in range(20):
    sim.step()
    for plate in loader.plates.values():
        plate.update(dt)

for name, plate in loader.plates.items():
    pos = plate.data.root_pos_w[0].detach().cpu().numpy()
    vel = float(torch.norm(plate.data.root_lin_vel_w[0]))
    passed = all(abs(float(v)) < 1000.0 for v in pos)
    ok = ok and passed
    print(
        f"  {name}: pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) "
        f"|v|={vel:.4f} {'PASS' if passed else 'FAIL'}",
        flush=True,
    )

simulation_app.close()
raise SystemExit(0 if ok else 1)
