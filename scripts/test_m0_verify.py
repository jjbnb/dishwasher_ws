#!/usr/bin/env python3
"""M0 verification on the native competition all.usd scene.

This script does not respawn assets, shift coordinates, add a ground plane, or
rewrite physics APIs.  It opens the provided all.usd as the active stage and
validates that the native scene can be used as the simulation baseline.
"""

from __future__ import annotations

import argparse
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="M0 native all.usd verification")
parser.add_argument(
    "--assets",
    default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher"),
)
parser.add_argument(
    "--keep-runtime-helpers",
    action="store_true",
    help="Keep nested PhysicsScene and ROS2 ActionGraph active.",
)
parser.add_argument(
    "--wrap-right",
    action="store_true",
    help="Also wrap the right Piper articulation.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import torch

from dishwasher.scene.native_loader import NativeSceneLoader


results: dict[str, bool] = {}


def check(name: str, passed: bool, detail: str = ""):
    results[name] = passed
    marker = "PASS" if passed else "FAIL"
    print(f"  [{marker}] {name}: {detail}", flush=True)


def count_active_physics_scenes(stage) -> list[str]:
    return [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.GetTypeName() == "PhysicsScene"
    ]


print("=" * 72)
print("M0 native all.usd verification")
print("=" * 72)

loader = NativeSceneLoader(args_cli.assets)

print("\n[1] USD structure contract", flush=True)
report = loader.validate()
check("all.usd exists", report.exists, report.path)
check("metersPerUnit = 0.01", report.meters_per_unit == 0.01, str(report.meters_per_unit))
check("upAxis = Z", report.up_axis == "Z", str(report.up_axis))
check("required prims present", not report.missing_prims, ", ".join(report.missing_prims))
check("Piper articulation roots >= 2", len(report.articulation_roots) >= 2, str(report.articulation_roots))
check("native cameras >= 3", len(report.cameras) >= 3, str(report.cameras))
check("native rigid bodies >= 6", len(report.rigid_bodies) >= 6, str(len(report.rigid_bodies)))
check("native collisions >= 6", len(report.collisions) >= 6, str(len(report.collisions)))

if not report.ok:
    print("\n[FAIL] Native USD structure is not valid enough for M0.", flush=True)
    simulation_app.close()
    sys.exit(1)

print("\n[2] Open native all.usd as active stage", flush=True)
try:
    stage = loader.open_stage()
    check("open_stage(all.usd)", True, loader.all_usd_path)
except Exception as exc:
    check("open_stage(all.usd)", False, str(exc))
    simulation_app.close()
    sys.exit(1)

print("\n[3] Runtime preparation", flush=True)
if args_cli.keep_runtime_helpers:
    print("  Keeping nested PhysicsScene and ROS2 ActionGraph active.", flush=True)
else:
    changed = loader.prepare_for_direct_isaaclab(
        stage,
        deactivate_nested_physics_scenes=True,
        deactivate_action_graphs=True,
    )
    check(
        "deactivate nested PhysicsScene runtime-only",
        True,
        str(changed["nested_physics_scenes"]),
    )
    check(
        "deactivate ROS2 ActionGraph runtime-only",
        True,
        str(changed["action_graphs"]),
    )

active_physics = count_active_physics_scenes(stage)
check("active PhysicsScene count = 1", len(active_physics) == 1, str(active_physics))

print("\n[4] SimulationContext", flush=True)
try:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    check("SimulationContext", True, args_cli.device)
except Exception as exc:
    check("SimulationContext", False, str(exc))
    simulation_app.close()
    sys.exit(1)

print("\n[5] Wrap native assets", flush=True)
try:
    loader.wrap_assets(wrap_left=True, wrap_right=args_cli.wrap_right)
    check("left Piper wrapper", loader.piper_l is not None, loader.paths.left_piper_articulation)
    check("native plate wrappers", len(loader.plates) == 3, str(list(loader.plates)))
except Exception as exc:
    check("wrap native assets", False, str(exc))
    simulation_app.close()
    sys.exit(1)

print("\n[6] sim.reset and native state readback", flush=True)
try:
    sim.reset()
    dt = sim.get_physics_dt()
    check("sim.reset()", True, f"dt={dt:.6f}")
except Exception as exc:
    check("sim.reset()", False, str(exc))
    simulation_app.close()
    sys.exit(1)

piper = loader.piper
if piper is not None:
    check("Piper joints = 8", piper.num_joints == 8, str(piper.joint_names))
    check("Piper bodies >= 9", piper.num_bodies >= 9, str(piper.num_bodies))
    root_pos = piper.data.root_pos_w[0].detach().cpu().numpy()
    check(
        "Piper native root near left-arm origin",
        abs(float(root_pos[0]) - 90.0) < 8.0
        and abs(float(root_pos[1]) - 137.0) < 8.0,
        (
            f"stage=({root_pos[0]:.3f}, {root_pos[1]:.3f}, {root_pos[2]:.3f}) "
            "meters=(0.900, 1.370, 0.144)"
        ),
    )

print("\n[7] Short native stepping smoke test", flush=True)
try:
    if piper is not None:
        piper.set_joint_position_target(piper.data.default_joint_pos.clone())
    for _ in range(10):
        if piper is not None:
            piper.write_data_to_sim()
        sim.step()
        if piper is not None:
            piper.update(dt)
        for plate in loader.plates.values():
            plate.update(dt)
    check("10 simulation steps", True)
except Exception as exc:
    check("10 simulation steps", False, str(exc))

print("\n[8] Native plate readback", flush=True)
for name, plate in loader.plates.items():
    try:
        pos = plate.data.root_pos_w[0].detach().cpu().numpy()
        vel = float(torch.norm(plate.data.root_lin_vel_w[0]))
        check(
            f"{name} native rigid body readable",
            all(abs(float(v)) < 1000.0 for v in pos),
            f"pos=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}), |v|={vel:.4f}",
        )
    except Exception as exc:
        check(f"{name} native rigid body readable", False, str(exc))

print("\n" + "=" * 72)
passed = sum(results.values())
total = len(results)
print(f"M0 native verification: {passed}/{total} passed", flush=True)
for name, ok in results.items():
    print(f"  {'PASS' if ok else 'FAIL'} {name}", flush=True)

simulation_app.close()
raise SystemExit(0 if passed == total else 1)
