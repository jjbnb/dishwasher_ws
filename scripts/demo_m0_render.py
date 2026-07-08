#!/usr/bin/env python3
"""M0 native all.usd headless smoke demo and USD snapshot export."""

from __future__ import annotations

import argparse
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="M0 native all.usd headless demo")
parser.add_argument(
    "--assets",
    default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher"),
)
parser.add_argument(
    "--output-dir",
    default=os.path.expanduser("~/dishwasher_ws/results/m0_demo"),
)
parser.add_argument(
    "--keep-runtime-helpers",
    action="store_true",
    help="Keep nested PhysicsScene and ROS2 ActionGraph active.",
)
AppLauncher.add_app_launcher_args(parser)
if "--headless" not in sys.argv:
    sys.argv.append("--headless")
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
import omni.usd

from dishwasher.scene.native_loader import NativeSceneLoader


def export_stage(label: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    path = os.path.join(output_dir, f"{label}.usd")
    stage.Export(path, False)
    print(f"  exported: {path} ({os.path.getsize(path) / 1024:.1f} KB)", flush=True)


print("=" * 72)
print("M0 native all.usd demo")
print("=" * 72)

loader = NativeSceneLoader(args_cli.assets)
report = loader.validate()
if not report.ok:
    print("[FAIL] Native all.usd validation failed.", flush=True)
    simulation_app.close()
    raise SystemExit(1)

print("\n[1] Open native all.usd", flush=True)
stage = loader.open_stage()
print(f"  opened: {loader.all_usd_path}", flush=True)
export_stage("00_native_all_usd_opened", args_cli.output_dir)

print("\n[2] Runtime preparation", flush=True)
if args_cli.keep_runtime_helpers:
    print("  kept nested PhysicsScene and ROS2 ActionGraph active", flush=True)
else:
    changed = loader.prepare_for_direct_isaaclab(
        stage,
        deactivate_nested_physics_scenes=True,
        deactivate_action_graphs=True,
    )
    print(f"  disabled nested PhysicsScene: {changed['nested_physics_scenes']}", flush=True)
    print(f"  disabled ActionGraph: {changed['action_graphs']}", flush=True)
export_stage("01_native_runtime_prepared", args_cli.output_dir)

print("\n[3] Simulation smoke test", flush=True)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
loader.wrap_assets(wrap_left=True, wrap_right=False)
sim.reset()
dt = sim.get_physics_dt()
print(f"  dt={dt:.6f}", flush=True)

robot = loader.piper
robot.set_joint_position_target(robot.data.default_joint_pos.clone())
for _ in range(10):
    robot.write_data_to_sim()
    sim.step()
    robot.update(dt)
    for plate in loader.plates.values():
        plate.update(dt)

root = robot.data.root_pos_w[0].detach().cpu().numpy()
print(
    f"  left Piper root stage units=({root[0]:.3f}, {root[1]:.3f}, {root[2]:.3f})",
    flush=True,
)
for name, plate in loader.plates.items():
    pos = plate.data.root_pos_w[0].detach().cpu().numpy()
    print(
        f"  {name} stage units=({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})",
        flush=True,
    )

export_stage("02_native_after_10_steps", args_cli.output_dir)

print("\n[4] Summary", flush=True)
print(f"  snapshots: {args_cli.output_dir}", flush=True)
print("  source all.usd was not modified", flush=True)
print(
    "  note: USD export may not capture every PhysX Fabric tensor state; "
    "use readback logs for simulation state.",
    flush=True,
)

simulation_app.close()
