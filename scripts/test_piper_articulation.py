#!/usr/bin/env python3
"""Piper Articulation 测试 v6 — 正确 PD 控制: set_joint_position_target + write_data_to_sim"""
import argparse, os, sys
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

ASSETS = os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher")

print("Step 1: SimulationContext", flush=True)
sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))

sim_utils.GroundPlaneCfg().func("/World/defaultGroundPlane", sim_utils.GroundPlaneCfg())
sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)).func("/World/Light", sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75)))

# Spawn Piper
piper_usd = f"{ASSETS}/piper/piper_description_v100_realsense_camera_v2.usd"
sim_utils.UsdFileCfg(usd_path=piper_usd).func("/World/Piper",
    sim_utils.UsdFileCfg(usd_path=piper_usd), translation=(0.0, 0.0, 0.0))

# ArticulationCfg
piper_cfg = ArticulationCfg(
    prim_path="/World/Piper",
    spawn=None,
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={"joint[1-8]": 0.0},
    ),
    actuators={
        "piper_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit_sim=100.0, velocity_limit_sim=5.0,
            stiffness=800.0, damping=80.0,
        ),
        "piper_gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint[7-8]"],
            effort_limit_sim=10.0, velocity_limit_sim=1.0,
            stiffness=500.0, damping=50.0,
        ),
    },
)
piper = Articulation(cfg=piper_cfg)
print("Step 2: Articulation created OK", flush=True)

# sim.reset
print("Step 3: sim.reset()", flush=True)
sim.reset()
dt = sim.get_physics_dt()
import torch

print(f"  joints: {piper.num_joints}, names: {piper.joint_names}", flush=True)

# Settle at default position using PD control
print("Step 4: Settle at default pose (PD control)", flush=True)
piper.set_joint_position_target(piper.data.default_joint_pos.clone())
for i in range(60):
    piper.write_data_to_sim()
    sim.step()
    piper.update(dt)
jpos = piper.data.joint_pos[0, :6].cpu().numpy()
print(f"  After settle: [{', '.join(f'{a:.4f}' for a in jpos)}]", flush=True)

# Move joint1 using PD control
print("\nStep 5: Move joint1 to 0.3 via PD control", flush=True)
target = piper.data.default_joint_pos.clone()
target[0, 0] = 0.3
piper.set_joint_position_target(target)

for i in range(200):
    piper.write_data_to_sim()
    sim.step()
    piper.update(dt)
    if i % 40 == 0:
        jpos = piper.data.joint_pos[0, :6].cpu().numpy()
        print(f"  Step {i:3d}: [{', '.join(f'{a:.3f}' for a in jpos)}]", flush=True)

jpos = piper.data.joint_pos[0, :6].cpu().numpy()
print(f"\n  Final: joint1={jpos[0]:.3f} (target=0.300)", flush=True)
if abs(jpos[0] - 0.3) < 0.05:
    print(f"[ALL GOOD] ✅", flush=True)
else:
    print(f"[PARTIAL] joint1={jpos[0]:.3f}", flush=True)

# Move joint2
print("\nStep 6: Move joint2 to 0.5", flush=True)
target = piper.data.joint_pos.clone()
target[0, 1] = 0.5
piper.set_joint_position_target(target)

for i in range(200):
    piper.write_data_to_sim()
    sim.step()
    piper.update(dt)
    if i % 40 == 0:
        jpos = piper.data.joint_pos[0, :6].cpu().numpy()
        print(f"  Step {i:3d}: [{', '.join(f'{a:.3f}' for a in jpos)}]", flush=True)

jpos = piper.data.joint_pos[0, :6].cpu().numpy()
print(f"\n  Final: joint1={jpos[0]:.3f}, joint2={jpos[1]:.3f}", flush=True)
if abs(jpos[0] - 0.3) < 0.05 and abs(jpos[1] - 0.5) < 0.05:
    print(f"[ALL GOOD] ✅", flush=True)
else:
    print(f"[PARTIAL]", flush=True)

simulation_app.close()
