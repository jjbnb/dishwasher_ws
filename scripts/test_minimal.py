#!/usr/bin/env python3
"""
最小测试：只创建 SimulationContext + step，不加载任何组件。
完全参照 dishwasher_test.py 的模式。
"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Minimal test.")
parser.add_argument("--num_dishes", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_assets import FRANKA_PANDA_CFG

print("\n[TEST] Creating SimulationContext...", flush=True)
sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
sim = sim_utils.SimulationContext(sim_cfg)
print("[TEST] SimulationContext created!", flush=True)

# 最小场景：地面+灯光+Franka
cfg = sim_utils.GroundPlaneCfg()
cfg.func("/World/defaultGroundPlane", cfg)
cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
cfg.func("/World/Light", cfg)

franka_cfg = FRANKA_PANDA_CFG.replace(prim_path="/World/Robot")
franka_cfg.init_state.pos = (0.0, 0.0, 0.0)
franka = Articulation(cfg=franka_cfg)

print("[TEST] Scene designed. Calling sim.reset()...", flush=True)
sim.reset()
print("[TEST] sim.reset() done!", flush=True)

import time
print("[TEST] Stepping 10 times...", flush=True)
for i in range(10):
    t0 = time.time()
    sim.step()
    franka.update(sim.get_physics_dt())
    print(f"  Step {i+1}: {time.time()-t0:.3f}s", flush=True)

print("[TEST] All good!", flush=True)
simulation_app.close()
