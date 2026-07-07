#!/usr/bin/env python3
"""
M0 验证：打开 all.usd → 关闭嵌套 PhysicsScene 和 ROS2 ActionGraph →
初始化 SimulationContext → play + step → 观察是否卡死。

不修改原始 all.usd 文件。
Usage:
    conda activate env_isaaclab
    python scripts/test_deactivate_physx.py
"""

import argparse
import os
import sys

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="M0: 测试 deactivate 方案")
parser.add_argument("--scene", type=str,
                    default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher/all.usd"))
AppLauncher.add_app_launcher_args(parser)
args_cli, unknown = parser.parse_known_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.usd
from pxr import Usd

# ============================================================
# Step 1: 打开 all.usd
# ============================================================
print("\n" + "=" * 60)
print("Step 1: 打开 all.usd")
print("=" * 60)

scene_path = os.path.expanduser(args_cli.scene)
if not os.path.exists(scene_path):
    print(f"[ERR] 场景不存在: {scene_path}")
    simulation_app.close()
    sys.exit(1)

ctx = omni.usd.get_context()
ctx.open_stage(scene_path)
stage = ctx.get_stage()
print(f"[OK] 场景已打开: {scene_path}")
print(f"  Prim 总数: {len(list(stage.Traverse()))}")

# ============================================================
# Step 2: 找到需要关闭的 prim
# ============================================================
print("\n" + "=" * 60)
print("Step 2: 扫描嵌套 PhysicsScene + ActionGraph / ROS2 节点")
print("=" * 60)

nested_physics_scenes = []
action_graphs = []
ros2_prims = []

for prim in stage.Traverse():
    path = prim.GetPath().pathString
    ptype = prim.GetTypeName()
    pname = prim.GetName()

    # 找 PhysicsScene（排除 /World/physicsScene 即根层的）
    if ptype == "PhysicsScene" and path != "/World/physicsScene":
        nested_physics_scenes.append(path)
        print(f"  [PhysicsScene] {path}  ← 嵌套！")

    # 找 ActionGraph
    if "ActionGraph" in ptype:
        action_graphs.append(path)
        print(f"  [ActionGraph]  {path}")

    # 找 ROS2 相关
    if "ROS" in pname or "ros2" in pname.lower() or "ROS2" in ptype or "OmniGraph" in ptype:
        ros2_prims.append((ptype, path))
        print(f"  [ROS2]         {path}  ({ptype})")

if not nested_physics_scenes:
    print("  (未发现嵌套 PhysicsScene)")
if not action_graphs:
    print("  (未发现 ActionGraph)")
if not ros2_prims:
    print("  (未发现 ROS2 节点)")

# ============================================================
# Step 3: 关闭嵌套 PhysicsScene 和 ActionGraph
# ============================================================
print("\n" + "=" * 60)
print("Step 3: 执行 SetActive(False)")
print("=" * 60)

# 关闭嵌套 PhysicsScene
for path in nested_physics_scenes:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        prim.SetActive(False)
        print(f"  [DEACTIVATED] PhysicsScene: {path}")

# 关闭 ActionGraph
for path in action_graphs:
    prim = stage.GetPrimAtPath(path)
    if prim.IsValid():
        prim.SetActive(False)
        print(f"  [DEACTIVATED] ActionGraph:  {path}")

# 验证：再次遍历，确认只剩一个 PhysicsScene
remaining_phys = []
for prim in stage.Traverse():
    if prim.GetTypeName() == "PhysicsScene":
        remaining_phys.append(prim.GetPath().pathString)

if len(remaining_phys) == 1:
    print(f"\n  [OK] 只剩 1 个 PhysicsScene: {remaining_phys[0]}")
else:
    print(f"\n  [WARN] PhysicsScene 数量: {len(remaining_phys)} → {remaining_phys}")

# ============================================================
# Step 4: 尝试初始化 SimulationContext
# ============================================================
print("\n" + "=" * 60)
print("Step 4: 初始化 SimulationContext")
print("=" * 60)

import isaaclab.sim as sim_utils

try:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    print("[OK] SimulationContext 初始化成功！")
except Exception as e:
    print(f"[FAIL] SimulationContext 初始化失败: {e}")
    simulation_app.close()
    sys.exit(1)

# ============================================================
# Step 5: play + 多步 step，观察是否卡死
# ============================================================
print("\n" + "=" * 60)
print("Step 5: Play + Step 测试 (10 步，每步超时 5s)")
print("=" * 60)

import time

sim.reset()
sim_dt = sim.get_physics_dt()
print(f"  物理时间步长: {sim_dt:.4f}s")

STEP_TIMEOUT = 5.0  # 单步超时秒数
NUM_STEPS = 10

for i in range(NUM_STEPS):
    t0 = time.time()
    try:
        sim.step()
        elapsed = time.time() - t0
        if elapsed > 1.0:
            print(f"  Step {i+1:2d}/{NUM_STEPS}  ⚠️ {elapsed:.1f}s (偏慢)")
        else:
            print(f"  Step {i+1:2d}/{NUM_STEPS}  ✅ {elapsed:.3f}s")
    except Exception as e:
        print(f"  Step {i+1:2d}/{NUM_STEPS}  ❌ 异常: {e}")
        break

print("\n" + "=" * 60)
print("结论")
print("=" * 60)
print("如果以上 10 步都正常通过 → deactivate 方案可行 ✅")
print("如果有卡死/超时/异常 → deactivate 方案不可行 ❌")
print()

simulation_app.close()
