#!/usr/bin/env python3
"""
M0 场景可视化 — 在 Isaac Sim GUI 中打开 all.usd。
关闭窗口即退出。

Usage:
    conda activate env_isaaclab
    python scripts/view_scene.py
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="M0: 在 GUI 中查看赛方场景")
parser.add_argument("--scene", type=str,
                    default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher/all.usd"))
args, unknown = parser.parse_known_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.usd
from pxr import Usd

# 打开场景
if not os.path.exists(args.scene):
    print(f"[ERR] 场景文件不存在: {args.scene}")
    simulation_app.close()
    sys.exit(1)

ctx = omni.usd.get_context()
ctx.open_stage(args.scene)
print(f"[OK] 场景已打开: {args.scene}")

stage = ctx.get_stage()

# 打印场景中的关键 prim 供参考
print("\n===== 场景关键资产 =====")
for prim in stage.Traverse():
    if prim.GetName() in ["piper_ros2_", "piper_ros2__04", "dishwasher_desk_1_",
                           "plate_1", "plate_02", "plate_03", "roomScene"]:
        print(f"  [{prim.GetTypeName()}] {prim.GetPath().pathString}")

print("\n场景已显示。关闭 Isaac Sim 窗口退出...")
print("提示：鼠标滚轮缩放 | 中键旋转 | 右键平移\n")

# 阻塞直到用户关闭窗口 — 使用 omniverse kit 的事件循环
import omni.kit.app
kit_app = omni.kit.app.get_app()

while simulation_app.is_running():
    kit_app.update()

print("窗口已关闭，退出。")
simulation_app.close()
