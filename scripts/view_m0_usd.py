#!/usr/bin/env python3
"""在 Isaac Sim GUI 中打开 M0 验证场景 USD 文件。"""
import os, sys, argparse
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])  # GUI 模式
app = AppLauncher(args)
sim_app = app.app

import omni.usd

# 打开第一个 USD：展示 Piper 关节移动后的最终效果
usd_dir = os.path.expanduser("~/dishwasher_ws/results/m0_demo")
usd_files = sorted([f for f in os.listdir(usd_dir) if f.endswith(".usd")])

print("=" * 60, flush=True)
print("M0 验证场景 — Isaac Sim GUI", flush=True)
print("=" * 60, flush=True)
for i, f in enumerate(usd_files):
    print(f"  [{i}] {f}", flush=True)

# 默认打开 piper_joints_moved（最有说服力的一张）
target = os.path.join(usd_dir, usd_files[-2] if len(usd_files) >= 5 else usd_files[0])
print(f"\n📂 打开: {target}", flush=True)
print("   窗口保持 30 秒供观察，或按 Ctrl+C 退出", flush=True)

omni.usd.get_context().open_stage(target)

# 保持窗口打开
import time
time.sleep(30)

sim_app.close()
print("Done", flush=True)
