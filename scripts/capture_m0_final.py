#!/usr/bin/env python3
"""最终方案: 打开USD→timeline播放→捕获swapchain→保存PNG"""
import os, sys, argparse, time
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
app = AppLauncher(args)
sim_app = app.app

time.sleep(12)  # 等 GUI 完全就绪

import omni.usd
import omni.timeline
import omni.kit.renderer.capture as rcap

usd_dir = os.path.expanduser("~/dishwasher_ws/results/m0_demo")
output_dir = os.path.expanduser("~/dishwasher_ws/results/m0_screenshots")
os.makedirs(output_dir, exist_ok=True)

cap = rcap.acquire_renderer_capture_interface()
cap.start_frame_updates()
timeline = omni.timeline.get_timeline_interface()

# 只截两张最有说服力的:
targets = [
    ("02_plates_landed.usd", "plates_landed"),   # 盘子着地证明碰撞
    ("04_piper_joints_moved.usd", "piper_moved"), # 关节移动证明PD控制
]

for usd_file, label in targets:
    usd_path = os.path.join(usd_dir, usd_file)
    png_path = os.path.join(output_dir, f"{label}.png")

    print(f"📂 {usd_file} → {label}.png", flush=True)
    omni.usd.get_context().open_stage(usd_path)
    time.sleep(3)

    # 播放几帧触发渲染
    timeline.play()
    time.sleep(1.5)
    timeline.pause()
    time.sleep(1)

    # 捕获 swapchain
    cap.capture_next_frame_swapchain_to_file(png_path)
    time.sleep(2)  # 等待异步写入

    if os.path.exists(png_path):
        size_kb = os.path.getsize(png_path) / 1024
        print(f"  ✅ {size_kb:.0f} KB", flush=True)
    else:
        print(f"  ❌ 文件未生成, 再试 async...", flush=True)
        # 用 callback 方式
        cap.capture_next_frame_swapchain_callback(lambda p=png_path: print(f"  async saved: {p}") if os.path.exists(p) else None)
        time.sleep(3)
        if os.path.exists(png_path):
            print(f"  ✅ (delayed) {os.path.getsize(png_path)/1024:.0f} KB", flush=True)
        else:
            print(f"  ❌ 最终失败", flush=True)

print(f"\n结果: {output_dir}", flush=True)
for f in sorted(os.listdir(output_dir)):
    fpath = os.path.join(output_dir, f)
    md5 = os.popen(f"md5sum {fpath}").read().split()[0][:8]
    print(f"  {f} ({os.path.getsize(fpath)/1024:.0f} KB) md5={md5}", flush=True)

sim_app.close()
