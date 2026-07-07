#!/usr/bin/env python3
"""综合尝试所有截图方式：syntheticdata + renderer.capture + print transforms"""
import os, sys, argparse, time
os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args([])
app = AppLauncher(args)
sim_app = app.app

time.sleep(10)
import omni.usd
from pxr import UsdGeom

# ---- 方法 1: syntheticdata (extension 已加载) ----
print("=== 方法1: omni.syntheticdata ===", flush=True)
try:
    import omni.syntheticdata as sd
    print(f"  ✅ import OK, methods: {[m for m in dir(sd) if not m.startswith('_')][:20]}", flush=True)
except Exception as e:
    print(f"  ❌ {e}", flush=True)

# ---- 方法 2: renderer.capture + frame_updates ----
print("\n=== 方法2: renderer.capture + frame_updates ===", flush=True)
try:
    import omni.kit.renderer.capture as rcap
    cap = rcap.acquire_renderer_capture_interface()
    cap.start_frame_updates()
    print("  ✅ start_frame_updates called", flush=True)
except Exception as e:
    print(f"  ❌ {e}", flush=True)

# ---- 方法 3: 干脆打印 USD 中的关键 transform 数据 ----
print("\n=== 方法3: USD transform 数据验证 ===", flush=True)

usd_dir = os.path.expanduser("~/dishwasher_ws/results/m0_demo")
output_dir = os.path.expanduser("~/dishwasher_ws/results/m0_screenshots")
os.makedirs(output_dir, exist_ok=True)

# 重点验证 3 个关键文件
key_files = [
    "02_plates_landed.usd",   # 盘子落地
    "04_piper_joints_moved.usd",  # 关节移动
    "05_piper_restored.usd",  # 回到默认
]

for usd_file in key_files:
    usd_path = os.path.join(usd_dir, usd_file)
    print(f"\n📂 {usd_file}", flush=True)
    omni.usd.get_context().open_stage(usd_path)
    time.sleep(2)

    stage = omni.usd.get_context().get_stage()

    # 检查盘子位置
    for plate_name in ["/World/Objects/Plate_0", "/World/Objects/Plate_1", "/World/Objects/Plate_2"]:
        prim = stage.GetPrimAtPath(plate_name)
        if prim.IsValid():
            xform = UsdGeom.Xformable(prim)
            ops = xform.GetOrderedXformOps()
            translate = None
            for op in ops:
                if op.GetOpName() == "xformOp:translate":
                    translate = op.Get()
            if translate:
                print(f"  {plate_name}: pos=({translate[0]:.3f}, {translate[1]:.3f}, {translate[2]:.3f})", flush=True)

    # 检查 Piper joint 状态
    for j in range(1, 9):
        joint_path = f"/World/Piper/joint{j}"
        prim = stage.GetPrimAtPath(joint_path)
        if prim.IsValid():
            # 检查 joint 的 rotate/translate 属性
            for attr_name in ["physics:angularPosition", "angle", "state:angular:physics:position"]:
                attr = prim.GetAttribute(attr_name)
                if attr.IsValid():
                    val = attr.Get()
                    print(f"  {joint_path}: {attr_name}={val}", flush=True)
                    break

    # 检查 Piper root 位置
    root = stage.GetPrimAtPath("/World/Piper")
    if root.IsValid():
        xform = UsdGeom.Xformable(root)
        ops = xform.GetOrderedXformOps()
        for op in ops:
            if op.GetOpName() == "xformOp:translate":
                t = op.Get()
                print(f"  /World/Piper root: pos=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})", flush=True)

    # 尝试截图方法2
    png_path = os.path.join(output_dir, usd_file.replace(".usd", "_method2.png"))
    try:
        cap.capture_next_frame_swapchain_to_file(png_path)
        time.sleep(1)
        if os.path.exists(png_path):
            print(f"  📸 method2 OK: {os.path.getsize(png_path)/1024:.0f} KB", flush=True)
        else:
            print(f"  ⚠️ method2: 文件未生成", flush=True)
    except Exception as e:
        print(f"  ❌ method2: {e}", flush=True)

    # 尝试截图方法1
    png_path = os.path.join(output_dir, usd_file.replace(".usd", "_method1.png"))
    try:
        import omni.syntheticdata as sd
        sd.initialize()
        # Try to get default render product
        rp = sd.get_default_render_product_path()
        print(f"     default render product: {rp}", flush=True)
    except Exception as e:
        print(f"  ❌ method1: {e}", flush=True)

# ---- 最终尝试: 用 omni.kit.app 截图 ----
print("\n=== 方法4: omni.kit.app ===", flush=True)
try:
    import omni.kit.app
    kit_app = omni.kit.app.get_app()
    print(f"  kit_app methods: {[m for m in dir(kit_app) if 'capture' in m.lower() or 'screenshot' in m.lower() or 'render' in m.lower()]}", flush=True)
except Exception as e:
    print(f"  ❌ {e}", flush=True)

print(f"\n输出目录: {output_dir}", flush=True)
for f in sorted(os.listdir(output_dir)):
    print(f"  {f} ({os.path.getsize(os.path.join(output_dir,f))/1024:.0f} KB)", flush=True)

sim_app.close()
