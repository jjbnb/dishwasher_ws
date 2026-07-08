#!/usr/bin/env python3
"""Directly open the competition-provided all.usd scene."""

from __future__ import annotations

import argparse
import os
import sys
import time

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
sys.path.insert(0, os.path.expanduser("~/dishwasher_ws/src"))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Direct all.usd viewer")
parser.add_argument(
    "--assets",
    default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher"),
)
parser.add_argument("--seconds", type=float, default=0.0)
parser.add_argument(
    "--native-camera",
    action="store_true",
    help="Use the native /World/Camera instead of a runtime overview camera.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from dishwasher.scene.native_loader import NativeSceneLoader


def configure_view(stage, use_native_camera: bool):
    camera_path = "/World/Camera" if use_native_camera else "/World/M0_OverviewCamera"
    if not use_native_camera:
        from pxr import Gf, UsdGeom

        camera = UsdGeom.Camera.Define(stage, camera_path)
        camera.GetFocalLengthAttr().Set(20.0)
        camera.GetClippingRangeAttr().Set(Gf.Vec2f(1.0, 100000.0))
        view = Gf.Matrix4d().SetLookAt(
            Gf.Vec3d(35.0, -85.0, 120.0),
            Gf.Vec3d(85.0, 105.0, -5.0),
            Gf.Vec3d(0.0, 0.0, 1.0),
        )
        xform = UsdGeom.Xformable(camera)
        xform.ClearXformOpOrder()
        xform.AddTransformOp().Set(view.GetInverse())

    try:
        from omni.kit.viewport.utility import get_active_viewport

        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_path
            print(f"viewport camera: {camera_path}", flush=True)
    except Exception as exc:
        print(f"[WARN] Could not set active viewport camera: {exc}", flush=True)


loader = NativeSceneLoader(args_cli.assets)
report = loader.validate()
if not report.ok:
    print("[FAIL] Native all.usd validation failed.", flush=True)
    simulation_app.close()
    raise SystemExit(1)

stage = loader.open_stage()
configure_view(stage, args_cli.native_camera)
print(f"[OK] opened native all.usd: {loader.all_usd_path}", flush=True)
print("No coordinate conversion, respawn, extra ground plane, or collision helper was added.")

if args_cli.headless:
    simulation_app.close()
elif args_cli.seconds > 0:
    deadline = time.time() + args_cli.seconds
    while simulation_app.is_running() and time.time() < deadline:
        simulation_app.update()
        time.sleep(0.01)
    simulation_app.close()
else:
    try:
        while simulation_app.is_running():
            simulation_app.update()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
    simulation_app.close()
