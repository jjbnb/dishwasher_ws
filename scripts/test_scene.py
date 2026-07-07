#!/usr/bin/env python3
"""
M0 环境检查脚本 — Prim 结构探查版。
只打开 all.usd 查看 prim 结构，不做物理模拟。

Usage:
    conda activate env_isaaclab
    python scripts/test_scene.py
"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description="M0: 赛方场景 Prim 结构探查")
parser.add_argument("--scene", type=str,
                    default=os.path.expanduser("~/dishwasher_ws/assets/isaac_dishwisher/all.usd"))
args = parser.parse_args()

os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"

from isaaclab.app import AppLauncher
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import omni.usd

# ---- 辅助函数 ----
def H(title: str): print(f"\n{'='*60}\n  {title}\n{'='*60}")
def ok(msg: str):   print(f"  [OK]    {msg}")
def warn(msg: str): print(f"  [WARN]  {msg}")
def info(msg: str): print(f"          {msg}")

# ---- 打开场景 ----
H("M0: 赛方场景 Prim 结构探查")

if not os.path.exists(args.scene):
    print(f"  [ERR] 场景文件不存在: {args.scene}")
    simulation_app.close()
    sys.exit(1)
ok(f"场景文件: {args.scene}")

ctx = omni.usd.get_context()
ctx.open_stage(args.scene)
stage = ctx.get_stage()
ok("场景已打开")

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema

# ---- 遍历统计 ----
H("Prim 类型统计")

prim_types = {}
all_prims = []
for prim in stage.TraverseAll():
    t = prim.GetTypeName()
    p = prim.GetPath().pathString
    prim_types[t] = prim_types.get(t, 0) + 1
    all_prims.append((t, p))

for t, c in sorted(prim_types.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")
print(f"  ────────────────────")
print(f"  Total: {len(all_prims)} prims")

# ---- 顶层结构 ----
H("/World 子 Prim")

world = stage.GetPrimAtPath("/World")
if world.IsValid():
    for child in world.GetChildren():
        cp = child.GetPath().pathString
        ct = child.GetTypeName()
        # 检查是否有 Articulation API
        from pxr import UsdPhysics
        has_articulation = child.HasAPI(UsdPhysics.ArticulationRootAPI)
        has_rigid = child.HasAPI(UsdPhysics.RigidBodyAPI)
        apis = []
        if has_articulation: apis.append("ArticulationRoot")
        if has_rigid: apis.append("RigidBody")
        api_str = f" ({', '.join(apis)})" if apis else ""
        print(f"  [{ct}]{api_str} {cp}")
else:
    root = stage.GetPseudoRoot()
    for child in root.GetChildren():
        print(f"  [{child.GetTypeName()}] {child.GetPath().pathString}")

# ---- 关键资产逐类查询 ----
searches = {
    "Piper 机械臂":    ["piper"],
    "相机/传感器":     ["camera", "sensor", "realsense", "rgb", "depth", "d435", "d455"],
    "盘子":            ["plate"],
    "洗碗机/水槽/桌子": ["dishw", "desk", "sink", "basin", "ground", "table"],
    "关节":            ["joint"],
    "夹爪/手":         ["gripper", "hand", "finger", "grip"],
    "链接":            ["link"],
}

for label, keywords in searches.items():
    H(label)
    found = []
    for t, p in all_prims:
        for kw in keywords:
            if kw in p.lower():
                found.append((t, p))
                break
    if found:
        for t, p in found:
            prim = stage.GetPrimAtPath(p)
            # 检查物理属性
            extras = []
            mass_api = UsdPhysics.MassAPI.Get(stage, p)
            if mass_api:
                try:
                    m = mass_api.GetMassAttr().Get()
                    extras.append(f"mass={m:.3f}kg")
                except: pass
            rigid_api = UsdPhysics.RigidBodyAPI.Get(stage, p)
            if rigid_api:
                extras.append("RigidBody")
            art_api = UsdPhysics.ArticulationRootAPI.Get(stage, p)
            if art_api:
                extras.append("Articulation")
            # 检查父子关系
            prim_children = prim.GetChildren()
            extra_str = f"  [{', '.join(extras)}]" if extras else ""
            child_info = f"  children={len(prim_children)}" if prim_children else ""
            print(f"  [{t}]{extra_str} {p}{child_info}")
    else:
        warn(f"未找到与 '{keywords[0]}' 相关的 prim")

# ---- Joint 分析 ----
H("关节 (Joint) 分析")

joint_prims = [p for t, p in all_prims if "Joint" in t]
if joint_prims:
    for p in joint_prims[:30]:
        prim = stage.GetPrimAtPath(p)
        # 尝试读取关节属性
        try:
            from pxr import UsdPhysics
            drive_api = UsdPhysics.DriveAPI.Get(stage, p, "angular") or UsdPhysics.DriveAPI.Get(stage, p, "linear")
            if drive_api:
                info(f"Joint (有 Drive): {p}")
            else:
                info(f"Joint: {p}")
        except:
            info(f"Joint: {p}")
else:
    warn("未找到 Joint prim")

# ---- 材质 ----
H("材质/着色器")
mat_prims = [(t, p) for t, p in all_prims if "Material" in t or "Shader" in t or "material" in p.lower()]
if mat_prims:
    for t, p in mat_prims[:10]:
        info(f"[{t}] {p}")
else:
    info("(无显式材质 prim)")

# ---- 物理场景 ----
H("物理场景")
phys_prims = [(t, p) for t, p in all_prims if "Physics" in t or "Physx" in t or "physx" in p.lower()]
if phys_prims:
    for t, p in phys_prims:
        info(f"[{t}] {p}")
else:
    info("(未找到物理场景 prim)")

# ---- 总结 ----
H("M0 Prim 探查总结")
print(f"""
  场景 prim 总数:    {len(all_prims)}
  Prim 类型数:       {len(prim_types)}
  根 Prim 类型:      {world.GetTypeName() if world else 'N/A'}

  下一步:
  1. 确认 Piper 关节结构 (joint 名称、数量、父子关系)
  2. 确认相机 prim 路径 (用于 Isaac Lab Camera wrapper)
  3. 确认盘子 prim 路径 (用于 RigidObject wrapper)
  4. 研究如何用 Isaac Lab 加载预建的 USD 场景 (可能需要 reference/payload 方式)
""")

simulation_app.close()
print("M0 Prim 探查完成。")
