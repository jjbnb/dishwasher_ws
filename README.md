# ZX-2026-0303 · 多模态感知洗碗机摆盘

[![M0](https://img.shields.io/badge/M0-环境验证通过-brightgreen)]()
[![Isaac Sim](https://img.shields.io/badge/Isaac_Sim-5.1.0-blue)]()
[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-2.3.2-blue)]()

基于 Isaac Sim + Isaac Lab 的机器人竞赛方案：Piper 机械臂通过多模态感知（RGB-D + 实例分割）自主完成「从水槽抓取盘子 → 放入洗碗机卡槽」的全流程。

## 硬件要求

| GPU | 显存 | GUI + 物理 | 纯物理 |
|-----|------|-----------|--------|
| RTX 5070 | 12 GB | ❌ CUDA OOM | ✅ |
| **A40**（推荐） | 48 GB | ✅ | ✅ |

> RTX 5070 可跑 headless 物理模拟和静态 USD 查看，但无法同时 GUI 渲染 + 物理仿真。

## 环境配置

```bash
# 1. Conda 环境（Isaac Sim 5.1.0 + Isaac Lab 2.3.2）
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab

# 2. 安装 Isaac Lab
pip install -e ~/IsaacLab/source/isaaclab/
pip install -e ~/IsaacLab/source/isaaclab_assets/
pip install -e ~/IsaacLab/source/isaaclab_tasks/

# 3. 安装本项目
cd ~/dishwasher_ws
pip install -e src/

# 4. 赛方资产软链接
ln -s /path/to/isaac_dishwisher assets/isaac_dishwisher
```

## 快速开始

```bash
conda activate env_isaaclab
cd ~/dishwasher_ws

# 校验赛方原生 all.usd 结构（不启动 Isaac）
python scripts/validate_all_usd.py

# 验证原生 all.usd 能作为仿真基座
python scripts/test_m0_verify.py --headless

# 验证 Piper 关节控制
python scripts/test_piper_articulation.py --headless

# 验证盘子物理
python scripts/test_plate_physics.py --headless

# M0 原生 all.usd headless 快照导出
python scripts/demo_m0_render.py

# M1 / Level 1 原生 all.usd smoke（detect + plan + IK）
python scripts/run_level1_native.py --headless --num_plates 1
```

## M0 验证结果 ✅

| 验证项 | 状态 | 详情 |
|--------|------|------|
| 原生 all.usd 结构 | ✅ | `metersPerUnit=0.01`, `/World` 默认 prim, 359 prim |
| 原生场景内容 | ✅ | 房间 + 桌子/水槽/卡槽 + 双 Piper + 3 盘 + 相机 + ROS2 图 |
| Piper 包装 | ✅ | 直接包装 `/World/piper_ros2_/piper_camera` 的 8 关节 Articulation |
| 盘子包装 | ✅ | 直接包装原生 plate mesh RigidBody，不移动物理 API |
| USD 导出 | ✅ | 导出原生场景打开/运行时准备/短步进快照 |

输出位于 `results/m0_demo/`，含 `00_native_all_usd_opened.usd` 等原生场景快照。

## 项目结构

```
dishwasher_ws/
├── README.md                   # 本文件
├── docs/ROADMAP.md             # 完整技术路线（含三级策略细节）
├── Makefile                    # 常用命令
├── setup.py                    # pip install -e .
├── configs/                    # Level 1/2/3 配置文件
│
├── src/dishwasher/
│   ├── scene/                  # 场景管理
│   │   ├── native_loader.py    #   NativeSceneLoader: 直接打开赛方 all.usd
│   │   ├── loader.py           #   Legacy SceneLoader: 旧组件化装配（待迁移）
│   │   └── piper_cfg.py        #   Piper ArticulationCfg (8 joints)
│   ├── perception/             # 感知：检测 + 位姿估计 + 深度补全
│   ├── grasping/               # 抓取：生成 + 过滤 + 排序
│   ├── motion/                 # 运动：IK 控制器 + 轨迹规划
│   ├── control/                # 控制：状态机 + 夹爪 + 恢复
│   └── utils/                  # 配置管理 + 指标 + 日志
│
├── scripts/                    # 运行脚本
│   ├── test_*.py               #   各模块独立验证
│   ├── validate_all_usd.py     #   原生 all.usd 结构校验
│   ├── demo_m0_render.py       #   M0 native all.usd headless 演示
│   ├── demo_m0_gui.py          #   M0 GUI 直接查看原生 all.usd
│   ├── run_level1_native.py    #   M1 native all.usd detect/plan/IK smoke
│   └── view_m0_usd.py          #   查看导出的 USD 场景
│
├── results/m0_demo/            # M0 原生 all.usd 快照输出
└── docker/                     # Docker 打包
```

## 架构

```
RGB-D 相机 ──► [感知] 检测+分割+位姿 ──► [规划] 抓取生成+碰撞规避 ──► [控制] IK+夹爪 ──► Piper 机械臂
```

三级策略：
- **Level 1** — 规则策略（深度阈值 + 模板匹配 + 固定抓取 + 顺序放置），目标 100%
- **Level 2** — 启发式（多模态融合 + 物体分类 + 域随机化），目标 95%
- **Level 3** — RL/IL 增强（深度补全 + 主动感知 + 力控 + 恢复），目标 85%

详见 [docs/ROADMAP.md](docs/ROADMAP.md)。

## 开发流程

```bash
# 开分支
git checkout -b feature/xxx

# 写代码 → 验证
python scripts/test_scene.py
python scripts/demo_m0_render.py

# 提交
git add -A && git commit -m "feat: xxx"
git push origin feature/xxx
```

## Git 分支

```
main
├── m0-env-setup        # ✅ 已完成
├── m1-level1           # ← 当前
├── m2-level2
├── m3-level3
└── m4-delivery
```

## 已知问题

- **RTX 5070 (12GB)**: GUI + 完整物理场景 CUDA OOM，需在 A40 上做 GUI 验证
- **原生 all.usd 单位**: `metersPerUnit=0.01`，Isaac Lab 张量读数按 stage units 显示为厘米尺度；不要再手动 `÷100 + Z_SHIFT` 重建场景
- **原生 all.usd 运行时辅助节点**: 文件内含嵌套 `PhysicsScene` 和 ROS2 `ActionGraph`；直接 Isaac Lab 控制时脚本会运行时临时禁用这些节点，但不保存、不改源 USD
- **Isaac Sim 5.1 截图 API**: `omni.syntheticdata`、`omni.replicator`、`renderer.capture` 均不可靠
- **USD 导出不捕获物理状态**: `stage.Export()` 只导出 USD prim transform，非 PhysX Fabric 张量状态
