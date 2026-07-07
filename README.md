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

# 验证场景能加载
python scripts/test_scene.py

# 验证 Piper 关节控制
python scripts/test_piper_articulation.py

# 验证盘子物理
python scripts/test_plate_physics.py

# M0 完整验证（headless 模式，RTX 5070 可用）
python scripts/demo_m0_render.py
```

## M0 验证结果 ✅

| 验证项 | 状态 | 详情 |
|--------|------|------|
| 场景加载 | ✅ | 地面 + 灯光 + 桌子 + Piper + 3 盘 |
| Piper PD 控制 | ✅ | 8 关节 PD 控制，误差 < 0.05 |
| 盘子物理 | ✅ | 重力下落 + 碰撞检测 |
| USD 导出 | ✅ | 6 个关键帧场景文件 |

输出位于 `results/m0_demo/`，含 00_init_scene 到 05_piper_restored 共 6 个 USD 文件。

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
│   │   ├── loader.py           #   SceneLoader: 组件化场景装配
│   │   └── piper_cfg.py        #   Piper ArticulationCfg (8 joints)
│   ├── perception/             # 感知：检测 + 位姿估计 + 深度补全
│   ├── grasping/               # 抓取：生成 + 过滤 + 排序
│   ├── motion/                 # 运动：IK 控制器 + 轨迹规划
│   ├── control/                # 控制：状态机 + 夹爪 + 恢复
│   └── utils/                  # 配置管理 + 指标 + 日志
│
├── scripts/                    # 运行脚本
│   ├── test_*.py               #   各模块独立验证
│   ├── demo_m0_render.py       #   M0 headless 物理演示
│   ├── demo_m0_gui.py          #   M0 GUI 演示（需 A40）
│   └── view_m0_usd.py          #   查看导出的 USD 场景
│
├── results/m0_demo/            # M0 验证输出（6 个 USD 文件）
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
- **Isaac Sim 5.1 截图 API**: `omni.syntheticdata`、`omni.replicator`、`renderer.capture` 均不可靠
- **USD 导出不捕获物理状态**: `stage.Export()` 只导出 USD prim transform，非 PhysX Fabric 张量状态
