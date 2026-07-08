# ZX-2026-0303 比赛全貌与实施路线

> 本文档面向 AI Agent，包含完整的技术决策上下文和三级策略细节。
> 开发者请阅读 [README.md](../README.md)。

## 一、这个比赛要做什么？

```
                感知 (Perception)           规划 (Planning)          控制 (Control)
              ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
  RGB-D相机   │  物体检测 + 分割  │    │  抓取点生成 + 排序 │    │  IK 逆运动学      │
  深度相机  ──►  6D位姿估计      ──►  碰撞规避 + 运动规划 ──►  轨迹执行          │
  实例分割   │  透明/反光补偿    │    │  任务状态机编排     │    │  夹爪控制         │
              └──────────────────┘    └──────────────────┘    └──────────────────┘

场景：水槽洗碗机 + 左水槽3个盘子 → Piper机械臂抓取 → 放入右侧洗碗机卡槽
```

**核心一句话：** 用多模态感知（RGB + Depth + Segmentation）让机械臂在仿真中自主完成「从水槽抓盘子 → 放入洗碗机卡槽」的全流程。

---

## 二、赛方给了什么 vs 你需要做什么

### 赛方已提供（拿来即用）

| 资源 | 位置 | 说明 |
|------|------|------|
| 完整 3D 场景 | `isaac_dishwisher/all.usd` | 桌子 + 水槽 + 洗碗机 + 地面 已装配 |
| Piper 机械臂 USD | `isaac_dishwisher/piper/` | 含 RealSense 相机 + ROS2 接口版本的 model |
| 盘子模型 × 2 | `isaac_dishwisher/plate.usdc`, `plate_1.usdc` | bowl_plate 和 plate 两个变体 |
| 示教器代码 | `tech.py` | ROS2 双臂键盘遥控器（订阅 `/joint_states`, 发布 `/joint_command`） |
| 原始 URDF | `isaac_dishwisher/urdf/` | 所有组件的 URDF 源文件 |
| ROS2 接口预留 | README 说明 | 相机图像话题 + 关节控制话题 已预留 |

### 你需要自己做的

| 需要开发 | 说明 | 难度 |
|----------|------|------|
| **1. 感知模块** | 从相机数据中检测盘子、估计 6D 位姿、处理堆叠/遮挡/透明/反光 | ⭐⭐⭐⭐⭐ |
| **2. 抓取规划** | 针对盘子/碗/杯/勺的几何形状生成抓取姿态，碰撞检测与过滤 | ⭐⭐⭐⭐ |
| **3. 运动规划** | IK 逆运动学求解（Piper 6DOF），无碰撞轨迹生成 | ⭐⭐⭐ |
| **4. 控制执行** | 夹爪控制、力控/位置控制、抓取验证 | ⭐⭐⭐ |
| **5. 任务状态机** | IDLE→DETECT→PLAN→APPROACH→GRASP→LIFT→PLACE→VERIFY | ⭐⭐ |
| **6. 域随机化** | 光照/纹理/颜色/物理参数的随机化（Level 2/3 需要） | ⭐⭐⭐ |
| **7. 三级策略** | Level 1 规则策略、Level 2 启发式、Level 3 RL/IL 增强 | ⭐⭐⭐⭐ |
| **8. 交付物打包** | Docker + 视频 + 文档 + 报告 | ⭐⭐ |

---

## 三、三关怎么过（核心差异）

### Level 1：基础关 → 目标 100% 成功率

```
场景：5 个同款白盘子，整齐排列在水槽中，光照固定良好
策略：纯规则化，不需要 AI

具体做法：
├── 感知：固定相机视角，用深度图阈值 + 模板匹配找到盘子位置
├── 抓取：预定义抓取姿态（盘子边缘，垂直于盘面）
├── 放置：固定关节轨迹 → 洗碗机下层盘槽
├── 状态机：简单的顺序执行，每次只处理一个盘子
└── 关键：不需要处理异常，确定性地一步步跑

代码量估算：~500 行（最简单的 pipeline）
```

### Level 2：进阶关 → 目标 95% 成功率

```
场景：盘/碗/杯/勺 混合，随机位姿，部分遮挡，光照/纹理/颜色随机化
策略：启发式抓取 + 多模态融合感知 + 运动规划

具体做法：
├── 感知：RGB+Depth+InstanceSeg 三通道融合
│   ├── 实例分割获取物体 ID 和掩码
│   ├── 深度图 → 点云 → 6D 位姿估计（ICP/PCA）
│   └── 物体分类（盘/碗/杯/勺）→ 选择抓取策略
├── 抓取：基于物体类型的几何启发式抓取点生成
│   ├── 盘子：边缘两点抓取
│   ├── 碗：边缘或内侧
│   ├── 杯子：杯身侧面
│   └── 勺子：手柄末端
├── 规划：碰撞检测 + 可达性过滤 + 候选排序
├── 控制：Differential IK（阻尼最小二乘）+ 关节位置控制
├── 容错：抓取失败自动重试（最多 2 次），遮挡时先移开上层物体
└── 域随机化：光照强度/色温 + 物体颜色 + 摩擦系数 ±20%

代码量估算：~2000 行
```

### Level 3：挑战关 → 目标 85% 成功率

```
场景：透明玻璃杯 + 高反光勺子 + 杂乱堆叠 + 纹理干扰背景
策略：RL/IL 增强 + 主动感知 + 鲁棒恢复

具体做法：
├── 感知：在 Level 2 基础上增加
│   ├── 透明物体：RGB 边缘检测 → 深度补全（邻域平滑 + 法向量约束）
│   ├── 反光物体：多视角融合（移动手腕相机重观察）
│   └── 堆叠场景：深度分层 → 优先抓取最顶层物体
├── 抓取：可选 RL 训练的策略网络（处理感知噪声）
│   └── 或：规则增强版（主动感知 + 更多候选 + 更多试探）
├── 置信度机制：感知输出附带置信度
│   └── 低置信度 → 触发主动感知（手腕相机移动观察）
├── 力控抓取：接触力阈值判断抓取成功（不只是位置）
├── 恢复策略：
│   ├── 物体掉落 → 重新检测 + 重新规划
│   ├── 碰撞 → 立即停止 → 回溯 → 换候选抓取
│   └── 连续失败 3 次 → 跳过当前物体 → 处理下一个
└── 域随机化：Level 2 全部 + 强光/弱光 + 背景纹理干扰

代码量估算：~3000-5000 行（含 RL 训练代码）
```

---

## 四、技术架构总览

### 方案选择：两种路径

| | **路径 A：纯规则（推荐先做）** | **路径 B：RL/IL 增强**
|---|---|---
| 做法 | 传统 CV + 启发式抓取 + IK | 路径 A + RL 抓取策略训练 |
| Level 1 | ✅ 轻松 100% | 杀鸡用牛刀 |
| Level 2 | ✅ 可达 95% | ✅ 更稳 |
| Level 3 | ⚠️ 需要精心调参 | ✅ 天然适合 |
| 开发周期 | 2-3 周 | 4-6 周 |
| 风险 | 透明/反光场景需手工处理 | 训练不稳定/需要大量数据 |
| 建议 | **先跑通 Level 1+2，再决定是否做 RL** | |

**推荐路径：** 先纯规则通 L1+L2，L3 用规则增强版试一试，不行再加 RL。

### 代码模块划分

```
dishwasher_project/
├── perception/              # 感知模块
│   ├── detector.py          # 物体检测（instance seg + depth clustering）
│   ├── pose_estimator.py    # 6D 位姿估计（ICP + PCA）
│   └── depth_completion.py  # 深度补全（透明/反光物体）
│
├── grasping/                # 抓取规划
│   ├── grasp_generator.py   # 基于物体类型的抓取点生成
│   ├── grasp_filter.py      # 碰撞/可达性/质量过滤
│   └── grasp_ranker.py      # 候选排序
│
├── motion/                  # 运动规划
│   ├── ik_controller.py     # Piper IK 控制器
│   └── trajectory_planner.py # 路径插值 + 碰撞规避
│
├── control/                 # 执行控制
│   ├── state_machine.py     # 任务状态机
│   └── gripper_control.py   # 夹爪控制
│
├── scene/                   # 场景管理
│   ├── scene_loader.py      # 加载赛方提供的 USD 场景
│   ├── domain_randomizer.py # 域随机化配置
│   └── camera_manager.py    # 相机配置与管理
│
├── eval/                    # 评估与交付
│   ├── run_level1.py        # Level 1 评估脚本
│   ├── run_level2.py        # Level 2 评估脚本
│   ├── run_level3.py        # Level 3 评估脚本
│   └── metrics.py           # 成功率/耗时/失败模式统计
│
└── docker/                  # Docker 打包
    ├── Dockerfile
    └── entrypoint.sh
```

---

## 五、工作空间结构

```
~/dishwasher_ws/                          # ← 项目根目录
│
├── README.md                             # 项目说明（比赛信息、环境配置、快速开始）
├── Makefile                              # 常用命令快捷入口
├── requirements.txt                      # Python 依赖
├── setup.py                              # pip install -e . 可编辑安装
│
├── assets/                               # 赛方提供的数字资产（只读）
│   └── isaac_dishwisher -> ../../下载/isaac/isaac_dishwisher/   # 软链接到赛方资料
│
├── src/dishwasher/                       # 核心代码包
│   ├── __init__.py
│   │
│   ├── scene/                            # —— 场景管理 ——
│   │   ├── __init__.py
│   │   ├── loader.py                     # 加载 all.usd + 初始化 Piper
│   │   ├── piper_cfg.py                  # Piper 机械臂 ArticulationCfg 配置
│   │   ├── camera_manager.py             # 多相机配置（俯视 + 手腕 RealSense）
│   │   └── domain_randomizer.py          # 域随机化（光照/纹理/颜色/物理）
│   │
│   ├── perception/                       # —— 感知模块 ——
│   │   ├── __init__.py
│   │   ├── detector.py                   # 物体检测 + 实例分割
│   │   ├── pose_estimator.py             # 6D 位姿估计（ICP + PCA）
│   │   ├── depth_completion.py           # 深度补全（透明/反光物体）
│   │   └── visualizer.py                 # 感知结果可视化（RGB/Depth/Seg/位姿）
│   │
│   ├── grasping/                         # —— 抓取规划 ——
│   │   ├── __init__.py
│   │   ├── generator.py                  # 基于物体类型的抓取姿态生成
│   │   ├── filter.py                     # 碰撞检测 + 可达性过滤
│   │   └── ranker.py                     # 候选排序
│   │
│   ├── motion/                           # —— 运动规划 ——
│   │   ├── __init__.py
│   │   ├── ik_controller.py              # Piper 6DOF Differential IK 控制器
│   │   └── trajectory.py                 # 路径插值 + 碰撞规避
│   │
│   ├── control/                          # —— 执行控制 ——
│   │   ├── __init__.py
│   │   ├── state_machine.py              # 任务状态机（IDLE→...→DONE）
│   │   ├── gripper.py                    # 夹爪控制（开/合/力控）
│   │   └── recovery.py                   # 异常恢复策略
│   │
│   └── utils/                            # —— 通用工具 ——
│       ├── __init__.py
│       ├── config.py                     # 全局配置管理
│       ├── metrics.py                    # 成功率/耗时/失败模式统计
│       └── logger.py                     # 日志与记录
│
├── scripts/                              # 运行脚本
│   ├── run_level1.py                     # Level 1 评估运行
│   ├── run_level2.py                     # Level 2 评估运行
│   ├── run_level3.py                     # Level 3 评估运行
│   ├── test_scene.py                     # 场景加载验证
│   ├── test_perception.py                # 感知模块验证
│   ├── test_grasp.py                     # 抓取模块验证
│   └── collect_demos.py                  # 遥操作/演示数据采集（RL 备选）
│
├── configs/                              # 配置文件
│   ├── default.yaml                      # 默认配置
│   ├── level1.yaml                       # Level 1 配置
│   ├── level2.yaml                       # Level 2 配置
│   └── level3.yaml                       # Level 3 配置
│
├── tests/                                # 单元测试
│   ├── test_perception.py
│   ├── test_grasping.py
│   ├── test_motion.py
│   └── test_state_machine.py
│
├── results/                              # 评估结果输出
│   ├── level1/                           # Level 1 100 次评估结果
│   ├── level2/                           # Level 2 100 次评估结果
│   └── level3/                           # Level 3 100 次评估结果
│
├── videos/                               # 演示视频
│   ├── level1_demo.mp4
│   ├── level2_demo.mp4
│   └── level3_demo.mp4
│
├── docs/                                 # 技术文档
│   ├── architecture.md                   # 方案架构文档
│   ├── perception.md                     # 感知模块设计
│   ├── evaluation_report.md              # 评估报告模板
│   └── submission_checklist.md           # 提交前检查清单
│
└── docker/                               # Docker 打包
    ├── Dockerfile
    ├── entrypoint.sh
    └── .dockerignore
```

### 开发周期里程碑

```
M0: 环境验证          M1: Level 1          M2: Level 2          M3: Level 3          M4: 交付
 ──●───────────────────●───────────────────●───────────────────●───────────────────●──►
  [Week 1]            [Week 2]            [Week 3-4]          [Week 5-6]          [Week 7]
  
  场景跑通             100% 成功率          95% 成功率           85% 成功率           Docker+视频+文档
  ├─ 原生 all.usd 校验 ├─ 规则化策略        ├─ 多模态融合感知     ├─ RL/IL 增强       ├─ Dockerfile
  ├─ Piper Articulation ├─ 固定抓取姿态     ├─ 启发式抓取        ├─ 深度补全         ├─ 录制视频
  ├─ 相机输出验证      ├─ 顺序放置          ├─ 域随机化          ├─ 主动感知         ├─ 技术文档
  ├─ 盘子物理验证      ├─ 状态机闭环        ├─ 碰撞检测          ├─ 力控抓取         ├─ 评估报告
  └─ IK 控制器         └─ 100 次评估        ├─ 重试恢复          ├─ 堆叠处理         └─ 提交检查
                                             └─ 100 次评估        └─ 100 次评估
```

**当前状态**: M0 已完成 ✅ → 进入 M1

### Git 分支策略

```
main
  ├── m0-env-setup        # 环境搭建 + 场景跑通 ✅
  ├── m1-level1           # Level 1 规则策略 → 合并后 tag: v1.0-level1
  ├── m2-level2           # Level 2 进阶策略 → 合并后 tag: v2.0-level2
  ├── m3-level3           # Level 3 挑战策略 → 合并后 tag: v3.0-level3
  └── m4-delivery         # Docker + 文档 + 视频 → tag: v4.0-submission
```

---

## 六、实施顺序（从零到完成）

### Step 1：场景跑通（1-2 天）✅ 已完成

- [x] 用 Python API 打开原生 all.usd，确认所有资产加载正常
- [x] 确认 Piper 关节能通过 PD 控制移动
- [x] 确认盘子模型的物理属性（质量、碰撞）正确
- [x] NativeSceneLoader 直接以赛方 all.usd 为仿真基座

### Step 2：感知模块（2-3 天）← 当前优先

**目标：** 能检测到水槽里的盘子，估算它们的位置和姿态

**交付物：** `detector.py` + `pose_estimator.py` — 输入相机数据，输出物体列表 `[{id, type, position, quaternion}]`

### Step 3：抓取 + 运动（2-3 天）

**目标：** 针对检测到的盘子生成抓取姿态，通过 IK 到达目标

**交付物：** `grasp_generator.py` + `ik_controller.py` — 输入物体位姿，输出机械臂能到达的抓取姿态

**当前 native 进展：** `scripts/run_level1_native.py` 已能在原生 `all.usd` 上完成 RigidObject 检测、规则抓取/放置目标生成、Piper IK smoke solve。

### Step 4：Level 1 闭环（2-3 天）

**目标：** 端到端跑通 Level 1 — 规则策略，100% 成功率

### Step 5：Level 2 升级（3-5 天）

**目标：** 增加多物体类型、随机位姿、域随机化、容错

### Step 6：Level 3 攻关（5-7 天）

**目标：** 处理透明/反光/堆叠，RL 增强或规则增强

### Step 7：交付打包（2-3 天）

**目标：** Docker 镜像、演示视频、技术文档、评估报告

---

## 七、最终提交什么

| 交付物 | 内容 | 格式 |
|--------|------|------|
| **Docker 镜像** | 完整可复现环境 | `.tar` 或 registry |
| **源码** | 所有模块代码 | Docker 内 `/workspace/` |
| **演示视频** | Level 1/2/3 各一个完整运行视频 | `.mp4` |
| **技术文档** | 方案架构、多模态融合设计、三级策略说明 | `.pdf` |
| **评估报告** | 每级 100 次评估统计（成功率/耗时/失败模式） | `.pdf` / `.csv` |

### 视频要求

演示视频需要多窗口同时展示：
- 左上：RGB 主视图（第三人称场景视角）
- 右上：深度图（JET 伪彩色）
- 左下：实例分割可视化（不同颜色掩码）
- 右下：任务状态面板（当前状态、目标物体、抓取候选、进度 N/3）
- 底部：实时成功率统计

---

## 八、关于 Piper 机械臂的关键信息

### 与 Franka 的对比

| | Franka Panda | Piper (AgileX) |
|---|---|---|
| 自由度 | 7 DOF | **6 DOF** |
| 构型 | 单臂 | **双臂**（赛方可能只用一臂） |
| 夹爪 | 2 指平行 | 额外关节（订阅 8 轴含夹爪） |
| 控制接口 | Isaac Lab IK 控制器 | ROS2 `/joint_command` 话题 |
| 相机 | 需外挂 | **自带 RealSense RGB-D** |

### Piper 在 Isaac Lab 中的使用方式

赛方提供了两套 Piper USD：
1. `piper_description_v100_realsense_camera_v2.usd` — 纯 model（无 ROS2）
2. `piper_ros2 .usd` — ROS2 集成版本（含 ActionGraph + ROS2 Bridge 节点）

**推荐方案：** 用 model 版本，在 Isaac Lab 中通过 Articulation 包装，绕过 ROS2。

好处：
- 不需要启动 ROS2 节点
- Isaac Lab 原生支持 DifferentialIKController（适配 6 DOF）
- 相机数据直接通过 Isaac Lab Camera Sensor 获取

**已实现**: `src/dishwasher/scene/piper_cfg.py` — PIPER_CFG ArticulationCfg（8 joints, PD gains）

---

## 九、需要确认的未知信息

| 问题 | 重要程度 | 影响 |
|------|----------|------|
| 提交截止日期？ | 🔴 关键 | 决定开发节奏 |
| Level 3 的具体物体列表？ | 🟡 重要 | 需要知道有哪些透明/反光物体 |
| 评分标准细节？（成功率权重？时间权重？感知模态权重？） | 🟡 重要 | 影响方案权衡 |
| 是否一定要用 ROS2？还是可以纯 Isaac Lab？ | 🟡 重要 | 决定架构选择 |
| 洗碗机卡槽数量/布局？ | 🟢 一般 | 影响放置策略 |
| 评估是在赛方环境还是提交 Docker 自评？ | 🔴 关键 | 决定交付形式 |

---

## 十、M0 技术笔记

### 已验证的技术模式

**PD 控制模式**:
```python
p.set_joint_position_target(target)  # 设置目标
p.write_data_to_sim()                # 写入 PhysX
sim.step()                           # 推进物理
p.update(dt)                         # 读取状态
```
注意：必须使用 `write_data_to_sim()` 而非 `write_joint_position_to_sim()`（后者是 teleport 模式）。

**Plate 物理修复**:
递归清理子 prim 上的物理 API → 仅 root prim 应用 RigidBody + Collision + Mass API。

**NativeSceneLoader 模式**:
直接打开赛方 `all.usd` 作为仿真基座；不重新 spawn 桌子、盘子、Piper，不做 `Z_SHIFT`，不改原始物理 API。直接 Isaac Lab 控制时仅运行时临时禁用嵌套 `PhysicsScene` 和 ROS2 `ActionGraph`，不保存源 USD。

### 已知限制

- **RTX 5070 12GB**: GUI + 完整 PhysX 场景 CUDA OOM。Headless 物理正常。A40 48GB 可解决。
- **原生 all.usd 单位**: `metersPerUnit=0.01`，张量读数按 stage units 为厘米尺度；后续规划/控制需要统一处理单位。
- **Isaac Sim 5.1 截图 API**: `omni.syntheticdata`、`omni.replicator`、`renderer.capture` 均不可靠。
- **USD 导出**: `stage.Export()` 只捕获 USD prim transform，不捕获 PhysX Fabric 张量状态。
