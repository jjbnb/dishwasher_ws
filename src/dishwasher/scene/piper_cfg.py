"""Piper 机械臂 ArticulationCfg 配置。

基于赛方 piper_description_v100_realsense_camera_v2.usd 模型，
通过 ArticulationCfg 包装以在 Isaac Lab 中实现关节控制。

关键发现：
- root_joint (PhysicsFixedJoint, ArticulationRootAPI) 的 body0 为空 → 连接 World
- 使用 set_joint_position_target + write_data_to_sim 实现 PD 控制
- write_joint_position_to_sim 是"瞬移"模式，不适合力控
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

PIPER_CFG = ArticulationCfg(
    prim_path="/World/Piper",
    spawn=None,  # 由 loader.py 预先 spawn USD
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "joint[1-6]": 0.0,   # 6 个臂关节
            "joint[7-8]": 0.0,   # 2 个夹爪指关节
        },
    ),
    actuators={
        "piper_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-6]"],
            effort_limit_sim=100.0,   # URDF: effort=100
            velocity_limit_sim=5.0,   # URDF: velocity=5
            stiffness=800.0,
            damping=80.0,
        ),
        "piper_gripper": ImplicitActuatorCfg(
            joint_names_expr=["joint[7-8]"],
            effort_limit_sim=10.0,    # URDF: effort=10
            velocity_limit_sim=1.0,   # URDF: velocity=1
            stiffness=500.0,
            damping=50.0,
        ),
    },
)
"""Piper 6-DOF 机械臂 + 2-DOF 夹爪配置。

关节名（USD 中）：
- joint1-joint6: revolute 臂关节
- joint7: prismatic 左指 (0 ~ 0.035)
- joint8: prismatic 右指 (-0.035 ~ 0)

用法：
    from dishwasher.scene.piper_cfg import PIPER_CFG
    piper_cfg = PIPER_CFG.copy()
    piper_cfg.init_state.pos = (x, y, z)
    piper = Articulation(cfg=piper_cfg)
"""
