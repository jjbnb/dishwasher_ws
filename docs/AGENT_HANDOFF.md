# Agent Handoff: Native `all.usd` Baseline And Camera Observation Work

Last updated: 2026-07-11

This document is for the next agent/conversation. It summarizes the current
project state, decisions, verified commands, known traps, and the recommended
next steps.

## Current Repository State

- Branch: `main`
- Remote: `origin https://github.com/jjbnb/dishwasher_ws.git`
- Latest pushed commit at handoff time:

```text
2c19e90 feat: add camera-guided observation planning
```

- `main` is synced with `origin/main`.
- There are still local untracked diagnostic/experimental files that were
  deliberately not committed:

```text
scripts/analyze_all_usd.py
scripts/analyze_world_positions.py
scripts/check_allusd.py
scripts/check_allusd2.py
scripts/demo_level1_gui.py
scripts/demo_level1_native_pregrasp_gui.py
scripts/diag_scene.py
scripts/discover_piper.py
scripts/find_place_position.py
scripts/inspect_scene.py
scripts/inspect_scene_v2.py
scripts/minimal_test.py
scripts/run_level1.py
scripts/sample_native_reachability.py
scripts/test_ik.py
src/dishwasher/utils/config.py
```

Do not delete or overwrite these without checking with the user; they may be
useful scratch work.

## Main Decisions

The project is now based on the competition-provided native USD scene:

```text
assets/isaac_dishwisher/all.usd
```

Avoid the old M0 pattern of reconstructing the scene manually. In particular:

- Do not respawn the table, sink, rack, Piper arms, or plates.
- Do not apply `Z_SHIFT`.
- Do not manually divide coordinates by 100 and rebuild a meter-stage scene.
- Do not move plate physics APIs from their native prims.
- Do not add replacement collision cuboids unless there is a very explicit,
  justified runtime-only experiment.

The source `all.usd` should remain the baseline and should not be saved after
runtime changes.

The project also now has an explicit official-first development rule in:

```text
agent.md
```

This was added after debugging the poor observation posture and custom IK
behavior. It says:

- Prefer official / asset-provided ROS2, ActionGraph, controller, and example
  interfaces before writing custom control wrappers.
- Treat custom Isaac Lab IK or direct joint control as experimental fallback
  unless the official path has been inspected and rejected with evidence.
- Do not conclude that a pose is unreachable only because the current custom
  IK/control chain failed.
- Keep the native USD and official asset structure intact.

This matters because the current poor "lying down to look" camera behavior is
not solved by more blind joint-angle tweaking. The next control/planning work
should validate the official controller path before making strong reachability
claims.

## 2026-07-11 Recent Progress

Committed and pushed:

```text
2c19e90 feat: add camera-guided observation planning
```

Added camera and perception plumbing:

```text
src/dishwasher/perception/native_camera.py
src/dishwasher/perception/camera_plate_detector.py
scripts/test_camera_plate_detection.py
```

These modules read the native wrist cameras and use RGB-D / semantic-style
signals to produce plate observations. The intent is to stop hard-coding plate
positions from rigid body tensors as the main task path.

Added active observation and camera-aware planning modules:

```text
src/dishwasher/planning/active_view_planner.py
src/dishwasher/planning/camera_view_planner.py
src/dishwasher/planning/grasp_planner.py
scripts/sample_camera_view_reachability.py
scripts/test_level1_native_pregrasp.py
```

The current Level 1 test can run both wrist cameras, score candidate
observations, fuse detections, and generate rim-grasp candidates. A short
headless validation selected a documented joint profile such as
`joint_pair_context_wider` and produced detections from both cameras.

Important limitation: this is not a final solution. The user correctly pointed
out that the arm still tends to observe from a low, awkward, collision-prone
posture. Detection scores are low largely because the camera pose is bad:
too close, too low, insufficient top-down context, and sometimes missing the
sink / full plate semantic scene. The current joint-profile fallback is better
documented than arbitrary deltas, but it is still not the principled planner
the project needs.

Current diagnosis:

- The main blocker is observation pose planning and control, not only plate
  segmentation.
- The end-effector depth cameras should be used for perception; rigid body
  plate positions may be used for validation/debugging, not as the main answer.
- Both left and right wrist cameras should participate. A single left-camera
  demo is not enough for the final task.
- The next serious improvement is to inspect and use the official ROS2 /
  ActionGraph / asset-provided control path, then re-test reachable observation
  poses with collision-aware constraints.

Useful ignored output directories from recent runs:

```text
outputs/joint_strategy_profile_test/
outputs/solve_prone_default_pass/
outputs/gui_prone_fix_demo/
```

These are intentionally ignored by Git and should not be committed.

## Important Native USD Facts

Native `all.usd` metadata:

```text
metersPerUnit = 0.01
upAxis = Z
defaultPrim = /World
```

Isaac Lab tensor positions read in stage units, which are centimeters for this
stage. For example, the left Piper root reads approximately:

```text
stage=(90.000, 137.000, 14.450)
meters=(0.900, 1.370, 0.144)
```

Key native prim paths:

```text
Left Piper root:          /World/piper_ros2_
Left Piper articulation:  /World/piper_ros2_/piper_camera
Left wrist camera:        /World/piper_ros2_/piper_camera/link6/d435_camera_link/Camera

Right Piper root:         /World/piper_ros2__04
Right Piper articulation: /World/piper_ros2__04/piper_camera
Right wrist camera:       /World/piper_ros2__04/piper_camera/link6/d435_camera_link/Camera

Scene camera:             /World/Camera
Sink mesh:                /World/dishwasher_desk_1_/洗碗机水槽/洗碗机水槽
Rack mesh:                /World/dishwasher_desk_1_/dishwasher_basin_step/dishwasher_basin_step

Plate body paths:
  plate_1:                /World/plate_1/plate/plate
  plate_02:               /World/plate_02/plate/plate
  plate_03:               /World/plate_03/plate/plate
```

The native file contains nested `PhysicsScene` prims and ROS2 `ActionGraph`
nodes. For direct Isaac Lab control, the current scripts disable these at
runtime only:

```text
/World/piper_ros2_/physicsScene
/World/piper_ros2__04/physicsScene
/World/piper_ros2_/ActionGraph
/World/piper_ros2__04/ActionGraph
```

This is not saved to the source USD.

## Implemented Files And Purpose

### Native Scene Loading

```text
src/dishwasher/scene/native_loader.py
```

Provides:

- native prim path constants
- `NativeSceneLoader.validate()`
- `NativeSceneLoader.open_stage()`
- runtime-only helper deactivation
- native Isaac Lab wrapping for Piper and plates

### M0 Native Validation And Demos

```text
scripts/validate_all_usd.py
scripts/test_m0_verify.py
scripts/demo_m0_gui.py
scripts/demo_m0_render.py
scripts/view_all_usd.py
scripts/test_piper_articulation.py
scripts/test_plate_physics.py
```

These now use native `all.usd`, not the old reconstructed scene.

`demo_m0_gui.py` and `view_all_usd.py` were fixed to avoid black viewport:

- keep calling `simulation_app.update()`
- create/use runtime overview camera `/World/M0_OverviewCamera`
- default to GPU through Isaac Lab AppLauncher

### Level 1 Native Foundation

```text
scripts/run_level1_native.py
src/dishwasher/grasping/generator.py
src/dishwasher/control/pipeline.py
```

`run_level1_native.py` is the current safe Level 1 entry point. Default mode
does a smoke test only:

```text
native all.usd -> wrap assets -> detect plate -> generate grasp/place -> IK solve
```

It does not execute the full pick/place state machine unless `--execute` is
explicitly passed.

`generator.py` now supports `unit_scale`. Use:

```text
unit_scale=100.0
```

for native `all.usd` because stage units are centimeters.

`pipeline.py` now accepts:

```text
unit_scale
place_position
```

so the old state machine can eventually run against the native scene.

### Legacy Loader

```text
src/dishwasher/scene/loader.py
```

This is now explicitly marked legacy. It still exists so older experiments do
not break, but it should not be used as the M0/M1 baseline.

## Verified Commands

Run inside the Isaac Lab conda environment unless noted otherwise:

```bash
conda activate env_isaaclab
```

Pure Python syntax check:

```bash
python -m compileall -q src scripts/validate_all_usd.py scripts/test_scene.py scripts/test_m0_verify.py scripts/test_piper_articulation.py scripts/test_plate_physics.py scripts/demo_m0_render.py scripts/demo_m0_gui.py scripts/view_all_usd.py scripts/run_level1_native.py
```

Native USD validation:

```bash
python scripts/validate_all_usd.py
```

Observed result:

```text
metersPerUnit:    0.01
upAxis:           Z
defaultPrim:      /World
total prims:      359
cameras:          6
physics scenes:   3
articulationRoot: 2
rigid bodies:     26
collisions:       17
result: PASS
```

M0 native headless verification:

```bash
python scripts/test_m0_verify.py --headless
```

Observed result:

```text
M0 native verification: 23/23 passed
```

GUI native scene:

```bash
python scripts/demo_m0_gui.py --seconds 120
```

Observed:

```text
Using device: cuda:0
viewport camera: /World/M0_OverviewCamera
opened: .../assets/isaac_dishwisher/all.usd
exit code 0
```

Level 1 native smoke:

```bash
python -u scripts/run_level1_native.py --headless --num_plates 1
```

Observed:

```text
plates selected: ['plate_02']
plate_02: stage=(77.320, 97.606, -0.464) meters=(0.773, 0.976, -0.005)
pre_grasp: stage=(77.320, 97.606, 9.536) meters=(0.773, 0.976, 0.095)
place: stage=(113.555, 74.844, -10.436) meters=(1.136, 0.748, -0.104)
IK finite: True
```

Known nuisance: some Isaac scripts complete their main work but hang during
`simulation_app.close()`. It has been safe to interrupt after the useful output
is printed. Do not confuse this with failure of the actual smoke test.

## Current Technical Position

Completed:

- Native `all.usd` validation.
- Native GUI viewing with GPU and non-black viewport.
- Native M0 headless smoke/verification.
- Native Piper/plate wrapping.
- Native Level 1 detect/plan/IK smoke.
- Native wrist camera capture path and simple plate observation pipeline.
- Two-camera observation scoring/fusion path for plate candidates.
- Rim-grasp candidate generation from camera/planning outputs.
- Documented joint-profile fallback for camera observation tests.

Not completed:

- Full L1 physical pick-and-place.
- Reliable gripper contact/plate retention.
- Robust top-down / context-rich wrist-camera observation pose.
- Official ROS2 / ActionGraph / asset-provided controller validation.
- Collision-aware arm motion to observation and grasp-preparation poses.
- True placement verification in the rack.
- A final policy that looks like it is "thinking" instead of getting stuck in
  low folded postures.

## Recommended Next Step

Do not jump straight to full `--execute` pick-and-place.

The earlier next step, `scripts/test_level1_native_pregrasp.py`, has already
been created and expanded. Do not spend time recreating it.

The next best task is now:

```text
validate the official control path, then plan better observation poses
```

Recommended sequence:

1. Inspect the native `all.usd` ROS2 `ActionGraph`, controller nodes, and any
   asset-provided Piper examples.
2. Identify whether the official path exposes joint targets, Cartesian targets,
   gripper commands, camera topics, or MoveIt-style planning hooks.
3. Build a minimal official-control smoke test that commands one Piper to a
   safe high observation posture without direct custom IK.
4. Validate with both wrist cameras that the plate and sink are visible with
   useful context.
5. Only after that, compare the custom `PiperIKController` fallback against
   the official path.
6. Add collision / clearance checks before descending toward grasp poses.

Observation pose requirements from the user:

- The plate should be near the camera image center.
- The view should include the full plate and enough sink/workspace context for
  semantic understanding.
- The camera should not be pressed close to the plate.
- The arm should not lie across the table or collide with scene geometry.
- The observation posture should preserve options for the next grasp and
  handoff/transfer step, not only maximize immediate detection score.

## RL / IL Decision Context

User asked whether L1 should be solved directly with RL/IL.

Current recommendation:

```text
L1: rules + IK baseline
L2: rules + perception + retries
L3: consider IL/RL as enhancement
```

Reason:

- L1 is simple and deterministic; IK is faster and easier to debug.
- L1 builds the infrastructure needed by later RL/IL anyway.
- Successful L1 trajectories can later become imitation-learning demos.

Do not replace L1 with RL/IL yet.

## Important User Preferences

- Prefer the native competition `all.usd`.
- Prefer official / asset-provided control functions before custom IK or
  manual joint-angle strategies.
- Avoid hidden scene reconstruction.
- Prefer GPU for simulation.
- Keep source USD unmodified.
- Be explicit about runtime-only changes.
- The user values careful verification over pretending the full pipeline works.
- The user does not want repeated GUI demos unless they are necessary and show
  concrete progress.

## 2026-07-08 Follow-up: Native Pre-grasp Diagnostics

Added:

```text
scripts/test_level1_native_pregrasp.py
scripts/sample_native_reachability.py
```

Updated:

```text
src/dishwasher/motion/ik_controller.py
src/dishwasher/control/pipeline.py
scripts/run_level1_native.py
```

Key IK controller changes are backwards-compatible defaults plus native-use
options:

- `position_scale`: native `all.usd` should use `0.01` because body poses are
  read in centimeters while the PhysX Jacobian behaves meter-scaled.
- `command_type`: can be `"pose"` or `"position"`.
- `delta_gain`: scales each IK joint delta for safer differential-IK stepping.
- Jacobian body row is now selected from actual Jacobian shape instead of
  always assuming a fixed-base `body_idx - 1`.

Verified commands:

```bash
python -m compileall -q src/dishwasher/motion/ik_controller.py src/dishwasher/control/pipeline.py scripts/run_level1_native.py scripts/test_level1_native_pregrasp.py scripts/sample_native_reachability.py
```

Native smoke still reaches the useful output:

```bash
conda run --no-capture-output -n env_isaaclab python -u scripts/run_level1_native.py --headless --num_plates 1
```

Observed:

```text
IK finite: True
joint target: 0.136, -0.260, 0.536, -0.362, -1.759, 0.357
```

It still hangs during `simulation_app.close()` and was interrupted after the
useful output, matching the known shutdown nuisance.

Important finding:

Direct differential IK from the native default posture to plate_02 pre-grasp
does not produce a stable pre-grasp. Even with position-only IK, interpolated
waypoints, lower IK delta gain, and extra `link6` Z offset, the arm drifts into
a low folded configuration and final `link6` error remains tens of centimeters.

Reachability sampling shows the target itself is reachable:

```bash
conda run --no-capture-output -n env_isaaclab python -u scripts/sample_native_reachability.py --headless --skip-close --samples 5000 --ee-z-offset 32 --seed 11
```

Observed best sample:

```text
target link6: stage=(77.320, 97.606, 41.536)
link6:        stage=(80.193, 100.718, 41.868)
error:        4.248 stage units (0.0425 m)
q:            -0.268, 1.641, -1.108, -0.007, 0.006, -2.970
```

This means the next best step is not more blind DifferentialIK tuning. Use the
sampled/reachable joint-space seed as a pre-grasp target or add a proper global
IK / joint-space planning stage before descending. Also account for the fact
that `PiperIKController` controls `link6`, not the plate contact point; native
pre-grasp for `link6` needed about `+32cm` extra Z over the generated plate
pre-grasp to keep the wrist at the original safe height.
