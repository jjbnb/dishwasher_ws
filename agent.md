# Agent Development Policy

## Primary Strategy: Official First

This project must prefer official, competition-provided, and asset-provided
interfaces before adding custom control or planning layers.

For this dishwasher workspace, "official first" means:

- Use the competition-provided native `all.usd` scene as the source of truth.
- Use the Piper/RealSense assets, prim paths, ROS2 interfaces, ActionGraphs,
  and example control code provided with the assets before writing replacement
  controllers.
- Prefer official Isaac Sim / Isaac Lab / ROS2 APIs over ad-hoc wrappers.
- Treat custom IK, custom joint targets, runtime camera offsets, and direct
  articulation control as experimental fallbacks, not as proof that the robot or
  asset cannot perform a motion.
- Do not conclude that a target is unreachable until it has been tested through
  the official or asset-provided control path, or until there is a documented
  collision/reachability analysis explaining why that path cannot be used.

## Required Order Of Work

When a task touches robot motion, camera observation, perception, grasping, or
scene behavior, agents must follow this order:

1. Inspect the native asset-provided mechanism first.
2. Verify whether ROS2 topics, ActionGraphs, controllers, or example scripts
   already provide the requested behavior.
3. Use official APIs and asset-provided control paths for the baseline
   implementation.
4. Add custom wrappers only when the official path is missing, broken, or
   explicitly unsuitable.
5. Clearly label any custom wrapper as an experiment or fallback.
6. Validate custom behavior against the official path before using it to make
   design conclusions.

## Motion And Observation Rules

- Observation pose failures must be treated first as planning/control-chain
  failures, not perception failures.
- Low detection confidence from wrist cameras should trigger official-control
  observation replanning before changing perception heuristics.
- Joint-angle profiles are allowed only as documented, reproducible candidates
  generated from reachability or official-control tests. They must not be used
  as unexplained magic poses.
- If an IK target fails under a custom controller, report it as:
  "failed under the current custom control chain."
  Do not report it as physically unreachable unless verified through the
  official path or a proper reachability/collision analysis.

## Native Asset Safety

- Do not rewrite or save the source `all.usd` after runtime experiments.
- Runtime-only changes are allowed for diagnosis, but they must be clearly
  marked and reversible.
- Do not respawn or replace the table, sink, rack, plates, Piper arms, or wrist
  cameras unless the user explicitly asks for that experiment.

## Reporting Standard

When reporting results, distinguish these cases explicitly:

- Official-control result
- Direct Isaac Lab result
- Custom IK result
- Runtime-only camera or scene modification
- Rigid-body ground-truth debug result
- Real camera RGB-D / segmentation result

This prevents experimental shortcuts from being mistaken for project facts.
