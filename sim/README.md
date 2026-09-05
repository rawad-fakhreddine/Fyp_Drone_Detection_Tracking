# Simulation assets (Gazebo Classic 11 / PX4 SITL)

Custom Gazebo models and worlds used by this project (dual-drone chaser/target SITL).
**SDF/world/config only** — the binary meshes (`iris.stl`, prop `.dae`) are the unmodified
standard PX4 iris meshes and are intentionally not tracked here; they ship with PX4.

## Install paths

Copy each model directory into the PX4 Gazebo model path, and the worlds into the world path:

```bash
PX4=~/PX4-Autopilot/Tools/sitl_gazebo
cp -r sim/models/iris_chaser                 $PX4/models/
cp -r sim/models/iris_chaser_nolockstep      $PX4/models/
cp -r sim/models/target_iris_sitl            $PX4/models/
cp -r sim/models/target_iris_sitl_nolockstep $PX4/models/
cp -r sim/models/iris_fpv_cam_nolockstep     $PX4/models/
cp    sim/worlds/rl_empty.world              $PX4/worlds/
cp    sim/worlds/baylands.world              $PX4/worlds/
# spectator_cam.sdf -> catkin_ws/src/drone_tracking/models/
```

The mesh files must exist under each model's `meshes/` folder for Gazebo to render the drone.
Reuse the standard PX4 iris meshes (from `$PX4/models/iris/meshes/`) — the chaser/target
models are the standard iris airframe with an added FPV camera + per-role tweaks in the SDF.

## Files

| File | Role |
|---|---|
| `models/iris_chaser/iris.sdf` | Chaser drone (FPV camera, lockstep) |
| `models/iris_chaser_nolockstep/iris.sdf` | Chaser, `enable_lockstep=false` (RL 4× speedup path) |
| `models/target_iris_sitl/iris.sdf` | Target drone (lockstep) |
| `models/target_iris_sitl_nolockstep/iris.sdf` | Target, `enable_lockstep=false` |
| `models/iris_fpv_cam_nolockstep/iris_fpv_cam_nolockstep.sdf` | FPV camera model (nolockstep) |
| `models/spectator_cam.sdf` | Third-person spectator camera (recording/multiview) |
| `worlds/rl_empty.world` | Flat empty world for RL training (`real_time_factor=4`) |
| `worlds/baylands.world` | Baylands world used for evaluation |
