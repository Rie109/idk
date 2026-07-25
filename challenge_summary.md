# AI Challenge Racing Kart - Complete Implementation Summary

This document provides a comprehensive summary of the issues encountered, architectural improvements designed, files modified, and verification methods implemented during the racing kart challenge.

---

## 1. Challenge Overview & Goals
The objective of this challenge is to safely navigate a racing kart through a highly winding, multi-lobe track at a target speed of **30 km/h** without hitting walls or losing control.

---

## 2. Diagnosed Issues & Root Causes

### A. Sensor/Filter Lag
* **Problem**: The vehicle oscillated and collided with barriers during high-speed cornering.
* **Root Cause**: The cascading 1D Kalman filter and EMA filter in `gyro_odometer` were configured with heavy smoothing (`vx_ema_alpha = 0.3`, `wz_ema_alpha = 0.6`). This introduced a feedback delay, causing control commands to lag behind the physical vehicle state.

### B. Waypoint Kidnapping / Lobe Jumping
* **Problem**: The kart would get stuck against the inner barriers after a crash.
* **Root Cause**: The track features a tight spiral layout where inbound and outbound lanes sit within just a few meters of each other. A global fallback search or a wide index search window incorrectly snapped the vehicle to the adjacent lane through the wall. This caused the MPC to compute a prediction horizon that cut straight through the wall, pushing the vehicle into the barrier.

### C. Spurious Lap Boundary Reset
* **Problem**: The vehicle lost tracking stability or jumped to incorrect lobes when passing the starting line.
* **Root Cause**: The re-anchor trigger condition was set to `if is_reset or self.wp_id == 0:`. Since `wp_id` legitimately wraps around to `0` once per lap, this incorrectly fired a global waypoint search at the starting line every lap, reintroducing the lobe-jumping bug.

### D. Flat Speed Profiling
* **Problem**: The vehicle regularly crashed at the entrance of sharp curves.
* **Root Cause**: The reference velocity profile was assigned as a flat, constant speed across whole sections of the track. Because the velocity limit dropped instantly when entering a curve section, the vehicle did not slow down in advance, entering corners too fast and slamming the brakes too late.

### E. Static Search Window Mismatch
* **Problem**: At high speeds, waypoint tracking would fall behind the actual position, causing steering errors.
* **Root Cause**: The waypoint search window was defined as a static size. When traveling at 30 km/h, the distance covered per control loop exceeded the search window's span, causing the tracker to fall behind.

### F. Uncaught ValueError silently freezing the control loop
* **Problem**: The vehicle bumped/recovered on Lap 1 but stalled completely (0 km/h, clean halt) on Lap 2.
* **Root Cause**: Any `ValueError` raised by `self.optimizer.solve()` propagated uncaught because `except TypeError or ValueError:` only caught `TypeError`. The control thread crashed/silently froze, keeping the last command active indefinitely.

### G. Spurious Infeasibility Retry Loop
* **Problem**: The safety margin relaxation retry loop was incorrectly triggered continuously.
* **Root Cause**: The solver checked feasibility via `if not np.all(use_control_signals):`, treating exactly `0.0` steering angle (straightaways) as falsy/infeasible.

### H. Waypoint ID Accumulation Drift
* **Problem**: Waypoint indexes drifted far ahead of the kart's physical position.
* **Root Cause**: When the spurious infeasibility check triggered the relaxation retry loop, it called `_init_problem()` multiple times in a single control cycle. Since `_init_problem()` increments `self.model.wp_id` by `self.wp_id_offset` on each call without resetting, the reference indexes drifted rapidly forward, making the solver completely infeasible on Lap 2.

### I. Low-Speed Deadlock on Sharp Curves
* **Problem**: The vehicle would sometimes get stuck at 0 km/h in sharp hairpins/spirals with no prediction trail.
* **Root Cause**: At zero or low velocities, the coordinate frames of future waypoints rotate rapidly. Because the vehicle's position is slightly off the centerline, the projection `e_y` exceeds the track boundaries. Since the vehicle cannot move fast enough at `v = 0` to reach the feasible bounds within the horizon $N$, the solver remains permanently infeasible, deadlocking the vehicle at `0.0` speed.

---

## 3. Implemented Architectural Solutions

### Filter Smoothing Optimization (C++)
* Tuned the filters in [gyro_odometer_core.cpp](file:///aichallenge/workspace/src/aichallenge_submit/gyro_odometer/src/gyro_odometer_core.cpp) to moderate values (`vx_ema_alpha = 0.55`, `wz_ema_alpha = 0.75`). This filters high-frequency sensor noise while keeping state-estimation latency low, eliminating control feedback delays.

### Curvature-Aware Speed Profile (Python)
* **Radius Estimation**: Added `_estimate_local_radius(self, idx, window=3)` to calculate local track curvature using three circumscribed waypoints.
* **Friction Limits**: Incorporated a `friction_coefficient` ($\mu=0.9$) parameter to dynamically limit target cornering speeds:
  $$v_{max\_dynamic} = \sqrt{\mu g R}$$
* **Forward/Backward Smoothing**: Implemented `_smooth_and_clip_speed_profile(self)` using kinematics equations:
  - Backward pass (braking): $v_{prev} = \sqrt{v_{next}^2 + 2 |a_{min}| d}$
  - Forward pass (accelerating): $v_{next} = \sqrt{v_{prev}^2 + 2 a_{max} d}$
  The vehicle now automatically and smoothly decelerates before entering curves and accelerates out of them.

### Speed-Scaling Dynamic Search Window (Python)
* Scaled the local search window dynamically inside `get_closest_waypoint`:
  $$\text{dynamic\_window} = \max\left(15, \text{int}\left(\frac{\text{speed} \times dt}{\text{avg\_wp\_spacing}}\right) + 10\right)$$
  This guarantees that the index tracker never falls behind, regardless of velocity.

### Protected Lap Transitions (Python)
* Added a dedicated `_wp_id_initialized` state boolean to the vehicle model. The re-anchoring search only fires at startup or on simulator teleport, preventing incorrect resets at the starting line (`wp_id == 0`).

### Blended Raceline Tracking (Python)
* Blended the reference lateral coordinate (`e_y`) between the optimal raceline ($0.0$) and the corridor centerline:
  $$e_{y\_ref} = \text{blend\_ratio} \times 0.0 + (1.0 - \text{blend\_ratio}) \times \left(\frac{lb + ub}{2}\right)$$
  Using a ratio of `0.65` allows the vehicle to cut corners for speed while maintaining a safe distance from the boundaries.

### Solver and Error Recovery Fixes (Python)
* Catching both exceptions: Changed `except TypeError or ValueError:` to `except (TypeError, ValueError):` inside `get_control()`.
* Checking status directly: Replaced the steering check with validation of `status != 'solved'` and `dec.x is None`.
* Caching base index: Cached `base_wp_id` at the start of `get_control()`, and restored it before each retry call inside the safety margin relaxation loop.

### Active Fallback Recovery Control (Python)
* Guided Recovery: Replaced the static zero-output fallback in `MPC.py` with an active controller. When the solver is infeasible, it commands `1.5 m/s` velocity and steers towards the centerline waypoints (`wp_id + 3`). This drives the vehicle back inside the track boundaries, allowing the MPC solver to resume normal operation.

---

## 4. Summary of Modified Files

* [gyro_odometer_core.cpp](file:///home/harry_ngyx/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/gyro_odometer/src/gyro_odometer_core.cpp)
  - Tuned vx/wz Kalman noise bounds and EMA alpha coefficients for smooth, low-latency filtering.
* [spatial_bicycle_models.py](file:///home/harry_ngyx/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/multi_purpose_mpc_ros/core/spatial_bicycle_models.py)
  - Added circumradius-based local radius estimation.
  - Implemented dynamic, speed-scaled waypoint search windowing.
  - Integrated `_wp_id_initialized` to isolate global re-anchoring to resets/cold starts.
* [mpc_controller.py](file:///home/harry_ngyx/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/multi_purpose_mpc_ros/mpc_controller.py)
  - Added `_update_waypoint_velocities` and kinematic profile smoothing.
  - Integrated `friction_coefficient` parameters and updated parameter callbacks.
  - Forwarded measured velocity and loop execution periods into `update_states`.
* [MPC.py](file:///home/harry_ngyx/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/multi_purpose_mpc_ros/core/MPC.py)
  - Restructured `get_control()` to cache and restore `base_wp_id` on retries, perform correct solve status validation, and catch `ValueError` solver exceptions.
  - Added active waypoint steering and velocity recovery fallback inside the fallback exception handler.
  - Integrated `raceline_blend_ratio` parameter and added lateral reference state blending in `_init_problem`.
  - Cleared stale predictions (`self.current_prediction = None`) on optimizer failure.
* [config.yaml](file:///home/harry_ngyx/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/config/config.yaml) & [sim_config.yaml](file:///home/harry_ngyx/aichallenge-racingkart/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros/config/sim_config.yaml)
  - Reverted horizon `N` to `20`.
  - Added `raceline_blend_ratio: 0.65` and `friction_coefficient: 0.9`.
  - Adjusted acceleration bounds (`a_max: 0.8`, `a_min: -1.6` $m/s^2$) for smooth speed profiles.

---

## 5. How to Run & Verify

1. **Rebuild the Workspace**:
   ```bash
   make autoware-build
   ```
2. **Start simulator and Autoware**:
   ```bash
   make dev
   ```
3. **Trigger Initial Pose & Control Mode**:
   *(Handled automatically by the background autostart orchestrator, or manually via):*
   ```bash
   make autoware-request-initialpose
   make autoware-request-control
   ```
4. **Monitor Trajectory & Log outputs**:
   - The vehicle should proactively decelerate before hairpins and accelerate on straightaways.
   - The MPC predictions (blue dots in RViz) should remain locked to the active lane.
   - Stop/reset commands should hide/clear prediction trails cleanly.
