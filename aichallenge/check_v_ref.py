#!/usr/bin/env python3
import sys
import os

# Add paths to sys
workspace_path = "/aichallenge/workspace/src/aichallenge_submit/multi_purpose_mpc_ros"
sys.path.append(workspace_path)
sys.path.append(os.path.join(workspace_path, "multi_purpose_mpc_ros"))

import numpy as np
import yaml
from core.map import Map
from core.reference_path import ReferencePath
from core.utils import load_ref_path, kmh_to_m_per_sec, m_per_sec_to_kmh

# Load config
config_path = os.path.join(workspace_path, "config/config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Set share path
share_path = "/aichallenge/workspace/install/multi_purpose_mpc_ros/share/multi_purpose_mpc_ros"

# Load map
map_yaml_path = os.path.join(share_path, config['map']['yaml_path'])
print("Map path:", map_yaml_path)
m = Map(map_yaml_path)

# Load reference path
csv_path = os.path.join(share_path, config['reference_path']['csv_path'])
print("CSV path:", csv_path)
wp_x, wp_y, csv_psi, csv_kappa = load_ref_path(csv_path)

ref_path = ReferencePath(
    m,
    wp_x,
    wp_y,
    config['reference_path']['resolution'],
    config['reference_path']['smoothing_distance'],
    config['reference_path']['max_width'],
    config['reference_path']['circular']
)

# Populate ref_vel configulator
# Let's get ref_vel config
ref_vel_cfg_path = os.path.join(workspace_path, "config/ref_vel.yaml")
if os.path.exists(ref_vel_cfg_path):
    print("Found ref_vel.yaml")
    with open(ref_vel_cfg_path, 'r') as f:
        ref_vel_data = yaml.safe_load(f)
else:
    print("ref_vel.yaml not found!")
    ref_vel_data = None

# Mimic ReferenceVelocityConfigulator
class RefVelConfigulator:
    def __init__(self, data):
        self.points = []
        if data and 'ref_vel_configulator' in data:
            cfg = data['ref_vel_configulator']
            for name, val in cfg.items():
                self.points.append((val['wp_id'], val['ref_vel']))
        self.points.sort()

    def get_ref_vel(self, wp_id):
        if not self.points:
            return 30.0
        # find the section
        current_val = self.points[-1][1]
        for p_id, val in self.points:
            if wp_id >= p_id:
                current_val = val
            else:
                break
        return current_val

configulator = RefVelConfigulator(ref_vel_data)

ay_max = config['mpc']['ay_max'] * 0.75
v_max = config['mpc']['v_max']
v_ref = []
for i, wp in enumerate(ref_path.waypoints):
    ref_vel_kmph = configulator.get_ref_vel(i)
    section_speed_ms = min(
        kmh_to_m_per_sec(ref_vel_kmph),
        kmh_to_m_per_sec(v_max)
    )
    kappa_abs = abs(wp.kappa) if wp.kappa is not None else 0.0
    v_max_kappa = np.sqrt(ay_max / (kappa_abs + 1e-9))
    v_ref.append(min(section_speed_ms, v_max_kappa))

ref_path.set_v_ref(v_ref)

# Let's perform smooth and clip (circular wrap-around matching mpc_controller.py)
waypoints = ref_path.waypoints
n = len(waypoints)
max_accel = config['mpc']['a_max']
max_decel = abs(config['mpc']['a_min']) * 0.65

curvature_ceiling = [wp.v_ref for wp in waypoints]

# Circular Backward pass
for step in range(2 * n - 1, -1, -1):
    i = step % n
    next_i = (i + 1) % n
    dist = np.hypot(waypoints[next_i].x - waypoints[i].x, waypoints[next_i].y - waypoints[i].y)
    max_reachable = np.sqrt(waypoints[next_i].v_ref**2 + 2 * max_decel * dist)
    waypoints[i].v_ref = min(waypoints[i].v_ref, max_reachable)

# Circular Forward pass
for step in range(1, 2 * n):
    i = step % n
    prev_i = (i - 1 + n) % n
    dist = np.hypot(waypoints[i].x - waypoints[prev_i].x, waypoints[i].y - waypoints[prev_i].y)
    max_reachable = np.sqrt(waypoints[prev_i].v_ref**2 + 2 * max_accel * dist)
    waypoints[i].v_ref = min(curvature_ceiling[i], max_reachable)

print(f"\nTotal waypoints after interpolation/savgol: {n}")
kappas = [wp.kappa for wp in waypoints]
print(f"Computed kappa min: {min(kappas):.3f}, max: {max(kappas):.3f}, mean: {np.mean(kappas):.3f}")

print("\nWaypoint Velocities (km/h):")
for i in range(0, n, max(1, n // 20)):
    wp = waypoints[i]
    print(f"WP {i:3d}: x={wp.x:6.2f}, y={wp.y:6.2f}, kappa={wp.kappa:6.3f}, original_v_ref={v_ref[i]*3.6:5.1f}, smooth_v_ref={wp.v_ref*3.6:5.1f}")

# Find any waypoints with extremely low velocity
low_vel_wps = [(i, wp.v_ref*3.6) for i, wp in enumerate(waypoints) if wp.v_ref * 3.6 < 10.0]
print(f"\nWaypoints with velocity < 10 km/h: {len(low_vel_wps)}")
if low_vel_wps:
    print("Sample low velocity waypoints:")
    for idx, vel in low_vel_wps[:30]:
        wp = waypoints[idx]
        print(f"  WP {idx:3d}: smooth_v_ref={vel:.2f} km/h, kappa={wp.kappa:.3f}")
