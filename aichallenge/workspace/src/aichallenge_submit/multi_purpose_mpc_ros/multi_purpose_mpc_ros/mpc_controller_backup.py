#!/usr/bin/env python3

import yaml
from typing import List, Tuple, Optional, NamedTuple, Dict, Any
from collections import deque
import dataclasses
from scipy import sparse
from scipy.sparse import dia_matrix
import json
import numpy as np
import math
import copy
import os
import shutil
from datetime import datetime

# ROS 2
import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from rclpy.parameter import Parameter
from visualization_msgs.msg import Marker, MarkerArray
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy

from std_msgs.msg import Empty, Bool, Float32MultiArray, Int32, Float64MultiArray, String
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Quaternion, Pose2D, Point, Vector3
from std_msgs.msg import ColorRGBA

from rcl_interfaces.msg import SetParametersResult
from rclpy.parameter import Parameter

# autoware
from autoware_auto_control_msgs.msg import AckermannControlCommand
from autoware_auto_vehicle_msgs.msg import GearCommand
from autoware_auto_planning_msgs.msg import Trajectory
from v2x_msgs.msg import V2XVehiclePositionArray
from multi_purpose_mpc_ros.v2x_vehicle_tracker import (
    V2XVehicleTracker,
    predictions_to_obstacles,
)

# Multi_Purpose_MPC
from multi_purpose_mpc_ros.core.map import Map, Obstacle
from multi_purpose_mpc_ros.core.reference_path import ReferencePath
from multi_purpose_mpc_ros.core.spatial_bicycle_models import BicycleModel
from multi_purpose_mpc_ros.core.MPC import MPC
from multi_purpose_mpc_ros.core.utils import load_waypoints, kmh_to_m_per_sec, load_ref_path

# Project
from multi_purpose_mpc_ros.common import convert_to_namedtuple, file_exists
from multi_purpose_mpc_ros.simulation_logger import SimulationLogger
from multi_purpose_mpc_ros.obstacle_manager import ObstacleManager
from multi_purpose_mpc_ros.exexution_stats import ExecutionStats
from multi_purpose_mpc_ros_msgs.msg import AckermannControlBoostCommand, PathConstraints, BorderCells
from multi_purpose_mpc_ros.tools.reference_velocity_configulator import ReferenceVelocityConfigulator
from multi_purpose_mpc_ros.lidar_road_mapper import LidarRoadMapper
from multi_purpose_mpc_ros.opponent_tracker import OpponentTracker
from multi_purpose_mpc_ros.map_path_generator import MapPathGenerator
from sensor_msgs.msg import PointCloud2


RED = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)
YELLOW = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
CYAN = ColorRGBA(r=0.0, g=156.0 / 255.0, b=209.0 / 255.0, a=1.0)


class FrictionEstimator:
    """Integrated friction estimation for adaptive control"""
    
    def __init__(self, window_size=50):
        self.window_size = window_size
        self.velocity_history = deque(maxlen=window_size)
        self.steering_history = deque(maxlen=window_size)
        self.lateral_accel_history = deque(maxlen=window_size)
        self.estimated_friction = 0.9
        self.friction_confidence = 0.0
        self.road_condition = "unknown"
    
    def update(self, velocity, steering_angle, lateral_accel):
        """Update friction estimate"""
        self.velocity_history.append(velocity)
        self.steering_history.append(steering_angle)
        self.lateral_accel_history.append(lateral_accel)
        
        if len(self.velocity_history) < 10:
            return
        
        # Simple friction estimation from lateral acceleration
        g = 9.81
        lateral_accel = abs(lateral_accel)
        instant_friction = lateral_accel / g if g > 0 else 0.9
        instant_friction = np.clip(instant_friction, 0.5, 1.2)
        
        # Smooth estimation
        alpha = 0.1
        self.estimated_friction = alpha * instant_friction + (1 - alpha) * self.estimated_friction
        
        # Calculate confidence
        window_fullness = len(self.velocity_history) / self.window_size
        self.friction_confidence = window_fullness
        
        # Determine road condition
        if self.estimated_friction > 0.9:
            self.road_condition = "dry"
        elif self.estimated_friction > 0.7:
            self.road_condition = "damp"
        elif self.estimated_friction > 0.5:
            self.road_condition = "wet"
        else:
            self.road_condition = "slippery"
    
    def get_speed_factor(self):
        """Get speed adaptation factor based on friction"""
        return np.clip(self.estimated_friction / 0.9, 0.5, 1.0)


class LapLearner:
    """Integrated lap learning for racing line optimization"""
    
    def __init__(self, max_laps=10):
        self.max_laps = max_laps
        self.current_lap_data = []
        self.lap_history = []
        self.lap_count = 0
        self.lap_start_time = None
        self.learned_path = None
    
    def start_lap(self):
        """Start recording a new lap"""
        self.current_lap_data = []
        self.lap_start_time = None
    
    def update(self, x, y, yaw, velocity):
        """Record current state"""
        if self.lap_start_time is None:
            self.lap_start_time = 0.0
        
        self.current_lap_data.append((x, y, yaw, velocity))
    
    def complete_lap(self):
        """Process completed lap"""
        if len(self.current_lap_data) < 100:
            return
        
        self.lap_history.append(self.current_lap_data)
        self.lap_count += 1
        
        # Keep only recent laps
        if len(self.lap_history) > self.max_laps:
            self.lap_history.pop(0)
        
        # Learn optimal path if we have enough data
        if len(self.lap_history) >= 2:
            self._learn_optimal_path()
        
        # Reset for next lap
        self.current_lap_data = []
        self.lap_start_time = None
    
    def _learn_optimal_path(self):
        """Learn optimal racing line from lap history"""
        # Simple implementation: average trajectories weighted by velocity
        if not self.lap_history:
            return
        
        # Resample all trajectories to common length
        target_length = 100
        aligned_trajectories = []
        
        for lap in self.lap_history:
            if len(lap) < 10:
                continue
            # Simple resampling
            indices = np.linspace(0, len(lap)-1, target_length, dtype=int)
            resampled = [lap[i] for i in indices]
            aligned_trajectories.append(resampled)
        
        if not aligned_trajectories:
            return
        
        # Average trajectories
        self.learned_path = []
        for i in range(target_length):
            points = [traj[i] for traj in aligned_trajectories if i < len(traj)]
            if points:
                avg_x = np.mean([p[0] for p in points])
                avg_y = np.mean([p[1] for p in points])
                avg_yaw = np.mean([p[2] for p in points])
                avg_vel = np.mean([p[3] for p in points])
                self.learned_path.append((avg_x, avg_y, avg_yaw, avg_vel))
    
    def get_learned_waypoint(self, index):
        """Get learned waypoint at index"""
        if self.learned_path and index < len(self.learned_path):
            return self.learned_path[index]
        return None

def array_to_ackermann_control_command(stamp, u: np.ndarray, acc: float) -> AckermannControlCommand:
    msg = AckermannControlCommand()
    msg.stamp = stamp
    msg.lateral.stamp = stamp
    msg.lateral.steering_tire_angle = u[1]
    msg.lateral.steering_tire_rotation_rate = 0.5
    msg.longitudinal.stamp = stamp
    msg.longitudinal.speed = u[0]
    msg.longitudinal.acceleration = acc
    return msg

def yaw_from_quaternion(q: Quaternion):
    sqx = q.x * q.x
    sqy = q.y * q.y
    sqz = q.z * q.z
    sqw = q.w * q.w

    # Cases derived from https://orbitalstation.wordpress.com/tag/quaternion/
    sarg = -2 * (q.x*q.z - q.w*q.y) / (sqx + sqy + sqz + sqw) # normalization added from urdfom_headers

    if sarg <= -0.99999:
        yaw = -2. * np.arctan2(q.y, q.x)
    elif sarg >= 0.99999:
        yaw = 2. * np.arctan2(q.y, q.x)
    else:
        yaw = np.arctan2(2. * (q.x*q.y + q.w*q.z), sqw + sqx - sqy - sqz)

    return yaw

def odom_to_pose_2d(odom: Odometry) -> Pose2D:
    pose = Pose2D()
    pose.x = odom.pose.pose.position.x
    pose.y = odom.pose.pose.position.y
    pose.theta = yaw_from_quaternion(odom.pose.pose.orientation)

    return pose

@dataclasses.dataclass
class MPCConfig:
    N: int
    Q: dia_matrix
    R: dia_matrix
    QN: dia_matrix
    v_max: float
    a_min: float
    a_max: float
    ay_max: float
    delta_max: float
    steer_rate_max: float
    control_rate: float
    steering_tire_angle_gain_var: float
    accel_low_pass_gain: float
    steer_low_pass_gain: float
    wp_id_offset: int
    use_max_kappa_pred: bool
    raceline_blend_ratio: float
    friction_coefficient: float


class MPCController(Node):

    PKG_PATH: str = get_package_share_directory('multi_purpose_mpc_ros') + "/"
    # MAX_LAPS = 6
    MAX_LAPS = 10000
    BUG_VEL = 40.0 # km/h
    BUG_ACC = 400.0

    SHOW_PLOT_ANIMATION = False
    PLOT_RESULTS = False
    ANIMATION_INTERVAL = 20

    KP = 100.0

    def __init__(self, config_path: str, ref_vel_config_path: Optional[str]) -> None:
        super().__init__("mpc_controller") # type: ignore

        # declare parameters
        self.declare_parameter("use_boost_acceleration", False)
        self.declare_parameter("use_obstacle_avoidance", False)
        self.declare_parameter("use_stats", False)

        # get parameters
        self.use_sim_time = self.get_parameter("use_sim_time").get_parameter_value().bool_value
        self.USE_BUG_ACC = self.get_parameter("use_boost_acceleration").get_parameter_value().bool_value
        self.USE_OBSTACLE_AVOIDANCE = self.get_parameter("use_obstacle_avoidance").get_parameter_value().bool_value
        self.use_stats = self.get_parameter("use_stats").get_parameter_value().bool_value

        self._config_path = config_path
        self._ref_vel_config_path: Optional[str] = ref_vel_config_path
        self._cfg = self._load_config()
        self._odom: Optional[Odometry] = None
        self._enable_control = True
        self._initialize()
        
        # Initialize Phase 1 Lidar-based systems BEFORE setup_pub_sub
        self._use_lidar_constraints = getattr(self._cfg.mpc, 'use_lidar_constraints', False)
        self._use_map_path_generation = getattr(self._cfg.mpc, 'use_map_path_generation', False)
        
        self._lidar_mapper = LidarRoadMapper(
            map_resolution=0.1,
            max_range=50.0,
            update_rate=10.0
        )
        self._opponent_tracker = OpponentTracker(
            max_opponents=10,
            prediction_horizon=2.0,
            update_rate=20.0
        )
        
        # Initialize Phase 2 Map-based path generator
        self._map_path_generator = None
        if self._use_map_path_generation and self._map is not None:
            self._map_path_generator = MapPathGenerator(
                map_obj=self._map,
                resolution=0.5,
                smoothing_distance=3.0
            )
            self.get_logger().warn("MAP-BASED PATH GENERATION ENABLED")
        
        # Lidar data storage
        self._latest_lidar_data = None
        self._lidar_data_timestamp = 0.0
        
        if self._use_lidar_constraints:
            self._lidar_mapper.enable_lidar_constraints(True)
            self._opponent_tracker.enable_tracking(True)
            self.get_logger().warn("LIDAR-BASED CONSTRAINTS ENABLED")
            self.get_logger().warn("Subscribing to Lidar topics for dynamic constraints")
        
        self._setup_parameters_callback()
        self._setup_pub_sub()
        
        # Initialize optimization systems
        self._friction_estimator = FrictionEstimator(window_size=50)
        self._lap_learner = LapLearner(max_laps=10)
        self._lap_learner.start_lap()
        self._road_conditions = []  # Store road conditions for dynamic acceleration
        
        # Force enable control immediately for debugging
        self.get_logger().warn("FORCE ENABLING CONTROL - Kart should start moving")
        self.get_logger().warn("OPTIMIZATION SYSTEMS ENABLED: Friction Estimator + Lap Learner")
        self._enable_control = True

        if self.use_sim_time:
            self.get_logger().warn("------------------------------------")
            self.get_logger().warn("use_sim_time is enabled!")
            self.get_logger().warn("------------------------------------")
        if self.USE_BUG_ACC:
            self.get_logger().warn("------------------------------------")
            self.get_logger().warn("USE_BUG_ACC is enabled!")
            self.get_logger().warn("------------------------------------")
        if self.USE_OBSTACLE_AVOIDANCE:
            self.get_logger().warn("------------------------------------")
            self.get_logger().warn("USE_OBSTACLE_AVOIDANCE is enabled!")
            self.get_logger().warn("------------------------------------")

    def _load_config(self) -> NamedTuple:

        # logging content
        with open(self._config_path, "r") as f:
            config_content = f.read()
            self.get_logger().info(
                "\n" +
                "----- config.yaml -----\n"+
                config_content + "\n" +
                "-----------------------")

        if self._ref_vel_config_path is not None:
            with open(self._ref_vel_config_path, "r") as f:
                ref_vel_config_content = f.read()
                self.get_logger().info(
                    "\n" +
                    "----- ref_vel.yaml -----\n"+
                    ref_vel_config_content + "\n" +
                    "-----------------------")

        with open(self._config_path, "r") as f:
            cfg: NamedTuple = convert_to_namedtuple(yaml.safe_load(f)) # type: ignore

        # Check if the files exist
        mandatory_files = [cfg.map.yaml_path, cfg.waypoints.csv_path] # type: ignore
        for file_path in mandatory_files:
            file_exists(self.in_pkg_share(file_path))
        return cfg

    def _create_reference_path_from_autoware_trajectory(self, trajectory: Trajectory) -> Optional[ReferencePath]:
        wp_x = [0] * len(trajectory.points)
        wp_y = [0] * len(trajectory.points)
        for i, p in enumerate(trajectory.points):
            wp_x[i] = p.pose.position.x
            wp_y[i] = p.pose.position.y

        cfg_ref_path = self._cfg.reference_path # type: ignore
        reference_path = ReferencePath(
            self._map,
            wp_x,
            wp_y,
            cfg_ref_path.resolution,
            cfg_ref_path.smoothing_distance,
            cfg_ref_path.max_width,
            cfg_ref_path.circular)

        mpc_config = self._mpc_cfg
        speed_profile_constraints = {
            "a_min": mpc_config.a_min, "a_max": mpc_config.a_max,
            "v_min": 0.0, "v_max": mpc_config.v_max, "ay_max": mpc_config.ay_max}

        if not reference_path.compute_speed_profile(speed_profile_constraints):
            return None

        return reference_path

    def _setup_parameters_callback(self) -> None:
        def declatre_parameters():
            cfg_mpc = self._cfg.mpc
            self.declare_parameter("v_max", cfg_mpc.v_max)
            self.declare_parameter("steering_tire_angle_gain_var", cfg_mpc.steering_tire_angle_gain_var)
            self.declare_parameter("Q0", cfg_mpc.Q[0])
            self.declare_parameter("Q1", cfg_mpc.Q[1])
            self.declare_parameter("Q2", cfg_mpc.Q[2])
            self.declare_parameter("R0", cfg_mpc.R[0])
            self.declare_parameter("R1", cfg_mpc.R[1])
            self.declare_parameter("QN0", cfg_mpc.QN[0])
            self.declare_parameter("QN1", cfg_mpc.QN[1])
            self.declare_parameter("QN2", cfg_mpc.QN[2])

            mpc_cfg = self._mpc_cfg
            self.declare_parameter("ay_max", mpc_cfg.ay_max)
            self.declare_parameter("accel_low_pass_gain", mpc_cfg.accel_low_pass_gain)
            self.declare_parameter("steer_low_pass_gain", mpc_cfg.steer_low_pass_gain)
            self.declare_parameter("wp_id_offset", mpc_cfg.wp_id_offset)
            self.declare_parameter("raceline_blend_ratio", mpc_cfg.raceline_blend_ratio)
            self.declare_parameter("friction_coefficient", mpc_cfg.friction_coefficient)

        def param_cb(parameters):
            cfg_mpc = self._cfg.mpc # type: ignore
            mpc_cfg = self._mpc_cfg

            def update_Q(index: int, value: float):
                cfg_mpc.Q[index] = value
                mpc_cfg.Q = sparse.diags(cfg_mpc.Q)
                self._mpc.update_Q(mpc_cfg.Q)
                self.get_logger().warn(f"Q[{index}] was updated to '{value}'")

            def update_R(index: int, value: float):
                cfg_mpc.R[index] = value
                mpc_cfg.R = sparse.diags(cfg_mpc.R)
                self._mpc.update_R(mpc_cfg.R)
                self.get_logger().warn(f"R[{index}] was updated to '{value}'")

            def update_QN(index: int, value: float):
                cfg_mpc.QN[index] = value
                mpc_cfg.QN = sparse.diags(cfg_mpc.QN)
                self._mpc.update_QN(mpc_cfg.QN)
                self.get_logger().warn(f"QN[{index}] was updated to '{value}'")

            ref_vel_changed = False
            for param in parameters:
                if param.name.startswith("ref_vel/"):
                    ref_vel_changed = True
                elif param.name == "v_max" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.v_max = param.value
                    self._mpc.update_v_max(kmh_to_m_per_sec(param.value))
                    v_ref: List[float] = [kmh_to_m_per_sec(param.value)] * len(self._reference_path.waypoints)
                    self._reference_path.set_v_ref(v_ref)

                    self.get_logger().warn(f"v_max was updated to '{param.value}' [km/h]")

                elif param.name == "steering_tire_angle_gain_var" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.steering_tire_angle_gain_var = param.value
                    self.get_logger().warn(f"steering_tire_angle_gain_var was updated to '{param.value}'")

                elif param.name == "Q0" and param.type_ == Parameter.Type.DOUBLE:
                    update_Q(0, param.value)
                elif param.name == "Q1" and param.type_ == Parameter.Type.DOUBLE:
                    update_Q(1, param.value)
                elif param.name == "Q2" and param.type_ == Parameter.Type.DOUBLE:
                    update_Q(2, param.value)


                elif param.name == "R0" and param.type_ == Parameter.Type.DOUBLE:
                    update_R(0, param.value)
                elif param.name == "R1" and param.type_ == Parameter.Type.DOUBLE:
                    update_R(1, param.value)

                elif param.name == "QN0" and param.type_ == Parameter.Type.DOUBLE:
                    update_QN(0, param.value)
                elif param.name == "QN1" and param.type_ == Parameter.Type.DOUBLE:
                    update_QN(1, param.value)
                elif param.name == "QN2" and param.type_ == Parameter.Type.DOUBLE:
                    update_QN(2, param.value)

                elif param.name == "ay_max" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.ay_max = param.value
                    self._mpc.update_ay_max(param.value)
                    self.get_logger().warn(f"ay_max was updated to '{param.value}'")

                elif param.name == "accel_low_pass_gain" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.accel_low_pass_gain = param.value
                    self.get_logger().warn(f"accel_low_pass_gain was updated to '{param.value}'")

                elif param.name == "steer_low_pass_gain" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.steer_low_pass_gain = param.value
                    self.get_logger().warn(f"steer_low_pass_gain was updated to '{param.value}'")

                elif param.name == "wp_id_offset" and param.type_ == Parameter.Type.INTEGER:
                    mpc_cfg.wp_id_offset = param.value
                    self._mpc.update_wp_id_offset(param.value)
                    self.get_logger().warn(f"wp_id_offset was updated to '{param.value}'")

                elif param.name == "raceline_blend_ratio" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.raceline_blend_ratio = param.value
                    self._mpc.raceline_blend_ratio = param.value
                    self.get_logger().warn(f"raceline_blend_ratio was updated to '{param.value}'")

                elif param.name == "friction_coefficient" and param.type_ == Parameter.Type.DOUBLE:
                    mpc_cfg.friction_coefficient = param.value
                    self.get_logger().warn(f"friction_coefficient was updated to '{param.value}'")
                    ref_vel_changed = True

            if ref_vel_changed:
                self._update_waypoint_velocities()

            return SetParametersResult(successful=True)

        declatre_parameters()
        self.add_on_set_parameters_callback(param_cb)

    def _initialize(self) -> None:
        def create_map() -> Map:
            return Map(self.in_pkg_share(self._cfg.map.yaml_path)) # type: ignore

        def create_ref_path(map: Map) -> ReferencePath:
            cfg_ref_path = self._cfg.reference_path # type: ignore

            is_ref_path_given = cfg_ref_path.csv_path != "" # type: ignore
            if is_ref_path_given:
                print("Using given reference path")
                wp_x, wp_y, _, _ = load_ref_path(self.in_pkg_share(self._cfg.reference_path.csv_path)) # type: ignore
                return ReferencePath(
                    map,
                    wp_x,
                    wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular)

            else:
                print("Using waypoints to create reference path")
                wp_x, wp_y = load_waypoints(self.in_pkg_share(self._cfg.waypoints.csv_path)) # type: ignore

                return ReferencePath(
                    map,
                    wp_x,
                    wp_y,
                    cfg_ref_path.resolution,
                    cfg_ref_path.smoothing_distance,
                    cfg_ref_path.max_width,
                    cfg_ref_path.circular)


        def create_obstacles() -> List[Obstacle]:
            use_csv_obstacles = self._cfg.obstacles.csv_path != "" # type: ignore
            if use_csv_obstacles:
                obstacles_file_path = self.in_pkg_share(self._cfg.obstacles.csv_path) # type: ignore
                obs_x, obs_y = load_waypoints(obstacles_file_path)
                obstacles = []
                for cx, cy in zip(obs_x, obs_y):
                    obstacles.append(Obstacle(cx=cx, cy=cy, radius=self._cfg.obstacles.radius)) # type: ignore
                self._obstacle_manager = ObstacleManager(self._map, obstacles)
                return obstacles
            else:
                return []

        def create_car(ref_path: ReferencePath) -> BicycleModel:
            cfg_model = self._cfg.bicycle_model # type: ignore
            car = BicycleModel(
                ref_path,
                cfg_model.length,
                cfg_model.width,
                1.0 / self._cfg.mpc.control_rate) # type: ignore
            car.a_min = self._cfg.mpc.a_min
            car.a_max = self._cfg.mpc.a_max
            return car

        def create_mpc(car: BicycleModel) -> Tuple[MPCConfig, MPC]:
            cfg_mpc = self._cfg.mpc # type: ignore

            mpc_cfg = MPCConfig(
                cfg_mpc.N,
                sparse.diags(cfg_mpc.Q),
                sparse.diags(cfg_mpc.R),
                sparse.diags(cfg_mpc.QN),
                kmh_to_m_per_sec(self.BUG_VEL if self.USE_BUG_ACC else cfg_mpc.v_max),
                cfg_mpc.a_min,
                cfg_mpc.a_max,
                cfg_mpc.ay_max,
                np.deg2rad(cfg_mpc.delta_max_deg),
                cfg_mpc.steer_rate_max,
                cfg_mpc.control_rate,
                cfg_mpc.steering_tire_angle_gain_var,
                cfg_mpc.accel_low_pass_gain,
                cfg_mpc.steer_low_pass_gain,
                cfg_mpc.wp_id_offset,
                cfg_mpc.use_max_kappa_pred,
                cfg_mpc.raceline_blend_ratio,
                cfg_mpc.friction_coefficient)

            state_constraints = {
                "xmin": np.array([-np.inf, -np.inf, -np.inf]),
                "xmax": np.array([np.inf, np.inf, np.inf])}
            input_constraints = {
                "umin": np.array([0.0, -np.tan(mpc_cfg.delta_max) / car.length]),
                "umax": np.array([mpc_cfg.v_max, np.tan(mpc_cfg.delta_max) / car.length])}

            # mpcからのsteer指令出力は、gainを掛けて出力され、その状態で車体のsteer rate limit が適用されるため、
            # mpcの制御計算におけるsteer_rate_maxは、実際のsteer_rate_maxをgainで除した値で設定する
            scaled_steer_rate_max = mpc_cfg.steer_rate_max / mpc_cfg.steering_tire_angle_gain_var

            mpc = MPC(
                car,
                mpc_cfg.N,
                mpc_cfg.Q,
                mpc_cfg.R,
                mpc_cfg.QN,
                state_constraints,
                input_constraints,
                mpc_cfg.ay_max,
                scaled_steer_rate_max,
                mpc_cfg.wp_id_offset,
                self.USE_OBSTACLE_AVOIDANCE,
                self._cfg.reference_path.use_path_constraints_topic,
                mpc_cfg.use_max_kappa_pred,
                mpc_cfg.raceline_blend_ratio)

            return mpc_cfg, mpc

        def compute_speed_profile(car: BicycleModel, mpc_config: MPCConfig) -> None:
            speed_profile_constraints = {
                "a_min": mpc_config.a_min, "a_max": mpc_config.a_max,
                "v_min": 0.0, "v_max": mpc_config.v_max, "ay_max": mpc_config.ay_max}
            car.reference_path.compute_speed_profile(speed_profile_constraints)

        def create_ref_vel_configulator() -> Optional[ReferenceVelocityConfigulator]:
            if self._ref_vel_config_path is None:
                return None
            return ReferenceVelocityConfigulator(self, self._config_path, self._ref_vel_config_path)

        self._map = create_map()
        self._reference_path = create_ref_path(self._map)
        self._car = create_car(self._reference_path)
        self._mpc_cfg, self._mpc = create_mpc(self._car)

        # Create ref_vel_configulator FIRST so we know whether a custom profile exists.
        # Only fall back to the OSQP-based compute_speed_profile when no custom profile
        # is configured — the curvature-aware profile built by _update_waypoint_velocities()
        # is more accurate and will overwrite it anyway.
        self._ref_vel_configulator: Optional[ReferenceVelocityConfigulator] = create_ref_vel_configulator()
        if self._ref_vel_configulator is None:
            compute_speed_profile(self._car, self._mpc_cfg)
        self._update_waypoint_velocities()

        self._trajectory: Optional[Trajectory] = None
        self._path_constraints = None

        # Obstacles
        if self.USE_OBSTACLE_AVOIDANCE:
            self._static_obstacles: List[Obstacle] = create_obstacles()
            self._dynamic_obstacles: List[Obstacle] = []
            self._detected_obstacles: List[Obstacle] = []
            self._last_obstacles_msgs_raw = None
            self._obstacles_updated = bool(self._static_obstacles)
            self._latest_opponent_forecast = None
            v2x_cfg = self._cfg.v2x_obstacle_avoidance  # type: ignore
            self._v2x_tracker = V2XVehicleTracker(
                v_max_safety=float(v2x_cfg.v_max_safety),
                position_jump_threshold=float(v2x_cfg.position_jump_threshold),
                warn_callback=self.get_logger().warn,
            )
            self._v2x_vehicle_radius = float(v2x_cfg.vehicle_radius)
            mpc_N = int(self._cfg.mpc.N)  # type: ignore
            t_horizon = mpc_N / float(self._cfg.mpc.control_rate)  # type: ignore
            self._v2x_t_samples = [
                k * t_horizon / max(mpc_N - 1, 1) for k in range(mpc_N)
            ]
            # コリドー外の V2X 障害物で MPC のコリドー狭窄/反転が起きないよう、
            # ref-path 近傍のみに絞り込む。閾値 = max_width/2 + vehicle_radius + 余白。
            ref_max_width = float(self._cfg.reference_path.max_width)  # type: ignore
            self._v2x_corridor_threshold_sq = (
                ref_max_width / 2.0 + self._v2x_vehicle_radius + 0.5
            ) ** 2
            wps = self._reference_path.waypoints
            self._waypoint_xy = np.asarray(
                [(wp.x, wp.y) for wp in wps], dtype=np.float64)

        # Laps
        self._current_laps = 1
        self._last_lap_time = 0.0
        self._lap_times = [None] * (self.MAX_LAPS + 1) # +1 means include lap 0

        # condition
        self._last_condition = None
        self._last_colliding_time = None

        # reverse recovery state
        self._stuck_ticks = 0
        self._reverse_ticks = 0
        self._realign_ticks = 0
        self._reverse_steer_dir = 1.0
        self._was_reversing = False

        # stats
        self._stats = ExecutionStats(self.get_logger(), window_size=50, record_count_threshold=1000)

        # save config
        if self._cfg.common.save_config:
            self._save_config()

    def _update_waypoint_velocities(self) -> None:
        ay_max = self._mpc_cfg.ay_max
        v_ref = []
        road_conditions = []  # Store road condition for each waypoint
        
        for i, wp in enumerate(self._reference_path.waypoints):
            if self._ref_vel_configulator is not None:
                ref_vel_kmph = self._ref_vel_configulator.get_ref_vel(i)
                section_speed_ms = min(
                    kmh_to_m_per_sec(ref_vel_kmph),
                    self._mpc_cfg.v_max
                )
            else:
                section_speed_ms = self._mpc_cfg.v_max
            
            # Use the CSV kappa (already smooth, matches MPC runtime kappa_ref)
            # to calculate dynamic curvature speed ceiling
            kappa_abs = abs(wp.kappa) if wp.kappa is not None else 0.0
            
            # Classify road condition based on curvature
            # Green (straight): kappa < 0.05
            # Yellow (moderate curve): 0.05 <= kappa < 0.15
            # Red (sharp curve): kappa >= 0.15
            if kappa_abs < 0.05:
                road_condition = "green"  # Straight - max acceleration
            elif kappa_abs < 0.15:
                road_condition = "yellow"  # Moderate - reduce by 5-10%
            else:
                road_condition = "red"  # Sharp - reduce by 10-15%
            road_conditions.append(road_condition)
            
            # Use 75% of ay_max for the speed profile to give the MPC solver steering headroom!
            # If they are exactly the same, the solver's steering constraint will lock up at high speeds
            # causing the kart to understeer (wobble) when it enters the turn slightly too fast.
            v_max_kappa = np.sqrt((ay_max * 0.75) / (kappa_abs + 1e-9))
            
            # Apply road condition-based speed adjustments
            if road_condition == "green":
                # Straight road - use max speed
                adjusted_speed = min(section_speed_ms, v_max_kappa)
            elif road_condition == "yellow":
                # Moderate curve
                reduction_factor = 1.15
                adjusted_speed = min(section_speed_ms, v_max_kappa) * reduction_factor
            else:  # red
                # Sharp curve
                reduction_factor = 1.25 
                adjusted_speed = min(section_speed_ms, v_max_kappa) * reduction_factor
            
            v_ref.append(adjusted_speed)
        
        self._reference_path.set_v_ref(v_ref)
        self._road_conditions = road_conditions  # Store for dynamic acceleration
        self._smooth_and_clip_speed_profile()

    def _smooth_and_clip_speed_profile(self) -> None:
        waypoints = self._car.reference_path.waypoints
        n = len(waypoints)
        if n < 2:
            return

        # Get road conditions for dynamic acceleration
        road_conditions = getattr(self, '_road_conditions', ['green'] * n)
        
        max_accel = self._mpc_cfg.a_max
        # Use a more conservative deceleration for the speed profile so the kart starts braking
        # earlier and doesn't hit the physical limit (a_min) and overshoot into corners.
        max_decel = abs(self._mpc_cfg.a_min) * 0.65

        # Snapshot the per-waypoint curvature ceiling BEFORE the backward pass
        curvature_ceiling = [wp.v_ref for wp in waypoints]

        # Perform 2 passes for seamless circular wrap-around consistency
        is_circular = getattr(self._reference_path, 'circular', True)
        passes = 2 if is_circular else 1

        for _ in range(passes):
            # Backward pass: ensure braking is physically achievable before a corner
            for i in range(n - 1, -1, -1):
                next_i = (i + 1) % n if is_circular else min(i + 1, n - 1)
                dist = np.hypot(waypoints[next_i].x - waypoints[i].x, waypoints[next_i].y - waypoints[i].y)
                max_reachable = np.sqrt(
                    waypoints[next_i].v_ref**2 + 2 * max_decel * dist
                )
                waypoints[i].v_ref = min(waypoints[i].v_ref, max_reachable)

            # Forward pass: accelerate back up to the curvature ceiling on straights
            for i in range(n):
                prev_i = (i - 1) % n if is_circular else max(i - 1, 0)
                dist = np.hypot(waypoints[i].x - waypoints[prev_i].x, waypoints[i].y - waypoints[prev_i].y)
                
                # Dynamic acceleration based on road condition (currently set to 1.0 for testing max limits)
                current_condition = road_conditions[i] if i < len(road_conditions) else 'green'
                if current_condition == 'green':
                    # Straight road
                    dynamic_accel = max_accel
                elif current_condition == 'yellow':
                    # Moderate curve
                    dynamic_accel = max_accel * 1.0
                else:  # red
                    # Sharp curve
                    dynamic_accel = max_accel * 1.0
                
                max_reachable = np.sqrt(
                    waypoints[prev_i].v_ref**2 + 2 * dynamic_accel * dist
                )
                waypoints[i].v_ref = min(curvature_ceiling[i], max_reachable)

    def _update_reference_path(self) -> None:
        """Update reference path from map if enabled, otherwise use CSV."""
        if self._use_map_path_generation and self._map_path_generator is not None:
            self._generate_map_based_path()
        else:
            # Use existing CSV path
            pass
    
    def _generate_map_based_path(self) -> None:
        """Generate racing line from map data."""
        try:
            # Get current vehicle position as start
            if self._car.temporal_state is not None:
                start_pos = (self._car.temporal_state.x, self._car.temporal_state.y)
            else:
                start_pos = (0.0, 0.0)
            
            # Generate racing line from map
            world_path = self._map_path_generator.generate_racing_line(start_pos)
            
            if len(world_path) > 0:
                # Generate waypoints with curvature
                waypoints = self._map_path_generator.generate_waypoints(world_path)
                
                if len(waypoints) > 0:
                    # Update reference path with map-generated waypoints
                    self.get_logger().info(f"Generated map-based path with {len(waypoints)} waypoints")
                    # Note: This would require updating the reference path structure
                    # For now, just log that generation succeeded
                    
        except Exception as e:
            self.get_logger().error(f"Map-based path generation failed: {e}")
            self.get_logger().warn("Falling back to CSV path")
    
    def _process_lidar_data(self, now):
        """Process Lidar data for dynamic constraints (Phase 1)."""
        if not self._use_lidar_constraints:
            return
        
        # Check if we have recent Lidar data
        current_time = now.nanoseconds / 1e9
        if self._latest_lidar_data is None or (current_time - self._lidar_data_timestamp) > 0.5:
            self.get_logger().warn("No recent Lidar data available, using CSV fallback")
            return
        
        # Process Lidar data
        vehicle_pose = (self._car.temporal_state.x, self._car.temporal_state.y, self._car.temporal_state.psi)
        
        try:
            # Process for road mapping
            road_constraints = self._lidar_mapper.process_lidar_scan(self._latest_lidar_data, vehicle_pose)
            
            # Process for opponent tracking
            opponent_constraints = self._opponent_tracker.process_sensor_data(self._latest_lidar_data, vehicle_pose)
            
            # Log processing results
            if self._loop % 50 == 0:
                self.get_logger().info(f"Lidar processing: {len(road_constraints['obstacles'])} obstacles, "
                                     f"{len(opponent_constraints['opponents'])} opponents detected")
                
        except Exception as e:
            self.get_logger().error(f"Lidar processing failed: {e}")
    
    def _lidar_callback(self, msg: PointCloud2):
        """Callback for Lidar point cloud data."""
        try:
            # Convert PointCloud2 to numpy array
            import sensor_msgs_py.point_cloud2 as pc2
            points = pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)
            self._latest_lidar_data = np.array(list(points))
            self._lidar_data_timestamp = self.get_clock().now().nanoseconds / 1e9
            
            if self._loop % 100 == 0:
                self.get_logger().info(f"Received Lidar scan with {len(self._latest_lidar_data)} points")
                
        except Exception as e:
            self.get_logger().error(f"Lidar callback failed: {e}")
    
    def _save_config(self) -> None:
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        dst_dir = self.PKG_PATH + f"log/{now}"
        os.makedirs(dst_dir, exist_ok=True)
        shutil.copy(self._config_path, os.path.join(dst_dir, "config.yaml"))

    def _setup_pub_sub(self) -> None:
        # Publishers
        if self.USE_BUG_ACC:
          self._command_pub = self.create_publisher(
            AckermannControlBoostCommand, "/boost_commander/command", 1)
        else:
          self._command_pub = self.create_publisher(
            AckermannControlCommand, "/control/command/control_cmd", 1)
          self._ackermann_pub = self.create_publisher(
            AckermannControlCommand, "/ackermann_cmd", 1)
          print("use normal ackermann control command")

        self._command_raw_pub = self.create_publisher(
          AckermannControlCommand, "/control/command/control_cmd_raw", 1)

        self._gear_pub = self.create_publisher(
            GearCommand, "/control/command/gear_cmd", 1)
        self._control_mode_pub = self.create_publisher(
            Bool, "/awsim/control_mode_request_topic", 1)

        # NOTE:評価環境での可視化のためにダミーのトピック名を使用
        self._mpc_pred_pub = self.create_publisher(
            MarkerArray, "/mpc/prediction", 1)
        self._mpc_pred_pub_dummy = self.create_publisher(
            MarkerArray, "/planning/scenario_planning/lane_driving/motion_planning/obstacle_stop_planner/virtual_wall", 1)

        latching_qos = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        # NOTE:評価環境での可視化のためにダミーのトピック名を使用
        self._ref_path_pub = self.create_publisher(
            MarkerArray, "/mpc/ref_path", latching_qos)
        self._ref_path_pub_dummy = self.create_publisher(
            MarkerArray, "/planning/scenario_planning/lane_driving/behavior_planning/behavior_path_planner/debug/bound", latching_qos)

        # Subscribers
        self._odom_sub = self.create_subscription(
            Odometry, "/localization/kinematic_state", self._odom_callback, 1)
        
        # Phase 1: Subscribe to Lidar if enabled
        if self._use_lidar_constraints:
            self._lidar_sub = self.create_subscription(
                PointCloud2, "/sensing/lidar/top/pointcloud_raw", self._lidar_callback, 10)
            self.get_logger().info("Subscribed to Lidar topic: /sensing/lidar/top/pointcloud_raw")
        
        self._control_mode_request_sub = self.create_subscription(
            Bool, "/awsim/control_mode_request_topic", self._control_mode_request_callback, 1)
        self._control_mode_request_sub_alt = self.create_subscription(
            Bool, "/control/control_mode_request_topic", self._control_mode_request_callback, 1)
        # simple_trajectory_generator publishes with BEST_EFFORT/KEEP_LAST(1) — match it
        # so the subscription is QoS-compatible (rclpy default is RELIABLE).
        trajectory_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._trajectory_sub = self.create_subscription(
            Trajectory, "planning/scenario_planning/trajectory", self._trajectory_callback, trajectory_qos)
        self._stop_request_sub = self.create_subscription(
            Empty, "/control/mpc/stop_request", self._stop_request_callback, 1)

        if self.use_sim_time:
            self._awsim_status_sub = self.create_subscription(
                Float32MultiArray, "/awsim/status", self._awsim_status_callback, 1)
            self._condition_sub = self.create_subscription(
                Int32, "/aichallenge/pitstop/condition", self._condition_callback, 1)

        if self.USE_OBSTACLE_AVOIDANCE:
            if self._cfg.reference_path.use_path_constraints_topic: # type: ignore
                self._path_constraints_sub = self.create_subscription(
                    PathConstraints, "/path_constraints_provider/path_constraints", self._path_constraints_callback, 1)

            if self._cfg.reference_path.use_border_cells_topic: # type: ignore
                self._border_cells_sub = self.create_subscription(
                    BorderCells, "/path_constraints_provider/border_cells", self._border_cells_callback, 1)

            if not self._cfg.reference_path.use_path_constraints_topic: # type: ignore
                self._obstacles_sub = self.create_subscription(
                    Float64MultiArray,
                    "/aichallenge/objects",
                    self._obstacles_callback,
                    1)

            self._forecast_sub = self.create_subscription(
                String,
                "/opponent_forecast",
                self._forecast_callback,
                1)

    def _create_ackerman_control_command(self, stamp, u, acc, bug_acc_enabled):
        v_cmd = abs(u[0]) if u[0] < -0.1 else u[0]
        steer_cmd = u[1]

        ackerman_cmd = array_to_ackermann_control_command(stamp.to_msg(), [v_cmd, steer_cmd], acc)

        if not self.USE_BUG_ACC:
            return ackerman_cmd

        ackerman_boost_cmd = AckermannControlBoostCommand()
        ackerman_boost_cmd.command = ackerman_cmd
        ackerman_boost_cmd.boost_mode = bug_acc_enabled
        return ackerman_boost_cmd

    def _publish_control_command(self, stamp, u, acc, bug_acc_enabled):
        cmd = self._create_ackerman_control_command(stamp, u, acc, bug_acc_enabled)

        # publish raw control command
        if self.USE_BUG_ACC:
            self._command_raw_pub.publish(cmd.command)
        else:
            self._command_raw_pub.publish(cmd)

        # compensate steering angle for the real vehicle
        # AWSIMにおいても後段のactuation_cmd_converter でgainを考慮した指令を生成するため、実機/sim問わず
        # gain を掛ける
        if self.USE_BUG_ACC:
            cmd.command.lateral.steering_tire_angle *= self._mpc_cfg.steering_tire_angle_gain_var
        else:
            cmd.lateral.steering_tire_angle *= self._mpc_cfg.steering_tire_angle_gain_var
        self._command_pub.publish(cmd)
        if hasattr(self, '_ackermann_pub'):
            self._ackermann_pub.publish(cmd)

        # Publish gear command (DRIVE=2, REVERSE=20)
        gear_msg = GearCommand()
        gear_msg.stamp = stamp.to_msg()
        if u[0] < -0.1:
            gear_msg.command = 20  # GearCommand.REVERSE
        else:
            gear_msg.command = GearCommand.DRIVE
        self._gear_pub.publish(gear_msg)

        # Periodically publish control mode request (AUTONOMOUS=True)
        if hasattr(self, '_loop') and self._loop % 20 == 0:
            cm_msg = Bool()
            cm_msg.data = True
            self._control_mode_pub.publish(cm_msg)


    def _odom_callback(self, msg: Odometry) -> None:
        self._odom = msg

    def _control_mode_request_callback(self, msg):
        if msg.data and not self._enable_control:
            self.get_logger().info("Control mode request received")
            self._enable_control = True

    def _path_constraints_callback(self, msg: PathConstraints):
        self._reference_path.set_path_constraints(
            msg.upper_bounds, msg.lower_bounds, msg.rows, msg.cols)

    def _forecast_callback(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
            self._latest_opponent_forecast = data
            
            predictions = data.get('opponents', {})
            self._dynamic_obstacles = []
            for vid, path in predictions.items():
                for pt in path:
                    self._dynamic_obstacles.append(Obstacle(cx=pt[0], cy=pt[1], radius=self._v2x_vehicle_radius))
            
            recommended_side = data.get('recommended_pass_side', 'none')
            if recommended_side != 'none':
                effective_blend = 0.35
            else:
                effective_blend = float(self._cfg.mpc.raceline_blend_ratio) # type: ignore
                
            self._mpc.raceline_blend_ratio = effective_blend
            self._obstacles_updated = True
        except Exception as e:
            self.get_logger().error(f"Failed to parse opponent forecast: {e}")

    def _obstacles_callback(self, msg: Float64MultiArray) -> None:
        obstacles_updated = (self._last_obstacles_msgs_raw != msg.data) and (len(msg.data) > 0)
        if obstacles_updated:
            self._last_obstacles_msgs_raw = msg.data
            self._detected_obstacles = []
            for i in range(0, len(msg.data), 4):
                x = msg.data[i]
                y = msg.data[i + 1]
                self._detected_obstacles.append(Obstacle(cx=x, cy=y, radius=float(self._cfg.obstacles.radius))) # type: ignore
            self._obstacles_updated = True

    def _filter_obstacles_to_corridor(self, obstacles: List[Obstacle]) -> List[Obstacle]:
        if not obstacles or self._waypoint_xy.size == 0:
            return obstacles
        thr_sq = self._v2x_corridor_threshold_sq
        wps = self._waypoint_xy
        kept: List[Obstacle] = []
        car_x = getattr(self._car.temporal_state, 'x', None)
        car_y = getattr(self._car.temporal_state, 'y', None)
        for ob in obstacles:
            if car_x is not None and car_y is not None:
                if np.hypot(ob.cx - car_x, ob.cy - car_y) < 0.8:
                    continue
            dxy = wps - np.array([ob.cx, ob.cy], dtype=np.float64)
            if np.min(np.einsum('ij,ij->i', dxy, dxy)) <= thr_sq:
                kept.append(ob)
        return kept

    def _border_cells_callback(self, msg: BorderCells):
        self._reference_path.set_border_cells(
            msg.dynamic_upper_bounds, msg.dynamic_lower_bounds, msg.rows, msg.cols)

    def _trajectory_callback(self, msg):
        self._trajectory = msg

    def _awsim_status_callback(self, msg):
        laps = int(msg.data[1])
        lap_time = msg.data[2]
        # section = int(msg.data[3])

        if self._current_laps is None:
            self._current_laps = 1 if laps == 0 else laps

        if laps > self._current_laps:
            self.get_logger().info(f'\033[32mLap {self._current_laps} completed! Lap time: {self._last_lap_time} s\033[0m')
            self._lap_times[self._current_laps] = self._last_lap_time
            self._current_laps = laps

        self._last_lap_time = lap_time

    def _condition_callback(self, msg: Int32):
        if self._last_condition is None:
            self._last_condition = msg.data

        diff_condition = abs(msg.data - self._last_condition)
        if diff_condition > 0:
            self._last_colliding_time = self.get_clock().now()
            self.get_logger().warning(f"Collision detected! Condition changed by {diff_condition}")
        self._last_condition = msg.data

    def _stop_request_callback(self, msg: Empty) -> None:
        if self._enable_control:
            self.get_logger().warn(f"Stop request received {self._enable_control}")
            self._enable_control = False

    def _wait_until_clock_received(self) -> None:
        if self.use_sim_time:
            self.get_logger().info(f"wait until clock received...")
            rate = self.create_rate(10)
            rate.sleep()
            self.get_logger().info(f">> OK!")

    def _wait_until_message_received(self, message_getter, message_name: str, timeout: float = 60.0, rate_hz: int = 30) -> None:
        t_start = self.get_clock().now()
        last_warn = t_start
        rate = self.create_rate(rate_hz)

        self.get_logger().info(f"wait until {message_name} received...")

        while message_getter() is None:
            now = self.get_clock().now()
            if (now - last_warn).nanoseconds > 5.0 * 1e9:
                self.get_logger().warn(f"Still waiting for {message_name} message...")
                last_warn = now
            if timeout > 0 and (now - t_start).nanoseconds > timeout * 1e9:
                self.get_logger().error(f"Timeout while waiting for {message_name} message after {timeout}s!")
                raise TimeoutError(f"Timeout while waiting for {message_name} message")
            rate.sleep()

        self.get_logger().info(f">> {message_name} OK!")

    def _wait_until_odom_received(self, timeout: float = 30.) -> None:
        self._wait_until_message_received(lambda: self._odom, 'odometry', timeout)

    def _wait_until_trajectory_received(self, timeout: float = 30.) -> None:
        if self._cfg.reference_path.update_by_topic:
            self._wait_until_message_received(lambda: self._trajectory, 'trajectory', timeout)

    def _wait_until_path_constraints_received(self, timeout: float = 30.) -> None:
        if self.USE_OBSTACLE_AVOIDANCE and self._cfg.reference_path.use_path_constraints_topic: # type: ignore
            self._wait_until_message_received(lambda: self._reference_path.path_constraints, 'path constraints', timeout)

    def _compute_path_curvatures(self, x_pts: List[float], y_pts: List[float]) -> List[float]:
        n = len(x_pts)
        if n < 3:
            return [0.0] * n
        kappas = [0.0] * n
        for i in range(1, n - 1):
            dx1, dy1 = x_pts[i] - x_pts[i - 1], y_pts[i] - y_pts[i - 1]
            dx2, dy2 = x_pts[i + 1] - x_pts[i], y_pts[i + 1] - y_pts[i]
            ds = math.hypot(dx1, dy1) + 1e-6
            th1 = math.atan2(dy1, dx1)
            th2 = math.atan2(dy2, dx2)
            dth = math.atan2(math.sin(th2 - th1), math.cos(th2 - th1))
            kappas[i] = abs(dth / ds)
        kappas[0] = kappas[1]
        kappas[-1] = kappas[-2]
        return kappas

    def _evaluate_best_recovery_steer(self, pose_x: float, pose_y: float, pose_theta: float, e_y: float) -> float:
        candidates = [np.deg2rad(25.0), 0.0, np.deg2rad(-25.0)]
        best_steer = -1.0 if e_y > 0 else 1.0
        best_cost = float('inf')
        v_rev = -2.5
        dt_sim = 1.0
        L = float(getattr(self._cfg.bicycle_model, 'length', 1.087)) # type: ignore

        for delta in candidates:
            theta_sim = pose_theta + (v_rev / max(L, 0.1)) * np.tan(delta) * dt_sim
            x_sim = pose_x + v_rev * np.cos(theta_sim) * dt_sim
            y_sim = pose_y + v_rev * np.sin(theta_sim) * dt_sim

            closest_idx = self._car.get_closest_waypoint(x_sim, y_sim)
            wp = self._reference_path.get_waypoint(closest_idx)

            dx_sim = x_sim - wp.x
            dy_sim = y_sim - wp.y
            e_y_sim = -np.sin(wp.psi) * dx_sim + np.cos(wp.psi) * dy_sim
            dpsi = math.atan2(math.sin(theta_sim - wp.psi), math.cos(theta_sim - wp.psi))

            cost = 2.0 * abs(e_y_sim) + 1.5 * abs(dpsi)
            if cost < best_cost:
                best_cost = cost
                best_steer = np.sign(delta) if abs(delta) > 1e-3 else (-1.0 if e_y > 0 else 1.0)

        return best_steer

    def _publish_mpc_pred_marker(self, x_pred, y_pred):
        pred_marker_array = MarkerArray()
        m_base = Marker()
        m_base.header.frame_id = "map"
        m_base.ns = "mpc_pred"
        m_base.type = Marker.SPHERE
        m_base.action = Marker.ADD
        m_base.pose.position.z = 0.0
        m_base.scale = Vector3(x=0.5, y=0.5, z=0.5)

        kappas = self._compute_path_curvatures(list(x_pred), list(y_pred))

        GREEN = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.95)
        YELLOW = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.95)
        RED = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.95)

        for i in range(len(x_pred)):
            m = copy.deepcopy(m_base)
            m.id = i
            m.pose.position.x = x_pred[i]
            m.pose.position.y = y_pred[i]

            k = kappas[i] if i < len(kappas) else 0.0
            if k < 0.06:
                m.color = GREEN
            elif k < 0.15:
                m.color = YELLOW
            else:
                m.color = RED

            pred_marker_array.markers.append(m) # type: ignore
        self._mpc_pred_pub.publish(pred_marker_array)
        self._mpc_pred_pub_dummy.publish(pred_marker_array)

    def _publish_ref_path_marker(self, ref_path: ReferencePath):
        WP_SPHERE_ENABLED = False

        ref_path_marker_array = MarkerArray()

        m_base = Marker()
        m_base.header.frame_id = "map"
        m_base.ns = "ref_path"
        m_base.type = Marker.LINE_STRIP
        m_base.action = Marker.ADD
        m_base.pose.position.z = 0.0
        m_base.scale.x = 0.2
        m_base.color = ColorRGBA(r=0.0, g=0.0, b=1.0, a=0.7)

        for i in range(len(ref_path.waypoints) - 1):
            m = copy.deepcopy(m_base)
            m.id = i
            start = Point()
            start.x = ref_path.waypoints[i].x
            start.y = ref_path.waypoints[i].y
            end = Point()
            end.x = ref_path.waypoints[i + 1].x
            end.y = ref_path.waypoints[i + 1].y
            m.points.append(start) # type: ignore
            m.points.append(end) # type: ignore
            ref_path_marker_array.markers.append(m) # type: ignore

        if WP_SPHERE_ENABLED:
            spheres = Marker()
            spheres.header.frame_id = "map"
            spheres.ns = "ref_path_point"
            spheres.type = Marker.SPHERE_LIST
            spheres.action = Marker.ADD
            radius = 0.2
            spheres.scale = Vector3(x=radius, y=radius, z=radius)
            spheres.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=0.7)
            for i in range(len(ref_path.waypoints) - 1):
                p = Point()
                p.x = ref_path.waypoints[i].x
                p.y = ref_path.waypoints[i].y
                p.z = 0.
                spheres.points.append(p) #type: ignore
            ref_path_marker_array.markers.append(spheres) # type: ignore

        self._ref_path_pub.publish(ref_path_marker_array)
        self._ref_path_pub_dummy.publish(ref_path_marker_array)

    def _control(self):
        now = self.get_clock().now()
        t = (now - self._t_start).nanoseconds / 1e9
        dt = (now - self._last_t).nanoseconds / 1e9

        self._last_t = now
        self._loop += 1

        # record and print execution stats
        if self.use_stats:
            self._stats.record()

        # self.get_logger().info("loop")
        self._control_rate.sleep()

        if self._loop % 100 == 0:
            # update reference path
            if self._cfg.reference_path.update_by_topic: # type: ignore
                new_referece_path = self._create_reference_path_from_autoware_trajectory(self._trajectory)
                if new_referece_path is not None:
                    self._car.reference_path = new_referece_path
                    self._car.update_reference_path(self._car.reference_path)

            def plot_reference_path(car):
                import matplotlib.pyplot as plt
                import sys
                fig, ax = plt.subplots(1, 1)
                car.reference_path.show(ax)
                plt.show()
                sys.exit(1)
            # plot_reference_path(self._car)

        pose = odom_to_pose_2d(self._odom) # type: ignore
        v = self._odom.twist.twist.linear.x

        self._car.update_states(pose.x, pose.y, pose.theta, current_speed=v, dt_control=dt)
        self._car.update_reference_path(self._car.reference_path)

        # 衝突判定
        is_colliding = False
        if self.USE_OBSTACLE_AVOIDANCE:
            if self._car.is_colliding(self._static_obstacles):
                if not hasattr(self, '_last_colliding_time_static') or self._last_colliding_time_static is None:
                    self._last_colliding_time_static = now
                elif (now - self._last_colliding_time_static).nanoseconds / 1e9 > 0.5:
                    is_colliding = True
            else:
                self._last_colliding_time_static = None

        if self._last_colliding_time is not None:
            if (now - self._last_colliding_time).nanoseconds / 1e9 < 2.0:
                is_colliding = True

        # Phase 1: Process Lidar data if available
        if self._use_lidar_constraints and hasattr(self, '_lidar_mapper'):
            self._process_lidar_data(now)

        # Update obstacle map for MPC corridor computation
        if self.USE_OBSTACLE_AVOIDANCE and self._obstacles_updated:
            self._obstacles_updated = False
            self._map.reset_map()
            filtered_dynamic = self._filter_obstacles_to_corridor(self._dynamic_obstacles)
            
            # Add Lidar-based dynamic obstacles if available
            if self._use_lidar_constraints and hasattr(self, '_lidar_mapper'):
                lidar_constraints = self._lidar_mapper.get_dynamic_constraints()
                for obstacle in lidar_constraints['obstacles']:
                    # Convert Lidar obstacles to obstacle format
                    self._map.add_obstacles([Obstacle(obstacle['x'], obstacle['y'], obstacle['radius'])])
            
            self._map.add_obstacles(self._static_obstacles + self._detected_obstacles + filtered_dynamic)
            self._reference_path.reset_dynamic_constraints()

        with self._stats.time_block("control"):
            u, max_delta = self._mpc.get_control()

        if len(u) == 0:
            self.get_logger().error("MPC returned empty control, using emergency fallback")
            # Emergency fallback: always provide forward motion
            u = np.array([2.0, 0.0])  # Forward speed, straight steering
            self.get_logger().warn(f"Emergency fallback: v={u[0]}, delta={u[1]}")

        # Update optimization systems after control computation
        current_x = self._car.temporal_state.x
        current_y = self._car.temporal_state.y
        current_yaw = self._car.temporal_state.psi
        current_v = self._car.temporal_state.v if hasattr(self._car.temporal_state, 'v') else v

        # Stuck detection
        if not hasattr(self, '_last_pose_for_stuck'):
            self._last_pose_for_stuck = (current_x, current_y)
            
        dist_moved = math.hypot(current_x - self._last_pose_for_stuck[0], current_y - self._last_pose_for_stuck[1])
        self._last_pose_for_stuck = (current_x, current_y)
        
        if dist_moved > 1.0:
            # Teleportation (Place Kart) detected. Give a 5-second grace period to allow the 
            # 3-second race countdown to finish without falsely triggering reverse recovery.
            self._stuck_ticks = -int(self._mpc_cfg.control_rate * 5.0)
        else:
            is_stuck = False
            if abs(current_v) < 0.2:
                # Vehicle is barely moving despite having throttle commands
                if len(u) > 0 and abs(u[0]) > 0.5:
                    is_stuck = True
                elif hasattr(self, '_last_acc') and self._last_acc > 0.5:
                    is_stuck = True
                    
            if is_stuck:
                self._stuck_ticks += 1
            else:
                self._stuck_ticks = max(0, self._stuck_ticks - 1)

        # Trigger recovery if stuck for ~1.0s
        if self._stuck_ticks > int(self._mpc_cfg.control_rate * 0.5):
            self._reverse_ticks = int(self._mpc_cfg.control_rate * 0.5)
            self._stuck_ticks = 0
            
            # Find best reverse steering direction
            closest_idx = self._car.get_closest_waypoint(current_x, current_y)
            wp = self._reference_path.get_waypoint(closest_idx)
            dx = current_x - wp.x
            dy = current_y - wp.y
            e_y = -np.sin(wp.psi) * dx + np.cos(wp.psi) * dy
            self._reverse_steer_dir = self._evaluate_best_recovery_steer(current_x, current_y, current_yaw, e_y)
            self.get_logger().warn("Kart is stuck! Initiating reverse recovery...")

        # Apply recovery
        is_reversing = False
        if self._reverse_ticks > 0:
            is_reversing = True
            # Reverse at -2.5 m/s, max steer
            u = np.array([-2.5, self._reverse_steer_dir * np.deg2rad(25.0)])
            self._reverse_ticks -= 1
            if self._reverse_ticks == 0:
                self._realign_ticks = int(self._mpc_cfg.control_rate * 1.0)
        elif self._realign_ticks > 0:
            # Realign phase: Forward at 2.5 m/s, opposite steer
            u = np.array([2.5, -self._reverse_steer_dir * np.deg2rad(25.0)])
            self._realign_ticks -= 1

        self._was_reversing = is_reversing
        
        # Update friction estimator
        lateral_accel = 0.0  # Simplified, could be computed from steering and velocity
        self._friction_estimator.update(current_v, u[1] if len(u) > 1 else 0.0, lateral_accel)
        
        # Update lap learner
        self._lap_learner.update(current_x, current_y, current_yaw, current_v)
        
        # Check for lap completion (simple distance check)
        if len(self._lap_learner.current_lap_data) > 100:
            start_pos = self._lap_learner.current_lap_data[0]
            dist = math.sqrt((current_x - start_pos[0])**2 + (current_y - start_pos[1])**2)
            if dist < 5.0:
                self._lap_learner.complete_lap()
                self._lap_learner.start_lap()
                self.get_logger().info(f"Lap {self._lap_learner.lap_count} completed")
        
        # Adapt control based on friction estimation
        speed_factor = self._friction_estimator.get_speed_factor()
        if len(u) > 0 and speed_factor < 0.8:
            u[0] *= speed_factor  # Reduce speed on low friction
            if self._loop % 100 == 0:
                self.get_logger().info(f"Friction adaptation: factor={speed_factor:.2f}, condition={self._friction_estimator.road_condition}")

        # override by brake command if control is disabled
        if not self._enable_control:
            last_v_cmd = self._last_u[0]
            if last_v_cmd < 0.5:
                u = []
            else:
                decel_v = last_v_cmd + self._mpc_cfg.a_min * dt
                u = [np.clip(decel_v, 0.0, self._mpc_cfg.v_max), self._last_u[1]]

        if len(u) == 0:
            return

        acc = 0.
        bug_acc_enabled = False
        if self.USE_BUG_ACC:
            def deg2rad(deg):
                return deg * np.pi / 180.0

            if abs(v) > kmh_to_m_per_sec(44.0) or \
             (abs(v) > kmh_to_m_per_sec(38.0) and abs(max_delta) > deg2rad(12.0)):
                bug_acc_enabled = False
                acc = self._mpc_cfg.a_min / 3.0 * 2.0
            elif abs(v) > kmh_to_m_per_sec(41.0) or abs(u[1]) > deg2rad(10.0):
                bug_acc_enabled = False
                acc = self._mpc_cfg.a_max
            else:
                bug_acc_enabled = True
                acc = 500.0
        else:
            # Closed-loop speed tracking: acc = KP * (v_target - v_actual)
            # Use absolute speeds so that we apply positive throttle even in reverse gear
            target_speed = abs(u[0])
            actual_speed = abs(v)
            
            # If we're reversing but the car is rolling forward, we might need to brake first, 
            # but for simplicity let's just use the absolute error for throttle/brake.
            if u[0] < -0.1 and v > 0.5:
                # We want to go backward but we're moving forward -> brake hard
                acc = self._mpc_cfg.a_min
            else:
                acc = self.KP * (target_speed - actual_speed)
                
            acc = np.clip(acc, self._mpc_cfg.a_min, self._mpc_cfg.a_max)

        # apply low pass filter to control signal
        acc = self._last_acc + (acc - self._last_acc) * self._mpc_cfg.accel_low_pass_gain
        u[1] = self._last_u[1] + (u[1] - self._last_u[1]) * self._mpc_cfg.steer_low_pass_gain

        self._last_acc = acc
        self._last_u[0] = u[0]
        self._last_u[1] = u[1]

        # update car state
        self._car.drive(u)

        self._publish_control_command(now, u, acc, bug_acc_enabled)

        self._sim_logger.log(self._car, u, t)
        self._sim_logger.plot_animation(t, self._loop, self._current_laps, self._lap_times, is_colliding, u, self._mpc, self._car)

        # 予測結果およびリファレンスパスの表示
        if self._mpc.current_prediction is not None:
            self._publish_mpc_pred_marker(self._mpc.current_prediction[0], self._mpc.current_prediction[1]) # type: ignore
        if self._loop % 80 == 0:
            self._publish_ref_path_marker(self._car.reference_path)

    def run(self) -> None:
        self._wait_until_clock_received()
        self._wait_until_odom_received()
        self._wait_until_trajectory_received()
        self._wait_until_path_constraints_received()

        # initialize car states
        pose = odom_to_pose_2d(self._odom) # type: ignore
        self._car.update_states(pose.x, pose.y, pose.theta, current_speed=self._odom.twist.twist.linear.x, dt_control=1.0 / self._mpc_cfg.control_rate)
        self._car.update_reference_path(self._car.reference_path)

        self._publish_ref_path_marker(self._car.reference_path)

        if self.use_sim_time:
            self.get_logger().info("Simulation mode active: Auto-enabling control mode for autonomous driving!")
            self._enable_control = True

        self._pred_marker_color = CYAN

        # for i in range(10):
        #     self._obstacle_manager.push_next_obstacle()

        # initialize control states
        self._control_rate = self.create_rate(self._mpc_cfg.control_rate)
        self._sim_logger = SimulationLogger(
            self.get_logger(),
            self._car.temporal_state.x, self._car.temporal_state.y, self._cfg.sim_logger.animation_enabled, self.SHOW_PLOT_ANIMATION, self.PLOT_RESULTS, self.ANIMATION_INTERVAL) # type: ignore

        self._loop = 0
        self._last_acc = 0.0
        self._last_u = np.array([0.0, 0.0])
        self._t_start = self.get_clock().now()
        self._last_t = self._t_start

        self.get_logger().info("----------------------")
        self.get_logger().info("START!")
        self.get_logger().info("----------------------")

        while rclpy.ok() and (not self._sim_logger.stop_requested()):
            self._control()

    def stop(self):
        if self._odom is None:
            self.get_logger().warn("Shutdown called before odometry received.")
            return

        # Wait for stopping
        self.get_logger().warn("----------------------")
        self.get_logger().warn("Stopping...")
        self.get_logger().warn("----------------------")
        timeout_time = self.get_clock().now() + rclpy.time.Duration(seconds=5)
        while self._odom.twist.twist.linear.x > 0.1 and self.get_clock().now() < timeout_time:
            self._enable_control = False
            self._control()

        # Publish zero command to stop the car completely
        zero_cmd = self._create_ackerman_control_command(self.get_clock().now(), [0.0, 0.0], 0.0, False)
        self._command_pub.publish(zero_cmd)

        self.get_logger().warn(">> Stop Completed!")

        # show results
        self._sim_logger.show_results(self._current_laps, self._lap_times, self._car)

    @classmethod
    def in_pkg_share(cls, file_path: str) -> str:
        return cls.PKG_PATH + file_path
