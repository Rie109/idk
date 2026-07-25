#!/usr/bin/env python3
"""
Lap Learning System for Autonomous Racing
Learns optimal racing lines over multiple laps and adapts reference trajectory.
"""

import numpy as np
import json
import os
from typing import List, Tuple, Dict
from dataclasses import dataclass
from collections import deque
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from autoware_auto_planning_msgs.msg import Trajectory
from geometry_msgs.msg import PoseStamped
import math


@dataclass
class LapData:
    """Stores data for a single lap"""
    lap_number: int
    trajectory: List[Tuple[float, float, float]]  # x, y, yaw
    velocities: List[float]
    steering_angles: List[float]
    lap_time: float
    timestamp: float


class LapLearner(Node):
    def __init__(self):
        super().__init__('lap_learner')
        
        # Parameters
        self.max_laps_to_store = 10
        self.learning_rate = 0.3  # How much to adapt based on new data
        self.min_samples_per_segment = 5
        self.curve_detection_threshold = 0.1  # radians/m
        
        # State
        self.current_lap_data = LapData(
            lap_number=0,
            trajectory=[],
            velocities=[],
            steering_angles=[],
            lap_time=0.0,
            timestamp=0.0
        )
        self.lap_history: List[LapData] = []
        self.lap_start_time = None
        self.lap_count = 0
        self.current_position = None
        self.lap_completed = False
        
        # Learned optimal path
        self.learned_path = None
        self.path_segments = []
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/localization/kinematic_state', self.odom_callback, 10)
        self.trajectory_sub = self.create_subscription(
            Trajectory, '/planning/scenario_planning/trajectory', self.trajectory_callback, 10)
        
        # Publishers
        self.learned_path_pub = self.create_publisher(
            Trajectory, '/planning/learned_trajectory', 10)
        
        # Timer for periodic analysis
        self.timer = self.create_timer(1.0, self.analyze_progress)
        
        self.get_logger().info('Lap Learner initialized')
    
    def odom_callback(self, msg):
        """Record vehicle state for learning"""
        if self.lap_start_time is None:
            self.lap_start_time = self.get_clock().now()
        
        # Extract position and orientation
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                        1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        
        # Extract velocity
        v = msg.twist.twist.linear.x
        
        # Store data
        self.current_lap_data.trajectory.append((x, y, yaw))
        self.current_lap_data.velocities.append(v)
        
        # Detect lap completion (simple distance-based check)
        if len(self.current_lap_data.trajectory) > 100:
            start_pos = self.current_lap_data.trajectory[0]
            current_pos = (x, y, yaw)
            dist = math.sqrt((current_pos[0] - start_pos[0])**2 + 
                           (current_pos[1] - start_pos[1])**2)
            
            if dist < 5.0 and not self.lap_completed:
                self.complete_lap()
    
    def trajectory_callback(self, msg):
        """Store reference trajectory for comparison"""
        self.reference_trajectory = msg
    
    def complete_lap(self):
        """Process completed lap data"""
        if self.lap_start_time is None:
            return
        
        lap_time = (self.get_clock().now() - self.lap_start_time).nanoseconds / 1e9
        self.current_lap_data.lap_time = lap_time
        self.current_lap_data.timestamp = self.get_clock().now().nanoseconds / 1e9
        
        # Store lap data
        self.lap_history.append(self.current_lap_data)
        self.lap_count += 1
        
        self.get_logger().info(f'Lap {self.lap_count} completed in {lap_time:.2f}s')
        
        # Learn from this lap
        if len(self.lap_history) >= 2:
            self.learn_optimal_path()
        
        # Reset for next lap
        self.current_lap_data = LapData(
            lap_number=self.lap_count + 1,
            trajectory=[],
            velocities=[],
            steering_angles=[],
            lap_time=0.0,
            timestamp=0.0
        )
        self.lap_start_time = None
        self.lap_completed = False
        
        # Keep only recent laps
        if len(self.lap_history) > self.max_laps_to_store:
            self.lap_history.pop(0)
    
    def learn_optimal_path(self):
        """Learn optimal racing line from lap history"""
        if len(self.lap_history) < 2:
            return
        
        self.get_logger().info('Learning optimal racing line from lap history...')
        
        # Align trajectories from different laps
        aligned_trajectories = self.align_trajectories()
        
        # Find optimal segments
        optimal_segments = self.find_optimal_segments(aligned_trajectories)
        
        # Create learned path
        self.learned_path = self.create_learned_trajectory(optimal_segments)
        
        # Publish learned path
        if self.learned_path is not None:
            self.learned_path_pub.publish(self.learned_path)
            self.get_logger().info('Published learned optimal trajectory')
    
    def align_trajectories(self) -> List[List[Tuple[float, float, float]]]:
        """Align trajectories from different laps to common reference points"""
        aligned = []
        
        for lap_data in self.lap_history:
            # Resample trajectory to uniform spacing
            resampled = self.resample_trajectory(lap_data.trajectory, spacing=1.0)
            aligned.append(resampled)
        
        return aligned
    
    def resample_trajectory(self, trajectory: List[Tuple[float, float, float]], 
                          spacing: float) -> List[Tuple[float, float, float]]:
        """Resample trajectory to uniform spacing"""
        if len(trajectory) < 2:
            return trajectory
        
        resampled = [trajectory[0]]
        current_idx = 0
        
        while current_idx < len(trajectory) - 1:
            # Find next point at desired spacing
            for i in range(current_idx + 1, len(trajectory)):
                dist = math.sqrt((trajectory[i][0] - resampled[-1][0])**2 +
                               (trajectory[i][1] - resampled[-1][1])**2)
                if dist >= spacing:
                    # Interpolate to exact spacing
                    ratio = spacing / dist
                    x = resampled[-1][0] + ratio * (trajectory[i][0] - resampled[-1][0])
                    y = resampled[-1][1] + ratio * (trajectory[i][1] - resampled[-1][1])
                    
                    # Interpolate yaw
                    yaw_diff = trajectory[i][2] - resampled[-1][2]
                    # Normalize yaw difference
                    yaw_diff = math.atan2(math.sin(yaw_diff), math.cos(yaw_diff))
                    yaw = resampled[-1][2] + ratio * yaw_diff
                    
                    resampled.append((x, y, yaw))
                    current_idx = i
                    break
            else:
                break
        
        return resampled
    
    def find_optimal_segments(self, aligned_trajectories: List[List[Tuple[float, float, float]]]) -> List[Tuple[float, float, float]]:
        """Find optimal segments by analyzing performance at each point"""
        if not aligned_trajectories:
            return []
        
        # Use longest trajectory as reference
        max_len = max(len(traj) for traj in aligned_trajectories)
        reference_traj = aligned_trajectories[0] if len(aligned_trajectories[0]) == max_len else aligned_trajectories[
            np.argmax([len(traj) for traj in aligned_trajectories])]
        
        optimal_segments = []
        
        for i in range(len(reference_traj)):
            # Collect corresponding points from all trajectories
            points = []
            velocities = []
            
            for j, traj in enumerate(aligned_trajectories):
                if i < len(traj):
                    points.append(traj[i])
                    velocities.append(self.lap_history[j].velocities[min(i, len(self.lap_history[j].velocities)-1)])
            
            if len(points) < self.min_samples_per_segment:
                optimal_segments.append(reference_traj[i])
                continue
            
            # Find optimal point based on velocity and smoothness
            optimal_idx = self.select_optimal_point(points, velocities)
            optimal_segments.append(points[optimal_idx])
        
        return optimal_segments
    
    def select_optimal_point(self, points: List[Tuple[float, float, float]], 
                           velocities: List[float]) -> int:
        """Select optimal point from multiple candidates"""
        if len(points) == 1:
            return 0
        
        # Score each point based on velocity and smoothness
        scores = []
        for i, (point, vel) in enumerate(zip(points, velocities)):
            # Higher velocity is better
            velocity_score = vel / max(velocities) if max(velocities) > 0 else 0
            
            # Smoother path is better (less deviation from mean)
            mean_x = np.mean([p[0] for p in points])
            mean_y = np.mean([p[1] for p in points])
            deviation = math.sqrt((point[0] - mean_x)**2 + (point[1] - mean_y)**2)
            smoothness_score = 1.0 / (1.0 + deviation)
            
            # Combined score
            score = 0.7 * velocity_score + 0.3 * smoothness_score
            scores.append(score)
        
        return np.argmax(scores)
    
    def create_learned_trajectory(self, optimal_segments: List[Tuple[float, float, float]]) -> Trajectory:
        """Create ROS trajectory message from learned segments"""
        if not optimal_segments:
            return None
        
        trajectory_msg = Trajectory()
        
        for i, (x, y, yaw) in enumerate(optimal_segments):
            point = Trajectory.Point()
            point.pose.position.x = x
            point.pose.position.y = y
            point.pose.position.z = 0.0
            
            # Create quaternion from yaw
            point.pose.orientation.w = math.cos(yaw / 2.0)
            point.pose.orientation.x = 0.0
            point.pose.orientation.y = 0.0
            point.pose.orientation.z = math.sin(yaw / 2.0)
            
            # Set longitudinal velocity (can be optimized further)
            point.longitudinal_velocity_mps = 5.0  # Default, can be learned
            
            trajectory_msg.points.append(point)
        
        return trajectory_msg
    
    def analyze_progress(self):
        """Periodic analysis of learning progress"""
        if len(self.lap_history) >= 2:
            # Compare lap times
            recent_times = [lap.lap_time for lap in self.lap_history[-3:]]
            if len(recent_times) >= 2:
                improvement = recent_times[-1] - recent_times[-2]
                if improvement < 0:
                    self.get_logger().info(f'Lap time improved by {-improvement:.2f}s')
                else:
                    self.get_logger().info(f'Lap time increased by {improvement:.2f}s')


def main(args=None):
    rclpy.init(args=args)
    node = LapLearner()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
