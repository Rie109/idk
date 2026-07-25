#!/usr/bin/env python3
"""
Simple Pure Pursuit Line Follower
A robust fallback controller that follows reference paths and avoids walls.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from autoware_auto_planning_msgs.msg import Trajectory
from autoware_auto_control_msgs.msg import AckermannControlCommand
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import LaserScan
import math


class SimpleLineFollower(Node):
    def __init__(self):
        super().__init__('simple_line_follower')
        
        # Parameters
        self.lookahead_distance = 2.0  # meters
        self.target_speed = 3.0  # m/s
        self.max_steering_angle = 0.5  # radians (~28 degrees)
        self.steering_gain = 0.8
        self.safety_distance = 1.5  # meters from walls
        
        # State
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.0
        self.current_speed = 0.0
        self.trajectory = None
        self.laser_scan = None
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/localization/kinematic_state', self.odom_callback, 10)
        self.trajectory_sub = self.create_subscription(
            Trajectory, '/planning/scenario_planning/trajectory', self.trajectory_callback, 10)
        self.laser_sub = self.create_subscription(
            LaserScan, '/sensing/lidar/top/scan_raw', self.laser_callback, 10)
        
        # Publisher
        self.control_pub = self.create_publisher(
            AckermannControlCommand, '/control/command/control_cmd', 10)
        
        # Timer
        self.timer = self.create_timer(0.05, self.control_loop)  # 20 Hz
        
        self.get_logger().info('Simple Line Follower initialized')

    def odom_callback(self, msg):
        """Update vehicle state from odometry"""
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        # Extract yaw from quaternion
        q = msg.pose.pose.orientation
        self.current_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.current_speed = msg.twist.twist.linear.x

    def trajectory_callback(self, msg):
        """Update reference trajectory"""
        self.trajectory = msg

    def laser_callback(self, msg):
        """Update laser scan for obstacle avoidance"""
        self.laser_scan = msg

    def find_lookahead_point(self):
        """Find the lookahead point on the trajectory"""
        if self.trajectory is None or len(self.trajectory.points) == 0:
            return None
        
        # Find closest point on trajectory
        min_dist = float('inf')
        closest_idx = 0
        
        for i, point in enumerate(self.trajectory.points):
            dx = point.pose.position.x - self.current_x
            dy = point.pose.position.y - self.current_y
            dist = math.sqrt(dx*dx + dy*dy)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Find lookahead point
        lookahead_idx = closest_idx
        accumulated_dist = 0.0
        
        for i in range(closest_idx, len(self.trajectory.points)):
            if i == 0:
                continue
            prev_point = self.trajectory.points[i-1]
            curr_point = self.trajectory.points[i]
            
            segment_dist = math.sqrt(
                (curr_point.pose.position.x - prev_point.pose.position.x)**2 +
                (curr_point.pose.position.y - prev_point.pose.position.y)**2
            )
            accumulated_dist += segment_dist
            
            if accumulated_dist >= self.lookahead_distance:
                lookahead_idx = i
                break
        
        return self.trajectory.points[lookahead_idx]

    def check_obstacle_avoidance(self, target_steering):
        """Modify steering to avoid obstacles based on laser scan"""
        if self.laser_scan is None:
            return target_steering
        
        # Check front sector for obstacles
        angle_min = self.laser_scan.angle_min
        angle_increment = self.laser_scan.angle_increment
        ranges = self.laser_scan.ranges
        
        # Define front sector (e.g., -30 to +30 degrees)
        front_sector_start_idx = int((-math.pi/6 - angle_min) / angle_increment)
        front_sector_end_idx = int((math.pi/6 - angle_min) / angle_increment)
        
        front_sector_start_idx = max(0, front_sector_start_idx)
        front_sector_end_idx = min(len(ranges), front_sector_end_idx)
        
        # Find minimum distance in front sector
        min_front_dist = float('inf')
        for i in range(front_sector_start_idx, front_sector_end_idx):
            if ranges[i] < min_front_dist and ranges[i] > 0.1:  # Filter invalid readings
                min_front_dist = ranges[i]
        
        # If obstacle detected, steer away
        if min_front_dist < self.safety_distance:
            self.get_logger().warn(f'Obstacle detected at {min_front_dist:.2f}m, avoiding')
            
            # Check left vs right sector
            left_sector_start = int((math.pi/6 - angle_min) / angle_increment)
            left_sector_end = int((math.pi/2 - angle_min) / angle_increment)
            right_sector_start = int((-math.pi/2 - angle_min) / angle_increment)
            right_sector_end = int((-math.pi/6 - angle_min) / angle_increment)
            
            left_dist = min(ranges[max(0, left_sector_start):min(len(ranges), left_sector_end)])
            right_dist = min(ranges[max(0, right_sector_start):min(len(ranges), right_sector_end)])
            
            # Steer towards the side with more space
            if left_dist > right_dist:
                return self.max_steering_angle * 0.8  # Steer left
            else:
                return -self.max_steering_angle * 0.8  # Steer right
        
        return target_steering

    def control_loop(self):
        """Main control loop"""
        if self.trajectory is None or len(self.trajectory.points) == 0:
            return
        
        # Find lookahead point
        lookahead_point = self.find_lookahead_point()
        if lookahead_point is None:
            return
        
        # Calculate heading to lookahead point
        dx = lookahead_point.pose.position.x - self.current_x
        dy = lookahead_point.pose.position.y - self.current_y
        target_heading = math.atan2(dy, dx)
        
        # Calculate heading error
        heading_error = target_heading - self.current_yaw
        # Normalize to [-pi, pi]
        heading_error = math.atan2(math.sin(heading_error), math.cos(heading_error))
        
        # Calculate steering angle using pure pursuit
        curvature = 2.0 * math.sin(heading_error) / self.lookahead_distance
        steering_angle = math.atan(curvature * 1.5)  # 1.5 is wheelbase approximation
        
        # Apply steering gain and limits
        steering_angle *= self.steering_gain
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        # Apply obstacle avoidance
        steering_angle = self.check_obstacle_avoidance(steering_angle)
        
        # Adjust speed based on steering angle (slow down for sharp turns)
        speed_factor = 1.0 - abs(steering_angle) / self.max_steering_angle * 0.5
        target_speed = self.target_speed * speed_factor
        
        # Ensure minimum speed
        target_speed = max(target_speed, 1.0)
        
        # Create control command
        control_msg = AckermannControlCommand()
        control_msg.stamp = self.get_clock().now().to_msg()
        control_msg.longitudinal.speed = target_speed
        control_msg.longitudinal.acceleration = 1.0
        control_msg.lateral.steering_tire_angle = steering_angle
        
        # Publish control command
        self.control_pub.publish(control_msg)
        
        # Debug logging
        if self.get_clock().now().nanoseconds % 5000000000 < 50000000:  # Every 5 seconds
            self.get_logger().info(
                f'Speed: {target_speed:.2f} m/s, Steering: {steering_angle:.3f} rad, '
                f'Heading error: {heading_error:.3f} rad'
            )


def main(args=None):
    rclpy.init(args=args)
    node = SimpleLineFollower()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
