#!/usr/bin/env python3
"""
Sensor Fusion State Estimation
Combines GNSS, IMU, and wheel odometry for robust state estimation.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped
import math
from collections import deque
from typing import Optional, Tuple
import threading


class SensorFusion(Node):
    def __init__(self):
        super().__init__('sensor_fusion')
        
        # Parameters
        self.gnss_weight = 0.3  # Weight for GNSS position
        self.imu_weight = 0.4   # Weight for IMU orientation
        self.odom_weight = 0.3  # Weight for wheel odometry
        
        # State
        self.fused_position = np.array([0.0, 0.0, 0.0])  # x, y, z
        self.fused_orientation = np.array([0.0, 0.0, 0.0, 1.0])  # quaternion
        self.fused_velocity = np.array([0.0, 0.0, 0.0])
        self.fused_angular_velocity = np.array([0.0, 0.0, 0.0])
        
        # Sensor data buffers
        self.gnss_buffer = deque(maxlen=10)
        self.imu_buffer = deque(maxlen=20)
        self.odom_buffer = deque(maxlen=20)
        
        # Timestamps for synchronization
        self.last_gnss_time = None
        self.last_imu_time = None
        self.last_odom_time = None
        
        # Kalman filter state
        self.state = np.zeros(6)  # x, y, yaw, vx, vy, vyaw
        self.P = np.eye(6) * 0.1  # Covariance matrix
        
        # Subscribers
        self.gnss_sub = self.create_subscription(
            Odometry, '/localization/kinematic_state', self.gnss_callback, 10)
        self.imu_sub = self.create_subscription(
            Imu, '/sensing/imu/imu_data', self.imu_callback, 10)
        self.odom_sub = self.create_subscription(
            TwistWithCovarianceStamped, '/localization/twist_estimator/twist_with_covariance', 
            self.odom_callback, 10)
        
        # Publishers
        self.fused_odom_pub = self.create_publisher(
            Odometry, '/localization/fused_state', 10)
        
        # Timer for fusion and publishing
        self.timer = self.create_timer(0.02, self.fuse_and_publish)  # 50 Hz
        
        # Lock for thread safety
        self.lock = threading.Lock()
        
        self.get_logger().info('Sensor Fusion initialized')
    
    def gnss_callback(self, msg):
        """Receive GNSS position data"""
        with self.lock:
            self.gnss_buffer.append({
                'position': np.array([
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z
                ]),
                'orientation': np.array([
                    msg.pose.pose.orientation.x,
                    msg.pose.pose.orientation.y,
                    msg.pose.pose.orientation.z,
                    msg.pose.pose.orientation.w
                ]),
                'velocity': np.array([
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z
                ]),
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            })
            self.last_gnss_time = self.gnss_buffer[-1]['timestamp']
    
    def imu_callback(self, msg):
        """Receive IMU orientation and angular velocity data"""
        with self.lock:
            self.imu_buffer.append({
                'orientation': np.array([
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                    msg.orientation.w
                ]),
                'angular_velocity': np.array([
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ]),
                'linear_acceleration': np.array([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ]),
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            })
            self.last_imu_time = self.imu_buffer[-1]['timestamp']
    
    def odom_callback(self, msg):
        """Receive wheel odometry data"""
        with self.lock:
            self.odom_buffer.append({
                'linear': np.array([
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z
                ]),
                'angular': np.array([
                    msg.twist.twist.angular.x,
                    msg.twist.twist.angular.y,
                    msg.twist.twist.angular.z
                ]),
                'timestamp': self.get_clock().now().nanoseconds / 1e9
            })
            self.last_odom_time = self.odom_buffer[-1]['timestamp']
    
    def fuse_and_publish(self):
        """Fuse sensor data and publish fused state"""
        with self.lock:
            if len(self.gnss_buffer) == 0 or len(self.imu_buffer) == 0 or len(self.odom_buffer) == 0:
                return
            
            # Get latest synchronized data
            gnss_data = self.gnss_buffer[-1]
            imu_data = self.imu_buffer[-1]
            odom_data = self.odom_buffer[-1]
            
            # Fuse position (weighted average)
            self.fused_position = (
                self.gnss_weight * gnss_data['position'] +
                0.0 * imu_data['orientation'][:3] +  # IMU doesn't provide position
                0.0 * odom_data['linear']  # Odom doesn't provide absolute position
            )
            
            # Fuse orientation (IMU is primary, GNSS as backup)
            self.fused_orientation = (
                self.imu_weight * imu_data['orientation'] +
                self.gnss_weight * gnss_data['orientation'] +
                0.0 * odom_data['angular']  # Odom doesn't provide orientation
            )
            
            # Normalize quaternion
            norm = np.linalg.norm(self.fused_orientation)
            if norm > 0:
                self.fused_orientation /= norm
            
            # Fuse velocity (wheel odometry is primary, GNSS as backup)
            self.fused_velocity = (
                self.odom_weight * odom_data['linear'] +
                self.gnss_weight * gnss_data['velocity'] +
                0.0 * imu_data['linear_acceleration']  # IMU provides acceleration, not velocity
            )
            
            # Fuse angular velocity (IMU is primary, odom as backup)
            self.fused_angular_velocity = (
                self.imu_weight * imu_data['angular_velocity'] +
                self.odom_weight * odom_data['angular']
            )
            
            # Apply Kalman filter for improved estimation
            self.kalman_filter_update(gnss_data, imu_data, odom_data)
            
            # Publish fused odometry
            self.publish_fused_odometry()
    
    def kalman_filter_update(self, gnss_data, imu_data, odom_data):
        """Extended Kalman filter for state estimation"""
        # Prediction step (simplified)
        dt = 0.02  # 50 Hz
        
        # State transition matrix (constant velocity model)
        F = np.eye(6)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # yaw += vyaw * dt
        
        # Predict state
        self.state = F @ self.state
        
        # Predict covariance
        Q = np.eye(6) * 0.01  # Process noise
        self.P = F @ self.P @ F.T + Q
        
        # Measurement update (simplified)
        # Measurement from GNSS position
        z_pos = np.array([gnss_data['position'][0], gnss_data['position'][1], 
                        self.quaternion_to_yaw(gnss_data['orientation'])])
        
        # Measurement matrix (position only)
        H = np.zeros((3, 6))
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # yaw
        
        # Measurement noise
        R = np.eye(3) * 0.1
        
        # Kalman gain
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        
        # Update state
        y = z_pos - H @ self.state
        self.state = self.state + K @ y
        
        # Update covariance
        self.P = (np.eye(6) - K @ H) @ self.P
    
    def quaternion_to_yaw(self, q):
        """Convert quaternion to yaw angle"""
        yaw = math.atan2(2.0 * (q[3] * q[2] + q[0] * q[1]),
                        1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2]))
        return yaw
    
    def publish_fused_odometry(self):
        """Publish fused odometry message"""
        msg = Odometry()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        msg.child_frame_id = "base_link"
        
        # Position
        msg.pose.pose.position.x = self.state[0]
        msg.pose.pose.position.y = self.state[1]
        msg.pose.pose.position.z = 0.0
        
        # Orientation (yaw from state)
        yaw = self.state[2]
        msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
        msg.pose.pose.orientation.x = 0.0
        msg.pose.pose.orientation.y = 0.0
        msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
        
        # Velocity
        msg.twist.twist.linear.x = self.state[3]
        msg.twist.twist.linear.y = self.state[4]
        msg.twist.twist.linear.z = 0.0
        
        # Angular velocity
        msg.twist.twist.angular.x = 0.0
        msg.twist.twist.angular.y = 0.0
        msg.twist.twist.angular.z = self.state[5]
        
        self.fused_odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = SensorFusion()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
