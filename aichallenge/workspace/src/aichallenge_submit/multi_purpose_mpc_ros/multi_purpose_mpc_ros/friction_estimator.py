#!/usr/bin/env python3
"""
Real-time Friction Estimation System
Estimates tire grip and road conditions for adaptive control.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from autoware_auto_control_msgs.msg import AckermannControlCommand
from geometry_msgs.msg import AccelWithCovarianceStamped
import math
from collections import deque
from typing import Tuple


class FrictionEstimator(Node):
    def __init__(self):
        super().__init__('friction_estimator')
        
        # Parameters
        self.window_size = 50  # Number of samples for estimation
        self.min_velocity = 2.0  # m/s (below this, estimation is unreliable)
        self.max_friction = 1.2  # Maximum expected friction coefficient
        self.min_friction = 0.5  # Minimum expected friction coefficient
        
        # State
        self.velocity_history = deque(maxlen=self.window_size)
        self.steering_history = deque(maxlen=self.window_size)
        self.acceleration_history = deque(maxlen=self.window_size)
        self.lateral_accel_history = deque(maxlen=self.window_size)
        
        self.estimated_friction = 0.9  # Default value
        self.friction_confidence = 0.0
        self.road_condition = "unknown"
        
        # Subscribers
        self.odom_sub = self.create_subscription(
            Odometry, '/localization/kinematic_state', self.odom_callback, 10)
        self.control_sub = self.create_subscription(
            AckermannControlCommand, '/control/command/control_cmd', self.control_callback, 10)
        self.accel_sub = self.create_subscription(
            AccelWithCovarianceStamped, '/localization/acceleration', self.accel_callback, 10)
        
        # Timer for periodic estimation
        self.timer = self.create_timer(0.1, self.estimate_friction)
        
        self.get_logger().info('Friction Estimator initialized')
    
    def odom_callback(self, msg):
        """Record velocity data"""
        v = msg.twist.twist.linear.x
        self.velocity_history.append(v)
    
    def control_callback(self, msg):
        """Record steering data"""
        delta = msg.lateral.steering_tire_angle
        self.steering_history.append(delta)
    
    def accel_callback(self, msg):
        """Record acceleration data"""
        ax = msg.accel.accel.linear.x
        ay = msg.accel.accel.linear.y
        self.acceleration_history.append(ax)
        self.lateral_accel_history.append(ay)
    
    def estimate_friction(self):
        """Estimate friction coefficient from vehicle dynamics"""
        if len(self.velocity_history) < self.window_size:
            return
        
        # Get current velocity
        current_v = self.velocity_history[-1]
        
        # Only estimate when moving at sufficient speed
        if current_v < self.min_velocity:
            return
        
        # Calculate lateral acceleration from steering and velocity
        # ay = v^2 * tan(delta) / L (simplified bicycle model)
        if len(self.steering_history) > 0:
            delta = self.steering_history[-1]
            wheelbase = 1.087  # From config
            predicted_ay = (current_v ** 2) * math.tan(delta) / wheelbase
        else:
            predicted_ay = 0.0
        
        # Get measured lateral acceleration
        if len(self.lateral_accel_history) > 0:
            measured_ay = self.lateral_accel_history[-1]
        else:
            measured_ay = 0.0
        
        # Friction estimation based on lateral acceleration
        # mu = |ay| / g (simplified)
        g = 9.81
        lateral_accel = abs(measured_ay)
        
        # Smooth estimation
        instant_friction = lateral_accel / g
        
        # Apply limits
        instant_friction = np.clip(instant_friction, self.min_friction, self.max_friction)
        
        # Exponential moving average for smooth estimation
        alpha = 0.1  # Smoothing factor
        self.estimated_friction = alpha * instant_friction + (1 - alpha) * self.estimated_friction
        
        # Calculate confidence based on data quality
        self.calculate_confidence()
        
        # Determine road condition
        self.determine_road_condition()
        
        # Log periodically
        if self.get_clock().now().nanoseconds % 500000000 < 50000000:  # Every 0.5 seconds
            self.get_logger().info(
                f'Friction: {self.estimated_friction:.3f}, '
                f'Confidence: {self.friction_confidence:.2f}, '
                f'Condition: {self.road_condition}'
            )
    
    def calculate_confidence(self):
        """Calculate confidence in friction estimation"""
        # Confidence based on:
        # 1. Data window fullness
        # 2. Velocity stability
        # 3. Steering activity
        
        window_fullness = len(self.velocity_history) / self.window_size
        
        if len(self.velocity_history) > 10:
            velocity_std = np.std(list(self.velocity_history)[-10:])
            velocity_stability = 1.0 / (1.0 + velocity_std)
        else:
            velocity_stability = 0.5
        
        if len(self.steering_history) > 10:
            steering_activity = np.mean(np.abs(list(self.steering_history)[-10:])) / 0.5  # Normalized
            steering_activity = np.clip(steering_activity, 0.0, 1.0)
        else:
            steering_activity = 0.5
        
        self.friction_confidence = 0.4 * window_fullness + 0.3 * velocity_stability + 0.3 * steering_activity
    
    def determine_road_condition(self):
        """Determine road condition from friction estimate"""
        if self.friction_confidence < 0.3:
            self.road_condition = "unknown"
        elif self.estimated_friction > 0.9:
            self.road_condition = "dry"
        elif self.estimated_friction > 0.7:
            self.road_condition = "damp"
        elif self.estimated_friction > 0.5:
            self.road_condition = "wet"
        else:
            self.road_condition = "slippery"
    
    def get_adapted_parameters(self) -> dict:
        """Get control parameters adapted to current friction"""
        # Reduce max speed and lateral acceleration on low friction
        speed_factor = np.clip(self.estimated_friction / 0.9, 0.5, 1.0)
        lateral_accel_factor = np.clip(self.estimated_friction / 0.9, 0.6, 1.0)
        
        return {
            'speed_factor': speed_factor,
            'lateral_accel_factor': lateral_accel_factor,
            'friction': self.estimated_friction,
            'confidence': self.friction_confidence,
            'road_condition': self.road_condition
        }


def main(args=None):
    rclpy.init(args=args)
    node = FrictionEstimator()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
