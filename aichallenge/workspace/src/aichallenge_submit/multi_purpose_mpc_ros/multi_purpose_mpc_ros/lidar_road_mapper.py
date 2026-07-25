"""
Lidar-based Road Mapper for Dynamic Constraints
Phase 1: Hybrid approach - augment CSV path with Lidar-based dynamic constraints
"""

import numpy as np
from typing import List, Tuple, Optional
from collections import deque
import math


class LidarRoadMapper:
    """
    Real-time road mapping using Lidar data for dynamic constraints.
    Works alongside existing CSV-based path for hybrid approach.
    """
    
    def __init__(self, map_resolution=0.1, max_range=50.0, update_rate=10.0):
        """
        Initialize Lidar road mapper.
        
        Args:
            map_resolution: Grid cell size in meters
            max_range: Maximum Lidar range in meters
            update_rate: Map update rate in Hz
        """
        self.map_resolution = map_resolution
        self.max_range = max_range
        self.update_rate = update_rate
        
        # Dynamic obstacle storage
        self.dynamic_obstacles = deque(maxlen=100)
        self.obstacle_timestamps = deque(maxlen=100)
        
        # Road boundary detection
        self.road_boundaries = {'left': [], 'right': []}
        self.drivable_area = None
        
        # Integration flags
        self.use_lidar_constraints = False
        self.lidar_data_available = False
        
    def process_lidar_scan(self, point_cloud: np.ndarray, vehicle_pose: Tuple[float, float, float]) -> dict:
        """
        Process Lidar point cloud and extract dynamic constraints.
        
        Args:
            point_cloud: Nx3 array of (x, y, z) points in vehicle frame
            vehicle_pose: (x, y, yaw) vehicle pose in world frame
            
        Returns:
            Dictionary containing dynamic constraints
        """
        if point_cloud is None or len(point_cloud) == 0:
            return self._get_empty_constraints()
        
        # Transform points to world frame
        world_points = self._transform_to_world(point_cloud, vehicle_pose)
        
        # Filter ground points
        non_ground_points = self._filter_ground(world_points)
        
        # Detect dynamic obstacles
        obstacles = self._detect_obstacles(non_ground_points)
        
        # Update obstacle storage
        current_time = self._get_current_time()
        self._update_obstacle_storage(obstacles, current_time)
        
        # Generate dynamic constraints
        constraints = self._generate_dynamic_constraints(obstacles, vehicle_pose)
        
        self.lidar_data_available = True
        return constraints
    
    def _transform_to_world(self, points: np.ndarray, vehicle_pose: Tuple[float, float, float]) -> np.ndarray:
        """Transform points from vehicle frame to world frame."""
        x, y, yaw = vehicle_pose
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        # Rotation matrix
        R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        
        # Transform 2D points (x, y)
        points_2d = points[:, :2]
        transformed = (R @ points_2d.T).T
        
        # Translation
        transformed[:, 0] += x
        transformed[:, 1] += y
        
        return transformed
    
    def _filter_ground(self, points: np.ndarray, height_threshold=0.3) -> np.ndarray:
        """Filter out ground points based on height."""
        if points.shape[1] >= 3:
            # Keep points above ground threshold
            return points[points[:, 2] > height_threshold]
        return points
    
    def _detect_obstacles(self, points: np.ndarray, cluster_threshold=0.5) -> List[dict]:
        """
        Detect obstacles from point cloud using simple clustering.
        
        Args:
            points: Nx3 array of world frame points
            cluster_threshold: Distance threshold for clustering
            
        Returns:
            List of obstacle dictionaries
        """
        if len(points) == 0:
            return []
        
        obstacles = []
        
        # Simple distance-based clustering
        visited = np.zeros(len(points), dtype=bool)
        
        for i in range(len(points)):
            if visited[i]:
                continue
            
            # Find nearby points
            distances = np.sqrt(np.sum((points - points[i])**2, axis=1))
            cluster_indices = distances < cluster_threshold
            visited[cluster_indices] = True
            
            cluster_points = points[cluster_indices]
            
            # Extract obstacle properties
            center = np.mean(cluster_points[:, :2], axis=0)
            radius = np.max(np.sqrt(np.sum((cluster_points[:, :2] - center)**2, axis=1)))
            height = np.max(cluster_points[:, 2]) - np.min(cluster_points[:, 2])
            
            obstacles.append({
                'center': center,
                'radius': radius,
                'height': height,
                'point_count': len(cluster_points)
            })
        
        return obstacles
    
    def _update_obstacle_storage(self, obstacles: List[dict], timestamp: float):
        """Update obstacle storage with new detections."""
        for obstacle in obstacles:
            self.dynamic_obstacles.append(obstacle)
            self.obstacle_timestamps.append(timestamp)
        
        # Remove old obstacles
        current_time = self._get_current_time()
        while len(self.obstacle_timestamps) > 0 and current_time - self.obstacle_timestamps[0] > 2.0:
            self.dynamic_obstacles.popleft()
            self.obstacle_timestamps.popleft()
    
    def _generate_dynamic_constraints(self, obstacles: List[dict], vehicle_pose: Tuple[float, float, float]) -> dict:
        """
        Generate dynamic constraints for MPC from detected obstacles.
        
        Args:
            obstacles: List of detected obstacles
            vehicle_pose: Current vehicle pose
            
        Returns:
            Dictionary containing dynamic constraints
        """
        constraints = {
            'obstacles': [],
            'safety_margin': 1.0,
            'update_time': self._get_current_time()
        }
        
        vehicle_x, vehicle_y, _ = vehicle_pose
        
        # Filter obstacles within relevant range
        for obstacle in obstacles:
            obs_x, obs_y = obstacle['center']
            distance = math.sqrt((obs_x - vehicle_x)**2 + (obs_y - vehicle_y)**2)
            
            if distance < self.max_range:
                constraints['obstacles'].append({
                    'x': obs_x,
                    'y': obs_y,
                    'radius': obstacle['radius'] + constraints['safety_margin']
                })
        
        return constraints
    
    def _get_empty_constraints(self) -> dict:
        """Return empty constraints when no Lidar data available."""
        return {
            'obstacles': [],
            'safety_margin': 1.0,
            'update_time': 0.0
        }
    
    def _get_current_time(self) -> float:
        """Get current simulation time."""
        # This should be replaced with actual ROS time in integration
        import time
        return time.time()
    
    def enable_lidar_constraints(self, enable: bool):
        """Enable or disable Lidar-based constraints."""
        self.use_lidar_constraints = enable
        if not enable:
            self.dynamic_obstacles.clear()
            self.obstacle_timestamps.clear()
    
    def get_dynamic_constraints(self) -> dict:
        """Get current dynamic constraints."""
        if not self.use_lidar_constraints or not self.lidar_data_available:
            return self._get_empty_constraints()
        
        return {
            'obstacles': list(self.dynamic_obstacles),
            'safety_margin': 1.0,
            'update_time': self._get_current_time()
        }
