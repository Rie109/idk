"""
Opponent Tracker Module
Separates opponent prediction from MPC for cleaner architecture.
Phase 1: Standalone opponent tracking using Lidar data
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import math


class OpponentTracker:
    """
    Tracks opponent vehicles using Lidar and sensor data.
    Provides obstacle constraints to MPC without being embedded in control logic.
    """
    
    def __init__(self, max_opponents=10, prediction_horizon=2.0, update_rate=20.0):
        """
        Initialize opponent tracker.
        
        Args:
            max_opponents: Maximum number of opponents to track
            prediction_horizon: Prediction horizon in seconds
            update_rate: Update rate in Hz
        """
        self.max_opponents = max_opponents
        self.prediction_horizon = prediction_horizon
        self.update_rate = update_rate
        
        # Opponent storage
        self.opponents = {}  # {id: OpponentState}
        self.opponent_id_counter = 0
        
        # Tracking history for prediction
        self.tracking_history = deque(maxlen=50)
        
        # Integration flags
        self.tracking_enabled = False
        self.lidar_data_available = False
        
    def process_sensor_data(self, lidar_points: np.ndarray, vehicle_pose: Tuple[float, float, float], 
                           camera_detections: Optional[List[dict]] = None) -> Dict[str, List[dict]]:
        """
        Process sensor data to track opponents.
        
        Args:
            lidar_points: Nx3 array of Lidar points in vehicle frame
            vehicle_pose: (x, y, yaw) vehicle pose in world frame
            camera_detections: Optional list of camera-based detections
            
        Returns:
            Dictionary containing opponent constraints for MPC
        """
        if lidar_points is None or len(lidar_points) == 0:
            return self._get_empty_opponent_constraints()
        
        # Detect potential opponents from Lidar
        potential_opponents = self._detect_opponents_from_lidar(lidar_points, vehicle_pose)
        
        # Update tracking with new detections
        self._update_opponent_tracking(potential_opponents, vehicle_pose)
        
        # Predict opponent positions
        predicted_opponents = self._predict_opponent_positions()
        
        # Generate constraints for MPC
        constraints = self._generate_opponent_constraints(predicted_opponents)
        
        self.lidar_data_available = True
        return constraints
    
    def _detect_opponents_from_lidar(self, points: np.ndarray, vehicle_pose: Tuple[float, float, float]) -> List[dict]:
        """
        Detect opponent vehicles from Lidar point cloud.
        
        Args:
            points: Nx3 array of Lidar points
            vehicle_pose: Current vehicle pose
            
        Returns:
            List of detected opponent candidates
        """
        # Transform to world frame
        world_points = self._transform_to_world(points, vehicle_pose)
        
        # Filter points within reasonable opponent size range
        candidates = []
        
        # Simple clustering for opponent detection
        visited = np.zeros(len(world_points), dtype=bool)
        
        for i in range(len(world_points)):
            if visited[i]:
                continue
            
            # Find cluster
            distances = np.sqrt(np.sum((world_points - world_points[i])**2, axis=1))
            cluster_indices = distances < 1.0  # 1m clustering radius
            visited[cluster_indices] = True
            
            cluster_points = world_points[cluster_indices]
            
            # Check if cluster size matches vehicle dimensions
            cluster_size = self._estimate_cluster_size(cluster_points)
            
            # Vehicle-like dimensions: length 1-2m, width 0.5-1.5m, height 0.5-1.5m
            if (1.0 < cluster_size['length'] < 2.5 and 
                0.5 < cluster_size['width'] < 2.0 and
                0.3 < cluster_size['height'] < 2.0):
                
                center = np.mean(cluster_points[:, :2], axis=0)
                velocity = self._estimate_velocity(center, vehicle_pose)
                
                candidates.append({
                    'center': center,
                    'velocity': velocity,
                    'size': cluster_size,
                    'confidence': min(1.0, len(cluster_points) / 50.0)
                })
        
        return candidates
    
    def _transform_to_world(self, points: np.ndarray, vehicle_pose: Tuple[float, float, float]) -> np.ndarray:
        """Transform points from vehicle frame to world frame."""
        x, y, yaw = vehicle_pose
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        
        R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])
        points_2d = points[:, :2]
        transformed = (R @ points_2d.T).T
        transformed[:, 0] += x
        transformed[:, 1] += y
        
        return transformed
    
    def _estimate_cluster_size(self, points: np.ndarray) -> Dict[str, float]:
        """Estimate physical dimensions of point cluster."""
        if len(points) == 0:
            return {'length': 0, 'width': 0, 'height': 0}
        
        # Bounding box dimensions
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        return {
            'length': max_coords[0] - min_coords[0],
            'width': max_coords[1] - min_coords[1],
            'height': max_coords[2] - min_coords[2] if points.shape[1] >= 3 else 0.5
        }
    
    def _estimate_velocity(self, center: np.ndarray, vehicle_pose: Tuple[float, float, float]) -> np.ndarray:
        """Estimate velocity of detected opponent."""
        # Simple estimation based on relative motion
        # In full implementation, use tracking history
        return np.array([0.0, 0.0])  # Placeholder
    
    def _update_opponent_tracking(self, candidates: List[dict], vehicle_pose: Tuple[float, float, float]):
        """Update opponent tracking with new detections."""
        current_time = self._get_current_time()
        
        # Simple data association based on distance
        for candidate in candidates:
            matched = False
            
            for opp_id, opponent in self.opponents.items():
                distance = np.linalg.norm(candidate['center'] - opponent['position'])
                if distance < 2.0:  # Association threshold
                    # Update existing opponent
                    opponent['position'] = candidate['center']
                    opponent['velocity'] = candidate['velocity']
                    opponent['last_seen'] = current_time
                    opponent['confidence'] = candidate['confidence']
                    matched = True
                    break
            
            if not matched and len(self.opponents) < self.max_opponents:
                # Create new opponent
                self.opponent_id_counter += 1
                self.opponents[self.opponent_id_counter] = {
                    'position': candidate['center'],
                    'velocity': candidate['velocity'],
                    'size': candidate['size'],
                    'confidence': candidate['confidence'],
                    'last_seen': current_time,
                    'id': self.opponent_id_counter
                }
        
        # Remove old opponents
        to_remove = []
        for opp_id, opponent in self.opponents.items():
            if current_time - opponent['last_seen'] > 3.0:  # 3 second timeout
                to_remove.append(opp_id)
        
        for opp_id in to_remove:
            del self.opponents[opp_id]
    
    def _predict_opponent_positions(self) -> List[dict]:
        """Predict opponent positions for prediction horizon."""
        predicted = []
        current_time = self._get_current_time()
        
        for opponent in self.opponents.values():
            # Simple constant velocity prediction
            dt = self.prediction_horizon
            predicted_position = opponent['position'] + opponent['velocity'] * dt
            
            predicted.append({
                'position': predicted_position,
                'velocity': opponent['velocity'],
                'size': opponent['size'],
                'confidence': opponent['confidence'],
                'id': opponent['id']
            })
        
        return predicted
    
    def _generate_opponent_constraints(self, predicted_opponents: List[dict]) -> Dict[str, List[dict]]:
        """Generate obstacle constraints for MPC from predicted opponents."""
        constraints = {
            'opponents': [],
            'safety_margin': 2.0,  # Larger safety margin for opponents
            'update_time': self._get_current_time()
        }
        
        for opponent in predicted_opponents:
            # Create safety zone around predicted position
            safety_radius = max(opponent['size']['length'], opponent['size']['width']) / 2 + constraints['safety_margin']
            
            constraints['opponents'].append({
                'x': opponent['position'][0],
                'y': opponent['position'][1],
                'radius': safety_radius,
                'velocity': opponent['velocity'],
                'confidence': opponent['confidence'],
                'id': opponent['id']
            })
        
        return constraints
    
    def _get_empty_opponent_constraints(self) -> Dict[str, List[dict]]:
        """Return empty constraints when no sensor data available."""
        return {
            'opponents': [],
            'safety_margin': 2.0,
            'update_time': 0.0
        }
    
    def _get_current_time(self) -> float:
        """Get current simulation time."""
        import time
        return time.time()
    
    def enable_tracking(self, enable: bool):
        """Enable or disable opponent tracking."""
        self.tracking_enabled = enable
        if not enable:
            self.opponents.clear()
    
    def get_tracked_opponents(self) -> List[dict]:
        """Get currently tracked opponents."""
        return list(self.opponents.values())
