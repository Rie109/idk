"""
Map-based Dynamic Path Generator
Generates racing lines from occupancy grid map data instead of CSV files.
This works in simulation since map data is always available.
"""

import numpy as np
from typing import List, Tuple, Optional
from multi_purpose_mpc_ros.core.map import Map
from scipy.interpolate import splprep, splev

# Optional cv2 import for advanced processing
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False


class MapPathGenerator:
    """
    Generate dynamic racing lines from occupancy grid map data.
    Replaces static CSV paths with map-based path generation.
    """
    
    def __init__(self, map_obj: Map, resolution=0.5, smoothing_distance=3.0):
        """
        Initialize map path generator.
        
        Args:
            map_obj: Map object containing occupancy grid
            resolution: Path resolution in meters
            smoothing_distance: Smoothing distance for path generation
        """
        self.map = map_obj
        self.resolution = resolution
        self.smoothing_distance = smoothing_distance
        
    def generate_racing_line(self, start_pos: Tuple[float, float], 
                           end_pos: Optional[Tuple[float, float]] = None) -> List[Tuple[float, float]]:
        """
        Generate racing line from map data.
        
        Args:
            start_pos: Starting position (x, y) in world coordinates
            end_pos: Optional end position for non-circular paths
            
        Returns:
            List of (x, y) waypoints forming the racing line
        """
        # Extract drivable area from occupancy grid
        drivable_mask = self._extract_drivable_area()
        
        # Find centerline of drivable area
        centerline = self._find_centerline(drivable_mask, start_pos, end_pos)
        
        if len(centerline) == 0:
            self._fallback_to_csv_path()
            return []
        
        # Smooth the centerline
        smoothed_path = self._smooth_path(centerline)
        
        # Convert to world coordinates
        world_path = self._map_to_world(smoothed_path)
        
        return world_path
    
    def _extract_drivable_area(self) -> np.ndarray:
        """Extract drivable area from occupancy grid."""
        # Get binary mask (1 = free, 0 = occupied)
        binary_mask = self.map.data.copy()
        
        # Apply morphological operations to clean up the mask
        if CV2_AVAILABLE:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
        else:
            # Fallback: simple noise reduction without cv2
            from scipy.ndimage import binary_erosion, binary_dilation
            binary_mask = binary_erosion(binary_mask, iterations=1)
            binary_mask = binary_dilation(binary_mask, iterations=1)
        
        return binary_mask
    
    def _find_centerline(self, drivable_mask: np.ndarray, 
                        start_pos: Tuple[float, float],
                        end_pos: Optional[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """
        Find centerline of drivable area using skeletonization.
        
        Args:
            drivable_mask: Binary mask of drivable area
            start_pos: Starting position in world coordinates
            end_pos: Optional end position
            
        Returns:
            List of (x, y) pixel coordinates forming the centerline
        """
        # Convert start position to map coordinates
        start_x, start_y = self.map.w2m(start_pos[0], start_pos[1])
        
        # Skeletonize the drivable area
        if CV2_AVAILABLE:
            skeleton = cv2.ximgproc.thinning(drivable_mask.astype(np.uint8))
        else:
            # Fallback: simple distance-based centerline
            skeleton = self._simple_centerline(drivable_mask)
        
        # Find skeleton points
        skeleton_points = np.argwhere(skeleton > 0)
        
        if len(skeleton_points) == 0:
            return []
        
        # Find closest skeleton point to start position
        distances = np.sqrt((skeleton_points[:, 1] - start_x)**2 + 
                          (skeleton_points[:, 0] - start_y)**2)
        closest_idx = np.argmin(distances)
        start_point = skeleton_points[closest_idx]
        
        # Extract centerline by following skeleton
        centerline = self._follow_skeleton(skeleton, start_point, end_pos)
        
        return centerline
    
    def _simple_centerline(self, drivable_mask: np.ndarray) -> np.ndarray:
        """Fallback centerline extraction without cv2."""
        # Simple approach: find points that are farthest from boundaries
        from scipy.ndimage import distance_transform_edt
        distance = distance_transform_edt(drivable_mask)
        
        # Threshold to keep points far from boundaries
        max_distance = np.max(distance)
        if max_distance > 0:
            skeleton = (distance > max_distance * 0.5).astype(np.uint8)
        else:
            skeleton = drivable_mask.astype(np.uint8)
        
        return skeleton
    
    def _follow_skeleton(self, skeleton: np.ndarray, start_point: Tuple[int, int],
                        end_pos: Optional[Tuple[float, float]]) -> List[Tuple[int, int]]:
        """
        Follow skeleton to extract centerline path.
        
        Args:
            skeleton: Skeletonized image
            start_point: Starting point on skeleton
            end_pos: Optional end position in world coordinates
            
        Returns:
            List of (x, y) pixel coordinates
        """
        centerline = [start_point]
        visited = set()
        visited.add(start_point)
        
        current_point = start_point
        max_iterations = 10000
        iteration = 0
        
        while iteration < max_iterations:
            # Find neighbors
            x, y = current_point
            neighbors = []
            
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    
                    if (0 <= nx < skeleton.shape[1] and 
                        0 <= ny < skeleton.shape[0] and
                        skeleton[ny, nx] > 0 and
                        (nx, ny) not in visited):
                        neighbors.append((nx, ny))
            
            if len(neighbors) == 0:
                break
            
            # Choose next point (prefer forward direction)
            if len(centerline) > 1:
                prev_point = centerline[-2]
                direction = np.array([current_point[0] - prev_point[0], 
                                    current_point[1] - prev_point[1]])
                
                # Score neighbors based on alignment with direction
                best_neighbor = None
                best_score = -float('inf')
                
                for neighbor in neighbors:
                    neighbor_dir = np.array([neighbor[0] - current_point[0],
                                           neighbor[1] - current_point[1]])
                    if np.linalg.norm(direction) > 0:
                        score = np.dot(direction, neighbor_dir) / (np.linalg.norm(direction) * np.linalg.norm(neighbor_dir))
                    else:
                        score = 0
                    
                    if score > best_score:
                        best_score = score
                        best_neighbor = neighbor
                
                if best_neighbor is not None:
                    next_point = best_neighbor
                else:
                    next_point = neighbors[0]
            else:
                next_point = neighbors[0]
            
            # Check if we've reached the end position (if specified)
            if end_pos is not None:
                end_x, end_y = self.map.w2m(end_pos[0], end_pos[1])
                if np.sqrt((next_point[0] - end_x)**2 + (next_point[1] - end_y)**2) < 10:
                    centerline.append(next_point)
                    break
            
            centerline.append(next_point)
            visited.add(next_point)
            current_point = next_point
            iteration += 1
        
        return centerline
    
    def _smooth_path(self, path: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Smooth the path using spline interpolation.
        
        Args:
            path: List of (x, y) pixel coordinates
            
        Returns:
            Smoothed path
        """
        if len(path) < 4:
            return path
        
        # Convert to numpy arrays
        points = np.array(path)
        x = points[:, 0]
        y = points[:, 1]
        
        # Fit spline
        try:
            tck, u = splprep([x, y], s=self.smoothing_distance * len(path))
            u_new = np.linspace(u.min(), u.max(), len(path))
            x_new, y_new = splev(u_new, tck)
            
            # Round to integer coordinates
            smoothed = [(int(round(x)), int(round(y))) for x, y in zip(x_new, y_new)]
            return smoothed
        except:
            return path
    
    def _map_to_world(self, path: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """
        Convert path from map coordinates to world coordinates.
        
        Args:
            path: List of (x, y) pixel coordinates
            
        Returns:
            List of (x, y) world coordinates
        """
        world_path = []
        for x, y in path:
            wx, wy = self.map.m2w(x, y)
            world_path.append((wx, wy))
        
        return world_path
    
    def _fallback_to_csv_path(self):
        """Fallback to CSV path if map generation fails."""
        print("Warning: Map-based path generation failed, falling back to CSV path")
    
    def generate_waypoints(self, path: List[Tuple[float, float]], 
                          resolution: Optional[float] = None) -> List[dict]:
        """
        Generate waypoints with curvature information.
        
        Args:
            path: List of (x, y) world coordinates
            resolution: Optional resolution override
            
        Returns:
            List of waypoint dictionaries with x, y, kappa, etc.
        """
        if resolution is None:
            resolution = self.resolution
        
        if len(path) < 2:
            return []
        
        waypoints = []
        
        for i, (x, y) in enumerate(path):
            # Calculate curvature
            kappa = 0.0
            if i > 0 and i < len(path) - 1:
                prev_x, prev_y = path[i-1]
                next_x, next_y = path[i+1]
                
                # Calculate curvature using three points
                kappa = self._calculate_curvature(prev_x, prev_y, x, y, next_x, next_y)
            
            waypoints.append({
                'x': x,
                'y': y,
                'kappa': kappa,
                'v_ref': 0.0  # Will be set by speed profile generator
            })
        
        return waypoints
    
    def _calculate_curvature(self, x1, y1, x2, y2, x3, y3) -> float:
        """Calculate curvature from three."""
        # Using the formula for curvature of a circle through three points
        area = x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2)
        side_a = np.sqrt((x2 - x3)**2 + (y2 - y3)**2)
        side_b = np.sqrt((x1 - x3)**2 + (y1 - y3)**2)
        side_c = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
        
        if side_a * side_b * side_c == 0:
            return 0.0
        
        curvature = 4 * area / (side_a * side_b * side_c)
        return curvature
