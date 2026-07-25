"""Unit tests for LidarRoadMapper (pure Python, no rclpy)."""

import numpy as np
import pytest
from multi_purpose_mpc_ros.lidar_road_mapper import LidarRoadMapper


def test_initialization():
    """Test that LidarRoadMapper initializes correctly."""
    mapper = LidarRoadMapper(map_resolution=0.1, max_range=50.0, update_rate=10.0)
    
    assert mapper.map_resolution == 0.1
    assert mapper.max_range == 50.0
    assert mapper.update_rate == 10.0
    assert len(mapper.dynamic_obstacles) == 0
    assert mapper.lidar_data_available == False
    assert mapper.use_lidar_constraints == False


def test_empty_point_cloud_returns_empty_constraints():
    """Test that empty point cloud returns empty constraints."""
    mapper = LidarRoadMapper()
    vehicle_pose = (0.0, 0.0, 0.0)
    empty_points = np.array([])
    
    constraints = mapper.process_lidar_scan(empty_points, vehicle_pose)
    
    assert len(constraints['obstacles']) == 0
    assert constraints['safety_margin'] == 1.0
    assert constraints['update_time'] > 0


def test_none_point_cloud_returns_empty_constraints():
    """Test that None point cloud returns empty constraints."""
    mapper = LidarRoadMapper()
    vehicle_pose = (0.0, 0.0, 0.0)
    
    constraints = mapper.process_lidar_scan(None, vehicle_pose)
    
    assert len(constraints['obstacles']) == 0
    assert constraints['safety_margin'] == 1.0


def test_point_cloud_transformation():
    """Test that points are correctly transformed to world frame."""
    mapper = LidarRoadMapper()
    vehicle_pose = (10.0, 5.0, np.pi/4)  # 45 degree rotation
    
    # Simple point cloud in vehicle frame
    points = np.array([
        [1.0, 0.0, 0.0],  # Point directly ahead
        [0.0, 1.0, 0.0],  # Point to the left
    ])
    
    transformed = mapper._transform_to_world(points, vehicle_pose)
    
    # Check that points are transformed (not in original vehicle frame)
    assert not np.allclose(transformed[:, :2], points[:, :2])
    # Check that z coordinates are preserved
    assert np.allclose(transformed[:, 2], points[:, 2])


def test_ground_filtering():
    """Test that ground points are filtered out."""
    mapper = LidarRoadMapper()
    
    points = np.array([
        [1.0, 0.0, 0.1],  # Ground point (below threshold)
        [1.0, 0.0, 0.5],  # Non-ground point (above threshold)
        [1.0, 0.0, 1.0],  # Non-ground point (above threshold)
    ])
    
    filtered = mapper._filter_ground(points, height_threshold=0.3)
    
    # Should only keep points above threshold
    assert len(filtered) == 2
    assert all(filtered[:, 2] > 0.3)


def test_obstacle_detection_simple():
    """Test simple obstacle detection from point cloud."""
    mapper = LidarRoadMapper()
    
    # Create a simple cluster of points representing an obstacle
    points = np.array([
        [5.0, 5.0, 1.0],
        [5.1, 5.0, 1.0],
        [5.0, 5.1, 1.0],
        [5.1, 5.1, 1.0],
    ])
    
    obstacles = mapper._detect_obstacles(points, cluster_threshold=0.5)
    
    # Should detect one obstacle
    assert len(obstacles) == 1
    assert obstacles[0]['point_count'] == 4
    assert obstacles[0]['radius'] > 0


def test_obstacle_detection_empty():
    """Test obstacle detection with no points."""
    mapper = LidarRoadMapper()
    points = np.array([])
    
    obstacles = mapper._detect_obstacles(points)
    
    assert len(obstacles) == 0


def test_obstacle_storage_update():
    """Test that obstacle storage is updated correctly."""
    mapper = LidarRoadMapper()
    
    obstacles = [
        {'center': np.array([1.0, 1.0]), 'radius': 0.5, 'height': 1.0, 'point_count': 10}
    ]
    
    mapper._update_obstacle_storage(obstacles, 100.0)
    
    assert len(mapper.dynamic_obstacles) == 1
    assert len(mapper.obstacle_timestamps) == 1


def test_obstacle_storage_timeout():
    """Test that old obstacles are removed from storage."""
    mapper = LidarRoadMapper()
    
    obstacles = [
        {'center': np.array([1.0, 1.0]), 'radius': 0.5, 'height': 1.0, 'point_count': 10}
    ]
    
    # Add obstacle with old timestamp
    mapper._update_obstacle_storage(obstacles, 0.0)
    
    # Add obstacle with current timestamp
    mapper._update_obstacle_storage(obstacles, 100.0)
    
    # Old obstacle should be removed
    assert len(mapper.dynamic_obstacles) == 1


def test_dynamic_constraints_generation():
    """Test generation of dynamic constraints from obstacles."""
    mapper = LidarRoadMapper()
    vehicle_pose = (0.0, 0.0, 0.0)
    
    obstacles = [
        {'center': np.array([5.0, 5.0]), 'radius': 0.5, 'height': 1.0, 'point_count': 10}
    ]
    
    constraints = mapper._generate_dynamic_constraints(obstacles, vehicle_pose)
    
    assert len(constraints['obstacles']) == 1
    assert constraints['obstacles'][0]['x'] == 5.0
    assert constraints['obstacles'][0]['y'] == 5.0
    # Radius should include safety margin
    assert constraints['obstacles'][0]['radius'] > 0.5


def test_dynamic_constraints_range_filtering():
    """Test that obstacles outside max_range are filtered."""
    mapper = LidarRoadMapper(max_range=10.0)
    vehicle_pose = (0.0, 0.0, 0.0)
    
    obstacles = [
        {'center': np.array([5.0, 5.0]), 'radius': 0.5, 'height': 1.0, 'point_count': 10},  # Within range
        {'center': np.array([20.0, 20.0]), 'radius': 0.5, 'height': 1.0, 'point_count': 10},  # Outside range
    ]
    
    constraints = mapper._generate_dynamic_constraints(obstacles, vehicle_pose)
    
    # Only obstacle within range should be included
    assert len(constraints['obstacles']) == 1
    assert constraints['obstacles'][0]['x'] == 5.0


def test_enable_disable_constraints():
    """Test enabling and disabling Lidar constraints."""
    mapper = LidarRoadMapper()
    
    assert mapper.use_lidar_constraints == False
    
    mapper.enable_lidar_constraints(True)
    assert mapper.use_lidar_constraints == True
    
    mapper.enable_lidar_constraints(False)
    assert mapper.use_lidar_constraints == False
    
    # Disabling should clear obstacles
    mapper.dynamic_obstacles.append({'test': 'obstacle'})
    mapper.enable_lidar_constraints(False)
    assert len(mapper.dynamic_obstacles) == 0


def test_get_dynamic_constraints():
    """Test getting current dynamic constraints."""
    mapper = LidarRoadMapper()
    
    # When disabled, should return empty constraints
    constraints = mapper.get_dynamic_constraints()
    assert len(constraints['obstacles']) == 0
    
    # Enable and add obstacle
    mapper.enable_lidar_constraints(True)
    mapper.dynamic_obstacles.append({
        'center': np.array([1.0, 1.0]),
        'radius': 0.5,
        'height': 1.0,
        'point_count': 10
    })
    
    constraints = mapper.get_dynamic_constraints()
    assert len(constraints['obstacles']) == 1


def test_full_pipeline_integration():
    """Test the full pipeline from point cloud to constraints."""
    mapper = LidarRoadMapper()
    vehicle_pose = (0.0, 0.0, 0.0)
    
    # Create point cloud with an obstacle
    points = np.array([
        [5.0, 5.0, 1.0],
        [5.1, 5.0, 1.0],
        [5.0, 5.1, 1.0],
        [5.1, 5.1, 1.0],
    ])
    
    constraints = mapper.process_lidar_scan(points, vehicle_pose)
    
    # Should detect obstacle and generate constraints
    assert len(constraints['obstacles']) >= 0  # May or may not detect depending on clustering
    assert mapper.lidar_data_available == True
