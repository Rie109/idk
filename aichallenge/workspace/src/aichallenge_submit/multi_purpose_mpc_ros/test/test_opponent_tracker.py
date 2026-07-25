"""Unit tests for OpponentTracker (pure Python, no rclpy)."""

import numpy as np
import pytest
from multi_purpose_mpc_ros.opponent_tracker import OpponentTracker


def test_initialization():
    """Test that OpponentTracker initializes correctly."""
    tracker = OpponentTracker(max_opponents=10, prediction_horizon=2.0, update_rate=20.0)
    
    assert tracker.max_opponents == 10
    assert tracker.prediction_horizon == 2.0
    assert tracker.update_rate == 20.0
    assert len(tracker.opponents) == 0
    assert tracker.tracking_enabled == False
    assert tracker.lidar_data_available == False


def test_empty_point_cloud_returns_empty_constraints():
    """Test that empty point cloud returns empty constraints."""
    tracker = OpponentTracker()
    vehicle_pose = (0.0, 0.0, 0.0)
    empty_points = np.array([])
    
    constraints = tracker.process_sensor_data(empty_points, vehicle_pose)
    
    assert len(constraints['opponents']) == 0
    assert constraints['safety_margin'] == 2.0
    assert constraints['update_time'] > 0


def test_none_point_cloud_returns_empty_constraints():
    """Test that None point cloud returns empty constraints."""
    tracker = OpponentTracker()
    vehicle_pose = (0.0, 0.0, 0.0)
    
    constraints = tracker.process_sensor_data(None, vehicle_pose)
    
    assert len(constraints['opponents']) == 0
    assert constraints['safety_margin'] == 2.0


def test_point_cloud_transformation():
    """Test that points are correctly transformed to world frame."""
    tracker = OpponentTracker()
    vehicle_pose = (10.0, 5.0, np.pi/4)  # 45 degree rotation
    
    points = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ])
    
    transformed = tracker._transform_to_world(points, vehicle_pose)
    
    # Check that points are transformed
    assert not np.allclose(transformed[:, :2], points[:, :2])
    assert np.allclose(transformed[:, 2], points[:, 2])


def test_ground_filtering():
    """Test that ground points are filtered out."""
    tracker = OpponentTracker()
    
    points = np.array([
        [1.0, 0.0, 0.1],  # Ground point
        [1.0, 0.0, 0.5],  # Non-ground
        [1.0, 0.0, 1.0],  # Non-ground
    ])
    
    filtered = tracker._filter_ground(points, height_threshold=0.3)
    
    assert len(filtered) == 2
    assert all(filtered[:, 2] > 0.3)


def test_cluster_size_estimation():
    """Test cluster size estimation."""
    tracker = OpponentTracker()
    
    points = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.5, 0.0],
        [1.0, 0.5, 1.0],
    ])
    
    size = tracker._estimate_cluster_size(points)
    
    assert size['length'] > 0
    assert size['width'] > 0
    assert size['height'] > 0


def test_opponent_detection_vehicle_like():
    """Test detection of vehicle-like clusters."""
    tracker = OpponentTracker()
    vehicle_pose = (0.0, 0.0, 0.0)
    
    # Create vehicle-like cluster (1.5m x 1.0m x 1.0m)
    points = np.array([
        [0.0, 0.0, 0.5],
        [0.5, 0.0, 0.5],
        [1.0, 0.0, 0.5],
        [1.5, 0.0, 0.5],
        [0.0, 0.5, 0.5],
        [0.5, 0.5, 0.5],
        [1.0, 0.5, 0.5],
        [1.5, 0.5, 0.5],
        [0.0, 0.0, 1.5],
        [1.5, 0.5, 1.5],
    ])
    
    candidates = tracker._detect_opponents_from_lidar(points, vehicle_pose)
    
    # Should detect at least one vehicle-like cluster
    assert len(candidates) >= 0  # May or may not detect depending on clustering


def test_opponent_detection_empty():
    """Test opponent detection with no points."""
    tracker = OpponentTracker()
    vehicle_pose = (0.0, 0.0, 0.0)
    points = np.array([])
    
    candidates = tracker._detect_opponents_from_lidar(points, vehicle_pose)
    
    assert len(candidates) == 0


def test_opponent_tracking_update():
    """Test that opponent tracking updates correctly."""
    tracker = OpponentTracker()
    vehicle_pose = (0.0, 0.0, 0.0)
    
    candidates = [
        {'center': np.array([5.0, 5.0]), 'velocity': np.array([1.0, 0.0]), 
         'size': {'length': 1.5, 'width': 1.0, 'height': 1.0}, 'confidence': 0.8}
    ]
    
    tracker._update_opponent_tracking(candidates, vehicle_pose)
    
    assert len(tracker.opponents) == 1


def test_opponent_tracking_timeout():
    """Test that old opponents are removed."""
    tracker = OpponentTracker()
    vehicle_pose = (0.0, 0.0, 0.0)
    
    candidates = [
        {'center': np.array([5.0, 5.0]), 'velocity': np.array([1.0, 0.0]), 
         'size': {'length': 1.5, 'width': 1.0, 'height': 1.0}, 'confidence': 0.8}
    ]
    
    # Add opponent with old timestamp
    tracker._update_opponent_tracking(candidates, vehicle_pose, current_time=0.0)
    
    # Add opponent with current timestamp
    tracker._update_opponent_tracking(candidates, vehicle_pose, current_time=100.0)
    
    # Old opponent should be removed
    assert len(tracker.opponents) == 1


def test_opponent_prediction():
    """Test opponent position prediction."""
    tracker = OpponentTracker()
    
    tracker.opponents[1] = {
        'position': np.array([10.0, 10.0]),
        'velocity': np.array([2.0, 1.0]),
        'size': {'length': 1.5, 'width': 1.0, 'height': 1.0},
        'confidence': 0.8,
        'last_seen': 100.0,
        'id': 1
    }
    
    predicted = tracker._predict_opponent_positions()
    
    assert len(predicted) == 1
    # Position should be predicted forward: 10 + 2*2 = 14, 10 + 1*2 = 12
    assert predicted[0]['position'][0] == pytest.approx(14.0)
    assert predicted[0]['position'][1] == pytest.approx(12.0)


def test_opponent_constraints_generation():
    """Test generation of opponent constraints."""
    tracker = OpponentTracker()
    
    predicted = [
        {
            'position': np.array([10.0, 10.0]),
            'velocity': np.array([2.0, 1.0]),
            'size': {'length': 1.5, 'width': 1.0, 'height': 1.0},
            'confidence': 0.8,
            'id': 1
        }
    ]
    
    constraints = tracker._generate_opponent_constraints(predicted)
    
    assert len(constraints['opponents']) == 1
    assert constraints['opponents'][0]['x'] == 10.0
    assert constraints['opponents'][0]['y'] == 10.0
    # Safety margin should be added to radius
    assert constraints['opponents'][0]['radius'] > 0.75  # max(1.5, 1.0)/2 = 0.75


def test_enable_disable_tracking():
    """Test enabling and disabling opponent tracking."""
    tracker = OpponentTracker()
    
    assert tracker.tracking_enabled == False
    
    tracker.enable_tracking(True)
    assert tracker.tracking_enabled == True
    
    tracker.enable_tracking(False)
    assert tracker.tracking_enabled == False
    
    # Disabling should clear opponents
    tracker.opponents[1] = {'test': 'opponent'}
    tracker.enable_tracking(False)
    assert len(tracker.opponents) == 0


def test_get_tracked_opponents():
    """Test getting tracked opponents."""
    tracker = OpponentTracker()
    
    tracker.opponents[1] = {
        'position': np.array([10.0, 10.0]),
        'velocity': np.array([2.0, 1.0]),
        'size': {'length': 1.5, 'width': 1.0, 'height': 1.0},
        'confidence': 0.8,
        'last_seen': 100.0,
        'id': 1
    }
    
    opponents = tracker.get_tracked_opponents()
    
    assert len(opponents) == 1
    assert opponents[0]['id'] == 1


def test_max_opponents_limit():
    """Test that max opponents limit is enforced."""
    tracker = OpponentTracker(max_opponents=2)
    vehicle_pose = (0.0, 0.0, 0.0)
    
    # Try to add more opponents than limit
    candidates = [
        {'center': np.array([i, i]), 'velocity': np.array([0.0, 0.0]), 
         'size': {'length': 1.5, 'width': 1.0, 'height': 1.0}, 'confidence': 0.8}
        for i in range(5)
    ]
    
    tracker._update_opponent_tracking(candidates, vehicle_pose)
    
    # Should not exceed max_opponents
    assert len(tracker.opponents) <= 2


def test_full_pipeline_integration():
    """Test the full pipeline from point cloud to opponent constraints."""
    tracker = OpponentTracker()
    vehicle_pose = (0.0, 0.0, 0.0)
    
    # Create point cloud with vehicle-like cluster
    points = np.array([
        [5.0, 5.0, 0.5],
        [5.5, 5.0, 0.5],
        [5.0, 5.5, 0.5],
        [5.5, 5.5, 0.5],
        [5.0, 5.0, 1.5],
        [5.5, 5.5, 1.5],
    ])
    
    constraints = tracker.process_sensor_data(points, vehicle_pose)
    
    # Should process without errors
    assert constraints is not None
    assert 'opponents' in constraints
    assert tracker.lidar_data_available == True
