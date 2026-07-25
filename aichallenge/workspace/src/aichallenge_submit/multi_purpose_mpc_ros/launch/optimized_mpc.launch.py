from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
from pathlib import Path


def generate_launch_description():
    return LaunchDescription([
        # Sensor Fusion System
        Node(
            package='multi_purpose_mpc_ros',
            executable='sensor_fusion.py',
            name='sensor_fusion',
            output='both',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': True,
            }],
        ),
        
        # Friction Estimator
        Node(
            package='multi_purpose_mpc_ros',
            executable='friction_estimator.py',
            name='friction_estimator',
            output='both',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': True,
            }],
        ),
        
        # Lap Learner
        Node(
            package='multi_purpose_mpc_ros',
            executable='lap_learner.py',
            name='lap_learner',
            output='both',
            emulate_tty=True,
            parameters=[{
                'use_sim_time': True,
            }],
        ),
        
        # MPC Controller with optimization
        Node(
            package='multi_purpose_mpc_ros',
            executable='run_mpc_controller.bash',
            name='mpc_controller',
            output='both',
            emulate_tty=True,
            arguments=[
                "--config_path",
                str(Path(get_package_share_directory("multi_purpose_mpc_ros")) / "config/config.yaml"),
                "--ref_vel_path",
                str(Path(get_package_share_directory("multi_purpose_mpc_ros")) / "config/ref_vel.yaml"),
                "--ros-args",
                "--log-level",
                "info",
            ],
            parameters=[
                {"use_boost_acceleration": False},
                {"use_obstacle_avoidance": False},
                {"use_stats": False},
                {"use_sim_time": True},
            ],
        ),
    ])
