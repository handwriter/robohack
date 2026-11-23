"""Запуск одометрии T4:  ros2 launch go1_odometry odometry.launch.py use_lidar:=true"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_lidar', default_value='true'),
        Node(package='go1_odometry', executable='odometry_node', name='go1_odometry',
             output='screen',
             parameters=[{'use_sim_time': True,
                          'use_lidar': ParameterValue(LaunchConfiguration('use_lidar'),
                                                      value_type=bool)}]),
    ])
