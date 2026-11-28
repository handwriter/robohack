"""Run the T3 policy and the mandatory, unchanged T2 Safety Block."""

from launch import LaunchDescription
from launch.actions import Shutdown
from launch_ros.actions import Node


def generate_launch_description():
    sim_time = {'use_sim_time': True}
    return LaunchDescription([
        Node(package='go1_safety', executable='safety_node', output='screen',
             parameters=[sim_time], on_exit=Shutdown(reason='T3 Safety Block stopped')),
        Node(package='go1_locomotion', executable='walking_node', output='screen',
             parameters=[sim_time], on_exit=Shutdown(reason='T3 walking node stopped')),
    ])
