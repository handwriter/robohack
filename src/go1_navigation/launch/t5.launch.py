"""Полный запуск T5: одометрия + map->odom + Nav2 (/navigate_to_pose) + ArUco.

  ros2 launch go1_navigation t5.launch.py

Структура: одометрия T4 (go1_odometry) даёт /odom и TF odom -> trunk; map_odom_node - TF
map -> odom; Nav2 (controller, planner, behavior, bt_navigator, velocity_smoother) принимает
цель Action /navigate_to_pose и публикует /cmd_vel; aruco_node публикует /aruco/detections.
Узлы Nav2 запускаются напрямую (а не через nav2_bringup), чтобы отключить bond-таймаут
lifecycle_manager: на загруженной машине он ложно останавливает навигацию.
"""
import os
import tempfile

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

LIFECYCLE_NODES = ['controller_server', 'planner_server', 'behavior_server', 'bt_navigator',
                   'velocity_smoother']


def build_params_file(share_dir):
    """Копия nav2_params.yaml с путём к дереву поведения; возвращает путь к файлу."""
    with open(os.path.join(share_dir, 'config', 'nav2_params.yaml')) as f:
        params = yaml.safe_load(f)
    params['bt_navigator']['ros__parameters']['default_nav_to_pose_bt_xml'] = \
        os.path.join(share_dir, 'behavior_trees', 'navigate_no_backup.xml')
    out = os.path.join(tempfile.gettempdir(), 'go1_nav2_params_%d.yaml' % os.getpid())
    with open(out, 'w') as f:
        yaml.safe_dump(params, f)
    return out


def _setup(context):
    share = get_package_share_directory('go1_navigation')
    params = build_params_file(share)
    sim = {'use_sim_time': True}
    actions = [
        Node(package='go1_odometry', executable='odometry_node', name='go1_odometry',
             output='screen', parameters=[sim]),
        Node(package='go1_navigation', executable='map_odom_node', name='go1_map_odom',
             output='screen', parameters=[sim]),
        Node(package='go1_navigation', executable='aruco_node', name='go1_aruco',
             output='screen', parameters=[sim]),
    ]
    if LaunchConfiguration('nav').perform(context).lower() in ('true', '1'):
        actions += [
            Node(package='nav2_controller', executable='controller_server', output='screen',
                 parameters=[params], remappings=[('cmd_vel', 'cmd_vel_nav')]),
            Node(package='nav2_planner', executable='planner_server', name='planner_server',
                 output='screen', parameters=[params]),
            Node(package='nav2_behaviors', executable='behavior_server', name='behavior_server',
                 output='screen', parameters=[params]),
            Node(package='nav2_bt_navigator', executable='bt_navigator', name='bt_navigator',
                 output='screen', parameters=[params]),
            Node(package='nav2_velocity_smoother', executable='velocity_smoother',
                 name='velocity_smoother', output='screen', parameters=[params],
                 remappings=[('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')]),
            Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
                 name='lifecycle_manager_navigation', output='screen',
                 parameters=[{'use_sim_time': True, 'autostart': True,
                              'node_names': LIFECYCLE_NODES, 'bond_timeout': 0.0}]),
        ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('nav', default_value='true',
                              description='false - только одометрия, map->odom и ArUco (без Nav2)'),
        OpaqueFunction(function=_setup),
    ])
