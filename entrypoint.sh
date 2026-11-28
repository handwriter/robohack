#!/usr/bin/env bash
set -eu

if [ "$#" -ne 1 ]; then
    exit 64
fi

case "$1" in
    ping)
        printf '%s\n' '{"protocol_version":"1.0","supported_tasks":["T2","T4","T5"],"self_trained_rl":false,"preferred_speed_mps":null}'
        ;;
    T2)
        set +u
        source /opt/ros/humble/setup.bash
        source /opt/ws/install/setup.bash
        set -u
        exec ros2 run go1_safety safety_node
        ;;
    T4)
        set +u
        source /opt/ros/humble/setup.bash
        source /opt/ws/install/setup.bash
        set -u
        exec ros2 run go1_odometry odometry_node --ros-args -p use_sim_time:=true
        ;;
    T5)
        set +u
        source /opt/ros/humble/setup.bash
        source /opt/ws/install/setup.bash
        set -u
        exec ros2 launch go1_navigation t5.launch.py
        ;;
    *)
        exit 64
        ;;
esac
