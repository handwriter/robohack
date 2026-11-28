#!/usr/bin/env bash
set -eu

if [ "$#" -ne 1 ]; then
    exit 64
fi

case "$1" in
    ping)
        printf '%s\n' '{"protocol_version":"1.0","supported_tasks":["T2","T3","T4","T5"],"self_trained_rl":true,"preferred_speed_mps":0.4}'
        ;;
    T2)
        set +u
        source /opt/ros/humble/setup.bash
        source /opt/ws/install/setup.bash
        set -u
        exec ros2 run go1_safety safety_node
        ;;
    T3)
        set +u
        source /opt/ros/humble/setup.bash
        source /opt/ws/install/setup.bash
        set -u
        exec ros2 launch go1_locomotion t3.launch.py
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
