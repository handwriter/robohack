ARG BASE_IMAGE=registry.cookcloud.ru/base_images/ros2-base-humble-amd64@sha256:4d4a184901f154849d59477a077ce507f31fdf63b8bedd8caea460607eec243d
FROM ${BASE_IMAGE}

ARG UNITREE_ROS2_REF=668d1ec5a05d1c38d3306bdca7d59f2ba3581a88

ENV ROS_DISTRO=humble \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3-pip ros-humble-rosidl-generator-dds-idl \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir "numpy<2" "mujoco==3.3.7"

RUN git clone https://github.com/unitreerobotics/unitree_ros2.git /tmp/unitree_ros2 \
    && cd /tmp/unitree_ros2 && git checkout ${UNITREE_ROS2_REF} \
    && mkdir -p /opt/ws/src \
    && cp -r /tmp/unitree_ros2/cyclonedds_ws/src/unitree/unitree_go /opt/ws/src/ \
    && rm -rf /tmp/unitree_ros2

COPY go1_safety /opt/ws/src/go1_safety
RUN . /opt/ros/humble/setup.sh \
    && cd /opt/ws \
    && colcon build --packages-select unitree_go go1_safety \
    && rm -rf build log \
    && chmod -R a+rX /opt/ws

COPY entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh \
    && chown ros:ros /entrypoint.sh

USER ros
ENV ROS_HOME=/tmp/ros
WORKDIR /home/ros/app
ENTRYPOINT ["/entrypoint.sh"]
