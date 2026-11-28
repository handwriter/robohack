ARG BASE_IMAGE=registry.cookcloud.ru/base_images/ros2-base-humble-amd64@sha256:4d4a184901f154849d59477a077ce507f31fdf63b8bedd8caea460607eec243d
FROM ${BASE_IMAGE}

ARG UNITREE_ROS2_REF=668d1ec5a05d1c38d3306bdca7d59f2ba3581a88

ENV ROS_DISTRO=humble \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

USER root
RUN sed -i -e 's|http://archive.ubuntu.com|https://archive.ubuntu.com|g' \
           -e 's|http://security.ubuntu.com|https://security.ubuntu.com|g' \
           /etc/apt/sources.list /etc/apt/sources.list.d/* \
    && printf 'Acquire::Retries "3";\nAcquire::http::Timeout "60";\nAcquire::https::Timeout "60";\n' \
       > /etc/apt/apt.conf.d/80robohack \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3-pip python3-scipy python3-opencv python3-yaml \
        ros-humble-rosidl-generator-dds-idl \
        ros-humble-tf2-ros-py ros-humble-navigation2 \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir "numpy<2" "mujoco==3.3.7"

RUN git clone https://github.com/unitreerobotics/unitree_ros2.git /tmp/unitree_ros2 \
    && cd /tmp/unitree_ros2 && git checkout ${UNITREE_ROS2_REF} \
    && mkdir -p /opt/ws/src \
    && cp -r /tmp/unitree_ros2/cyclonedds_ws/src/unitree/unitree_go /opt/ws/src/ \
    && rm -rf /tmp/unitree_ros2

COPY go1_safety /opt/ws/src/go1_safety
COPY src/go1_locomotion /opt/ws/src/go1_locomotion
COPY src/go1_odometry /opt/ws/src/go1_odometry
COPY src/go1_navigation /opt/ws/src/go1_navigation
RUN . /opt/ros/humble/setup.sh \
    && cd /opt/ws \
    && colcon build --packages-select unitree_go go1_safety go1_locomotion go1_odometry go1_navigation \
    && rm -rf build log \
    && chmod -R a+rX /opt/ws

COPY entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh \
    && chown ros:ros /entrypoint.sh

USER ros
ENV ROS_HOME=/tmp/ros
WORKDIR /home/ros/app
ENTRYPOINT ["/entrypoint.sh"]
