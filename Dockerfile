ARG BASE_IMAGE=registry.cookcloud.ru/base_images/ros2-base-humble-amd64@sha256:4d4a184901f154849d59477a077ce507f31fdf63b8bedd8caea460607eec243d
FROM ${BASE_IMAGE}

ENV ROS_DISTRO=humble \
    RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

USER root
COPY entrypoint.sh /entrypoint.sh
RUN chmod 0755 /entrypoint.sh \
    && chown ros:ros /entrypoint.sh

USER ros
WORKDIR /home/ros/app
ENTRYPOINT ["/entrypoint.sh"]
