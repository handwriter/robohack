"""ROS 2 adapter: /cmd_vel + /lowstate -> /lowcmd_bare (never /lowcmd)."""

from pathlib import Path

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from unitree_go.msg import LowCmd, LowState

from .policy import CONTROL_PERIOD, MAX_DT, WalkingPolicy, load_weights

N_MOTORS = 12
MODE_SERVO = 0x0A
INPUT_QOS = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)


class WalkingNode(Node):
    def __init__(self):
        super().__init__('go1_walking')
        share = Path(get_package_share_directory('go1_locomotion'))
        self.declare_parameter('checkpoint', str(share / 'policy' / 't3_weights.json'))
        self.declare_parameter('state_timeout', 0.12)
        self.state_timeout = float(self.get_parameter('state_timeout').value)
        if not np.isfinite(self.state_timeout) or self.state_timeout <= 0:
            raise ValueError('state_timeout must be positive and finite')
        self.policy = WalkingPolicy(load_weights(self.get_parameter('checkpoint').value))
        self.state = None
        self.last_state_t = None
        self.last_tick = None
        self.command = np.zeros(3)
        self.pub = self.create_publisher(LowCmd, '/lowcmd_bare', 1)
        self.create_subscription(LowState, '/lowstate', self.on_state, INPUT_QOS)
        self.create_subscription(Twist, '/cmd_vel', self.on_command, INPUT_QOS)
        self.create_timer(CONTROL_PERIOD, self.on_timer)
        self.get_logger().info('T3 policy loaded; waiting for valid /lowstate')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def on_command(self, msg):
        command = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        # The Judge may send just one setpoint. Latch valid commands, stop
        # on an explicit zero Twist (or on invalid data), not on silence.
        self.command = command if np.isfinite(command).all() else np.zeros(3)

    def on_state(self, msg):
        motors = msg.motor_state[:N_MOTORS]
        if len(motors) != N_MOTORS:
            return
        q = np.array([motor.q for motor in motors])
        dq = np.array([motor.dq for motor in motors])
        quat = np.asarray(msg.imu_state.quaternion, dtype=float)
        gyro = np.asarray(msg.imu_state.gyroscope, dtype=float)
        if quat.shape != (4,) or gyro.shape != (3,):
            return
        if not all(np.isfinite(v).all() for v in (q, dq, quat, gyro)):
            return
        if not 0.5 <= np.linalg.norm(quat) <= 1.5 or np.max(np.abs(q)) > 6.3:
            return
        now = self.now()
        if self.last_state_t is None or not 0 <= now - self.last_state_t <= self.state_timeout:
            self.policy.reset()
            self.last_tick = None
        self.state = q, dq, quat, gyro
        self.last_state_t = now

    def on_timer(self):
        now = self.now()
        if self.state is None or not 0 <= now - self.last_state_t <= self.state_timeout:
            # Let the existing Safety Block enter its hold mode.
            if self.state is not None:
                self.policy.reset()
            self.state = None
            self.last_state_t = None
            self.last_tick = None
            return
        dt = CONTROL_PERIOD if self.last_tick is None else now - self.last_tick
        if dt <= 0:
            return
        if dt > MAX_DT:
            self.policy.reset()
            dt = CONTROL_PERIOD
        self.last_tick = now
        target = self.policy.step(dt, *self.state, self.command)
        msg = LowCmd()
        for i in range(N_MOTORS):
            motor = msg.motor_cmd[i]
            motor.mode = MODE_SERVO
            motor.q = float(target.q[i])
            motor.dq = float(target.dq[i])
            motor.tau = float(target.tau[i])
            motor.kp = target.kp
            motor.kd = target.kd
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = None
    try:
        node = WalkingNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
