import os
import time

import rclpy
from ament_index_python.packages import get_package_share_directory
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rosgraph_msgs.msg import Clock
from unitree_go.msg import LowCmd

from go1_safety.collision import CollisionChecker
from go1_safety.safety import N_JOINTS, SafetyFilter

MODE_SERVO = 0x0A


class SafetyNode(Node):
    def __init__(self):
        super().__init__('go1_safety')
        self.declare_parameter('rate', 500.0)
        self.declare_parameter('timeout', 0.25)
        self.declare_parameter('hold_kp', 40.0)
        self.declare_parameter('hold_kd', 2.0)
        model_dir = os.path.join(get_package_share_directory('go1_safety'), 'models', 'unitree_go1')
        self.declare_parameter('model_dir', model_dir)

        self.timeout = self.get_parameter('timeout').value
        checker = CollisionChecker(self.get_parameter('model_dir').value)
        self.filter = SafetyFilter(checker,
                                   hold_kp=self.get_parameter('hold_kp').value,
                                   hold_kd=self.get_parameter('hold_kd').value)

        self.last_wall = None
        self.last_sim = None
        self.sim_time = None

        in_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST)
        self.pub = self.create_publisher(LowCmd, '/lowcmd', 10)
        self.create_subscription(LowCmd, '/lowcmd_bare', self.on_cmd, in_qos)
        self.create_subscription(Clock, '/clock', self.on_clock, in_qos)
        self.create_timer(1.0 / self.get_parameter('rate').value, self.on_timer)
        self.get_logger().info('safety block started')

    def on_clock(self, msg):
        self.sim_time = msg.clock.sec + msg.clock.nanosec * 1e-9

    def on_cmd(self, msg):
        m = msg.motor_cmd
        ok = self.filter.command([c.q for c in m], [c.dq for c in m], [c.tau for c in m],
                                 [c.kp for c in m], [c.kd for c in m])
        if not ok:
            self.get_logger().warn('rejected command with NaN/Inf', throttle_duration_sec=1.0)
            return
        self.last_wall = time.monotonic()
        self.last_sim = self.sim_time
        self.publish()

    def stream_lost(self):
        if self.last_wall is None:
            return True
        if time.monotonic() - self.last_wall > self.timeout:
            return True
        if self.sim_time is not None and self.last_sim is not None:
            return self.sim_time - self.last_sim > self.timeout
        return False

    def on_timer(self):
        if self.stream_lost():
            self.filter.hold()
        else:
            self.filter.tick()
        self.publish()

    def publish(self):
        q, dq, tau, kp, kd = self.filter.output()
        msg = LowCmd()
        for i in range(N_JOINTS):
            c = msg.motor_cmd[i]
            c.mode = MODE_SERVO
            c.q = float(q[i])
            c.dq = float(dq[i])
            c.tau = float(tau[i])
            c.kp = float(kp[i])
            c.kd = float(kd[i])
        self.pub.publish(msg)


def main():
    rclpy.init()
    node = SafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
