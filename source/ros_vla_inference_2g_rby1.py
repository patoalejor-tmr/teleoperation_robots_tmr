from multiprocessing import Event
import time
from pathlib import Path
from queue import Empty
from typing import Optional
import numpy as np
import tyro
from loguru import logger
# from collections import deque
import yourdfpy
import pyroki as pk
from pyroki.collision import RobotCollision, HalfSpace
# import pyroki.examples.pyroki_snippets as pks
import viser
from viser.extras import ViserUrdf
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class JointStatePublisher(Node):
    def __init__(self):
        super().__init__('joint_state_subscriber')
        self._publisher = self.create_publisher(
            JointState,
            '/joint_states',
            10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.joints = []
        self.i = 0
        self.msg = JointState()
        self.msg.velocity = []
        self.msg.effort = []
        
    def timer_callback(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.msg.position = self.joints[self.i, :].tolist()
        self._publisher.publish(self.msg)
        self.i += 1
        if self.i >= self.joints.shape[0]:
            self.i = 0

        
def map_range(value, a_min, a_max, b_min, b_max):
    """Map a value from range [a_min, a_max] to [b_min, b_max]."""
    if a_max == a_min:
        raise ValueError("Input range cannot be zero.")
    
    scale = (b_max - b_min) / (a_max - a_min)
    return b_min + (value - a_min) * scale

def get_joints(joint_angles: np.array):
    # urdf_path = Path(__file__).absolute().parent.parent / 'assets' / 'rby1' / 'model.urdf'
    urdf_path = "/home/pato-tommoro/Documents/teleoperation_robots_tmr/assets/rby1/model.urdf"
    logger.info(f"{urdf_path = }")
    groot_output = ["right_arm_0","right_arm_1","right_arm_2","right_arm_3","right_arm_4","right_arm_5","right_arm_6","gripper_finger_r1","left_arm_0","left_arm_1","left_arm_2","left_arm_3","left_arm_4","left_arm_5","left_arm_6","gripper_finger_l1"]
    urdf = yourdfpy.URDF.load(urdf_path)
    robot = pk.Robot.from_urdf(urdf)
    
    actuated_names = robot.joints.actuated_names
    upper_limits = robot.joints.upper_limits
    lower_limits = robot.joints.lower_limits

    logger.debug(f"{actuated_names = }")
    logger.debug(f"{upper_limits = }")
    logger.debug(f"{lower_limits = }")
    logger.debug(f"{groot_output = }")

    retarget_groot_rby1 = [actuated_names.index(name)  for name in groot_output if name in actuated_names]
    
    left_index_1 = actuated_names.index('gripper_finger_r1')
    right_index_1 = actuated_names.index('gripper_finger_l1')
    left_index_2 = actuated_names.index('gripper_finger_r2')
    right_index_2 = actuated_names.index('gripper_finger_l2')

    
    rby1_joints = np.zeros((len(joint_angles), len(actuated_names)))

    for ix, joints in enumerate(joint_angles):
        _tmp = np.zeros(len(actuated_names))
        _tmp[retarget_groot_rby1] = joints
        
        rby1_joints[ix, :] = _tmp
        rby1_joints[ix, left_index_2] = rby1_joints[ix, left_index_1]
        rby1_joints[ix, right_index_2] = rby1_joints[ix, right_index_1]

        rby1_joints[ix, left_index_1] = map_range(rby1_joints[ix, left_index_1], 
                                            -0.01, 
                                            0.99, 
                                            upper_limits[left_index_1], 
                                            lower_limits[left_index_1])
        
        rby1_joints[ix, left_index_2] = map_range(rby1_joints[ix, left_index_2], 
                                            -0.01, 
                                            0.99, 
                                            lower_limits[left_index_2], 
                                            upper_limits[left_index_2])
    
        rby1_joints[ix, right_index_1] = map_range(rby1_joints[ix, right_index_1], 
                                            -0.01, 
                                            0.99, 
                                            upper_limits[right_index_1], 
                                            lower_limits[right_index_1])
        
        rby1_joints[ix, right_index_2] = map_range(rby1_joints[ix, right_index_2], 
                                            -0.01, 
                                            0.99, 
                                            lower_limits[right_index_2], 
                                            upper_limits[right_index_2])
    return rby1_joints

def data_loader(infer_path):
    data = np.load(infer_path)
    logger.debug(f"{data.shape = }")
    return data

def main(infer_path: str):
    shutdown_event = Event()
    joint_data = data_loader(infer_path)
    joints_rby1 = get_joints(joint_data)
    rclpy.init()
    node = JointStatePublisher()
    node.joints = joints_rby1
    try:
        if not shutdown_event.is_set():
            rclpy.spin(node)
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received, shutting down...")
        shutdown_event.set()
    finally:
        node.destroy_node()
        # rclpy.shutdown()

if __name__ == "__main__":
    tyro.cli(main)
