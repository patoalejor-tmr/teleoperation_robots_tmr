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

def map_range(value, a_min, a_max, b_min, b_max):
    """Map a value from range [a_min, a_max] to [b_min, b_max]."""
    if a_max == a_min:
        raise ValueError("Input range cannot be zero.")
    
    scale = (b_max - b_min) / (a_max - a_min)
    return b_min + (value - a_min) * scale

def teleop_robot_jts(joint_angles: np.array, shutdown_event: Event):
    # urdf_path = Path(__file__).absolute().parent.parent / 'assets' / 'rby1' / 'model.urdf'
    urdf_path = "/home/pato-tommoro/Documents/teleoperation_robots_tmr/assets/rby1/model.urdf"
    
    logger.info(f"{urdf_path = }")
    groot_output = ["right_arm_0","right_arm_1","right_arm_2","right_arm_3","right_arm_4","right_arm_5","right_arm_6","gripper_finger_r1","left_arm_0","left_arm_1","left_arm_2","left_arm_3","left_arm_4","left_arm_5","left_arm_6","gripper_finger_l1"]
    urdf = yourdfpy.URDF.load(urdf_path)
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)
    
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    robot = pk.Robot.from_urdf(urdf)
    # robot_coll = RobotCollision.from_urdf(urdf)
    # plane_coll = HalfSpace.from_point_and_normal(np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0]))

    actuated_names = robot.joints.actuated_names
    upper_limits = robot.joints.upper_limits
    lower_limits = robot.joints.lower_limits

    logger.debug(f"{actuated_names = }")
    logger.debug(f"{upper_limits = }")
    logger.debug(f"{lower_limits = }")
    logger.debug(f"{groot_output = }")

    retarget_groot_rby1 = [actuated_names.index(name)  for name in groot_output if name in actuated_names]
    _joints = np.zeros(len(actuated_names))
    left_index_1 = actuated_names.index('gripper_finger_r1')
    right_index_1 = actuated_names.index('gripper_finger_l1')
    left_index_2 = actuated_names.index('gripper_finger_r2')
    right_index_2 = actuated_names.index('gripper_finger_l2')

    try:
        while not shutdown_event.is_set():
            start_time = time.time()
            for joints in joint_angles:
                _joints[retarget_groot_rby1] = joints
                _joints[left_index_2] = _joints[left_index_1]
                _joints[right_index_2] = _joints[right_index_1]

                _joints[left_index_1] = map_range(_joints[left_index_1], 
                                                  -0.01, 
                                                  0.99, 
                                                  upper_limits[left_index_1], 
                                                  lower_limits[left_index_1])
                
                _joints[left_index_2] = map_range(_joints[left_index_2], 
                                                  -0.01, 
                                                  0.99, 
                                                  lower_limits[left_index_2], 
                                                  upper_limits[left_index_2])
            
                _joints[right_index_1] = map_range(_joints[right_index_1], 
                                                   -0.01, 
                                                   0.99, 
                                                   upper_limits[right_index_1], 
                                                   lower_limits[right_index_1])
                
                _joints[right_index_2] = map_range(_joints[right_index_2], 
                                                   -0.01, 
                                                   0.99, 
                                                   lower_limits[right_index_2], 
                                                   upper_limits[right_index_2])

                coll = []
                for _l, _v, _u in zip(lower_limits, _joints, upper_limits):
                    coll.append(not(_l<=_v and _v<=_u))
                if any(coll):
                    logger.warning(f"Index out of range {np.asarray(actuated_names)[coll]}")
                    logger.error(f"Joints out of range {np.asarray(_joints)[coll]}")

                urdf_vis.update_cfg(_joints)
                time.sleep(0.1)
                elapsed_time = time.time() - start_time
                timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)
    
    except KeyboardInterrupt:
        logger.warning("[Teleop] Ctrl+C received, shutting down...")
    
    finally:
        server.stop()
        logger.debug('[Teleop] Shutdown')

def data_loader(infer_path):
    data = np.load(infer_path)
    logger.debug(f"{data.shape = }")
    return data

def main(infer_path: str):

    logger.debug(Path(__file__).absolute().parent.parent)
    shutdown_event = Event()
    data_load = data_loader(infer_path)

    try:
        teleop_robot_jts(data_load, shutdown_event)
    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received, shutting down...")
        shutdown_event.set()

if __name__ == "__main__":
    tyro.cli(main)
