import sys
import os
# sys.path.append('/home/pato-tommoro/Documents/dex-retargeting/pyroki/examples/pyroki_snippets')
# base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
# sys.path.append(base_path)
# print(sys.path)

import multiprocessing
from multiprocessing import Process, Queue, Event
# import multiprocessing.synchronize 

import time
from pathlib import Path
from queue import Empty
from typing import Optional
from time import perf_counter
import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector
# from collections import deque
import yourdfpy
import pyroki as pk
# import pyroki.examples.pyroki_snippets as pks
from solve_ik_with_multiple_targets import solve_ik_with_multiple_targets

import viser
from viser.extras import ViserUrdf
from copy import copy


def get_latest_frame(q: Queue):
    latest = None
    while not q.empty():
        try:
            latest = q.get_nowait()
        except:
            break
    return latest


def start_retargeting(
        cam_queue: Queue, 
        qpos_queue: Queue, 
        xyz_queue: Queue,
        robot_dir: str, 
        config_path_right: str, 
        config_path_left: str, 
        shutdown_event: Event):
    # time.sleep(1)
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    # logger.info(f"Start retargeting with config {config_path_right}")
    retargeting_right = RetargetingConfig.load_from_file(config_path_right).build()
    retargeting_left = RetargetingConfig.load_from_file(config_path_left).build()

    # hand_type = "Right" if "right" in config_path_right.lower() else "Left"
    detector_right = SingleHandDetector(hand_type='Right', selfie=False)
    detector_left = SingleHandDetector(hand_type='Right', selfie=True)

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config_right = RetargetingConfig.load_from_file(config_path_right)
    config_left = RetargetingConfig.load_from_file(config_path_left)

    # Setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Load robot and set it to a good pose to take picture
    loader = scene.create_urdf_loader()
    loader.load_multiple_collisions_from_file = True
    # loader.scale = 1.5
    
    ############################################
    # Loading the right hand
    ############################################
    filepath = Path(config_right.urdf_path)
    filepath = str(filepath)
    robot_right = loader.load(filepath)
    robot_right.set_pose(sapien.Pose([0, -0.35, -0.13]))

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot_right.get_active_joints()]
    retargeting_joint_names = retargeting_right.joint_names

    retargeting_to_sapien_right = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    ############################################
    # Loading the right left
    ############################################
    filepath = Path(config_left.urdf_path)
    filepath = str(filepath)
    robot_left = loader.load(filepath)
    robot_left.set_pose(sapien.Pose([0, 0.35, -0.13]))

    # Different robot loader may have different orders for joints
    sapien_joint_names = [joint.get_name() for joint in robot_left.get_active_joints()]
    retargeting_joint_names = retargeting_left.joint_names

    retargeting_to_sapien_left = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)
    # logger.debug(f"{sapien_joint_names = }")
    qpos_queue.put_nowait([np.zeros(2), np.zeros(2)])
    qpos_r, qpos_l = np.zeros(2), np.zeros(2)

    try:

        while not shutdown_event.is_set():

            loop_time = perf_counter()
    
            try:
                bgr = cam_queue.get(timeout=5)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            except Empty:
                logger.error(
                    "Fail to fetch image from camera in 5 secs. Please check your web camera device."
                )
                return
            
            t_detect = perf_counter()
            _, joint_pos_r, keypoint_2d_r, _ = detector_right.detect(rgb)
            _, joint_pos_l, keypoint_2d_l, _ = detector_left.detect(rgb)
            
            bgr = detector_right.draw_skeleton_on_image(bgr, keypoint_2d_r, style="default")
            bgr = detector_left.draw_skeleton_on_image(bgr, keypoint_2d_l, style="default")

            logger.info(f"Time for hand detection {perf_counter() - t_detect:.4f}")
            cv2.imshow("realtime_retargeting_demo", bgr)
            cv2.waitKey(1)
            
            hand_palm_index = [0,5,9,13,17]
            _cl = np.asarray([0.35, 0.32, 1.0])
            _ll = np.asarray([0.0, 0.75, 0.50])
            _cr = np.asarray([0.35, -0.32, 1.0])
            _rr = np.asarray([0.0, 0.25, 0.50])
            xyz_r = []
            xyz_l = []

            if joint_pos_r is None:
                logger.warning(f"right hand is not detected.")
                xyz_r = _cr
            else:   
                xyz = []
                for idx, landmark in enumerate(keypoint_2d_r.landmark):
                    if ((landmark.HasField('visibility') and landmark.visibility < 0.5) or
                        (landmark.HasField('presence') and landmark.presence < 0.5)):
                        continue
                    if not(idx in hand_palm_index):
                        continue
            
                    _v = np.asarray([landmark.z, landmark.x, 1-landmark.y])
                    if len(xyz) == 0:
                        xyz = _v
                    else:
                        xyz = np.vstack((xyz, _v))

                xyz = np.mean(xyz, axis=0)
                # logger.debug(f"{xyz = }")
                xyz_r =_cr - (_rr - _v)
                # xyz_r[2] = -xyz_r[2]
                # logger.debug(f"{xyz_r = }")

                indices = retargeting_right.optimizer.target_link_human_indices
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos_r[task_indices, :] - joint_pos_r[origin_indices, :]
                
                r_detect = perf_counter()
                qpos_r = retargeting_right.retarget(ref_value)
                logger.info(f"Time for right hand retarget {perf_counter() - r_detect:.4f}")
                robot_right.set_qpos(qpos_r[retargeting_to_sapien_right])

            if joint_pos_l is None:
                logger.warning(f"left hand is not detected.")
                xyz_l = _cl
            else:
                xyz = []
                for idx, landmark in enumerate(keypoint_2d_l.landmark):
                    if ((landmark.HasField('visibility') and landmark.visibility < 0.5) or
                        (landmark.HasField('presence') and landmark.presence < 0.5)):
                        continue
                    if not(idx in hand_palm_index):
                        continue

                    _v = np.asarray([landmark.z, landmark.x, 1-landmark.y])
                    if len(xyz) == 0:
                        xyz = _v
                    else:
                        xyz = np.vstack((xyz, _v))

                xyz = np.mean(xyz, axis=0)
                # logger.debug(f"{xyz = }")
                xyz_l =_cl - (_ll - _v)
                # xyz_r[2] = -xyz_r[2]
                # logger.debug(f"{xyz_l = }")

                indices = retargeting_left.optimizer.target_link_human_indices
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos_l[task_indices, :] - joint_pos_l[origin_indices, :]
                
                r_detect = perf_counter()
                qpos_l = retargeting_left.retarget(ref_value)
                logger.info(f"Time for left hand retarget {perf_counter() - r_detect:.4f}")
                robot_left.set_qpos(qpos_l[retargeting_to_sapien_left])

            logger.debug(f"{retargeting_to_sapien_right = } {qpos_r = }")
            logger.debug(f"{retargeting_to_sapien_left = } {qpos_l = }")
            # logger.debug(f"{sapien_joint_names = } {retargeting_joint_names = }")

            if qpos_queue.full():
                _ = qpos_queue.get()
            qpos_queue.put([qpos_r, qpos_l])

            if xyz_queue.full():
                _ = xyz_queue.get()
            xyz_queue.put([np.asarray(xyz_r), np.asarray(xyz_l)])
                
            for _ in range(2):
                viewer.render()

            _loop = perf_counter() - loop_time
            logger.info(f"Loop time: {_loop:.3f} FPS: {int(1/_loop):d}")
    
    except KeyboardInterrupt:
        logger.warning("\n[Retargeting] Ctrl+C received, shutting down...")
    
    finally:
        logger.debug('[Retargeting] Shutdown')



def produce_frame(cam_queue: Queue, camera_path: str, shutdown_event: Event):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    try:
        while cap.isOpened() and not shutdown_event.is_set():
            success, image = cap.read()
            if not success:
                continue

            if cam_queue.full():
                _ = cam_queue.get()
            cam_queue.put(image)
            time.sleep(1 / 60.0)

    except KeyboardInterrupt:
        logger.warning("[Viewer] Ctrl+C received, shutting down...")
    
    finally:
        cv2.destroyAllWindows()
        cap.release()
        logger.debug("[Viewer] Shutdown.")

def map_range(value, a_min, a_max, b_min, b_max):
    """Map a value from range [a_min, a_max] to [b_min, b_max]."""
    if a_max == a_min:
        raise ValueError("Input range cannot be zero.")
    
    scale = (b_max - b_min) / (a_max - a_min)
    return b_min + (value - a_min) * scale

def teleop_robot(xyz_queue: Queue, qpos_queue: Queue, shutdown_event: Event):
    # time.sleep(2)
    urdf = yourdfpy.URDF.load("/home/pato-tommoro/Documents/teleoperation_robots_tmr/assets/rby1/model.urdf")
    target_link_names = ["link_right_arm_6", "link_left_arm_6"]
    robot = pk.Robot.from_urdf(urdf)

    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    ik_target_0 = server.scene.add_transform_controls(
        "/ik_target_0", scale=0.2, 
        position=(0.35, -0.32,  1.0), 
        wxyz=( 0.65727019, -0.02195583, -0.75134673, -0.0546987)
    )
    ik_target_1 = server.scene.add_transform_controls(
        "/ik_target_1", scale=0.2, 
        position=(0.35, 0.32, 1.0), 
        wxyz=( 0.70109401, -0.016592  , -0.7127362 , -0.01410571)
    )
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    logger.debug(f"PyRoki {robot = }")

    actuated_names = [
        'right_wheel', 'left_wheel', 
        'torso_0', 'torso_1', 'torso_2', 'torso_3', 'torso_4', 
        'right_arm_0', 'right_arm_1', 'right_arm_2', 'right_arm_3', 'right_arm_4', 'right_arm_5', 'right_arm_6', 
        'left_arm_0', 'left_arm_1', 'left_arm_2', 'left_arm_3', 'left_arm_4', 'left_arm_5', 'left_arm_6', 
        'gripper_finger_r1', 'gripper_finger_r2', 'gripper_finger_l1', 'gripper_finger_l2', 
        'head_0', 'head_1' ]
    
    upper_limits= np.array([
        3.1415927 , 3.1415927 , 
        1.5707964 , 1.5707964 , 1.5707964 , 0.5235988 , 2.3561945 , 
        3.1415927 , 0.01745329, 3.1415927 , 0.01745329, 3.1415927 , 1.9198622 , 2.7052603 , 
        3.1415927 , 3.1415927 , 3.1415927 , 0.01745329, 3.1415927 , 1.9198622 , 2.7052603 , 
        0.        , 0.05      , 0.        , 0.05      ,
        0.523     , 1.57      ], dtype=np.float32)
    
    lower_limits_all= np.array([
        -3.1415927 , -3.1415927 , 
        -0.5235988 , -2.6179938 , -0.7853982 , -0.5235988 , -2.3561945 , 
        -3.1415927 , -3.1415927 , -3.1415927 , -2.6179938 , -3.1415927 , -1.5707964 , -2.7052603 , 
        -3.1415927 ,-0.01745329, -3.1415927 , -2.6179938 , -3.1415927 , -1.5707964 , -2.7052603 , 
        -0.05      ,  0.        , -0.05      ,  0.        ,
        -0.523     , -0.35      ], dtype=np.float32)
    

    left_index_1 = actuated_names.index('gripper_finger_r1')
    right_index_1 = actuated_names.index('gripper_finger_l1')
    left_index_2 = actuated_names.index('gripper_finger_r2')
    right_index_2 = actuated_names.index('gripper_finger_l2')

    try:
        while not shutdown_event.is_set():
            try:
                demo_qops = qpos_queue.get_nowait()
                _memory_qpos_ = copy(demo_qops)
            except:
                logger.error('Error getting poses from buffer')
                demo_qops = _memory_qpos_
                # np.asarray([[0.,0.], [0.,0.]])
                pass

            try:
                xyz_points = xyz_queue.get_nowait()
                ik_target_0.position = xyz_points[0]
                ik_target_1.position = xyz_points[1]
            except:
                logger.error('Error getting poses from buffer')
                pass

            start_time = time.time()
            solution = solve_ik_with_multiple_targets(
                robot=robot,
                target_link_names=target_link_names,
                target_positions=np.array([ik_target_0.position, ik_target_1.position]),
                target_wxyzs=np.array([ik_target_0.wxyz, ik_target_1.wxyz]),
            )

            # logger.info(f"{solution = } {left_index_1 = } {right_index_1 = }")
            logger.info(f"{demo_qops = }")
            solution[left_index_1] = map_range(demo_qops[0][0], 0, 0.04, 0.0, 0.05)
            solution[left_index_2] = map_range(demo_qops[0][0], 0, 0.04, 0.0, -0.05)
        
            solution[right_index_1] = map_range(demo_qops[1][0], 0, 0.04, 0.0, 0.05)
            solution[right_index_2] = map_range(demo_qops[1][0], 0, 0.04, 0.0, -0.05)
            
            urdf_vis.update_cfg(solution)
            time.sleep(0.01)
            elapsed_time = time.time() - start_time
            timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

    
    except KeyboardInterrupt:
        logger.warning("[Teleop] Ctrl+C received, shutting down...")
    
    finally:
        server.stop()
        logger.debug('[Teleop] Shutdown')


def main(
        camera_path: Optional[int] = 0,
    ):

    # from multiprocessing import set_start_method
    # set_start_method("fork")  # or "fork" on Linux
    shutdown_event = Event()

    # config_path_left = get_default_config_path(robot_name, retargeting_type, hand_type)
    config_path_right = get_default_config_path(RobotName.panda, RetargetingType.dexpilot, HandType.right)
    config_path_left  = get_default_config_path(RobotName.panda, RetargetingType.dexpilot, HandType.left)
    # config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path("/home/pato-tommoro/Documents/teleoperation_robots_tmr/third-party/retarget/assets/robots/hands")
    
    )

    logger.debug(Path(__file__).absolute())
    # Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"

    cam_queue = multiprocessing.Queue(maxsize=10)
    qpos_queue = multiprocessing.Queue(maxsize=10)
    xyz_queue = multiprocessing.Queue(maxsize=10)

    producer_process = multiprocessing.Process(
        target=produce_frame, args=(cam_queue, camera_path, shutdown_event)
    )
    
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(cam_queue, qpos_queue, xyz_queue, str(robot_dir), str(config_path_right), str(config_path_left), shutdown_event)
    )

    teleop_process = multiprocessing.Process(
        target=teleop_robot, args=(xyz_queue, qpos_queue, shutdown_event)
    )

    producer_process.start()
    consumer_process.start()
    teleop_process.start()

    try:
        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n[Main] Ctrl+C received, shutting down...")
        shutdown_event.set()
    
        producer_process.terminate()
        consumer_process.terminate()
        teleop_process.terminate()

        producer_process.join(timeout=5)
        consumer_process.join(timeout=5)
        teleop_process.join(timeout=5)
    
    time.sleep(1)
    logger.info("Exited Application")


if __name__ == "__main__":
    tyro.cli(main)
