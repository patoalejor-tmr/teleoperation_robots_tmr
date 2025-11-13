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
    detector_left = SingleHandDetector(hand_type='Left', selfie=False)

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
    loader.scale = 1.5
    
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
            qpos_r, qpos_l = np.zeros(20), np.zeros(20)
            
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
                logger.debug(f"{xyz = }")
                xyz_r =_cr - (_rr - _v)
                # xyz_r[2] = -xyz_r[2]
                logger.debug(f"{xyz_r = }")

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
                logger.debug(f"{xyz = }")
                xyz_l =_cl - (_ll - _v)
                # xyz_r[2] = -xyz_r[2]
                logger.debug(f"{xyz_l = }")

                indices = retargeting_left.optimizer.target_link_human_indices
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos_l[task_indices, :] - joint_pos_l[origin_indices, :]
                
                r_detect = perf_counter()
                qpos_l = retargeting_left.retarget(ref_value)
                logger.info(f"Time for left hand retarget {perf_counter() - r_detect:.4f}")
                robot_left.set_qpos(qpos_l[retargeting_to_sapien_left])

            
            if qpos_queue.full():
                _ = qpos_queue.get()
            qpos_queue.put([np.asarray(qpos_r), np.asarray(qpos_l)])

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


def teleop_robot(xyz_queue: Queue, qpos_queue: Queue, shutdown_event: Event):
    # time.sleep(2)
    urdf_path = str(
        Path(__file__).absolute().parent.parent / "assets" / "aiw" / "ffw_bh5_rev1_follower" / "ffw_bh5_follower_alone.urdf"
    )
    urdf = yourdfpy.URDF.load(urdf_path)
    target_link_names = ["arm_r_link7", "arm_l_link7"]
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

    actuated_names = ['lift_joint', 'head_joint1', 'head_joint2', 
                    'arm_l_joint1', 'arm_l_joint2', 'arm_l_joint3', 'arm_l_joint4', 'arm_l_joint5', 'arm_l_joint6', 'arm_l_joint7', 
                    'arm_r_joint1', 'arm_r_joint2', 'arm_r_joint3', 'arm_r_joint4', 'arm_r_joint5', 'arm_r_joint6', 'arm_r_joint7',
                    'finger_l_joint1', 'finger_l_joint2', 'finger_l_joint3', 'finger_l_joint4', 
                    'finger_l_joint5', 'finger_l_joint6', 'finger_l_joint7', 'finger_l_joint8', 
                    'finger_l_joint9', 'finger_l_joint10', 'finger_l_joint11', 'finger_l_joint12', 
                    'finger_l_joint13', 'finger_l_joint14', 'finger_l_joint15', 'finger_l_joint16', 
                    'finger_l_joint17', 'finger_l_joint18', 'finger_l_joint19', 'finger_l_joint20',
                    'finger_r_joint1', 'finger_r_joint2', 'finger_r_joint3', 'finger_r_joint4', 
                    'finger_r_joint5', 'finger_r_joint6', 'finger_r_joint7', 'finger_r_joint8', 
                    'finger_r_joint9', 'finger_r_joint10', 'finger_r_joint11', 'finger_r_joint12', 
                    'finger_r_joint13', 'finger_r_joint14', 'finger_r_joint15', 'finger_r_joint16', 
                    'finger_r_joint17', 'finger_r_joint18', 'finger_r_joint19', 'finger_r_joint20']

    demo_svh_names = [
        'right_hand_Thumb_Opposition', 
        'right_hand_Thumb_Flexion', 
        'right_hand_j3', # mp
        'right_hand_j4', # dp
        'right_hand_index_spread', 
        'right_hand_Index_Finger_Proximal', 
        'right_hand_Index_Finger_Distal', 
        'right_hand_j14', 
        'right_hand_j5', 
        'right_hand_Finger_Spread', 
        'right_hand_Pinky', 
        'right_hand_j13', 
        'right_hand_j17', 
        'right_hand_ring_spread', 
        'right_hand_Ring_Finger', 
        'right_hand_j12', 
        'right_hand_j16', 
        'right_hand_Middle_Finger_Proximal', 
        'right_hand_Middle_Finger_Distal', 
        'right_hand_j15'
        ]
    
    demo_joints_svh = [
        # thumb finger
        ['right_hand_Thumb_Opposition','finger_l_joint1'], 
        ['right_hand_Thumb_Flexion','finger_l_joint2'], 
        ['right_hand_j3','finger_l_joint3'], 
        ['right_hand_j4','finger_l_joint4'], 
        
        ['right_hand_index_spread','finger_l_joint5'], 
        ['right_hand_Index_Finger_Proximal','finger_l_joint6'], 
        ['right_hand_Index_Finger_Distal','finger_l_joint7'], 
        ['right_hand_j14','finger_l_joint8'], 

        ['right_hand_Middle_Finger_Proximal','finger_l_joint10'], 
        ['right_hand_Middle_Finger_Distal','finger_l_joint11'], 
        ['right_hand_j15','finger_l_joint12'],

        ['right_hand_ring_spread','finger_l_joint13'], 
        ['right_hand_Ring_Finger','finger_l_joint14'], 
        ['right_hand_j12','finger_l_joint15'], 
        ['right_hand_j16','finger_l_joint16'], 
        
        ['right_hand_Finger_Spread','finger_l_joint17'], 
        ['right_hand_Pinky','finger_l_joint18'], 
        ['right_hand_j13','finger_l_joint19'], 
        ['right_hand_j17','finger_l_joint20'],   
    ]

    retargeting_from_svh = np.array(
        [demo_svh_names.index(_svh_aiw[0]) for _svh_aiw in demo_joints_svh]
    ).astype(int)

    retargeting_to_aiw_l = np.array(
        [actuated_names.index(_svh_aiw[1]) for _svh_aiw in demo_joints_svh]
    ).astype(int)

    retargeting_to_aiw_r = np.array(
        [actuated_names.index(_svh_aiw[1].replace('_l_','_r_')) for _svh_aiw in demo_joints_svh]
    ).astype(int)

    demo_qops_svh_right_init = np.array([-1.00000005e-03, -8.52410866e-01, -8.65290794e-01, -1.23504958e+00,
       -2.09251149e-01,  2.58693373e-01,  6.29845928e-03,  6.58188995e-03,
       -1.00000005e-03, -4.18502298e-01,  9.82749999e-01,  1.33536070e+00,
        1.39852204e+00, -2.09251149e-01,  9.81477499e-01,  1.33363163e+00,
        1.39461082e+00,  2.11896662e-01,  2.46621638e-03,  2.57818261e-03])
    demo_qops_svh_left_init = np.array([ 5.13297787e-01,  4.22011167e-01,  4.28387756e-01,  6.11447760e-01,
       -5.00000024e-04,  2.37154147e-01,  9.19086438e-04,  9.60445328e-04,
        5.13297787e-01, -1.00000005e-03,  9.82749999e-01,  1.33536070e+00,
        1.39852204e+00, -5.00000024e-04,  9.82749999e-01,  1.33536070e+00,
        1.39641896e+00,  7.99489975e-01,  1.33500004e+00,  1.39560904e+00])
    
    demo_qops_svh_right = demo_qops_svh_right_init
    demo_qops_svh_left = demo_qops_svh_left_init
    try:
        while not shutdown_event.is_set():
            try:
                demo_qops = qpos_queue.get_nowait()
                demo_qops_svh_right = demo_qops[0]
                demo_qops_svh_left = demo_qops[1]
            except:
                logger.error('Error getting poses from buffer')
                demo_qops_svh_right = demo_qops_svh_right_init
                demo_qops_svh_left = demo_qops_svh_left_init
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

            solution[retargeting_to_aiw_r] = demo_qops_svh_right[retargeting_from_svh]
            solution[retargeting_to_aiw_l] = demo_qops_svh_left[retargeting_from_svh]
            
            # Solution for the thumb
            th_r = [str(round(ff, 3)) for ff in solution[retargeting_to_aiw_r[0:4]]]
            logger.debug(f'Right thumb {th_r}')
            th_l = [str(round(ff, 3)) for ff in solution[retargeting_to_aiw_l[0:4]]]
            logger.debug(f'Left thumb {th_l}')

            # solution[retargeting_to_aiw_r[2]] = -solution[retargeting_to_aiw_r[2]]
            # solution[retargeting_to_aiw_r[3]] = -solution[retargeting_to_aiw_r[3]]
            solution[retargeting_to_aiw_r[1]] = -1.0
            solution[retargeting_to_aiw_r[0]] = -1.0 - solution[retargeting_to_aiw_r[0]]
            # solution[retargeting_to_aiw_r[0]] = -solution[retargeting_to_aiw_r[0]]
            
            solution[retargeting_to_aiw_l[0]] = 1.0 - solution[retargeting_to_aiw_l[1]]
            solution[retargeting_to_aiw_l[1]] = 1.0
            solution[retargeting_to_aiw_l[2]] = -solution[retargeting_to_aiw_l[2]]
            solution[retargeting_to_aiw_l[3]] = -solution[retargeting_to_aiw_l[3]]
            # solution[retargeting_to_aiw_l[0]] = -2.0 + solution[retargeting_to_aiw_l[0]]



            urdf_vis.update_cfg(solution)
            time.sleep(0.01)
            elapsed_time = time.time() - start_time
            timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

            demo_qops_svh_right_init = demo_qops_svh_right
            demo_qops_svh_left_init = demo_qops_svh_left
    
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
    config_path_right = get_default_config_path(RobotName.svh, RetargetingType.dexpilot, HandType.right)
    config_path_left  = get_default_config_path(RobotName.svh, RetargetingType.dexpilot, HandType.left)
    # config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path("/home/pato-tommoro/Documents/teleoperation_robots_tmr/third-party/retarget/assets/robots/hands")
    )
    # Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"

    cam_queue = multiprocessing.Queue(maxsize=2)
    qpos_queue = multiprocessing.Queue(maxsize=2)
    xyz_queue = multiprocessing.Queue(maxsize=2)

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
