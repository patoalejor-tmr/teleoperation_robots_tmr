# General Operation of simulation robots

The following will give steps for simulate and control a humanoid robot using RB-Y1 and AIW sources.

Note: there are also other non-related docs under 'extras' folder you can omit this directoy

## Table contents

- Create new environment for simulation
- Check camera connection and id
- Select your config and play with robot

### Create environment

```bash
python3 -m venv .venv 
source .venv/bin/activate 

# Go to submodule #1 for retargeting 
cd third-party/retarget
git submodule update --init --recursive
pip install dex_retargeting
pip install -e ".[example]"

# Go to submodue #2 for IK
cd ..
cd third-party/pyroki
pip install -e .

# Install extra libraries
cd ..
pip install -r requirements.txt
```

### Checking camera id

There are different packages to check your camera, I suggest using `v4l2-ctl --list-devices` and check the following shows in console and select the camera you want to use

```bash
Integrated Camera: Integrated C (usb-0000:00:14.0-11):
        /dev/video2
        /dev/video3
        /dev/media1

HD Pro Webcam C920 (usb-0000:00:14.0-8):
        /dev/video0
        /dev/video1
        /dev/media0
```

### Select config and simulate

Most of the dynamic config have been limited, for run the program use as follows
This will start the camera and the gripper retargeting

```bash
python source/show_realtime_retargeting_5g_aiw.py --camera-path 0 
```

If you want to test the output from a vla model use the following 

```bash
python source/show_vla_inference_2g_rby1.py --infer_path outputs/actions.npy  
```