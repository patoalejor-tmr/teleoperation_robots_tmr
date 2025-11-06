# ─────────────────────────────────────────────────────────────
# Base: NVIDIA CUDA 12.2 image with Ubuntu 22.04
# ─────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# ─────────────────────────────────────────────────────────────
# Install ROS 2 Humble
# ─────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y curl gnupg2 lsb-release
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key | apt-key add -
RUN echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
    > /etc/apt/sources.list.d/ros2.list

RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-venv \
    ros-humble-rosbridge-server \
    ros-humble-tf2-ros \
    ros-humble-joint-state-publisher \
    ros-humble-sensor-msgs \
    ros-humble-trajectory-msgs \
    git \
    curl \
    wget \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*
    
# ─────────────────────────────────────────────────────────────
# Upgrade pip
# ─────────────────────────────────────────────────────────────
RUN python3 -m pip install --upgrade pip

# ─────────────────────────────────────────────────────────────
# Set workspace
# ─────────────────────────────────────────────────────────────
WORKDIR /workspace

# ─────────────────────────────────────────────────────────────
# Copy project and install submodules
# ─────────────────────────────────────────────────────────────
COPY . /workspace/teleoperation_robots_tmr
WORKDIR /workspace/teleoperation_robots_tmr

# Make sure all submodules are cloned
RUN git submodule update --init --recursive

# ─────────────────────────────────────────────────────────────
# Install submodule: retarget
# ─────────────────────────────────────────────────────────────
WORKDIR /workspace/teleoperation_robots_tmr/third-party/retarget
RUN git submodule update --init --recursive \
 && pip install dex_retargeting \
 && pip install -e ".[example]"

# ─────────────────────────────────────────────────────────────
# Install submodule: pyroki (IK)
# ─────────────────────────────────────────────────────────────
WORKDIR /workspace/teleoperation_robots_tmr/third-party/pyroki
RUN pip install -e .

# ─────────────────────────────────────────────────────────────
# Install project-wide requirements
# ─────────────────────────────────────────────────────────────
WORKDIR /workspace/teleoperation_robots_tmr
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# ─────────────────────────────────────────────────────────────
# Set ROS environment sourcing
# ─────────────────────────────────────────────────────────────
SHELL ["/bin/bash", "-c"]
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# Default entrypoint
ENTRYPOINT ["/bin/bash"]
