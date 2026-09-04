# TEERS: Edge-Native Affective Human-Robot Interaction in ROS 2

[![ROS 2 Jazzy](https://img.shields.io/badge/ROS%202-Jazzy-blue.svg)](https://docs.ros.org/en/jazzy/)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![Google Colab](https://img.shields.io/badge/Model%20Training-Google%20Colab-orange.svg)](https://colab.research.google.com/drive/13oiHHziiQHgusVllPKGw0PxodJRWV4KN?usp=sharing)
[![Google Drive](https://img.shields.io/badge/Simulation%20Assets-Google%20Drive-blue.svg)](https://drive.google.com/drive/folders/11bST9Bkd25tQ-JYJnVvHypwgTcOlf-7P?usp=drive_link)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

## Project Description

**TEERS (Tactile & Emotion Engineered Robotic System)** is an edge-native Affective Human-Robot Interaction (HRI) framework designed to enable real-time, closed-loop teleoperation of mobile robots using human facial expressions. Built on **ROS 2 Jazzy** and evaluated with a **TurtleBot3 inside Gazebo**, the system bridges computer vision and kinematic motion control by extracting 468 3D facial landmarks via MediaPipe, classifying cognitive emotional states using an Echo State Network (ESN), and mapping predicted states directly into differential drive velocity vectors ($v, \omega$). 

This repository serves as the practical implementation and empirical validation for the authored research book chapter: *"Edge-Native Affective HRI: Closed-Loop Emotion Kinematics in ROS 2."*

---

## System Architecture

The pipeline operates across three primary stages to achieve low-latency execution on edge hardware:

1. **Perception Stage:** Captures video input to track 468 3D facial landmarks at 20+ FPS using MediaPipe FaceLandmarker.
2. **Cognition Stage:** Processes geometric variations across landmark frames through a lightweight Echo State Network (ESN) classifier.
3. **Action Stage (`emotion_teleop.py`):** Converts classified affect states into ROS 2 `geometry_msgs/msg/TwistStamped` messages to drive the robot base.

---

## Kinematic Mapping & Telemetry

### Emotion-to-Velocity Mapping
- **Happy:** Forward linear motion ($v_x = +0.20\text{ m/s}$)
- **Surprise:** Reverse linear motion ($v_x = -0.15\text{ m/s}$)
- **Angry:** Counter-clockwise yaw rotation ($\omega_z = +0.50\text{ rad/s}$)
- **Fear:** Clockwise yaw rotation ($\omega_z = -0.50\text{ rad/s}$)

### Measured Performance
* **Landmark Extraction Latency:** ~12–15 ms
* **ESN Inference Overhead:** < 5 ms
* **Command Publishing Rate:** 20 Hz (`/cmd_vel`)
* **Kinematic Mapping Accuracy:** 98.4%

---

## Repository Structure

```text
teers_ros2_emotion_control/
├── emotion_teleop.py      # ROS 2 inference and velocity control node
├── model_training.ipynb   # ESN feature extraction and training pipeline
├── setup.py               # Package build script
├── setup.cfg              # Setuptools configuration
├── package.xml            # ROS 2 dependency manifest
└── __init__.py
