---
layout: default
title: mujoco-examples
description: ROS-style robot simulations in plain Python with MuJoCo, FastAPI and OpenCV.
image: assets/tb4_sim_poster.jpg

# header
authors:
  - name: Eric Plaß
    url: https://github.com/erelbng
affiliations:
  - HTWK Leipzig
links:
  - name: Code
    url: https://github.com/erelbng/mujoco-examples
    icon: github
  - name: ROS version
    url: https://github.com/erelbng/ros-examples
  - name: BibTeX
    url: "#citation"

# teaser: videos side by side, then the one-line summary
videos:
  - title: TurtleBot 4
    caption: Driving with cmd_vel, streaming odometry and camera
    src: assets/tb4_sim.mp4
    poster: assets/tb4_sim_poster.jpg
  - title: PincherX 100
    caption: Pick and place through a sequence of joint poses
    src: assets/pincherx_sim.mp4
    poster: assets/pincherx_sim_poster.jpg
tldr: >-
  Ready-to-run MuJoCo simulations of a TurtleBot 4 and a PincherX 100,
  controlled over FastAPI WebSockets like ROS topics, without installing ROS.

footer: >-
  Built with [MuJoCo](https://mujoco.org), [FastAPI](https://fastapi.tiangolo.com) and [OpenCV](https://opencv.org).
---

## About

This repository mirrors [ros-examples](https://github.com/erelbng/ros-examples), but runs directly in Python. It is meant for quick experiments, visualization and cross-platform use, **Windows included**.

It is designed for:
- **Teaching and prototyping** robot control without a ROS installation
- **Computer vision** on rendered camera frames, decoded with OpenCV
- **Data science** on odometry and joint states streamed as JSON
- **Teleoperation** from any language that can open a WebSocket

## How it works

Every example has the same structure. The `*_sim.py` server loads the MJCF scene, opens the MuJoCo passive viewer and serves one WebSocket endpoint at `ws://localhost:8000/ws`. On each 20 ms tick, the server reads any pending command, advances the physics, renders the robot camera and sends back a state packet. This follows the ROS publish/subscribe model, with FastAPI in place of ROS DDS.

```
your client      *_client.py, or any language
   |    ^
   |    |        down: JSON command
   v    |        up:   state + JPEG camera frame, 50 Hz
FastAPI server   WebSocket /ws on port 8000
   |    ^
   |    |        down: ctrl
   v    |        up:   qpos, qvel, camera pixels
MuJoCo           mj_step, offscreen renderer, passive viewer
```

## Robots

### [TurtleBot 4](https://clearpathrobotics.com/turtlebot-4/)

A differential-drive mobile robot in a warehouse-style scene. It reports wheel odometry and a 480×480 onboard camera image. Source: [`turtlebot/`](https://github.com/erelbng/mujoco-examples/tree/main/turtlebot)

Command:
```json
{"cmd_vel": {"linear_x": 2.0, "angular_z": 1.0}}
```

State:
```json
{
  "odom":  {"x": 0.41, "y": 0.07, "theta": 0.33},
  "image": "<base64 JPEG>"
}
```

### [PincherX 100](https://www.trossenrobotics.com/pincherx100)

A compact 4-DOF arm with a gripper, mounted on a workbench. Use it for teleoperation and pick-and-place. It reports joint positions, joint velocities and a camera image. Source: [`pincherx/`](https://github.com/erelbng/mujoco-examples/tree/main/pincherx)

Command:
```json
{"joint_commands": {
  "names":     ["waist", "shoulder", "elbow", "wrist", "gripper"],
  "positions": [-0.8, 0.4, -0.3, 0.0, -1.0]
}}
```

State:
```json
{
  "joint_names":      ["..."],
  "joint_positions":  ["..."],
  "joint_velocities": ["..."],
  "image":            "<base64 JPEG>"
}
```

## Quick start

1. Clone the repository and install the dependencies:
   ```bash
   git clone https://github.com/erelbng/mujoco-examples.git
   cd mujoco-examples
   pip install mujoco numpy opencv-python fastapi uvicorn websockets
   ```
2. Start a simulation server. On macOS, use `mjpython` instead of `python3` if the viewer fails to open.
   ```bash
   python3 turtlebot/turtlebot_sim.py      # or: python3 pincherx/pincherx_sim.py
   ```
3. In a second terminal, connect the example client:
   ```bash
   python3 turtlebot/turtlebot_client.py   # or: python3 pincherx/pincherx_client.py
   ```

## Citation

If you use **mujoco-examples** in your work, please cite it as:

```bibtex
@misc{mujoco2025examples,
  author = {Eric Elbing},
  title  = {mujoco-examples},
  month  = {October},
  year   = {2025},
  url    = {https://github.com/erelbng/mujoco-examples}
}
```

## Contact

For technical support and other questions, contact [eric.elbing@htwk-leipzig.de](mailto:eric.elbing@htwk-leipzig.de) or [open an issue](https://github.com/erelbng/mujoco-examples/issues).
