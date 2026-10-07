**mujoco-examples** provides ready-to-run **MuJoCo** simulations of a **TurtleBot 4** mobile robot and a **PincherX 100** manipulator. Each robot runs behind a **FastAPI WebSocket** server that works like ROS topics, so you can control it and read its sensors from a few lines of Python, without installing ROS.

**Author:** Eric Plaß, HTWK Leipzig  
**Code:** [github.com/erelbng/mujoco-examples](https://github.com/erelbng/mujoco-examples)  
**ROS version:** [github.com/erelbng/ros-examples](https://github.com/erelbng/ros-examples)

---

## Demos

<div style="display: flex; flex-wrap: wrap; gap: 16px;">
  <div style="flex: 1; min-width: 260px; text-align: center;">
    <strong>TurtleBot 4: driving with <code>cmd_vel</code></strong>
    <video src="assets/tb4_sim.mp4" poster="assets/tb4_sim_poster.jpg" width="100%" autoplay muted loop playsinline></video>
  </div>
  <div style="flex: 1; min-width: 260px; text-align: center;">
    <strong>PincherX 100: pick &amp; place</strong>
    <video src="assets/pincherx_sim.mp4" poster="assets/pincherx_sim_poster.jpg" width="100%" autoplay muted loop playsinline></video>
  </div>
</div>

---

## About

This repository mirrors [ros-examples](https://github.com/erelbng/ros-examples), but runs directly in Python. It is meant for quick experiments, visualization and cross-platform use, **Windows included**.

It is designed for:
- **Teaching and prototyping** robot control without a ROS installation
- **Computer vision** on rendered camera frames, decoded with OpenCV
- **Data science** on odometry and joint states streamed as JSON
- **Teleoperation** from any language that can open a WebSocket

---

## How It Works

Every example has the same structure. The `*_sim.py` server loads the MJCF scene, opens the MuJoCo passive viewer and serves one WebSocket endpoint at `ws://localhost:8000/ws`. On each 20 ms tick, the server reads any pending command, advances the physics, renders the robot camera and sends back a state packet. This follows the ROS publish/subscribe model, with FastAPI in place of ROS DDS.

```
 your client                    FastAPI server                 MuJoCo
 (*_client.py, any language)    (WebSocket /ws, port 8000)     (mj_step, renderer, viewer)

        ---- JSON command ---->         ---- ctrl ---->
        <--- state + JPEG -----         <--- qpos, qvel, pixels ---
                 50 Hz
```

---

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

---

## Quick Start

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

---

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

---

## Contact

For technical support and other questions, contact [eric.elbing@htwk-leipzig.de](mailto:eric.elbing@htwk-leipzig.de) or [open an issue](https://github.com/erelbng/mujoco-examples/issues).
