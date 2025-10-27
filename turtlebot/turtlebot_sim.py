#!/usr/bin/env python3
import asyncio
import base64
import json
import math
import os
import time

import cv2
import mujoco
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket
from mujoco import viewer

app = FastAPI(title="TurtleBot MuJoCo WebSocket Server")


class TurtleBotSim:
    def __init__(self):
        # === Load MuJoCo model ===
        self.model_path = os.path.join(os.path.dirname(__file__), "example_scene.xml")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # === IDs ===
        self.forward_motor = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "forward"
        )
        self.turn_motor = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "turn"
        )
        self.left_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "left"
        )
        self.right_joint = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "right"
        )

        # === Parameters ===
        self.L = 0.234  # wheel separation
        self.R = 0.036  # wheel radius
        self.sim_dt = self.model.opt.timestep
        self.real_dt = 0.02
        self.steps_per_update = int(self.real_dt / self.sim_dt)

        # === State ===
        self.target_v = 0.0  # linear velocity
        self.target_omega = 0.0  # angular velocity
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0

        # === Camera ===
        self.renderer = mujoco.Renderer(self.model, 480, 480)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam"
        )

        # === Viewer ===
        self.viewer = viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        )

        self.last_image = None
        print("Simulation initialized.")

    # === Set velocity command (linear & angular) ===
    def set_cmd_vel(self, v, omega):
        self.target_v = v
        self.target_omega = omega

    # === Single simulation step ===
    def sim_step(self):
        # Apply commands directly to actuators (matches ROS code)
        self.data.ctrl[self.forward_motor] = self.target_v
        self.data.ctrl[self.turn_motor] = self.target_omega

        # Step physics
        for _ in range(self.steps_per_update):
            mujoco.mj_step(self.model, self.data)

        # Compute odometry
        ql = self.data.qpos[self.left_joint]
        qr = self.data.qpos[self.right_joint]
        dql = self.data.qvel[self.left_joint]
        dqr = self.data.qvel[self.right_joint]

        v_left = dql * self.R
        v_right = dqr * self.R
        v = (v_right + v_left) / 2.0
        omega = (v_right - v_left) / self.L

        self.x += v * math.cos(self.theta) * self.real_dt
        self.y += v * math.sin(self.theta) * self.real_dt
        self.theta += omega * self.real_dt

        # Render camera
        self.renderer.update_scene(self.data, camera=self.cam)
        img = (self.renderer.render()).astype(np.uint8)
        # img = cv2.flip(img, 0)
        self.last_image = img[..., ::-1]

        # Update Viewer
        self.viewer.sync()

    # === Package state for WebSocket ===
    def get_state_packet(self):
        img_b64 = None
        if self.last_image is not None:
            ret, buf = cv2.imencode(".jpg", self.last_image)
            if ret:
                img_b64 = base64.b64encode(buf).decode("utf-8")
        return {
            "odom": {"x": self.x, "y": self.y, "theta": self.theta},
            "image": img_b64,
        }


# === Initialize simulation ===
sim = TurtleBotSim()


# === WebSocket endpoint ===
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected.")
    try:
        while True:
            # --- Receive command (non-blocking with timeout) ---
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                data_json = json.loads(data)
                if "cmd_vel" in data_json:
                    v = data_json["cmd_vel"].get("linear_x", 0.0)
                    w = data_json["cmd_vel"].get("angular_z", 0.0)
                    sim.set_cmd_vel(v, w)
                    print(f"Received cmd_vel: v={v}, w={w}")
            except asyncio.TimeoutError:
                pass

            # --- Step simulation ---
            sim.sim_step()

            # --- Send state to client ---
            packet = sim.get_state_packet()
            await ws.send_text(json.dumps(packet))
            await asyncio.sleep(0.02)
    except Exception as e:
        print("WebSocket closed:", e)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
