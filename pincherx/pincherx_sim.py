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

app = FastAPI(title="PincherX MuJoCo WebSocket Server")


class PincherXSim:
    def __init__(self):
        # === Load MuJoCo model ===
        self.model_path = os.path.join(os.path.dirname(__file__), "example_scene.xml")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)

        # === Actuators ===
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]
        self.target_ctrl = np.zeros(self.model.nu)

        # === Simulation parameters ===
        self.sim_dt = self.model.opt.timestep
        self.real_dt = 0.02
        self.steps_per_update = int(self.real_dt / self.sim_dt)

        # === Camera ===
        self.renderer = mujoco.Renderer(self.model, 480, 480)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        self.cam.fixedcamid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "cam"
        )

        self.last_image = None

        # === MuJoCo Viewer ===
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        )

        print("PincherX simulation initialized.")

    # --- Set joint commands ---
    def set_joint_commands(self, names, positions):
        for name, pos in zip(names, positions):
            if name in self.actuator_names:
                act_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
                )
                self.target_ctrl[act_id] = pos

    # --- Single simulation step ---
    def sim_step(self):
        # Apply commands
        self.data.ctrl[:] = self.target_ctrl

        # Step physics
        for _ in range(self.steps_per_update):
            mujoco.mj_step(self.model, self.data)

        # Render camera
        self.renderer.update_scene(self.data, camera=self.cam)
        img = (self.renderer.render()).astype(np.uint8)
        self.last_image = img[..., ::-1]  # RGB -> BGR for OpenCV

        # Update viewer
        self.viewer.sync()

    # --- Package state for WebSocket ---
    def get_state_packet(self):
        joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            for j in range(self.model.njnt)
        ]
        qpos = self.data.qpos[: self.model.nq].tolist()
        qvel = self.data.qvel[: self.model.nv].tolist()

        img_b64 = None
        if self.last_image is not None:
            ret, buf = cv2.imencode(".jpg", self.last_image)
            if ret:
                img_b64 = base64.b64encode(buf).decode("utf-8")

        return {
            "joint_names": joint_names,
            "joint_positions": qpos,
            "joint_velocities": qvel,
            "image": img_b64,
        }


sim = PincherXSim()


# === WebSocket endpoint ===
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    print("Client connected.")
    try:
        while True:
            # --- Receive joint command (non-blocking with timeout) ---
            try:
                data = await asyncio.wait_for(ws.receive_text(), timeout=0.01)
                data_json = json.loads(data)
                if "joint_commands" in data_json:
                    names = data_json["joint_commands"].get("names", [])
                    positions = data_json["joint_commands"].get("positions", [])
                    sim.set_joint_commands(names, positions)
                    print(f"Received joint_commands: {list(zip(names, positions))}")
            except asyncio.TimeoutError:
                pass

            # --- Step simulation ---
            sim.sim_step()

            # --- Send state ---
            packet = sim.get_state_packet()
            await ws.send_text(json.dumps(packet))
            await asyncio.sleep(0.02)
    except Exception as e:
        print("WebSocket closed:", e)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
