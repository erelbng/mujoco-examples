import asyncio
import base64
import json

import cv2
import numpy as np
import websockets

def has_reached_target(current, target, tol=0.1):
    """Check if all joints are within tolerance of the target."""
    if len(current) != len(target):
        return False
    return all(abs(c - t) < tol for c, t in zip(current, target))

async def main():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        # Define key poses
        poses = [
            [0.0, 0.0, -0.0, 0.0, 0.0],   # Home
            [-0.8, 0.4, -0.3, 0.0, -1.0],  # Move to pick position
            [-0.8, 0.4, -0.3, 1.0, 2.0], # Close gripper (grab)
            [-0.8, 0.0, 0.0, 0.0, 1.0], # Lift
            [1.2, 0.4, -0.3, 0.0, 1.0],  # Move to place position
            [1.2, 0.4, -0.3, 1.0, -2.0],   # Open gripper (release)
            [0.0, 0.4, -0.6, 0.3, 0.0],   # Back to home
        ]

        pose_names = [
            "home", "pick", "grip_close", "lift", "place", "release", "home",
        ]

        current_step = 0
        current_target = poses[current_step]

        while True:
            # Send joint commands
            cmd = {
                "joint_commands": {
                    "names": ["waist", "shoulder", "elbow", "wrist", "gripper"],
                    "positions": current_target,
                }
            }
            await ws.send(json.dumps(cmd))

            # Receive state
            msg = await ws.recv()
            data = json.loads(msg)
            joint_positions = data["joint_positions"]
            print("Joint positions:", joint_positions)

            # Check if target reached
            if has_reached_target(joint_positions[:4], current_target[:4]):
                print(f"✅ Reached {pose_names[current_step]}")

                # Go to next pose
                current_step = (current_step + 1) % len(poses)
                current_target = poses[current_step]

            # Show camera image
            img_b64 = data.get("image")
            if img_b64:
                img_data = base64.b64decode(img_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is not None:
                    cv2.imshow("PincherX Camera", img)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
