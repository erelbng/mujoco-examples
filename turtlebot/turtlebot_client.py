import asyncio
import base64
import json

import cv2
import numpy as np
import websockets


async def main():
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        while True:
            # --- Send command ---
            cmd = {"cmd_vel": {"linear_x": 2.0, "angular_z": 1.0}}
            await ws.send(json.dumps(cmd))

            # --- Receive state ---
            msg = await ws.recv()
            data = json.loads(msg)

            # Print odometry
            print("Odom:", data["odom"])

            # --- Show camera image ---
            img_b64 = data.get("image")
            if img_b64:
                try:
                    img_data = base64.b64decode(img_b64)
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        cv2.imshow("TurtleBot Camera", img)
                        if cv2.waitKey(1) & 0xFF == 27:
                            break
                except Exception as e:
                    print("Image decode error:", e)

            await asyncio.sleep(0.01)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
