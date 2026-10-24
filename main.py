import asyncio
import cv2
import numpy as np
import serial.tools.list_ports as lp
from serial_asyncio import open_serial_connection as connect

BAUD, DEVICE_ID = 115200, "CH340"
TARGET_BALL_COUNT, COLOR_THRESHOLD = 5, 0.85
DETECT_RADIUS = 120  # px, sized to sit just inside the ball in the frame (ball is ~300px across)
CAM_OFFSET_X, Z_LOWER_DIST = 50, 20
CUP_OFFSET_LEFT, CUP_OFFSET_RIGHT = -100, 100
LEFT_COLOURS = ("Red", "Green", "Orange")  # everything else goes right

MOVES = {
    "SCAN": "G91 G1 Y-1\n",
    "CAM_TOOL": f"G91 G1 X{CAM_OFFSET_X}\n",
    "TOOL_CAM": f"G91 G1 X{-CAM_OFFSET_X}\n",
    "LOWER": f"G91 G1 Z{-Z_LOWER_DIST}\n",
    "RAISE": f"G91 G1 Z{Z_LOWER_DIST}\n",
    "LEFT": f"G91 G1 X{CUP_OFFSET_LEFT}\n",
    "RIGHT": f"G91 G1 X{CUP_OFFSET_RIGHT}\n",
}

COLOURS = {
    "Green":  [[35,  80, 70],   [85,  255, 255]],
    "Blue":   [[95,  80, 60],   [130, 255, 255]],
    "Yellow": [[20,  100, 100], [30,  255, 255]],
    "Orange": [[7,   140, 130], [21,  255, 255]],
    "Red":    [[0,   140, 100], [6,   255, 255]]
}

def detect_colour(frame):
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    mask = np.zeros((h, w), np.uint8)
    cv2.circle(mask, (cx, cy), DETECT_RADIUS, 255, -1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    area = cv2.countNonZero(mask)
    matches = {
        name: cv2.countNonZero(cv2.bitwise_and(
            cv2.inRange(hsv, np.array(bounds[0]), np.array(bounds[1])),
            mask
        )) / area
        for name, bounds in COLOURS.items()
    }
    colour = max(matches, key=matches.get) if matches else None
    return (colour if colour and matches[colour] >= COLOR_THRESHOLD else None), (cx, cy)

async def main():
    port = next((p.device for p in lp.comports() if DEVICE_ID in p.description), None)
    if not port: raise RuntimeError(f"No {DEVICE_ID} device found")

    reader, writer = await connect(url=port, baudrate=BAUD)
    cam = cv2.VideoCapture(0)

    try:
        if not cam.isOpened():
            raise RuntimeError("Camera unavailable")

        await asyncio.sleep(2)

        sorted_count = 0

        while sorted_count < TARGET_BALL_COUNT:
            success, frame = cam.read()
            if not success:
                raise RuntimeError("Camera read failed")

            colour, (cx, cy) = detect_colour(frame)
            cv2.circle(frame, (cx, cy), DETECT_RADIUS,
                       (0, 255, 0) if colour else (0, 0, 255), 2)
            cv2.putText(frame, colour or "Scanning...", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Sorting Camera", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            left = colour in LEFT_COLOURS
            cup = MOVES["LEFT"] if left else MOVES["RIGHT"]
            reset = f"G91 G1 X{-CUP_OFFSET_LEFT if left else -CUP_OFFSET_RIGHT}\n"

            commands = (
                [MOVES["CAM_TOOL"], MOVES["LOWER"], cup,
                 MOVES["RAISE"], reset, MOVES["TOOL_CAM"]]
                if colour else [MOVES["SCAN"]]
            )

            for cmd in commands:
                writer.write(cmd.encode())
                await writer.drain()

                response = await asyncio.wait_for(reader.readline(), 15)
                response = response.decode(errors="replace").strip()

                if response.lower().startswith("error") or not response:
                    raise RuntimeError(f"CNC response: {response or 'disconnected'}")

                if response.lower() != "ok":
                    raise RuntimeError(f"Unexpected CNC response: {response}")

            if colour:
                sorted_count += 1
                print(f"Sorted {sorted_count}/{TARGET_BALL_COUNT}: {colour}")

    finally:
        cam.release()
        cv2.destroyAllWindows()
        writer.close()
        await writer.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())