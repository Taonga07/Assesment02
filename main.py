from serial.tools.list_ports import comports as list_ports
import time, cv2, numpy as np, serial

BAUDRATE, DEVICE_ID, TARGET_BALLS = 115200, "CH340", 4
BALL_RADIUS, CAM_OFFSET, LOWER_DEPTH = 120, 50, 30
BIN_LEFT, BIN_RIGHT = -20, 20
IS_RIPE = {"LRed", "Green", "HRed"}

COLOUR_RANGES = {
    "Green":  [[78, 45, 35], [96, 255, 255]],  "Blue":  [[96, 100, 20], [130, 255, 255]],
    "Yellow": [[18, 40, 145], [38, 255, 255]], "LRed":  [[0, 70, 80], [10, 255, 200]],
    "HRed":   [[170, 70, 80], [179, 255, 200]],
}

MOVES = {ord("a"):"G91 X-5", ord("d"):"G91 X5", ord("w"):"G91 Y5", ord("s"):"G91 Y-5", ord("r"):"G91 Z5", ord("f"):"G91 Z-5"}

port = next((p.device for p in list_ports() if DEVICE_ID in p.description), None)
if not port: raise RuntimeError("CNC Device not found")
cnc = serial.Serial(port, BAUDRATE, timeout=30)
time.sleep(2); cnc.reset_input_buffer()

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
w, h = int(cam.get(3)), int(cam.get(4))
mask = cv2.circle(np.zeros((h, w), dtype=np.uint8), (w // 2, h // 2), BALL_RADIUS, 255, -1)
mask_pts = cv2.countNonZero(mask)

def tx(cmd):
    print(f"[tx] {cmd}")
    cnc.write(f"{cmd}\n".encode()); cnc.flush()
    while (resp := cnc.readline().decode(errors="ignore").strip()) != "ok":
        if not resp: raise RuntimeError(f"No response to '{cmd}'")
        print(f"[rx] {resp}")

while (k := cv2.waitKey(1) & 0xFF) != ord("q"):
    ok, frame = cam.read()
    if not ok: break
    if k in MOVES:
        tx(MOVES[k])
        tx("G4 P0")
    cv2.imshow("window", frame)
cv2.destroyAllWindows()

sorted_count = 0
while sorted_count < TARGET_BALLS:
    for _ in range(4): cam.grab()
    ok, frame = cam.read()
    if not ok: break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    detected = None
    for name, (low, high) in COLOUR_RANGES.items():
        c_mask = cv2.inRange(hsv, np.array(low), np.array(high))
        ratio = cv2.countNonZero(cv2.bitwise_and(c_mask, mask)) / mask_pts
        if ratio >= (0.2 if name == "Yellow" else 0.3):
            detected = name
            break

    if not detected:
        tx("G91 Y-0.5"); tx("G4 P0")
        continue

    b_off = BIN_LEFT if detected in IS_RIPE else BIN_RIGHT
    for cmd in [f"G91 X{-CAM_OFFSET}",
                f"G91 Z{-LOWER_DEPTH}",
                f"G91 X{b_off}",
                f"G91 Z{LOWER_DEPTH}",
                f"G91 X{-b_off + CAM_OFFSET}"]:
        tx(cmd)
    tx("G4 P0")

    sorted_count += 1
    print(f"Sorted ({sorted_count}/{TARGET_BALLS}): {detected}")

cam.release()
cnc.close()