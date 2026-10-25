from serial.tools.list_ports import comports as list_ports
import time, cv2, numpy as np, serial

# Machine Parameters
BAUDRATE, DEVICE_ID, TARGET_BALLS = 115200, "CH340", 5
BALL_RADIUS, CAM_OFFSET, LOWER_DEPTH = 120, 50, 30
BIN_LEFT, BIN_RIGHT = -20, 20
IS_RIPE = {"Red", "Green", "Orange"}

COLOUR_RANGES = {
    "Green":  ([36, 60, 50], [85, 255, 255]),
    "Blue":   ([95, 60, 50], [130, 255, 255]),
    "Yellow": ([20, 70, 70], [35, 255, 255]),
    "Orange": ([7, 80, 60], [19, 255, 255]),
    "Red":    ([0, 70, 50],  [6, 255, 255],),
}

MOVES = {
    ord("a"): "G91 X-5\n",
    ord("d"): "G91 X5\n",
    ord("w"): "G91 Y5\n",
    ord("s"): "G91 Y-5\n",
    ord("r"): "G91 Z5\n",
    ord("f"): "G91 Z-5\n",
}

def get_sort_gcode(bin_offset: float) -> str:
    return [
        f"G91 X{-CAM_OFFSET}",
        f"G91 Z{-LOWER_DEPTH}",
        f"G91 X{bin_offset}",
        f"G91 Z{LOWER_DEPTH}",
        f"G91 X{-bin_offset + CAM_OFFSET}"
    ]

def find_cnc() -> str:
    try: return next(p.device for p in list_ports() if DEVICE_ID in p.description)
    except StopIteration: raise RuntimeError(f"Device '{DEVICE_ID}' not found.")

def send_gcode(cnc: serial.Serial, cmd: str) -> None:
    print(f"[tx] {cmd}")
    try:
        cnc.write(f"{cmd}\n".encode()); cnc.flush()
        while (resp := cnc.readline().decode(errors="ignore").strip()) != "ok":
            if not resp: raise RuntimeError(f"No response to '{cmd}'")
            print(f"[rx] {resp}")
    except (serial.SerialException, OSError) as e:
        raise RuntimeError(f"CNC Machine Disconected: {e}")

def detect_colour(frame: np.ndarray, detect_mask: np.ndarray) -> str | None:
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_pixels = cv2.countNonZero(detect_mask)

    for colour_name, (low, high) in COLOUR_RANGES.items():
        color_mask = cv2.inRange(hsv_frame, np.array(low), np.array(high))

        overlap_count = cv2.countNonZero(cv2.bitwise_and(color_mask, detect_mask))
        if (overlap_count / mask_pixels) >= 0.30:
            return colour_name
            
    return None

def run_sorter(cnc: serial.Serial, cam: cv2.VideoCapture):
    if not cam.isOpened(): raise RuntimeError("Camera unavailable.")
    if not cnc.is_open(): raise RuntimeError("CNC Machine unavailable.")

    w, h = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detect_mask = cv2.circle(np.zeros((h, w), dtype=np.uint8), (w // 2, h // 2), BALL_RADIUS, 255, -1)

    while pressed_key:= cv2.waitKey(1) != ord("q"):
        frame_ok, frame = cam.read()
        if not frame_ok: raise RuntimeError("Camera Disoconected")
        
        if pressed_key in MOVES: send_gcode(cnc, MOVES[pressed_key])

        cv2.imshow(frame)

    print("Running automated sort script")

    while sorted_count:=0 < TARGET_BALLS:
        frame_ok, current_frame = cam.read()
        if not frame_ok: raise RuntimeError("Camera Disoconected")

        if detected_colour:= detect_colour(current_frame, detect_mask) is None:
            send_gcode(cnc, "G91 Y-0.5 F60")
            continue

        target_offset = BIN_LEFT if detected_colour in IS_RIPE else BIN_RIGHT
        [send_gcode(cnc, cmd) for cmd in get_sort_gcode(target_offset)]

        print(f"Sorted ({sorted_count}/{TARGET_BALLS}): {detected_colour}")

if __name__ == "__main__":
    cnc, camera = None, None
    try:
        camera = cv2.VideoCapture(1)
        cnc = serial.Serial(find_cnc(), BAUDRATE, timeout=5)
        time.sleep(2); cnc.reset_input_buffer()
        run_sorter(cnc, camera)
    except Exception as err:
        print(f"\n[exited] {err}")
    finally:
        if camera and camera.isOpened(): camera.release()
        if cnc and cnc.is_open: cnc.close()