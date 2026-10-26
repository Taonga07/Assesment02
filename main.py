from serial.tools.list_ports import comports as list_ports
import time, cv2, numpy as np, serial

# Machine Parameters
BAUDRATE, DEVICE_ID, TARGET_BALLS = 115200, "CH340", 4
BALL_RADIUS, CAM_OFFSET, LOWER_DEPTH = 120, 50, 30
BIN_LEFT, BIN_RIGHT = -20, 20
IS_RIPE = {"LRed", "Green", "HRed"}
FLUSH_FRAMES = 4

COLOUR_RANGES = {
    "Green":  [[78, 45, 35], [96, 255, 255]],
    "Blue":   [[96, 100, 20], [130, 255, 255]],
    "Yellow": [[18, 40, 145], [38, 255, 255]],
    "LRed":   [[0, 70, 80], [10, 255, 200]],
    "HRed":   [[170, 70, 80], [179, 255, 200]],
}

MOVES = {
    ord("a"): "G91 X-5", ord("d"): "G91 X5",
    ord("w"): "G91 Y5",  ord("s"): "G91 Y-5",
    ord("r"): "G91 Z5",  ord("f"): "G91 Z-5",
}

def get_sort_gcode(bin_offset: float) -> list[str]:
    return [
        f"G91 X{-CAM_OFFSET}",
        f"G91 Z{-LOWER_DEPTH}",
        f"G91 X{bin_offset}",
        f"G91 Z{LOWER_DEPTH}",
        f"G91 X{-bin_offset + CAM_OFFSET}",
    ]

def find_cnc() -> str:
    try: return next(p.device for p in list_ports() if DEVICE_ID in p.description)
    except StopIteration: raise RuntimeError(f"Device '{DEVICE_ID}' not found.")

def _txrx(cnc: serial.Serial, cmd: str) -> None:
    """Send one line, block until the controller replies 'ok'."""
    print(f"[tx] {cmd}")
    cnc.write(f"{cmd}\n".encode()); cnc.flush()
    while (resp := cnc.readline().decode(errors="ignore").strip()) != "ok":
        if not resp: raise RuntimeError(f"No response to '{cmd}'")
        print(f"[rx] {resp}")

def send_gcode(cnc: serial.Serial, cmd: str, wait: bool = True) -> None:
    """Send a command; if wait, also block until the machine has finished moving."""
    try:
        _txrx(cnc, cmd)
        if wait: _txrx(cnc, "G4 P0") # wait cmd
    except (serial.SerialException, OSError) as e:
        raise RuntimeError(f"CNC Machine Disconnected: {e}")

def send_sequence(cnc: serial.Serial, cmds: list[str]) -> None:
    """Queue every move back to back, sync only after the last one."""
    for i, cmd in enumerate(cmds): send_gcode(cnc, cmd, wait=(i == len(cmds) - 1))

def read_fresh(cam: cv2.VideoCapture) -> np.ndarray:
    """Discard buffered frames so detection uses an image taken after the move."""
    for _ in range(FLUSH_FRAMES): cam.grab()
    ok, frame = cam.read()
    if not ok: raise RuntimeError("Camera Disconnected")
    return frame

def detect_colour(frame: np.ndarray, detect_mask: np.ndarray) -> str | None:
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask_pixels = cv2.countNonZero(detect_mask)

    for colour_name, (low, high) in COLOUR_RANGES.items():
        color_mask = cv2.inRange(hsv_frame, np.array(low), np.array(high))
        overlap_count = cv2.countNonZero(cv2.bitwise_and(color_mask, detect_mask))
        print(overlap_count / mask_pixels, colour_name)
        if (overlap_count / mask_pixels) >= 0.2:
            return colour_name
    return None

def run_sorter(cnc: serial.Serial, cam: cv2.VideoCapture) -> None:
    if not cam.isOpened(): raise RuntimeError("Camera unavailable.")
    if not cnc.is_open: raise RuntimeError("CNC Machine unavailable.")

    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise camera lag
    send_gcode(cnc, "G91", wait=False)   # relative mode, set once

    w, h = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detect_mask = cv2.circle(np.zeros((h, w), dtype=np.uint8), (w // 2, h // 2), BALL_RADIUS, 255, -1)

    while (pressed_key := cv2.waitKey(1) & 0xFF) != ord("q"):
        frame_ok, frame = cam.read()
        if not frame_ok: raise RuntimeError("Camera Disconnected")
        if pressed_key in MOVES: send_gcode(cnc, MOVES[pressed_key])
        cv2.imshow("window", frame)

    cv2.destroyAllWindows()
    print("Running automated sort script")

    sorted_count = 0
    while sorted_count < TARGET_BALLS:
        detected_colour = detect_colour(read_fresh(cam), detect_mask)

        if detected_colour is None:
            send_gcode(cnc, f"G91 Y{-0.5}")  # waits so the next frame is fresh
            continue

        send_sequence(cnc, get_sort_gcode(BIN_LEFT if detected_colour in IS_RIPE else BIN_RIGHT))

        sorted_count += 1
        print(f"Sorted ({sorted_count}/{TARGET_BALLS}): {detected_colour}")

if __name__ == "__main__":
    cnc, camera = None, None
    try:
        camera = cv2.VideoCapture(1)
        cnc = serial.Serial(find_cnc(), BAUDRATE, timeout=30)
        time.sleep(2); cnc.reset_input_buffer()
        run_sorter(cnc, camera)
    finally:
        if camera and camera.isOpened(): camera.release()
        if cnc and cnc.is_open: cnc.close()

# stop if moved certain distance