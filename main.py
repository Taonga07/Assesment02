
import time
import cv2
import numpy as np
import serial
import serial.tools.list_ports as lp

BAUD, DEVICE_ID = 115200, "CH340"
TARGET_BALL_COUNT = 5
COLOR_THRESHOLD = 0.30
DETECT_RADIUS = 120
STABLE_FRAMES = 3
SCAN_DELAY = 0.15

CAM_OFFSET_X, Z_LOWER_DIST = -50, 33
CUP_OFFSET_LEFT, CUP_OFFSET_RIGHT = -20, 20
LEFT_COLOURS = ("Red", "Green", "Orange")

COLOURS = {
    "Green":  ([36, 60, 50], [85, 255, 255]),
    "Blue":   ([95, 60, 50], [130, 255, 255]),
    "Yellow": ([20, 70, 70], [35, 255, 255]),
    "Orange": ([7, 80, 60], [19, 255, 255]),
    "Red":    ([0, 70, 50], [6, 255, 255]),
}


def connect_cnc():
    port = next(
        (p.device for p in lp.comports()
         if DEVICE_ID.lower() in p.description.lower()),
        None,
    )
    if port is None:
        raise RuntimeError(f"No {DEVICE_ID} device found")

    cnc = serial.Serial(port, BAUD, timeout=0)
    time.sleep(2)
    cnc.reset_input_buffer()
    return cnc


def detect_colour(frame):
    h, w = frame.shape[:2]
    centre = (w // 2, h // 2)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    region = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(region, centre, DETECT_RADIUS, 255, -1)
    area = cv2.countNonZero(region)
    scores = {}

    for name, (lower, upper) in COLOURS.items():
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))

        if name == "Red":
            red2 = cv2.inRange(
                hsv, np.array([170, 70, 50]),
                np.array([180, 255, 255])
            )
            mask = cv2.bitwise_or(mask, red2)

        mask = cv2.bitwise_and(mask, region)
        scores[name] = cv2.countNonZero(mask) / area

    colour = max(scores, key=scores.get)
    return (
        colour if scores[colour] >= COLOR_THRESHOLD else None,
        centre,
        scores,
    )


def show_camera(cam):
    success, frame = cam.read()
    if not success:
        raise RuntimeError("Camera read failed")

    colour, centre, scores = detect_colour(frame)

    cv2.circle(
        frame, centre, DETECT_RADIUS,
        (0, 255, 0) if colour else (0, 0, 255), 2
    )
    cv2.putText(
        frame, colour or "Scanning...", (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8,
        (0, 255, 0) if colour else (0, 0, 255), 2
    )

    for i, (name, score) in enumerate(
        sorted(scores.items(), key=lambda item: -item[1])
    ):
        cv2.putText(
            frame, f"{name}: {score:.0%}", (20, 70 + i * 25),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
        )

    cv2.imshow("Sorting Camera", frame)
    return colour, (cv2.waitKey(1) & 0xFF) != ord("q")


def send_command(cnc, command, cam):
    print("Sending:", command.strip())
    cnc.write(command.encode())
    cnc.flush()

    deadline = time.monotonic() + 60
    response_buffer = b""

    while time.monotonic() < deadline:
        # Keep the camera updating while waiting for the CNC.
        success, frame = cam.read()
        if success:
            cv2.imshow("Sorting Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise KeyboardInterrupt

        # Read serial data without blocking.
        waiting = cnc.in_waiting
        if waiting:
            response_buffer += cnc.read(waiting)

            while b"\n" in response_buffer:
                line, response_buffer = response_buffer.split(b"\n", 1)
                response = line.decode(
                    errors="replace"
                ).strip()

                if not response:
                    continue

                print("CNC:", response)

                if response.lower().startswith("error"):
                    raise RuntimeError(response)

                if response.lower() == "ok":
                    return

        time.sleep(0.005)

    raise TimeoutError(f"No CNC acknowledgement for: {command.strip()}")


def confirm_colour(cam):
    readings = []

    for _ in range(STABLE_FRAMES):
        colour, running = show_camera(cam)
        if not running:
            raise KeyboardInterrupt

        readings.append(colour)
        time.sleep(0.1)

    confirmed = readings[0]
    return confirmed if confirmed and all(
        colour == confirmed for colour in readings
    ) else None


def sort_ball(cnc, cam, colour):
    left = colour in LEFT_COLOURS
    offset = CUP_OFFSET_LEFT if left else CUP_OFFSET_RIGHT

    commands = [
        f"G91 X{CAM_OFFSET_X}\n",
        f"G91 Z{-Z_LOWER_DIST}\n",
        f"G91 X{offset}\n",
        f"G91 Z{Z_LOWER_DIST}\n",
        f"G91 X{-offset}\n",
        f"G91 X{-CAM_OFFSET_X}\n",
    ]

    for command in commands:
        send_command(cnc, command, cam)


def main():
    cnc = connect_cnc()
    cam = cv2.VideoCapture(1)

    # Request a smaller camera buffer to reduce stale frames.
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cam.isOpened():
        cnc.close()
        raise RuntimeError("Camera unavailable")

    try:
        sorted_count = 0
        previous_colour = None

        while sorted_count < TARGET_BALL_COUNT:
            colour, running = show_camera(cam)
            if not running:
                break

            if colour is None:
                previous_colour = None
                send_command(cnc, "G91 Y-0.5 F60\n", cam)
                time.sleep(SCAN_DELAY)
                continue

            # Avoid sorting repeatedly while viewing the same ball.
            if colour == previous_colour:
                time.sleep(0.05)
                continue

            confirmed = confirm_colour(cam)
            if confirmed is None:
                previous_colour = None
                continue

            sort_ball(cnc, cam, confirmed)
            sorted_count += 1
            previous_colour = confirmed

            print(
                f"Sorted {sorted_count}/{TARGET_BALL_COUNT}: "
                f"{confirmed}"
            )

    except KeyboardInterrupt:
        print("Sorting stopped.")

    finally:
        cam.release()
        cv2.destroyAllWindows()
        cnc.close()


if __name__ == "__main__":
    main()
