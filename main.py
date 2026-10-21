import serial.tools.list_ports as lp
from serial import Serial
import cv2

BAUD = 115200
DEVICE_ID = "CH340"

def main(conn: Serial, cam: cv2.VideoCapture) -> bool:
    pressed_key = cv2.waitKey(1)

    success, frame = cam.read()

    if not success:
        print("Error Camera could not be read")
        return False

    cv2.imshow(frame)

    return pressed_key != ord("q")

if __name__ == "__main__":
    ports = filter(lambda p: DEVICE_ID in p.description, lp.comports())
    if not ports: raise SystemExit(f"No devices of ID '{DEVICE_ID}' found.")

    conn = Serial(ports[0], baudrate=BAUD, timeout=1) # select first device
    cam = cv2.VideoCapture(0)

    while main(conn, cam): pass

    conn.close()
    cam.release()

    cv2.destroyAllWindows()
    