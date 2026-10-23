import serial.tools.list_ports as lp
from serial import Serial
from time import sleep
import cv2

BAUD = 115200
DEVICE_ID = "CH340"

MOVES = {
    ord("a"): "G91 X-20\n",
    ord("d"): "G91 X20\n",
    ord("w"): "G91 Y20\n",
    ord("s"): "G91 Y-20\n",
    ord("r"): "G91 Z20\n",
    ord("f"): "G91 Z-20\n",
}

def send(conn: Serial, command: str):
    conn.write(command.encode())
    conn.flush()

    response = None
    while response != "ok":
        response = conn.readline().decode().strip()
        print(response)

def main(conn: Serial, cam: cv2.VideoCapture) -> bool:
    success, frame = cam.read()

    if not success:
        print("Error: Camera could not be read")
        return False

    cv2.imshow("window", frame)
    key = cv2.waitKey(1) & 0xFF

    if key in MOVES:
        send(conn, MOVES[key])

    return key != ord("q")

if __name__ == "__main__":
    ports = list(filter(lambda p: DEVICE_ID in p.description, lp.comports()))
    if not ports: raise SystemExit(f"No devices of ID '{DEVICE_ID}' found.")

    conn = Serial(ports[0].device, baudrate=BAUD, timeout=1)
    cam = cv2.VideoCapture(0)

    sleep(2)
    conn.reset_input_buffer()
    
    while main(conn, cam): pass
    
    conn.close()
    cam.release()
    cv2.destroyAllWindows()