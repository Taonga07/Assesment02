import serial.tools.list_ports as lp
from serial import Serial
from time import sleep
import numpy as np
import json
import cv2

BAUD = 115520
DEVICE_ID = "CH340"

MOVES = {
    ord("a"): "G91 X-5\n",
    ord("d"): "G91 X5\n",
    ord("w"): "G91 Y5\n",
    ord("s"): "G91 Y-5\n",
    ord("r"): "G91 Z5\n",
    ord("f"): "G91 Z-5\n",
}

def send(conn: Serial, command: str):
    conn.write(command.encode())
    conn.flush()

    response = None
    while response != "ok":
        response = conn.readline().decode().strip()
        print(response)

def main(conn: Serial, cam: cv2.VideoCapture, colours) -> bool:
    success, frame = cam.read()

    if not success:
        print("Error: Camera could not be read")
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for c in colours:
        mask = cv2.inRange(hsv, np.array(colours[c][0]), np.array(colours[c][1]))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Filter out tiny noise contours by checking area
            if cv2.contourArea(contour) > 500:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Draw bounding box and color name tag
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("window", frame)
    key = cv2.waitKey(1) & 0xFF

    if key in MOVES:
        send(conn, MOVES[key])

    return key != ord("q")

if __name__ == "__main__":
    ports = list(filter(lambda p: DEVICE_ID in p.description, lp.comports()))
    if not ports: raise SystemExit(f"No devices of ID '{DEVICE_ID}' found.")

    conn = Serial(ports[0].device, baudrate=BAUD, timeout=1)
    colours = json.load(open('colours.json', 'r'))
    cam = cv2.VideoCapture(0)

    sleep(2)
    conn.reset_input_buffer()
    
    while main(conn, cam, colours): pass
    
    conn.close()
    cam.release()
    cv2.destroyAllWindows()