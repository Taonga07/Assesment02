"""
CNC ball-sorter emulator: fake CH340 serial port + fake toolhead camera + device view.

Run your unmodified script on the emulated machine:

    python emulator.py                       # uses main.py + IMG_0846.JPEG next to this file
    python emulator.py --main main.py --image IMG_0846.JPEG --feed 3000

Windows opened:
  * "Sorting Camera"  - your script's own window (frames come from the bed photo)
  * "Device view"     - whole bed photo with gantry, camera FOV and tool

Device view controls (click the window first):
  left click / drag   place the CAMERA there (teleport, like sliding the gantry by hand)
  right click / drag  place the TOOL there
  W A S D             jog Y+ / X- / Y- / X+   ([ and ] change step 1/5/10/25 mm)
  R / F               Z up / down
  + / -               emulated machine speed x2 / x0.5
  0                   reset machine + balls
  Q                   quit (passed through to your script)
"""
import argparse
import asyncio
import collections
import importlib.util
import math
import os
import sys
import time
import types

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------------------------------------------------------- #
# CONFIG - tweak to match the real rig
# --------------------------------------------------------------------------- #
BALL_DIAMETER_MM = 15      # real ball size; sets the bed-photo scale (override with --px-per-mm)
PX_PER_MM = 160 / BALL_DIAMETER_MM   # bed-photo pixels per mm (ball is ~160 px across in the photo)
CAM_OFFSET_X = 50          # same as main.py: gantry X+50 moves tool onto the spot the camera saw
TOOL_OFFSET_Y = 0          # tool Y offset relative to camera (mm, +Y = image up)
START_PX, START_PY = 805, 40      # camera centre in the bed photo when machine is at X0 Y0 Z0
FOV_PX = (353, 263)        # bed-photo px covered by the camera (derived from your sample frame)
CAM_RES = (640, 480)       # frame size handed to main.py
CAM_HEIGHT_MM = 31         # camera height above bed at Z0 (FOV shrinks as Z lowers); 0 = no zoom
LIMITS = {"x": (-3000, 3000), "y": (-3000, 3000), "z": (-100, 5)}   # machine travel, mm - effectively none, like a rig without end stops
GRAB_Z = -5                # tool counts as "down" below this Z
BALL_R = 80                # ball radius in bed-photo px
TRACK_CLEAN_Y = 1385        # y of a ball-free stretch of the centre track, used to patch the holes
TOOL_R = 40                # tool pusher radius in bed-photo px (engages a ball if within BALL_R + TOOL_R)

# Minus Y moves DOWN the photo (towards the yellow ball), plus X moves RIGHT.
BALLS = [  # name, x, y (bed-photo px)
    ("Green", 810, 148), ("Red", 802, 647), ("Blue", 804, 1017),
    ("Orange", 803, 1276), ("Yellow", 800, 1750),
]
CUPS = [   # name, x, y, radius (bed-photo px)
    ("green cup", 535, 270, 150), ("orange cup", 540, 710, 150),
    ("green cup", 500, 1150, 150), ("orange cup", 535, 1670, 150),
    ("yellow cup", 1050, 300, 150), ("blue cup", 1040, 745, 150),
    ("yellow cup", 1030, 1150, 150), ("black cup", 1030, 1700, 150),
]


# --------------------------------------------------------------------------- #
# Simulated machine + scene
# --------------------------------------------------------------------------- #
class Ball:
    def __init__(self, name, x, y):
        self.name, self.x, self.y = name, float(x), float(y)
        self.home = (float(x), float(y))


class World:
    def __init__(self, image_path, feed, speed):
        photo = cv2.imread(image_path)
        if photo is None:
            raise SystemExit(f"Cannot read bed image: {image_path}")
        self.H, self.W = photo.shape[:2]
        self.photo = photo
        self.feed = feed / 60.0          # mm/s
        self.speed = speed
        self.relative = False
        self.pos = dict(x=0.0, y=0.0, z=0.0)
        self.balls = [Ball(*b) for b in BALLS]
        self.grabbed = None              # (ball, dx, dy) offset from tool in px
        self.log = collections.deque(maxlen=14)
        self.status = "idle"
        self.dirty = True
        self._build_sprites()
        self._scene = None
        self._small = None

    # --- sprites: cut balls out of the photo and inpaint the holes -------- #
    def _build_sprites(self):
        bg = self.photo.copy()
        self.sprites = {}
        hole = np.zeros(bg.shape[:2], np.uint8)
        r = BALL_R
        for b in self.balls:
            x, y = int(b.x), int(b.y)
            x0, y0 = max(x - r - 6, 0), max(y - r - 6, 0)
            patch = self.photo[y0:y + r + 7, x0:x + r + 7].copy()
            m = np.zeros(patch.shape[:2], np.uint8)
            cv2.circle(m, (x - x0, y - y0), r, 255, -1)
            m = cv2.GaussianBlur(m, (5, 5), 0)
            self.sprites[b.name] = (patch, m, x - x0, y - y0)
            cv2.circle(hole, (x, y), r + 22, 255, -1)
        bed = cv2.inpaint(bg, hole, 7, cv2.INPAINT_TELEA)
        # the track is vertically uniform: patch each hole from a clean stretch instead (looks far better)
        rr = r + 22
        for b in self.balls:
            x, y = int(b.x), int(b.y)
            src_y = TRACK_CLEAN_Y
            hole_m = np.zeros((2 * rr, 2 * rr), np.uint8)
            cv2.circle(hole_m, (rr, rr), rr, 255, -1)
            hole_m = cv2.GaussianBlur(hole_m, (15, 15), 0).astype(np.float32)[..., None] / 255.0
            dst = bed[y - rr:y + rr, x - rr:x + rr]
            src = self.photo[src_y:src_y + 2 * rr, x - rr:x + rr]
            if dst.shape == src.shape == (2 * rr, 2 * rr, 3):
                dst[:] = (src * hole_m + dst * (1 - hole_m)).astype(np.uint8)
        self.empty_bed = bed

    def _paste(self, dst, b):
        patch, m, ox, oy = self.sprites[b.name]
        h, w = patch.shape[:2]
        x0, y0 = int(b.x) - ox, int(b.y) - oy
        sx0, sy0 = max(0, -x0), max(0, -y0)
        ex, ey = min(w, self.W - x0), min(h, self.H - y0)
        if ex <= sx0 or ey <= sy0:
            return
        roi = dst[y0 + sy0:y0 + ey, x0 + sx0:x0 + ex]
        a = (m[sy0:ey, sx0:ex].astype(np.float32) / 255.0)[..., None]
        roi[:] = (patch[sy0:ey, sx0:ex] * a + roi * (1 - a)).astype(np.uint8)

    def scene(self):
        if self.dirty or self._scene is None:
            s = self.empty_bed.copy()
            for b in self.balls:
                self._paste(s, b)
            self._scene = s
            self._small = None
            self.dirty = False
        return self._scene

    # --- coordinate helpers ------------------------------------------------ #
    def cam_px(self):
        return START_PX + self.pos["x"] * PX_PER_MM, START_PY - self.pos["y"] * PX_PER_MM

    def tool_px(self):
        cx, cy = self.cam_px()
        return cx - CAM_OFFSET_X * PX_PER_MM, cy - TOOL_OFFSET_Y * PX_PER_MM

    def set_cam_px(self, px, py):
        self.pos["x"] = self._clamp("x", (px - START_PX) / PX_PER_MM)
        self.pos["y"] = self._clamp("y", (START_PY - py) / PX_PER_MM)
        self._after_move()

    def set_tool_px(self, px, py):
        self.set_cam_px(px + CAM_OFFSET_X * PX_PER_MM, py + TOOL_OFFSET_Y * PX_PER_MM)

    def _clamp(self, axis, v):
        lo, hi = LIMITS[axis]
        return min(max(v, lo), hi)

    def jog(self, dx=0, dy=0, dz=0):
        for a, d in zip("xyz", (dx, dy, dz)):
            self.pos[a] = self._clamp(a, self.pos[a] + d)
        self._after_move()

    def _after_move(self):
        down = self.pos["z"] <= GRAB_Z
        tx, ty = self.tool_px()
        if down and self.grabbed is None:
            best = min(self.balls, key=lambda b: math.hypot(b.x - tx, b.y - ty))
            if math.hypot(best.x - tx, best.y - ty) < BALL_R + TOOL_R:
                self.grabbed = (best, best.x - tx, best.y - ty)
                self.log.append(f"tool engaged {best.name} ball")
        elif not down and self.grabbed is not None:
            b = self.grabbed[0]
            cup = next((c for c in CUPS if math.hypot(b.x - c[1], b.y - c[2]) < c[3]), None)
            self.log.append(f"released {b.name} ball -> {cup[0] if cup else 'open bed'}")
            self.grabbed = None
        if self.grabbed:
            b, ox, oy = self.grabbed
            b.x, b.y = tx + ox, ty + oy
            self.dirty = True

    def reset(self):
        self.pos = dict(x=0.0, y=0.0, z=0.0)
        self.grabbed = None
        for b in self.balls:
            b.x, b.y = b.home
        self.dirty = True
        self.log.append("reset")

    # --- camera ------------------------------------------------------------ #
    def camera_frame(self):
        cx, cy = self.cam_px()
        zoom = 1.0
        if CAM_HEIGHT_MM:
            zoom = max(0.2, (CAM_HEIGHT_MM + self.pos["z"]) / CAM_HEIGHT_MM)
        w, h = int(FOV_PX[0] * zoom), int(FOV_PX[1] * zoom)
        crop = cv2.getRectSubPix(self.scene(), (max(w, 2), max(h, 2)), (cx, cy))
        return cv2.resize(crop, CAM_RES, interpolation=cv2.INTER_LINEAR)

    # --- G-code ------------------------------------------------------------ #
    async def execute(self, line):
        """Return the list of response lines for one G-code line."""
        line = line.split(";")[0].split("(")[0].strip().upper()
        if not line:
            return ["ok"]
        self.log.append("> " + line)
        words = line.replace(",", " ").split()
        if line in ("?", "$H"):
            return ["ok"]
        target, motion, feed = {}, None, None
        try:
            for w in words:
                c, v = w[0], w[1:]
                if c == "G":
                    g = int(float(v))
                    if g == 90:
                        self.relative = False
                    elif g == 91:
                        self.relative = True
                    elif g in (0, 1):
                        motion = g
                    elif g == 28:
                        target = {a: 0.0 for a in "xyz"}
                        motion, self.relative = 0, False
                elif c in "XYZ":
                    target[c.lower()] = float(v)
                elif c == "F":
                    feed = float(v) / 60.0
        except ValueError:
            return ["error:25"]
        if feed:
            self.feed = feed
        if motion is not None and target:
            dest = {a: (self.pos[a] + target[a] if self.relative else target[a]) for a in target}
            for a, v in dest.items():
                lo, hi = LIMITS[a]
                if not lo <= v <= hi:
                    self.log.append(f"SOFT LIMIT {a.upper()}={v:.1f}")
                    return ["error:Soft limit"]
            await self._move(dest)
        return ["ok"]

    async def _move(self, dest):
        start = dict(self.pos)
        delta = {a: dest.get(a, start[a]) - start[a] for a in "xyz"}
        dist = math.sqrt(sum(d * d for d in delta.values()))
        if dist < 1e-9:
            return
        dur = dist / (self.feed * self.speed)
        self.status = f"moving {dist:.1f} mm"
        t0, last = time.monotonic(), 0.0
        while last < 1.0:
            await asyncio.sleep(0.008)
            p = min((time.monotonic() - t0) / dur, 1.0)
            step = p - last
            for a in "xyz":
                self.pos[a] += delta[a] * step     # incremental => manual jogs survive
            last = p
            self._after_move()
        self.status = "idle"


# --------------------------------------------------------------------------- #
# Fake serial / camera plumbing
# --------------------------------------------------------------------------- #
class FakeReader:
    def __init__(self):
        self.q = asyncio.Queue()

    async def readline(self):
        return await self.q.get()


class FakeWriter:
    def __init__(self, world, reader):
        self.world, self.reader = world, reader
        self.buf = b""
        self.cmds = asyncio.Queue()
        self.task = asyncio.get_running_loop().create_task(self._worker())

    def write(self, data):
        self.buf += data
        while b"\n" in self.buf:
            line, self.buf = self.buf.split(b"\n", 1)
            self.cmds.put_nowait(line.decode(errors="replace"))

    async def drain(self):
        await asyncio.sleep(0)

    async def _worker(self):
        while True:
            line = await self.cmds.get()
            for resp in await self.world.execute(line):
                await self.reader.q.put((resp + "\r\n").encode())

    def close(self):
        self.task.cancel()

    async def wait_closed(self):
        pass


class FakeCapture:
    world = None

    def __init__(self, *a, **k):
        self.opened = True

    def isOpened(self):
        return self.opened

    def read(self):
        return True, self.world.camera_frame()

    def release(self):
        self.opened = False

    def set(self, *a):
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FRAME_WIDTH: CAM_RES[0], cv2.CAP_PROP_FRAME_HEIGHT: CAM_RES[1]}.get(prop, 0)


# --------------------------------------------------------------------------- #
# Device view
# --------------------------------------------------------------------------- #
VIEW_H = 900
PANEL_W = 330


class DeviceView:
    NAME = "Device view"

    def __init__(self, world, headless):
        self.w, self.headless = world, headless
        self.scale = VIEW_H / world.H
        self.step_i, self.steps = 1, [1, 5, 10, 25]
        self.pending = collections.deque()
        self.mouse_down = None
        if not headless:
            cv2.namedWindow(self.NAME, cv2.WINDOW_AUTOSIZE)
            cv2.setMouseCallback(self.NAME, self._mouse)

    def _mouse(self, event, x, y, flags, _):
        if x >= int(self.w.W * self.scale):
            return
        px, py = x / self.scale, y / self.scale
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = "L"
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_down = "R"
        elif event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            self.mouse_down = None
            return
        if event in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN) or (
                event == cv2.EVENT_MOUSEMOVE and self.mouse_down):
            (self.w.set_cam_px if self.mouse_down == "L" else self.w.set_tool_px)(px, py)

    def handle_key(self, k):
        k &= 0xFF
        w, s = self.w, self.steps[self.step_i]
        if k == 255:
            return
        c = chr(k).lower()
        if c == "w": w.jog(dy=s)
        elif c == "s": w.jog(dy=-s)
        elif c == "a": w.jog(dx=-s)
        elif c == "d": w.jog(dx=s)
        elif c == "r": w.jog(dz=s)
        elif c == "f": w.jog(dz=-s)
        elif c == "[": self.step_i = max(0, self.step_i - 1)
        elif c == "]": self.step_i = min(len(self.steps) - 1, self.step_i + 1)
        elif c in "=+": w.speed *= 2
        elif c in "-_": w.speed /= 2
        elif c == "0": w.reset()

    def render(self):
        w, sc = self.w, self.scale
        if w._small is None or w.dirty:
            w.scene()
            w._small = cv2.resize(w._scene, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
        img = w._small.copy()
        iw, ih = img.shape[1], img.shape[0]
        cx, cy = (v * sc for v in w.cam_px())
        tx, ty = (v * sc for v in w.tool_px())

        # gantry: X beam across the bed, Y rails down both sides
        ov = img.copy()
        cv2.rectangle(ov, (0, int(cy - 9)), (iw, int(cy + 9)), (60, 60, 60), -1)
        cv2.rectangle(ov, (0, 0), (8, ih), (60, 60, 60), -1)
        cv2.rectangle(ov, (iw - 8, 0), (iw, ih), (60, 60, 60), -1)
        # carriage block holding camera + tool
        x0, x1 = int(min(cx, tx) - 22), int(max(cx, tx) + 22)
        cv2.rectangle(ov, (x0, int(cy - 20)), (x1, int(cy + 20)), (90, 60, 30), -1)
        img = cv2.addWeighted(ov, 0.6, img, 0.4, 0)
        cv2.rectangle(img, (x0, int(cy - 20)), (x1, int(cy + 20)), (200, 200, 200), 1)

        # camera field of view
        zoom = max(0.2, (CAM_HEIGHT_MM + w.pos["z"]) / CAM_HEIGHT_MM) if CAM_HEIGHT_MM else 1.0
        fw, fh = FOV_PX[0] * zoom * sc / 2, FOV_PX[1] * zoom * sc / 2
        cv2.rectangle(img, (int(cx - fw), int(cy - fh)), (int(cx + fw), int(cy + fh)), (255, 255, 0), 2)
        cv2.circle(img, (int(cx), int(cy)), 6, (255, 255, 0), -1)
        cv2.putText(img, "CAM", (int(cx - fw), int(cy - fh - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # tool
        down = w.pos["z"] <= GRAB_Z
        col = (0, 0, 255) if down else (0, 165, 255)
        cv2.line(img, (int(cx), int(cy)), (int(tx), int(ty)), (255, 255, 255), 1)
        cv2.circle(img, (int(tx), int(ty)), 14, col, -1 if down else 3)
        cv2.drawMarker(img, (int(tx), int(ty)), (255, 255, 255), cv2.MARKER_CROSS, 22, 1)
        cv2.putText(img, "TOOL", (int(tx) - 20, int(ty) - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)

        # side panel
        panel = np.full((ih, PANEL_W, 3), 32, np.uint8)
        lines = [
            (f"X {w.pos['x']:8.1f} mm", (255, 255, 255)),
            (f"Y {w.pos['y']:8.1f} mm", (255, 255, 255)),
            (f"Z {w.pos['z']:8.1f} mm   {'DOWN' if down else 'up'}", col),
            (f"state: {w.status}", (180, 255, 180)),
            (f"speed x{w.speed:g}  feed {w.feed * 60:.0f} mm/min", (200, 200, 200)),
            (f"jog step {self.steps[self.step_i]} mm", (200, 200, 200)),
            ("", None),
            ("L-click: camera   R-click: tool", (150, 150, 150)),
            ("WASD jog  R/F Z  [ ] step", (150, 150, 150)),
            ("+/- speed   0 reset   Q quit", (150, 150, 150)),
            ("", None),
            ("G-code log:", (255, 255, 120)),
        ] + [(l, (220, 220, 220)) for l in w.log]
        for i, (t, c) in enumerate(lines):
            if t:
                cv2.putText(panel, t, (10, 28 + i * 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)
        return np.hstack([img, panel])


# --------------------------------------------------------------------------- #
# Wiring it all into main.py
# --------------------------------------------------------------------------- #
def install(world, view, headless):
    real_waitkey = cv2.waitKey

    def poll():
        if headless:
            return -1
        view_img = view.render()
        cv2.imshow(DeviceView.NAME, view_img)
        k = real_waitkey(1)
        if k != -1:
            view.handle_key(k)
        return k

    def wait_key(delay=0):
        if view.pending:
            return view.pending.popleft()
        k = poll()
        return k

    cv2.waitKey = wait_key
    if headless:
        cv2.imshow = lambda *a, **k: None
        cv2.destroyAllWindows = lambda: None
    FakeCapture.world = world
    cv2.VideoCapture = FakeCapture

    # serial.tools.list_ports.comports
    try:
        import serial.tools.list_ports as lp
    except ImportError:
        serial = types.ModuleType("serial"); tools = types.ModuleType("serial.tools")
        lp = types.ModuleType("serial.tools.list_ports")
        serial.tools, tools.list_ports = tools, lp
        sys.modules.update({"serial": serial, "serial.tools": tools, "serial.tools.list_ports": lp})

    class Port:
        device = "EMU-COM99"

        @property
        def description(self):
            m = sys.modules.get("main")
            return f"USB-SERIAL {getattr(m, 'DEVICE_ID', 'CH340')} (emulated)"

    lp.comports = lambda *a, **k: [Port()]

    # serial_asyncio.open_serial_connection
    async def open_serial_connection(*a, **k):
        reader = FakeReader()
        writer = FakeWriter(world, reader)

        async def viewer():
            while True:                       # keeps the device view animating during long moves
                k_ = poll()
                if k_ != -1 and (k_ & 0xFF) in (ord("q"), ord("Q")):
                    view.pending.append(k_)
                await asyncio.sleep(1 / 30)
        asyncio.get_running_loop().create_task(viewer())
        return reader, writer

    sa = types.ModuleType("serial_asyncio")
    sa.open_serial_connection = open_serial_connection
    sys.modules["serial_asyncio"] = sa


def load_main(path):
    spec = importlib.util.spec_from_file_location("main", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["main"] = mod
    sys.path.insert(0, os.path.dirname(os.path.abspath(path)))
    spec.loader.exec_module(mod)
    return mod


def find(*names):
    for n in names:
        for d in (HERE, os.getcwd()):
            p = os.path.join(d, n)
            if os.path.exists(p):
                return p
    return names[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main", default=find("main.py"))
    ap.add_argument("--image", default=find("IMG_0846.JPEG", "IMG_0846.jpeg", "IMG_0846.jpg"))
    ap.add_argument("--feed", type=float, default=3000, help="default feed mm/min (G-code F overrides)")
    ap.add_argument("--speed", type=float, default=1.0, help="emulated speed multiplier")
    ap.add_argument("--headless", action="store_true", help="no windows; save device-view PNG on exit")
    ap.add_argument("--snapshot", default="device_view.png", help="headless output path")
    ap.add_argument("--start-px", type=float, nargs=2, metavar=("X", "Y"), help="where the camera starts on the bed photo (px)")
    ap.add_argument("--px-per-mm", type=float, help="override bed-photo scale (default from 15 mm balls)")
    a = ap.parse_args()
    if a.start_px:
        globals()["START_PX"], globals()["START_PY"] = a.start_px
    if a.px_per_mm:
        globals()["PX_PER_MM"] = a.px_per_mm

    print("Loading bed photo and cutting out the balls (a few seconds)...")
    world = World(a.image, a.feed, a.speed)
    view = DeviceView(world, a.headless)
    install(world, view, a.headless)
    mod = load_main(a.main)
    try:
        asyncio.run(mod.main())
    finally:
        if a.headless:
            cv2.imwrite(a.snapshot, view.render())
            print("saved", a.snapshot)


if __name__ == "__main__":
    main()
