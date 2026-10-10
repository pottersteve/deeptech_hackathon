"""Live maze map from the Arduino maze robot.

    python maze_mapper.py                    # auto-detect the Arduino
    python maze_mapper.py --port COM5        # explicit port
    python maze_mapper.py --sim              # virtual maze, no hardware
    python maze_mapper.py --replay run.log   # play back a recorded run

Controls: mouse wheel = zoom, drag = pan, F = follow robot, R = reset map,
Q/E = nudge heading -/+5 deg, S = save screenshot, G = show the simulator's hidden maze (accuracy check),
L = wall lines on/off, H or the button = stop and drive back to START (again = cancel),
P or the button = click a point on the map and the robot drives there (again = cancel), Esc = quit.
"""
from __future__ import annotations

import argparse
import math
import queue
import sys
import threading
import time
from collections import deque
from datetime import datetime

import numpy as np
import pygame

from mapping import MazeMapper, RobotConfig, lattice_walls, parse_line, sensor_mounts
from navigation import DriveTo, nearest_visited

UNKNOWN = np.array([196, 199, 204], dtype=np.float32)
FREE = np.array([250, 250, 247], dtype=np.float32)
WALL = np.array([24, 32, 56], dtype=np.float32)
WALL_FAINT = np.array([150, 158, 178], dtype=np.float32)  # raw sonar hits, when wall lines are drawn on top
WALL_LINE = (20, 26, 48)
ROUTE_COLOR = (0, 150, 160)
BUTTON_GO = (200, 50, 50)
BUTTON_GOTO = (40, 100, 200)
TARGET_COLOR = (0, 150, 160)
BUTTON_CANCEL = (215, 140, 0)
BUTTON_OFF = (150, 150, 150)
PATH_COLOR = (235, 120, 30)
ROBOT_COLOR = (30, 110, 220)
SENSOR_COLORS = {"left": (40, 170, 80), "front": (220, 160, 0), "right": (150, 70, 200)}
START_COLOR = (30, 160, 60)
EXIT_COLOR = (210, 30, 120)
TRUTH_COLOR = (230, 40, 40)
TEXT = (20, 20, 20)

ARDUINO_VID = 0x2341


# ---------------------------------------------------------------- line sources

class LineSource(threading.Thread):
    """Background thread that pushes text lines into a queue."""

    def __init__(self):
        super().__init__(daemon=True)
        self.lines: queue.Queue[str] = queue.Queue()
        self.status = "starting"
        self.stopped = threading.Event()

    def stop(self) -> None:
        self.stopped.set()

    def send(self, line: str) -> bool:
        """Send a command to the robot (see navigation.py). False if this source can't."""
        return False


class SerialSource(LineSource):
    def __init__(self, port: str, baud: int):
        super().__init__()
        self.port, self.baud = port, baud
        self.ser = None

    def run(self) -> None:
        import serial
        while not self.stopped.is_set():
            try:
                with serial.Serial(self.port, self.baud, timeout=0.5) as ser:
                    self.ser = ser
                    self.status = f"connected {self.port} @ {self.baud}"
                    while not self.stopped.is_set():
                        raw = ser.readline()
                        if raw:
                            self.lines.put(raw.decode("utf-8", "replace").strip())
            except serial.SerialException as e:
                self.status = f"waiting for {self.port} ({e.__class__.__name__})"
                self.stopped.wait(1.0)
            finally:
                self.ser = None

    def send(self, line: str) -> bool:
        ser = self.ser
        if ser is None:
            return False
        try:
            ser.write((line + "\n").encode())
            return True
        except Exception:
            return False


class SimSource(LineSource):
    def __init__(self, sim, speed: float):
        super().__init__()
        self.sim, self.speed = sim, speed

    def run(self) -> None:
        self.status = f"simulator x{self.speed:g}"
        while not self.stopped.is_set():
            for line in self.sim.step().splitlines():
                self.lines.put(line)
            self.stopped.wait(self.sim.dt_ms / 1000.0 / self.speed)

    def send(self, line: str) -> bool:
        self.sim.command(line)
        return True


class ReplaySource(LineSource):
    def __init__(self, path: str, speed: float):
        super().__init__()
        self.path, self.speed = path, speed

    def run(self) -> None:
        self.status = f"replaying {self.path}"
        last_t = None
        with open(self.path, encoding="utf-8", errors="replace") as f:
            for line in f:
                if self.stopped.is_set():
                    return
                line = line.strip()
                t = parse_line(line)
                if t is not None:
                    if last_t is not None and 0 < t.t_ms - last_t < 2000:
                        self.stopped.wait((t.t_ms - last_t) / 1000.0 / self.speed)
                    last_t = t.t_ms
                self.lines.put(line)
        self.status = f"replay finished ({self.path})"


def find_arduino_port() -> str | None:
    from serial.tools import list_ports
    ports = list(list_ports.comports())
    for p in ports:
        if p.vid == ARDUINO_VID:
            return p.device
    for p in ports:
        text = f"{p.description} {p.manufacturer or ''}".lower()
        if "arduino" in text or "uno" in text:
            return p.device
    return None


# ---------------------------------------------------------------- viewer

class App:
    def __init__(self, source: LineSource, mapper: MazeMapper, log_path: str | None = None,
                 sim=None, maze_cell_cm: float = 35.0, size=(1280, 820)):
        self.source = source
        self.mapper = mapper
        self.sim = sim
        self.maze_cell = maze_cell_cm
        self.log = open(log_path, "a", encoding="utf-8") if log_path else None

        pygame.init()
        pygame.display.set_caption("Maze Mapper")
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self.font = pygame.font.SysFont("consolas,menlo,monospace", 15)
        self.small = pygame.font.SysFont("consolas,menlo,monospace", 13)
        self.clock = pygame.time.Clock()

        self.zoom = 3.0  # pixels per cm
        self.cam = [0.0, 0.0]
        self.follow = True
        self.show_truth = False  # G toggles the simulator's hidden maze, for checking accuracy
        self.show_lines = True   # L toggles wall lines snapped to the maze lattice
        self.dragging = False
        self.console: deque[str] = deque(maxlen=6)
        self.rate_times: deque[float] = deque(maxlen=50)
        self._map_surface = None
        self._map_key = None
        self._walls: list[tuple[float, float, float, float]] = []
        self._walls_version = -1
        self.drive: DriveTo | None = None  # current return-home / go-to-point trip
        self.drive_home = False             # ...and whether it's the return home
        self.picking = False                # waiting for a click on the map to pick a go-to point
        self.home_button = pygame.Rect(0, 0, 0, 0)
        self.goto_button = pygame.Rect(0, 0, 0, 0)

    # -- data

    def handle_line(self, line: str) -> None:
        if not line:
            return
        if self.log:
            self.log.write(line + "\n")
        t = parse_line(line)
        if t is None:
            self.console.append(line[:120])
            if line.startswith("EXIT"):
                self.mapper.mark_exit()
            return
        self.mapper.update(t)
        self.rate_times.append(time.monotonic())
        if self.drive:
            l, r = self.drive.command(self.mapper.pose)
            if self.drive.done:
                self.source.send("STOP")
                self.console.append(("back at START" if self.drive_home else "arrived") + " - robot stopped")
                self.drive = None
            else:
                self.source.send(f"M,{l},{r}")

    def start_drive(self, goal: tuple[int, int], home: bool) -> None:
        """Stop the robot and drive it to a visited maze cell."""
        self.picking = False
        if not self.source.send("STOP"):
            self.console.append("can't send commands to this source")
            return
        self.drive = DriveTo(self.mapper.path, self.mapper.pose, self.maze_cell, goal)
        self.drive_home = home
        where = "START" if home else f"cell {goal}"
        self.console.append(f"robot stopped - driving to {where} ({len(self.drive.waypoints)} waypoints)")
        if isinstance(self.source, SerialSource):
            self.console.append("note: needs maze_robot_nav.ino on the robot (maze_robot.ino ignores commands)")

    def cancel_drive(self) -> None:
        self.picking = False
        if self.drive:
            self.drive = None
            self.source.send("STOP")
            self.console.append("trip cancelled - robot stopped")

    def toggle_return_home(self) -> None:
        """Stop the robot and drive it back to START; pressed again while returning, just stop."""
        if self.drive and self.drive_home:
            self.cancel_drive()
        else:
            self.start_drive((0, 0), home=True)

    def toggle_go_to(self) -> None:
        """Arm point picking (the next map click sets the target); pressed again, cancel."""
        if self.picking or (self.drive and not self.drive_home):
            self.cancel_drive()
        elif isinstance(self.source, ReplaySource):
            self.console.append("can't send commands to this source")
        else:
            self.picking = True
            self.console.append("click a point on the map (Esc to cancel)")

    def pick_target(self, pos: tuple[int, int]) -> tuple[int, int]:
        """The visited cell a click at screen `pos` snaps to: routes can only end where the robot has been."""
        return nearest_visited(self.mapper.path, *self.to_world(*pos), self.maze_cell)

    def drain(self, limit: int = 1000) -> None:
        for _ in range(limit):
            try:
                self.handle_line(self.source.lines.get_nowait())
            except queue.Empty:
                break

    # -- coordinates

    def to_screen(self, x: float, y: float) -> tuple[int, int]:
        w, h = self.screen.get_size()
        return (int(w / 2 + (x - self.cam[0]) * self.zoom),
                int(h / 2 - (y - self.cam[1]) * self.zoom))

    def to_world(self, sx: float, sy: float) -> tuple[float, float]:
        w, h = self.screen.get_size()
        return self.cam[0] + (sx - w / 2) / self.zoom, self.cam[1] - (sy - h / 2) / self.zoom

    # -- drawing

    def _grid_surface(self) -> pygame.Surface:
        grid = self.mapper.grid
        key = (grid.version, self.show_lines)
        if key != self._map_key:
            wall = WALL_FAINT if self.show_lines else WALL
            a = np.clip(grid.logodds / 2.5, -1.0, 1.0)[..., None]
            rgb = np.where(a < 0, UNKNOWN + (FREE - UNKNOWN) * -a, UNKNOWN + (wall - UNKNOWN) * a)
            # rows are y-up; the screen is y-down, and surfarray wants [x, y]
            self._map_surface = pygame.surfarray.make_surface(
                np.ascontiguousarray(rgb[::-1].transpose(1, 0, 2)).astype(np.uint8))
            self._map_key = key
        return self._map_surface

    def draw_map(self) -> None:
        grid = self.mapper.grid
        surf = self._grid_surface()
        w, h = self.screen.get_size()
        (x0, y1), (x1, y0) = self.to_world(0, 0), self.to_world(w, h)
        c0 = max(0, int(math.floor(x0 / grid.cell)) + grid.origin)
        c1 = min(grid.n, int(math.ceil(x1 / grid.cell)) + grid.origin + 1)
        r0 = max(0, int(math.floor(y0 / grid.cell)) + grid.origin)
        r1 = min(grid.n, int(math.ceil(y1 / grid.cell)) + grid.origin + 1)
        if c1 <= c0 or r1 <= r0:
            return
        sub = surf.subsurface((c0, grid.n - r1, c1 - c0, r1 - r0))
        px = grid.cell * self.zoom
        scaled = pygame.transform.scale(sub, (max(1, round((c1 - c0) * px)), max(1, round((r1 - r0) * px))))
        self.screen.blit(scaled, self.to_screen((c0 - grid.origin) * grid.cell, (r1 - grid.origin) * grid.cell))

    def draw_walls(self) -> None:
        if not self.show_lines:
            return
        grid = self.mapper.grid
        if grid.version != self._walls_version:
            self._walls = lattice_walls(grid, self.maze_cell, self.mapper.path)
            self._walls_version = grid.version
        width = max(3, int(1.5 * self.zoom))
        for x1, y1, x2, y2 in self._walls:
            a, b = self.to_screen(x1, y1), self.to_screen(x2, y2)
            pygame.draw.line(self.screen, WALL_LINE, a, b, width)
            for end in (a, b):  # round ends so corners join cleanly
                pygame.draw.circle(self.screen, WALL_LINE, end, width // 2)

    def draw_route(self) -> None:
        if self.drive and self.drive.waypoints:
            pts = [self.to_screen(self.mapper.pose.x, self.mapper.pose.y)] + \
                  [self.to_screen(*p) for p in self.drive.waypoints]
            pygame.draw.lines(self.screen, ROUTE_COLOR, False, pts, 3)
            for p in pts[1:]:
                pygame.draw.circle(self.screen, ROUTE_COLOR, p, 4)
        if self.drive and not self.drive_home:
            self.draw_marker(*self.drive.goal, TARGET_COLOR, "TARGET")
        if self.picking:  # preview where a click would snap to
            gx, gy = self.pick_target(pygame.mouse.get_pos())
            self.draw_marker(gx * self.maze_cell, gy * self.maze_cell, TARGET_COLOR, "go here?")

    def _draw_button(self, label: str, color, top: int) -> pygame.Rect:
        w, _ = self.screen.get_size()
        text = self.font.render(label, True, (255, 255, 255))
        rect = pygame.Rect(0, 0, text.get_width() + 28, text.get_height() + 18)
        rect.topright = (w - 12, top)
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        self.screen.blit(text, text.get_rect(center=rect.center))
        return rect

    def draw_buttons(self) -> None:
        replay = isinstance(self.source, ReplaySource)
        if self.drive and self.drive_home:
            home = "CANCEL RETURN (H)", BUTTON_CANCEL
        elif replay:
            home = "RETURN HOME (replay: off)", BUTTON_OFF
        else:
            home = "STOP & RETURN HOME (H)", BUTTON_GO
        if self.picking:
            goto = "CLICK A POINT ON THE MAP (Esc)", BUTTON_CANCEL
        elif self.drive and not self.drive_home:
            goto = "CANCEL GO-TO (P)", BUTTON_CANCEL
        elif replay:
            goto = "GO TO POINT (replay: off)", BUTTON_OFF
        else:
            goto = "GO TO POINT (P)", BUTTON_GOTO
        self.home_button = self._draw_button(*home, 12)
        self.goto_button = self._draw_button(*goto, self.home_button.bottom + 8)

    def draw_marker(self, x: float, y: float, color, label: str) -> None:
        pos = self.to_screen(x, y)
        r = max(6, int(4 * self.zoom))
        pygame.draw.circle(self.screen, color, pos, r, 3)
        text = self.small.render(label, True, color)
        self.screen.blit(text, (pos[0] - text.get_width() // 2, pos[1] - r - text.get_height()))

    def draw_markers(self) -> None:
        self.draw_marker(0, 0, START_COLOR, "START")
        if self.mapper.exit_pos:
            self.draw_marker(*self.mapper.exit_pos, EXIT_COLOR, "EXIT")

    def draw_robot(self) -> None:
        m = self.mapper
        if len(m.path) > 1:
            pygame.draw.lines(self.screen, PATH_COLOR, False, [self.to_screen(*p) for p in m.path], 2)

        for name, ray in m.rays.items():
            color = SENSOR_COLORS[name]
            a, b = self.to_screen(*ray.start), self.to_screen(*ray.end)
            pygame.draw.line(self.screen, color, a, b, 1)
            if ray.hit:
                pygame.draw.circle(self.screen, color, b, 4)

        pose = m.pose
        r = max(4, int(m.cfg.wheelbase_cm / 2 * self.zoom))
        center = self.to_screen(pose.x, pose.y)
        pygame.draw.circle(self.screen, ROBOT_COLOR, center, r)
        pygame.draw.line(self.screen, (255, 255, 255), center,
                         self.to_screen(*pose.to_world(m.cfg.wheelbase_cm / 2, 0)), 3)

    def draw_truth(self) -> None:
        if not (self.sim and self.show_truth):
            return
        for x1, y1, x2, y2 in self.sim.walls:
            pygame.draw.line(self.screen, TRUTH_COLOR, self.to_screen(x1, y1), self.to_screen(x2, y2), 1)
        x1, y1, x2, y2 = self.sim.exit_gap
        pygame.draw.line(self.screen, EXIT_COLOR, self.to_screen(x1, y1), self.to_screen(x2, y2), 3)
        p = self.sim.pose
        pygame.draw.circle(self.screen, TRUTH_COLOR, self.to_screen(p.x, p.y), max(3, int(3 * self.zoom)), 2)

    def draw_scale_bar(self) -> None:
        w, h = self.screen.get_size()
        cm = next(c for c in (5, 10, 20, 50, 100, 200, 500, 1000) if c * self.zoom >= 60)
        x, y = w - 20 - cm * self.zoom, h - 30
        pygame.draw.line(self.screen, TEXT, (x, y), (w - 20, y), 3)
        label = self.small.render(f"{cm} cm", True, TEXT)
        self.screen.blit(label, (w - 20 - label.get_width(), y - 20))

    def draw_panel(self, lines: list[str], pos: tuple[int, int], font, anchor_bottom=False) -> None:
        if not lines:
            return
        rendered = [font.render(s, True, TEXT) for s in lines]
        width = max(r.get_width() for r in rendered) + 16
        height = sum(r.get_height() for r in rendered) + 12
        x, y = pos
        if anchor_bottom:
            y -= height
        panel = pygame.Surface((width, height), pygame.SRCALPHA)
        panel.fill((255, 255, 255, 215))
        self.screen.blit(panel, (x, y))
        y += 6
        for r in rendered:
            self.screen.blit(r, (x + 8, y))
            y += r.get_height()

    def draw_hud(self) -> None:
        m = self.mapper
        now = time.monotonic()
        recent = [t for t in self.rate_times if now - t < 2.0]
        rate = len(recent) / 2.0
        t = m.last

        def fmt(d):
            return "  ---" if d <= 0 else f"{d:5.1f}"

        lines = [self.source.status, f"{rate:4.1f} msg/s   samples {m.samples}"]
        if m.exit_pos:
            lines.append("EXIT REACHED - robot stopped")
        if self.drive:
            lines.append(("RETURNING TO START" if self.drive_home else "DRIVING TO TARGET") +
                         f" - {len(self.drive.waypoints)} waypoints left")
        if t:
            lines += [f"L {fmt(t.left)}  F {fmt(t.front)}  R {fmt(t.right)} cm",
                      f"motors L {t.motor_l:4d}  R {t.motor_r:4d}"]
        lines += [f"pose x {m.pose.x:6.1f}  y {m.pose.y:6.1f}  hdg {math.degrees(m.pose.theta) % 360:5.1f}",
                  f"path {self._path_length():.0f} cm"]
        self.draw_panel(lines, (10, 10), self.font)

        w, h = self.screen.get_size()
        help_text = "wheel zoom | drag pan | F follow" + (" (on)" if self.follow else "") + \
                    " | R reset | Q/E rotate | S save | L lines" + (" | G sim maze" if self.sim else "") + \
                    " | H home | P go to | Esc quit"
        self.draw_panel(list(self.console) + [help_text], (10, h - 10), self.small, anchor_bottom=True)

    def _path_length(self) -> float:
        p = np.array(self.mapper.path)
        return float(np.hypot(*np.diff(p, axis=0).T).sum()) if len(p) > 1 else 0.0

    def draw(self) -> None:
        if self.follow:
            self.cam = [self.mapper.pose.x, self.mapper.pose.y]
        self.screen.fill(tuple(int(c) for c in UNKNOWN))
        self.draw_map()
        self.draw_walls()
        self.draw_truth()
        self.draw_markers()
        self.draw_route()
        self.draw_robot()
        self.draw_scale_bar()
        self.draw_hud()
        self.draw_buttons()

    # -- input

    def save_screenshot(self) -> None:
        name = datetime.now().strftime("maze_%Y%m%d_%H%M%S.png")
        pygame.image.save(self.screen, name)
        self.console.append(f"saved {name}")

    def handle_event(self, e) -> bool:
        if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE and self.picking:
            self.picking = False
            self.console.append("go-to cancelled")
            return True
        if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE):
            return False
        if e.type == pygame.MOUSEWHEEL:
            before = self.to_world(*pygame.mouse.get_pos())
            self.zoom = min(40.0, max(0.3, self.zoom * (1.15 ** e.y)))
            if not self.follow:  # zoom around the cursor
                after = self.to_world(*pygame.mouse.get_pos())
                self.cam[0] += before[0] - after[0]
                self.cam[1] += before[1] - after[1]
        elif e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
            if self.home_button.collidepoint(e.pos):
                self.toggle_return_home()
            elif self.goto_button.collidepoint(e.pos):
                self.toggle_go_to()
            elif self.picking:
                self.start_drive(self.pick_target(e.pos), home=False)
            else:
                self.dragging = True
        elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
            self.dragging = False
        elif e.type == pygame.MOUSEMOTION and self.dragging:
            self.follow = False
            self.cam[0] -= e.rel[0] / self.zoom
            self.cam[1] += e.rel[1] / self.zoom
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_f:
                self.follow = not self.follow
            elif e.key == pygame.K_r:
                self.cancel_drive()  # the route is meaningless on a fresh map
                self.mapper.reset()
                self.console.append("map reset")
            elif e.key == pygame.K_h:
                self.toggle_return_home()
            elif e.key == pygame.K_p:
                self.toggle_go_to()
            elif e.key == pygame.K_l:
                self.show_lines = not self.show_lines
            elif e.key == pygame.K_q:
                self.mapper.rotate(5)
            elif e.key == pygame.K_e:
                self.mapper.rotate(-5)
            elif e.key == pygame.K_s:
                self.save_screenshot()
            elif e.key == pygame.K_g:
                self.show_truth = not self.show_truth
        return True

    def run(self) -> None:
        self.source.start()
        running = True
        try:
            while running:
                for e in pygame.event.get():
                    running = self.handle_event(e) and running
                self.drain()
                self.draw()
                pygame.display.flip()
                self.clock.tick(60)
        finally:
            self.source.stop()
            if self.log:
                self.log.close()
            pygame.quit()


# ---------------------------------------------------------------- CLI

def main() -> None:
    ap = argparse.ArgumentParser(description="Map a maze live from the Arduino robot's ultrasonic sensors.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--port", help="serial port, e.g. COM5 (default: auto-detect)")
    src.add_argument("--sim", action="store_true", help="use the built-in maze simulator")
    src.add_argument("--replay", metavar="LOG", help="replay a log recorded with --log")
    ap.add_argument("--baud", type=int, default=115200)
    ap.add_argument("--log", metavar="FILE", help="append every received line to FILE")
    ap.add_argument("--speed", type=float, default=20.0, help="robot ground speed at full power, cm/s (a rough value is fine: the sonars correct it)")
    ap.add_argument("--wheelbase", type=float, default=9.0, help="distance between wheels, cm")
    ap.add_argument("--side-angle", type=float, default=30.0,
                    help="how far the left/right sensors point out from straight ahead, degrees")
    ap.add_argument("--map-size", type=float, default=1000.0, help="map width/height, cm")
    ap.add_argument("--cell", type=float, default=2.0, help="map resolution, cm per cell")
    ap.add_argument("--playback-speed", type=float, default=1.0, help="time multiplier for --sim/--replay")
    ap.add_argument("--sim-seed", type=int, default=None)
    ap.add_argument("--sim-size", type=int, nargs=2, default=(6, 6), metavar=("COLS", "ROWS"))
    ap.add_argument("--maze-cell", type=float, default=35.0,
                    help="maze corridor width, cm: wall lines and return-home snap to this grid")
    ap.add_argument("--sim-cell", type=float, default=None, help="sim corridor width, cm (default: --maze-cell)")
    ap.add_argument("--sim-drift", type=float, default=0.0,
                    help="odometry error in the sim, e.g. 0.05 = robot really 5%% faster than assumed")
    args = ap.parse_args()

    cfg = RobotConfig(speed_cm_s=args.speed, wheelbase_cm=args.wheelbase,
                      sensors=sensor_mounts(args.side_angle))
    mapper = MazeMapper(cfg, args.map_size, args.cell, maze_cell_cm=args.maze_cell)

    sim = None
    if args.sim:
        from simulator import Simulator
        sim = Simulator(cfg, *args.sim_size, cell_cm=args.sim_cell or args.maze_cell,
                        seed=args.sim_seed, drift=args.sim_drift)
        source = SimSource(sim, args.playback_speed)
    elif args.replay:
        source = ReplaySource(args.replay, args.playback_speed)
    else:
        port = args.port or find_arduino_port()
        if port is None:
            from serial.tools import list_ports
            print("No Arduino found. Available ports:")
            for p in list_ports.comports():
                print(f"  {p.device}: {p.description}")
            sys.exit("Pass one with --port, or use --sim to try the simulator.")
        source = SerialSource(port, args.baud)

    App(source, mapper, log_path=args.log, sim=sim, maze_cell_cm=args.maze_cell).run()


if __name__ == "__main__":
    main()
