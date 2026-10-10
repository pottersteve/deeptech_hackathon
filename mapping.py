"""Telemetry parsing, pose tracking (motors predict, sonars correct) and occupancy-grid mapping for the maze robot.

Telemetry line format (one per Arduino loop, newline terminated):

    S,<millis>,<left_cm>,<front_cm>,<right_cm>,<motor_left>,<motor_right>

Distances are in cm (already averaged over the last few pings by the robot); <= 0 means "no echo". Motor values are -255..255, the
power each wheel was just set to (255 = full forward, 0 = stopped). Any other
line is treated as debug text; a line starting with "EXIT" means the robot has left the maze.

World frame: centimetres, x to the right, y up, robot starts at (0, 0) facing +y.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SENSOR_NAMES = ("left", "front", "right")


@dataclass
class Telemetry:
    t_ms: int
    left: float
    front: float
    right: float
    motor_l: int
    motor_r: int


def parse_line(line: str) -> Telemetry | None:
    parts = line.strip().split(",")
    if len(parts) != 7 or parts[0] != "S":
        return None
    try:
        return Telemetry(int(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]),
                         int(parts[5]), int(parts[6]))
    except ValueError:
        return None


def format_line(t: Telemetry) -> str:
    return f"S,{t.t_ms},{t.left:.1f},{t.front:.1f},{t.right:.1f},{t.motor_l},{t.motor_r}"


def sensor_mounts(side_angle_deg: float = 30.0) -> dict[str, tuple[float, float, float]]:
    """Sensor mounts relative to the robot centre: (forward cm, left cm, heading offset deg).

    One sensor points straight ahead; the other two point `side_angle_deg` out to the left and
    right of it.
    """
    return {
        "left": (5.0, 4.0, side_angle_deg),
        "front": (6.0, 0.0, 0.0),
        "right": (5.0, -4.0, -side_angle_deg),
    }


@dataclass
class RobotConfig:
    speed_cm_s: float = 20.0       # ground speed of a wheel at motor power 255 (rough is fine)
    wheelbase_cm: float = 9.0      # distance between the two wheels' contact points
    max_range_cm: float = 200.0    # readings at/above this are treated as "no echo"
    free_range_cm: float = 80.0    # don't mark space as free further out than this
    beam_half_angle_deg: float = 10.0  # map drawing: how wide each echo is smeared
    sonar_cone_deg: float = 8.0        # half-angle within which a sonar reports the nearest echo
    sensors: dict[str, tuple[float, float, float]] = field(default_factory=sensor_mounts)


@dataclass
class Pose:
    x: float = 0.0
    y: float = 0.0
    theta: float = math.pi / 2  # radians, 0 = +x

    def to_world(self, fwd: float, left: float) -> tuple[float, float]:
        """Robot-frame point (forward, left) -> world coordinates."""
        c, s = math.cos(self.theta), math.sin(self.theta)
        return self.x + fwd * c - left * s, self.y + fwd * s + left * c


def advance(pose: Pose, motor_l: int, motor_r: int, dt: float,
            speed_cm_s: float, wheelbase_cm: float) -> Pose:
    """Differential-drive kinematics: move `pose` for `dt` seconds at the given motor powers."""
    vl = speed_cm_s * motor_l / 255.0
    vr = speed_cm_s * motor_r / 255.0
    v = (vl + vr) / 2.0
    w = (vr - vl) / wheelbase_cm
    th = pose.theta
    if abs(w) < 1e-9:
        return Pose(pose.x + v * dt * math.cos(th), pose.y + v * dt * math.sin(th), th)
    th2 = th + w * dt
    r = v / w
    return Pose(pose.x + r * (math.sin(th2) - math.sin(th)),
                pose.y - r * (math.cos(th2) - math.cos(th)),
                th2)


class OccupancyGrid:
    """Log-odds occupancy grid centred on the start position. Indexed [row=y, col=x]."""

    L_FREE = -0.35
    L_OCC = 0.9
    L_CLAMP = 5.0

    def __init__(self, size_cm: float = 1000.0, cell_cm: float = 2.0):
        self.cell = cell_cm
        self.n = int(round(size_cm / cell_cm))
        self.origin = self.n // 2  # cell index of world 0
        self.logodds = np.zeros((self.n, self.n), dtype=np.float32)
        self.version = 0

    def clear(self) -> None:
        self.logodds.fill(0.0)
        self.version += 1

    def sample(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """Log-odds at world points (0 = unknown outside the map)."""
        xs, ys = np.broadcast_arrays(xs, ys)
        cols = np.floor(xs / self.cell).astype(np.int64) + self.origin
        rows = np.floor(ys / self.cell).astype(np.int64) + self.origin
        ok = (cols >= 0) & (cols < self.n) & (rows >= 0) & (rows < self.n)
        out = np.zeros(xs.shape, dtype=np.float32)
        out[ok] = self.logodds[rows[ok], cols[ok]]
        return out

    def _cells(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        cols = np.floor(xs / self.cell).astype(np.int64).ravel() + self.origin
        rows = np.floor(ys / self.cell).astype(np.int64).ravel() + self.origin
        ok = (cols >= 0) & (cols < self.n) & (rows >= 0) & (rows < self.n)
        return np.unique(rows[ok] * self.n + cols[ok])

    def integrate_reading(self, sx: float, sy: float, heading: float, dist: float,
                          cfg: RobotConfig) -> None:
        """Ultrasonic inverse sensor model: the cone up to the echo is free, the arc at the echo is occupied."""
        hit = 0 < dist < cfg.max_range_cm
        reach = dist - 1.5 * self.cell if hit else cfg.max_range_cm
        free_len = min(reach, cfg.free_range_cm)
        half = math.radians(cfg.beam_half_angle_deg)

        free = np.empty(0, dtype=np.int64)
        if free_len > 0:
            angles = heading + np.linspace(-half, half, 9)[:, None]
            ts = np.arange(0.0, free_len, self.cell * 0.5)[None, :]
            free = self._cells(sx + np.cos(angles) * ts, sy + np.sin(angles) * ts)

        occ = np.empty(0, dtype=np.int64)
        if hit:
            n_arc = max(3, int(dist * 2 * half / (self.cell * 0.5)) + 1)
            arc = heading + np.linspace(-half, half, n_arc)
            occ = self._cells(sx + dist * np.cos(arc), sy + dist * np.sin(arc))
            free = np.setdiff1d(free, occ, assume_unique=True)

        flat = self.logodds.reshape(-1)
        flat[free] = np.maximum(flat[free] + self.L_FREE, -self.L_CLAMP)
        flat[occ] = np.minimum(flat[occ] + self.L_OCC, self.L_CLAMP)
        self.version += 1


def lattice_walls(grid: OccupancyGrid, cell_cm: float, path: list[tuple[float, float]],
                  margin_cells: int = 2, threshold: float = 1.5) -> list[tuple[float, float, float, float]]:
    """Snap the occupancy grid onto the maze lattice: every cell edge with wall evidence along it.

    Maze cell (0, 0) is centred on the start, so edges lie half a cell either side of multiples of
    cell_cm. Only edges near the driven path are checked. An edge counts as a wall when at least
    half of the points sampled along it have an occupied cell within a few cm.
    Returns wall segments as (x1, y1, x2, y2).
    """
    p = np.asarray(path, dtype=float)
    lo = np.floor(p.min(axis=0) / cell_cm).astype(int) - margin_cells
    hi = np.ceil(p.max(axis=0) / cell_cm).astype(int) + margin_cells
    cx, cy = np.meshgrid(np.arange(lo[0], hi[0] + 1) * cell_cm,
                         np.arange(lo[1], hi[1] + 1) * cell_cm, indexing="ij")
    cx, cy = cx.ravel(), cy.ravel()
    half = cell_cm / 2
    # Skip the ends of each edge: sonar arcs from the neighbouring walls smear into the corners.
    along = (np.linspace(-0.35, 0.35, 9) * cell_cm)[None, :, None]
    across = np.linspace(-3.0, 3.0, 5)[None, None, :]

    segs = []
    # midpoint of each cell's bottom edge (horizontal) and left edge (vertical)
    for mx, my, horizontal in ((cx, cy - half, True), (cx - half, cy, False)):
        mx3, my3 = mx[:, None, None], my[:, None, None]
        xs, ys = (mx3 + along, my3 + across) if horizontal else (mx3 + across, my3 + along)
        wall = (grid.sample(xs, ys).max(axis=2) > threshold).mean(axis=1) >= 0.5
        for x, y in zip(mx[wall], my[wall]):
            segs.append((x - half, y, x + half, y) if horizontal else (x, y - half, x, y + half))
    return segs


def lattice_offset(v: float, cell_cm: float) -> float:
    """Signed distance from coordinate v to the nearest maze wall line (walls sit at (k + 1/2) * cell)."""
    u = v - cell_cm / 2
    return u - cell_cm * round(u / cell_cm)


def step_pose(p: Pose, ds: float, dth: float, dlat: float = 0.0) -> Pose:
    """Move `ds` forward while turning `dth`, plus an optional sideways shift `dlat`."""
    mid = p.theta + dth / 2
    c, s = math.cos(mid), math.sin(mid)
    return Pose(p.x + ds * c - dlat * s, p.y + ds * s + dlat * c, p.theta + dth)


def beam_range(p: Pose, mount: tuple[float, float, float], axis: int, line: float,
               half_angle: float) -> float | None:
    """What a sonar reads off the wall line x = line (axis 0) or y = line (axis 1).

    A sonar returns the nearest echo anywhere in its cone, so: the perpendicular distance if the
    line's normal is inside the cone, else the distance along the cone edge nearest the normal.
    """
    sx, sy = p.to_world(mount[0], mount[1])
    a = p.theta + math.radians(mount[2])
    gap = (line - sx) if axis == 0 else (line - sy)          # signed distance to the line
    normal = (0.0 if gap > 0 else math.pi) if axis == 0 else (math.pi / 2 if gap > 0 else -math.pi / 2)
    off = abs((a - normal + math.pi) % (2 * math.pi) - math.pi)  # beam vs the way to the line
    off = max(0.0, off - half_angle)
    if off > math.radians(75):  # facing away from / along the line
        return None
    return abs(gap) / math.cos(off)


class PoseTracker:
    """Motors predict, sonars correct.

    Each sample the pose is first moved by the motor powers (`advance`), which is only a rough
    guess: wheels slip, speed is uncalibrated, and the robot may be pushing against a wall.
    Then it's corrected so the sonar echoes land on maze wall lines (every wall lies on a
    lattice line at (k + 1/2) * cell). A few headings around the motor guess are tried, each
    echo is paired with the line nearest to it (dropped if it doesn't fit), the heading most
    echoes agree with wins, and Gauss-Newton fine-tunes (forward, sideways, turn). Readings are means of the last 3 pings, so each is compared with the
    mean of the ranges predicted from the last 3 poses.
    """

    GATE_CM = 7.0      # an echo further than this from every wall line is ignored
    FIT_GATE_CM = 3.0  # ...or further than this from what the motor guess predicts
    JUMP_CM = 4.0      # a reading that jumped this much since the last one probably averages two walls
    # How hard the correction (forward cm, sideways cm, turn rad) is held back per sample:
    # roughly 1 / (how wrong the motor guess can be)^2.
    PRIOR = np.diag([2.0, 4.0, 3000.0])
    AVG_N = 3          # the robot averages this many pings per reading
    HEADING_GUESSES_DEG = (-3, -1.5, 0, 1.5, 3)  # around the motor guess, each sample
    HEADING_GUESS_COST = 300.0  # tie-breaker towards the motor guess (per rad^2)
    FROZEN_CM = 0.6    # readings that all changed less than this since the last sample are "frozen"
    STAY_COST = 20.0   # how much better the echoes must fit to overrule the frozen/not-frozen call

    def __init__(self, cfg: RobotConfig, cell_cm: float):
        self.cfg = cfg
        self.cell = cell_cm
        self.half_angle = math.radians(cfg.sonar_cone_deg)
        self.correct = True    # False: plain motor odometry (for comparison)
        self.reset(Pose())

    def reset(self, pose: Pose) -> None:
        self.history = [pose]  # recent poses, newest last
        self.matched = 0       # echoes used for the last correction
        self.previous: dict[str, float] = {}

    def _pairs(self, guess: Pose, readings: dict[str, float]):
        pairs = []
        for name, mount in self.cfg.sensors.items():
            d = readings[name]
            if not 0 < d < self.cfg.max_range_cm or abs(d - self.previous.get(name, d)) > self.JUMP_CM:
                continue
            sx, sy = guess.to_world(mount[0], mount[1])
            a = guess.theta + math.radians(mount[2])
            px, py = sx + d * math.cos(a), sy + d * math.sin(a)
            rx, ry = lattice_offset(px, self.cell), lattice_offset(py, self.cell)
            axis, r = (0, rx) if abs(rx) < abs(ry) else (1, ry)
            if abs(r) < self.GATE_CM:
                pairs.append((mount, d, axis, (px if axis == 0 else py) - r))
        return pairs

    def _residuals(self, base: Pose, q: np.ndarray, pairs, history: list[Pose]) -> np.ndarray:
        # the reading averages pings taken at this pose and the previous AVG_N - 1 poses
        poses = [step_pose(base, q[0], q[2], q[1])] + history[::-1][:self.AVG_N - 1]
        out = []
        for mount, d, axis, line in pairs:
            ranges = [rg for rg in (beam_range(p, mount, axis, line, self.half_angle) for p in poses)
                      if rg is not None]
            out.append(d - sum(ranges) / len(ranges) if ranges else 0.0)
        return np.array(out)

    def update(self, readings: dict[str, float], motor_l: int, motor_r: int, dt: float) -> Pose:
        """`motor_*`: powers applied since the last sample, `dt`: seconds since it."""
        predicted = advance(self.history[-1], motor_l, motor_r, dt,
                            self.cfg.speed_cm_s, self.cfg.wheelbase_cm)
        usable = sum(0 < d < self.cfg.max_range_cm and abs(d - self.previous.get(n, d)) <= self.JUMP_CM
                     for n, d in readings.items())
        # Stuck? The motors say the robot moves, but the readings haven't changed: it's pushing
        # against a wall. (While driving straight the side readings can hold steady, so that only
        # counts if the front reading holds too, or the motors say it's turning.)
        valid = [n for n, d in readings.items() if 0 < d < self.cfg.max_range_cm]
        frozen = bool(self.previous) and (motor_l or motor_r) and \
            all(abs(readings[n] - self.previous.get(n, -1)) < self.FROZEN_CM for n in valid) and \
            ("front" in valid or motor_l != motor_r)
        here = self.history[-1]
        moves = [(predicted, self.STAY_COST if frozen else 0.0)]
        if frozen and self.correct:
            moves.append((here, 0.0))
        # Heading is what the motor guess gets most wrong (every turn adds a few degrees), so try
        # a few headings and keep the one most echoes agree with. Position stays with the motors.
        best = None
        for start, move_cost in moves:
            for off_deg in (self.HEADING_GUESSES_DEG if self.correct else (0,)):
                off = math.radians(off_deg)
                base = Pose(start.x, start.y, start.theta + off)
                history = [Pose(p.x, p.y, p.theta + off) for p in self.history]
                pairs = self._pairs(base, readings) if self.correct else []
                if pairs:  # keep only echoes this guess already roughly agrees with
                    fits = np.abs(self._residuals(base, np.zeros(3), pairs, history)) < self.FIT_GATE_CM
                    pairs = [pair for pair, ok in zip(pairs, fits) if ok]
                r = self._residuals(base, np.zeros(3), pairs, history) if pairs else np.zeros(0)
                score = (float(r @ r) + (usable - len(pairs)) * self.FIT_GATE_CM ** 2
                         + self.HEADING_GUESS_COST * off * off + move_cost)
                if best is None or score < best[0]:
                    best = (score, base, history, pairs)
        _, base, history, pairs = best
        self.previous = dict(readings)
        self.matched = len(pairs)

        q = np.zeros(3)  # fine correction: forward cm, sideways cm, turn rad
        for _ in range(3 if pairs else 0):
            r = self._residuals(base, q, pairs, history)
            J = np.empty((len(pairs), 3))
            for k, h in enumerate((0.1, 0.1, 0.002)):  # numeric Jacobian
                dq = np.zeros(3)
                dq[k] = h
                J[:, k] = (self._residuals(base, q + dq, pairs, history) - r) / h
            w = np.where(np.abs(r) < 2.0, 1.0, 2.0 / np.maximum(np.abs(r), 1e-9))  # Huber
            H = J.T @ (w[:, None] * J) + self.PRIOR
            q = q - np.linalg.solve(H, J.T @ (w * r) + self.PRIOR @ q)
        new = step_pose(base, q[0], q[2], q[1])
        self.history = (history + [new])[-self.AVG_N:]
        return new



@dataclass
class Ray:
    start: tuple[float, float]
    end: tuple[float, float]
    hit: bool


class MazeMapper:
    """Turns a stream of Telemetry into a pose estimate, a driven path and an occupancy grid.

    The motor values give a rough guess of each move; the sonars correct it (PoseTracker).
    """

    def __init__(self, cfg: RobotConfig, map_size_cm: float = 1000.0, cell_cm: float = 2.0,
                 maze_cell_cm: float = 35.0):
        self.cfg = cfg
        self.maze_cell = maze_cell_cm
        self.grid = OccupancyGrid(map_size_cm, cell_cm)
        self.tracker = PoseTracker(cfg, maze_cell_cm)
        self.reset()

    def reset(self) -> None:
        self.grid.clear()
        self.pose = Pose()
        self.tracker.reset(self.pose)
        self.path: list[tuple[float, float]] = [(0.0, 0.0)]
        self.rays: dict[str, Ray] = {}
        self.last: Telemetry | None = None
        self.samples = 0
        self.exit_pos: tuple[float, float] | None = None  # where the robot reported leaving the maze

    @property
    def matched(self) -> int:
        """Echoes that fixed the last pose."""
        return self.tracker.matched

    def update(self, t: Telemetry) -> None:
        prev = self.last
        self.last = t
        self.samples += 1
        dt = (t.t_ms - prev.t_ms) / 1000.0 if prev is not None else 0.0
        # dt <= 0: board was reset; huge dt: we lost data. Either way don't guess the move.
        motors = (prev.motor_l, prev.motor_r) if prev is not None and 0 < dt < 1.0 else (0, 0)
        self.pose = self.tracker.update({name: getattr(t, name) for name in SENSOR_NAMES},
                                        *motors, max(dt, 0.0))

        px, py = self.path[-1]
        if math.hypot(self.pose.x - px, self.pose.y - py) > 0.5:
            self.path.append((self.pose.x, self.pose.y))

        for name, (fwd, left, offset_deg) in self.cfg.sensors.items():
            dist = getattr(t, name)
            sx, sy = self.pose.to_world(fwd, left)
            heading = self.pose.theta + math.radians(offset_deg)
            self.grid.integrate_reading(sx, sy, heading, dist, self.cfg)
            hit = 0 < dist < self.cfg.max_range_cm
            length = dist if hit else min(self.cfg.free_range_cm, self.cfg.max_range_cm)
            self.rays[name] = Ray((sx, sy), (sx + length * math.cos(heading),
                                             sy + length * math.sin(heading)), hit)

    def mark_exit(self) -> None:
        self.exit_pos = (self.pose.x, self.pose.y)

    def rotate(self, degrees: float) -> None:
        """Manual heading correction for when the pose estimate has drifted."""
        self.pose = Pose(self.pose.x, self.pose.y, self.pose.theta + math.radians(degrees))
        self.tracker.reset(self.pose)
