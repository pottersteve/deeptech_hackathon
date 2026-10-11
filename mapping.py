"""Telemetry parsing, pose tracking (motors predict, sonars correct) and occupancy-grid mapping for the maze robot.

Telemetry line format (one per Arduino loop, newline terminated):

    S,<millis>,<left_cm>,<front_cm>,<right_cm>,<motor_left>,<motor_right>

Distances are in cm (already averaged over the last few pings by the robot); <= 0 means "no echo". Motor values are -255..255, the
power each wheel was just set to (255 = full forward, 0 = stopped). A line

    T,<millis>,<way>,<ms>

means the robot spun on the spot for <ms> ms (way 1 = left, -1 = right) without logging. Any other
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


def parse_turn(line: str) -> tuple[int, int, int] | None:
    """A T line -> (millis, way, ms): the robot spun left (way 1) or right (-1) for ms milliseconds."""
    parts = line.strip().split(",")
    if len(parts) != 4 or parts[0] != "T":
        return None
    try:
        return int(parts[1]), int(parts[2]), int(parts[3])
    except ValueError:
        return None


def format_line(t: Telemetry) -> str:
    return f"S,{t.t_ms},{t.left:.1f},{t.front:.1f},{t.right:.1f},{t.motor_l},{t.motor_r}"


def sensor_mounts(side_angle_deg: float = 30.0, front_fwd_cm: float = 12.0, side_fwd_cm: float = 10.0,
                  side_spacing_cm: float = 10.0) -> dict[str, tuple[float, float, float]]:
    """Sensor mounts relative to the midpoint between the wheels: (forward cm, left cm, heading offset deg).

    One sensor points straight ahead, `front_fwd_cm` ahead of the wheels; the other two sit
    `side_fwd_cm` ahead of the wheels, `side_spacing_cm` apart, pointing `side_angle_deg` out.
    Defaults are the measured robot: 16 x 12 cm body, wheels 12 cm behind the front, front sensor
    at the very front, side sensors 2 cm back from it and 10 cm apart.
    """
    return {
        "left": (side_fwd_cm, side_spacing_cm / 2, side_angle_deg),
        "front": (front_fwd_cm, 0.0, 0.0),
        "right": (side_fwd_cm, -side_spacing_cm / 2, -side_angle_deg),
    }


@dataclass
class RobotConfig:
    speed_cm_s: float = 20.0       # ground speed of a wheel at motor power 255 (rough is fine)
    wheelbase_cm: float = 9.0      # distance between the two wheels' contact points
    length_cm: float = 16.0        # body size, for drawing (the sensor positions come from sensor_mounts)
    width_cm: float = 12.0
    wheels_from_front_cm: float = 12.0  # midpoint between the wheels (the pose point), measured back from the front
    max_range_cm: float = 100.0    # readings at/above this are treated as "no echo" (longer ones leak out of the maze)
    free_range_cm: float = 40.0    # don't mark space as free further out than this
    beam_half_angle_deg: float = 10.0  # map drawing: how wide each echo is smeared
    sonar_cone_deg: float = 8.0        # half-angle within which a sonar reports the nearest echo
    spin_deg_s: float = 180.0          # how fast the robot turns when it spins on the spot (T lines)
    # How tunnel.ino turns (keep in step with it): a wall closer than front_turn_cm ahead -> stop and
    # spin, without logging, towards the first side reading over open_cm (left first), else turn
    # around. The line it logs straight after has that front reading, motors 0,0 and a time gap.
    front_turn_cm: float = 15.0        # FRONT_TURN
    open_cm: float = 25.0              # a side further than this (or no echo) is open
    turn_gap_ms: float = 400.0         # a gap this long before such a line = the robot spun 90 deg
    uturn_gap_ms: float = 900.0        # ...this long = it turned around (two spins' worth)
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

    # Sonar-only mode: the motor values are ignored completely. Forward speed comes from how fast
    # the wall ahead approaches. Heading: small wobbles from how fast the side walls approach as we
    # drive along them (closer by sin(a) cm per cm driven = angled a towards that wall); corners
    # from the wall ahead falling away after we got close to it (we turned towards the roomier side).
    SONAR_PRIOR = np.diag([0.3, 1.0, 1e6])  # the lattice fit may nudge position, not heading
    SONAR_MAX_STEP_CM = 5.0  # a front reading that shrank more than this in one sample hit a new wall
    SLOPE_SAMPLES = 6        # side readings fitted for the angle to a wall
    SLOPE_MIN_TRAVEL_CM = 3.0  # ...over at least this much driving
    SLOPE_MAX_SIN = 0.35     # steeper than ~20 deg we're turning, not driving along the wall
    HEADING_GAIN = 0.5       # how fast the heading follows the wall angle, per sample
    CORNER_TURN_DEG = 90.0   # a corner turns this much (maze walls meet at right angles)
    CORNER_LEAN_CM = 3.0     # one side must average this much more room, else it's a dead end

    def __init__(self, cfg: RobotConfig, cell_cm: float):
        self.cfg = cfg
        self.cell = cell_cm
        self.half_angle = math.radians(cfg.sonar_cone_deg)
        self.correct = True    # False: plain motor odometry (for comparison)
        self.use_motors = True  # False: ignore the motors, track the pose from the sonars alone
        self.turns_known = False  # the robot's spins are known (logged or seen in the log): don't guess corners
        self.reset(Pose())

    def reset(self, pose: Pose) -> None:
        self.history = [pose]  # recent poses, newest last
        self.matched = 0       # echoes used for the last correction
        self.previous: dict[str, float] = {}
        self.sonar_speed = 0.0  # cm per sample, from how fast the front wall approaches
        self.driven = 0.0       # sonar-only: total cm driven
        self.side_log: list[tuple[float, dict[str, float]]] = []  # (driven, readings), newest last
        self.wall_dir: float | None = None  # direction of the side wall(s) we're driving along
        self.corner = False     # sonar-only: the wall ahead is close, so we're at a corner
        self.corner_front = 0.0  # the closest front reading at this corner
        self.corner_lean = 0.0  # sum of (left - right) room at this corner: > 0 = more room left
        self.corner_samples = 0
        self.corner_gone = 0    # readings in a row with the wall ahead gone
        self.clear_ahead = 0    # readings in a row with no wall within a cell ahead
        self.clear_ahead_seen = True  # False after a corner, until the way ahead has been clear
        self.side_room: dict[str, float] = {}  # last good reading per side (missing echo = keep it)

    def _sonar_guess(self, readings: dict[str, float]) -> None:
        """Update the speed from the front sonar alone: the wall ahead got closer by as much as we drove."""
        now, before = readings["front"], self.previous.get("front", -1.0)
        if 0 < now < self.cfg.max_range_cm and 0 < before < self.cfg.max_range_cm \
                and -1.0 < before - now < self.SONAR_MAX_STEP_CM:
            self.sonar_speed = 0.7 * self.sonar_speed + 0.3 * max(before - now, 0.0)

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
        if not self.use_motors:
            return self._update_sonar(readings)
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
        return self._fit(readings, moves, self.HEADING_GUESSES_DEG, self.PRIOR, usable)

    def _wall_angle(self) -> float | None:
        """Our heading relative to the side wall(s), from how the side readings changed as we drove."""
        angles = []
        for name, (_, _, offset_deg) in self.cfg.sensors.items():
            if abs(offset_deg) < 60:
                continue  # only sensors pointing out sideways see a wall we drive along
            run = []  # newest readings back to the first missing echo or jump (a different wall)
            for driven, readings in reversed(self.side_log):
                d = readings[name]
                if not 0 < d < self.cfg.max_range_cm or (run and abs(d - run[-1][1]) > self.JUMP_CM):
                    break
                run.append((driven, d))
            if len(run) < 4 or run[0][0] - run[-1][0] < self.SLOPE_MIN_TRAVEL_CM:
                continue
            s, d = np.array(run).T
            slope = np.polyfit(s, d * math.sin(math.radians(abs(offset_deg))), 1)[0]
            if abs(slope) > self.SLOPE_MAX_SIN:
                continue
            # left wall getting closer = angled left (+); right wall getting closer = angled right
            angles.append(math.asin(-slope if offset_deg > 0 else slope))
        return sum(angles) / len(angles) if angles else None

    def spin(self, pose: Pose) -> None:
        """The robot spun on the spot to `pose`: readings from before the spin no longer fit."""
        self.turns_known = True
        self.history = [pose]
        self.previous = {}
        self.side_log = []
        self.wall_dir = None
        self.corner = False

    def _update_sonar(self, readings: dict[str, float]) -> Pose:
        here = self.history[-1]
        usable = sum(0 < d < self.cfg.max_range_cm and abs(d - self.previous.get(n, d)) <= self.JUMP_CM
                     for n, d in readings.items())
        for name, (_, _, offset_deg) in self.cfg.sensors.items():
            if offset_deg == 0:
                continue
            side = "left" if offset_deg > 0 else "right"  # where it's mounted (see --swap-sides)
            d = readings[name]
            self.side_room[side] = d if 0 < d < self.cfg.max_range_cm else \
                self.cfg.max_range_cm if d > 0 else self.side_room.get(side, 0.0)
        front = readings["front"]
        ahead = 0 < front < self.cfg.max_range_cm
        self.clear_ahead = 0 if ahead and front < self.cell else self.clear_ahead + 1
        self.clear_ahead_seen = self.clear_ahead_seen or self.clear_ahead >= 2
        # after a corner, wait for the new corridor to open up ahead: mid-turn the front sonar can
        # miss a couple of echoes and see the old wall again
        before = self.previous.get("front", -1.0)
        approached = 0 < before < self.cfg.max_range_cm and -1.0 < before - front < self.SONAR_MAX_STEP_CM
        # a corner starts when we drive up to the wall ahead (one that pops up from nowhere is a stray echo)
        if not self.corner and ahead and front < self.cell and approached and self.clear_ahead_seen                 and not self.turns_known:
            self.corner, self.corner_front, self.corner_lean, self.corner_gone = True, front, 0.0, 0
            self.corner_samples = 0
        if self.corner:
            self.corner_lean += self.side_room["left"] - self.side_room["right"]
            self.corner_samples += 1
            self.corner_gone = self.corner_gone + 1 if not ahead or front > self.corner_front + self.cell / 2 else 0
            if ahead and front < self.corner_front:
                self.corner_front = front
            lean = self.corner_lean / self.corner_samples
            if self.corner_gone >= 2 and abs(lean) < self.CORNER_LEAN_CM:
                self.corner = False  # a dead end where the wall ahead missed a couple of echoes
            elif self.corner_gone >= 2:
                # The wall ahead fell away: we turned the corner, towards the side with more room.
                # The side walls only show how far we've turned, not which way, so the corner is
                # taken as a whole right angle.
                side = 1 if lean > 0 else -1
                self.wall_dir = round(here.theta / (math.pi / 2)) * (math.pi / 2) \
                    + side * math.radians(self.CORNER_TURN_DEG)
                here = Pose(here.x, here.y, self.wall_dir)
                self.history = [here]
                self.side_log = []
                self.previous = {}
                self.corner = False
                self.clear_ahead_seen = False

        self._sonar_guess(readings)
        self.driven += self.sonar_speed
        self.side_log = (self.side_log + [(self.driven, dict(readings))])[-self.SLOPE_SAMPLES:]
        turn = 0.0
        angle = None if self.corner else self._wall_angle()  # mid-corner the side readings mislead
        if angle is None:
            self.wall_dir = None
        else:
            if self.wall_dir is None:  # a new wall: which way does it run?
                self.wall_dir = here.theta - angle
                if self.correct:  # maze walls run along the grid
                    self.wall_dir = round(self.wall_dir / (math.pi / 2)) * (math.pi / 2)
            off = (self.wall_dir + angle - here.theta + math.pi) % (2 * math.pi) - math.pi
            turn = self.HEADING_GAIN * off
        guess = step_pose(here, self.sonar_speed, turn)
        return self._fit(readings, [(guess, 0.0)], (0,), self.SONAR_PRIOR, usable)

    def _fit(self, readings: dict[str, float], moves, heading_guesses_deg, prior: np.ndarray,
             usable: int) -> Pose:
        best = None
        for start, move_cost in moves:
            for off_deg in (heading_guesses_deg if self.correct else (0,)):
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
            H = J.T @ (w[:, None] * J) + prior
            q = q - np.linalg.solve(H, J.T @ (w * r) + prior @ q)
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

    WALL_JOIN_CM = 12.0  # echoes from one sensor closer than this are joined into a wall line
    EDGE_GATE_CM = 8.0   # an echo this close to a maze wall line votes for that wall
    EDGE_VOTES = 3       # votes before a maze wall is drawn
    DISTANCE_MARK_CM = 25.0  # pin the sensor distances to the map every this many cm driven

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
        self.last_echo: dict[str, tuple[float, float]] = {}  # each sensor's previous echo point
        self.walls: list[tuple[float, float, float, float]] = []  # traced from the echoes; never erased
        self.distance_marks: list[tuple[float, float, float, float, float]] = []  # (x, y, left, front, right)
        self._since_mark = 0.0  # cm driven since the last distance mark
        self.edge_votes: dict[tuple[str, int, int], int] = {}  # maze wall -> echoes that landed on it
        self._crossed: set[tuple[str, int, int]] = set()
        self._crossed_upto = 0
        self._crossed_at = -1
        self.last: Telemetry | None = None
        self.samples = 0
        self.turn_lines = False  # the robot logs its turns as T lines (else they're read from the S lines)
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
        turn = self._silent_turn(t, dt)
        if turn:
            dt = 0.1  # most of the gap was spent spinning on the spot, not driving
        # dt <= 0: board was reset; huge dt: we lost data. Either way don't guess the move.
        motors = (prev.motor_l, prev.motor_r) if prev is not None and 0 < dt < 1.0 else (0, 0)
        self.pose = self.tracker.update({name: getattr(t, name) for name in SENSOR_NAMES},
                                        *motors, max(dt, 0.0))

        px, py = self.path[-1]
        step = math.hypot(self.pose.x - px, self.pose.y - py)
        if step > 0.5:
            self.path.append((self.pose.x, self.pose.y))
            self._since_mark += step
        if self._since_mark >= self.DISTANCE_MARK_CM or len(self.distance_marks) == 0:
            self.distance_marks.append((self.pose.x, self.pose.y, t.left, t.front, t.right))
            self._since_mark = 0.0

        for name, (fwd, left, offset_deg) in self.cfg.sensors.items():
            dist = getattr(t, name)
            sx, sy = self.pose.to_world(fwd, left)
            heading = self.pose.theta + math.radians(offset_deg)
            hit = 0 < dist < self.cfg.max_range_cm
            self.grid.integrate_reading(sx, sy, heading, dist, self.cfg)
            if hit:
                # Consecutive echoes from one sensor trace the wall it's sweeping along: join them.
                ex, ey = sx + dist * math.cos(heading), sy + dist * math.sin(heading)
                px, py = self.last_echo.get(name, (ex, ey))
                if math.hypot(ex - px, ey - py) > self.WALL_JOIN_CM:
                    px, py = ex, ey  # too far apart: a different wall, start a new line
                self.walls.append((px, py, ex, ey))
                self._vote(ex, ey)
                self.last_echo[name] = (ex, ey)
            else:
                self.last_echo.pop(name, None)
            length = dist if hit else min(self.cfg.free_range_cm, self.cfg.max_range_cm)
            self.rays[name] = Ray((sx, sy), (sx + length * math.cos(heading),
                                             sy + length * math.sin(heading)), hit)
        if turn:  # these readings were taken just before the spin, so turn after mapping them
            self.turn(turn)

    def _silent_turn(self, t: Telemetry, dt: float) -> float:
        """Degrees the robot spun (+ = left) just before logging `t`, if this line follows a turn.

        tunnel.ino doesn't log its turns: at a wall it stops, spins, and only then logs the reading
        that made it turn, with the motors at 0,0 and a gap in the timestamps while it spun.
        """
        cfg = self.cfg
        # (distances are logged to 0.1 cm, so a 14.97 that made it turn shows up as 15.0)
        stopped_at_wall = 0 < t.front < cfg.front_turn_cm + 0.5 and (t.motor_l, t.motor_r) == (0, 0)
        if self.turn_lines or not stopped_at_wall or not cfg.turn_gap_ms / 1000 <= dt < 5.0:
            return 0.0
        if dt >= cfg.uturn_gap_ms / 1000:
            return 180.0
        # A quarter turn, towards the first open side (it checks its left sensor first) - wherever
        # that sensor points. It decides on readings of its own, taken just after this one; when
        # neither side looks open here, one of those missed an echo: go by the roomier side.
        room = {n: math.inf if getattr(t, n) < 0 else getattr(t, n) for n in ("left", "right")}
        open_sides = [n for n in ("left", "right") if room[n] > cfg.open_cm]
        side = open_sides[0] if open_sides else max(room, key=room.get)
        return math.copysign(90.0, cfg.sensors[side][2])

    def _vote(self, x: float, y: float) -> None:
        """Count an echo at (x, y) towards the maze wall it lies on, if it lies near one."""
        c = self.maze_cell
        rx, ry = lattice_offset(x, c), lattice_offset(y, c)
        if min(abs(rx), abs(ry)) > self.EDGE_GATE_CM:
            return
        if abs(rx) < abs(ry):  # on a vertical wall line x = (i + 1/2) * c
            key = ("v", round((x - rx - c / 2) / c), round(y / c))
        else:                  # on a horizontal wall line y = (j + 1/2) * c
            key = ("h", round(x / c), round((y - ry - c / 2) / c))
        self.edge_votes[key] = self.edge_votes.get(key, 0) + 1

    def _crossed_edges(self) -> set[tuple[str, int, int]]:
        """Maze wall slots the driven path passes through. Cached per sample."""
        if self._crossed_at != self.samples:
            c = self.maze_cell
            for (x1, y1), (x2, y2) in zip(self.path[self._crossed_upto:], self.path[self._crossed_upto + 1:]):
                for axis, a1, a2, b in ((0, x1, x2, (y1 + y2) / 2), (1, y1, y2, (x1 + x2) / 2)):
                    k1, k2 = math.floor((a1 - c / 2) / c), math.floor((a2 - c / 2) / c)
                    if k1 != k2:  # stepped across the wall line at (max(k1, k2) + 1/2) * c
                        self._crossed.add(("v" if axis == 0 else "h", max(k1, k2), round(b / c)) if axis == 0
                                          else ("h", round(b / c), max(k1, k2)))
            self._crossed_upto = max(0, len(self.path) - 1)
            self._crossed_at = self.samples
        return self._crossed

    def straight_walls(self) -> list[tuple[float, float, float, float]]:
        """Maze walls with enough echoes on them, as straight cell-edge segments (x1, y1, x2, y2)."""
        c, h = self.maze_cell, self.maze_cell / 2
        crossed = self._crossed_edges()  # the robot drove through these, so they can't be walls
        segs = []
        for (kind, i, j), votes in self.edge_votes.items():
            if votes < self.EDGE_VOTES or (kind, i, j) in crossed:
                continue
            if kind == "v":
                x = (i + 0.5) * c
                segs.append((x, j * c - h, x, j * c + h))
            else:
                y = (j + 0.5) * c
                segs.append((i * c - h, y, i * c + h, y))
        return segs

    def mark_exit(self) -> None:
        self.exit_pos = (self.pose.x, self.pose.y)

    def spin(self, way: int, ms: int) -> None:
        """The robot reported spinning on the spot (a T line): turn by as much."""
        self.turn(way * self.cfg.spin_deg_s * ms / 1000.0)
        self.turn_lines = True

    def turn(self, degrees: float) -> None:
        """The robot spun on the spot by `degrees` (+ = left)."""
        self.pose = Pose(self.pose.x, self.pose.y, self.pose.theta + math.radians(degrees))
        self.tracker.spin(self.pose)
        self.last_echo.clear()  # don't join echoes from before and after the spin into one wall

    def rotate(self, degrees: float) -> None:
        """Manual heading correction for when the pose estimate has drifted."""
        self.pose = Pose(self.pose.x, self.pose.y, self.pose.theta + math.radians(degrees))
        self.tracker.reset(self.pose)
