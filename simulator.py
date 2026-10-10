"""Virtual maze + robot that emits the same telemetry as maze_robot.ino.

Lets you test the mapper without hardware: `python maze_mapper.py --sim`.
"""
from __future__ import annotations

import math
import random

import numpy as np

from mapping import SENSOR_NAMES, Pose, RobotConfig, Telemetry, advance, format_line


def generate_maze(cols: int, rows: int, cell_cm: float,
                  rng: random.Random) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Perfect maze (recursive backtracker). Cell (0, 0) is centred on the world origin.

    The robot starts in the closed corner cell (0, 0), facing +y; the exit is a gap in the outer
    wall at the top of the opposite corner cell.
    Returns the wall segments as an (N, 4) array of x1, y1, x2, y2, and the exit gap as x1, y1, x2, y2.
    """
    # h[r][c]: wall along the bottom of cell (r, c), r in 0..rows
    # v[r][c]: wall along the left of cell (r, c), c in 0..cols
    h = [[True] * cols for _ in range(rows + 1)]
    v = [[True] * (cols + 1) for _ in range(rows)]
    seen = {(0, 0)}
    stack = [(0, 0)]
    while stack:
        r, c = stack[-1]
        options = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))
                   if 0 <= r + dr < rows and 0 <= c + dc < cols and (r + dr, c + dc) not in seen]
        if not options:
            stack.pop()
            continue
        nr, nc = rng.choice(options)
        if nr != r:
            h[max(r, nr)][c] = False
        else:
            v[r][max(c, nc)] = False
        seen.add((nr, nc))
        stack.append((nr, nc))
    h[rows][cols - 1] = False   # exit

    x0 = y0 = -cell_cm / 2
    segs = []
    for r in range(rows + 1):
        for c in range(cols):
            if h[r][c]:
                segs.append((x0 + c * cell_cm, y0 + r * cell_cm, x0 + (c + 1) * cell_cm, y0 + r * cell_cm))
    for r in range(rows):
        for c in range(cols + 1):
            if v[r][c]:
                segs.append((x0 + c * cell_cm, y0 + r * cell_cm, x0 + c * cell_cm, y0 + (r + 1) * cell_cm))
    exit_gap = (x0 + (cols - 1) * cell_cm, y0 + rows * cell_cm, x0 + cols * cell_cm, y0 + rows * cell_cm)
    return np.array(segs, dtype=float), exit_gap


class Averager:
    """Mean of the last N valid pings, so one noisy or missed echo doesn't jerk the robot.

    Mirror of the averaging in maze_robot.ino.
    """

    def __init__(self, n: int = 3):
        self.readings: list[float] = []
        self.n = n

    def add(self, d: float) -> float:
        self.readings = (self.readings + [d])[-self.n:]
        valid = [r for r in self.readings if r > 0]
        return sum(valid) / len(valid) if valid else -1.0


class WallFollower:
    """Python mirror of decide() in maze_robot.ino. Keep the constants in sync.

    Left-hand wall follower with a front sensor and two sensors pointing SIDE_ANGLE_DEG out to
    the left and right of straight ahead. Each side reading splits into a sideways part
    (reading * sin) and a forward part (reading * cos). A wall ahead makes BOTH side sensors read
    short, so their longer forward part is a second opinion on the distance ahead (one side
    sensor reading short just means a side wall is close). When all three sensors see open space
    for a while, the robot has left the maze through the exit and stops (state DONE).
    Distances are in cm, tuned for ~35 cm corridors and a 10 cm wide robot.
    """

    SIDE_ANGLE_DEG = 30
    FRONT_STOP = 10      # distance ahead below this -> pivot
    FRONT_CLEAR = 12     # ...until it's at least this clear
    SIDE_OPEN = 14       # sideways distance above this = opening on that side (25 misses openings in the sim)
    LEFT_TARGET = 11     # hug the left wall (side sensors sit ~mid-robot: 11 = ~5 cm gap)
    STEER_BAND = 2.5     # within this of the target -> drive straight (no shaking)
    STEER_KD = 3         # weight of "how fast is the left wall approaching"
    LEFT_MIN = 7         # too close to the left wall -> veer right
    RIGHT_MIN = 7        # too close to the right wall -> veer left
    OPEN_CONFIRM = 2     # readings in a row before trusting a left opening
    PRE_LEFT_MS = 650    # drive on past the start of an opening before turning into it
    TURN_90_MS = 710     # ~90 deg pivot: (pi/2) * wheelbase / speed - calibrate on your floor
    POST_LEFT_MS = 900   # drive on after a left turn to pick up the new left wall
    MIN_PIVOT_MS = 250   # blocked: pivot at least this long (and per extra 90), then until clear
    FRONT_MARGIN = 6     # the front sensor overrides the side estimate only when it reads this much closer
    EXIT_CLEAR = 80      # all three sensors see nothing closer than this...
    EXIT_CONFIRM = 6     # ...this many readings in a row -> we're out of the maze: stop

    def __init__(self):
        self.side_scale = math.sin(math.radians(self.SIDE_ANGLE_DEG))
        self.ahead_scale = math.cos(math.radians(self.SIDE_ANGLE_DEG))
        self.state = "FOLLOW"
        self.state_start = 0
        self.motors = (0, 0)
        self.pivot = (255, 0)
        self.prev_left = None
        self.open_count = 0
        self.exit_count = 0
        self.seen_wall = False  # in the maze yet? Until then, no echoes != EXIT

    def _enter(self, state: str, now: int) -> None:
        self.state = state
        self.state_start = now

    def _drive_straight(self, L: float, R: float) -> tuple[int, int]:
        """Forward, but steer off a wall that's too close (headings drift during timed moves)."""
        if R < self.RIGHT_MIN:
            return (0, 255)
        if L < self.LEFT_MIN:
            return (255, 0)
        return (255, 255)

    def _start_pivot(self, L: float, R: float, now: int) -> None:
        # Pivot towards the side with more room (pivoting on one wheel moves the robot that way).
        self.pivot = (255, 0) if R >= L else (0, 255)
        self._enter("PIVOT", now)
        self.motors = self.pivot

    def decide(self, left: float, front: float, right: float, now: int) -> tuple[int, int]:
        # Don't start blind: no echoes at all would read as open space and trigger a false EXIT.
        self.seen_wall = self.seen_wall or any(0 < d < self.EXIT_CLEAR for d in (left, front, right))
        if not self.seen_wall:
            self.state_start = now
            self.motors = (0, 0)
            return self.motors
        left, front, right = (999.0 if d <= 0 else d for d in (left, front, right))
        L, R = left * self.side_scale, right * self.side_scale  # sideways distances
        left_open, right_open = L > self.SIDE_OPEN, R > self.SIDE_OPEN
        # Distance ahead: the longer forward part among the side sensors that still see a wall.
        # The front sensor only takes over when something is right in front of the nose (e.g. a
        # wall end between the side beams) - trusting it fully makes the robot stop too often.
        seen = [d for d, is_open in ((left, left_open), (right, right_open)) if not is_open]
        ahead = min(front + self.FRONT_MARGIN, max(seen) * self.ahead_scale if seen else 999.0)
        in_state = now - self.state_start
        self.open_count = self.open_count + 1 if left_open else 0
        clear = min(left, front, right) > self.EXIT_CLEAR
        self.exit_count = self.exit_count + 1 if clear else 0
        # How fast the left wall is approaching tells us our angle to it (only measured while
        # following the wall; pivots and timed moves would make it meaningless).
        following = self.state == "FOLLOW" and not left_open
        d_left = L - self.prev_left if following and self.prev_left is not None else 0.0
        self.prev_left = L if following else None

        if self.state == "DONE":
            self.motors = (0, 0)
        elif self.exit_count >= self.EXIT_CONFIRM:
            self._enter("DONE", now)
            self.motors = (0, 0)
        elif self.state == "FOLLOW":
            if ahead < self.FRONT_STOP:
                self._start_pivot(L, R, now)
            elif self.open_count >= self.OPEN_CONFIRM:
                self._enter("PRE_LEFT", now)
                self.motors = (255, 255)
            elif left_open:
                self.motors = (255, 255)  # possible opening - wait for confirmation
            elif R < self.RIGHT_MIN or L < self.LEFT_MIN:
                self.motors = self._drive_straight(L, R)  # about to hit a wall
            else:
                steer = (L - self.LEFT_TARGET) + self.STEER_KD * d_left
                if steer > self.STEER_BAND:
                    self.motors = (0, 255)    # veer left
                elif steer < -self.STEER_BAND:
                    self.motors = (255, 0)    # veer right
                else:
                    self.motors = (255, 255)  # straight
        elif self.state == "PIVOT":
            self.motors = self.pivot
            if in_state > self.MIN_PIVOT_MS and ahead > self.FRONT_CLEAR:
                self._enter("FOLLOW", now)
            elif in_state > self.TURN_90_MS:
                self.state_start = now  # dead end: keep pivoting, ignore glimpses for MIN_PIVOT_MS again
        elif self.state == "PRE_LEFT":
            self.motors = self._drive_straight(L, R)
            if ahead < self.FRONT_STOP or in_state > self.PRE_LEFT_MS:
                self._enter("TURN_LEFT", now)
        elif self.state == "TURN_LEFT":
            self.motors = (0, 255)
            if in_state > self.TURN_90_MS:
                self._enter("POST_LEFT", now)
        elif self.state == "POST_LEFT":
            self.motors = self._drive_straight(L, R)
            if ahead < self.FRONT_STOP:
                self._start_pivot(L, R, now)
            elif not left_open or in_state > self.POST_LEFT_MS:
                self._enter("FOLLOW", now)
        return self.motors


class Simulator:
    """Steps a virtual robot through a random maze and returns telemetry lines."""

    dt_ms = 60
    robot_radius_cm = 5.5  # 10 cm wide robot

    def __init__(self, cfg: RobotConfig, cols: int = 6, rows: int = 6, cell_cm: float = 35.0,
                 seed: int | None = None, drift: float = 0.0):
        self.cfg = cfg
        self.rng = random.Random(seed)
        self.cell_cm = cell_cm
        self.walls, self.exit_gap = generate_maze(cols, rows, cell_cm, self.rng)
        # The "real" robot differs from what the mapper assumes by `drift`, to mimic odometry error.
        self.true_speed = cfg.speed_cm_s * (1.0 + drift)
        self.true_wheelbase = cfg.wheelbase_cm * (1.0 + drift / 2)
        self.pose = Pose()
        self.t_ms = 0
        self.motors = (0, 0)
        WallFollower.SIDE_ANGLE_DEG = cfg.sensors["left"][2]
        self.controller = WallFollower()
        self.averagers = {name: Averager() for name in SENSOR_NAMES}
        self.override: tuple[int, int] | None = None  # motor powers commanded by the PC

    def command(self, line: str) -> None:
        """Serial commands from the PC (see navigation.py): STOP, M,<left>,<right>, GO."""
        parts = line.strip().split(",")
        if parts[0] == "STOP":
            self.override = (0, 0)
        elif parts[0] == "M" and len(parts) == 3:
            self.override = (int(parts[1]), int(parts[2]))
        elif parts[0] == "GO":
            self.override = None

    def _raycast(self, px: float, py: float, angle: float) -> float:
        x1, y1, x2, y2 = self.walls.T
        dx, dy = math.cos(angle), math.sin(angle)
        ex, ey = x2 - x1, y2 - y1
        denom = dx * ey - dy * ex
        with np.errstate(divide="ignore", invalid="ignore"):
            t = ((x1 - px) * ey - (y1 - py) * ex) / denom
            u = ((x1 - px) * dy - (y1 - py) * dx) / denom
        valid = (np.abs(denom) > 1e-9) & (t > 0) & (u >= 0) & (u <= 1)
        return float(t[valid].min()) if valid.any() else math.inf

    def _clearance(self, px: float, py: float) -> float:
        x1, y1, x2, y2 = self.walls.T
        ex, ey = x2 - x1, y2 - y1
        u = np.clip(((px - x1) * ex + (py - y1) * ey) / (ex * ex + ey * ey), 0, 1)
        return float(np.hypot(x1 + u * ex - px, y1 + u * ey - py).min())

    def _sense(self, name: str) -> float:
        fwd, left, offset = self.cfg.sensors[name]
        sx, sy = self.pose.to_world(fwd, left)
        heading = self.pose.theta + math.radians(offset)
        cone = math.radians(8)
        d = min(self._raycast(sx, sy, heading + a) for a in np.linspace(-cone, cone, 5))
        if d >= self.cfg.max_range_cm or self.rng.random() < 0.02:
            return -1.0  # no echo
        return max(2.0, d + self.rng.gauss(0, 0.5))

    @property
    def escaped(self) -> bool:
        """Ground truth: the robot is in or past the exit gap."""
        x1, y, x2, _ = self.exit_gap
        return x1 < self.pose.x < x2 and self.pose.y > y - self.cell_cm / 3

    def step(self) -> str:
        """One loop of the robot. Returns its serial output (telemetry, plus any text lines)."""
        new = advance(self.pose, *self.motors, self.dt_ms / 1000.0, self.true_speed, self.true_wheelbase)
        if self._clearance(new.x, new.y) < self.robot_radius_cm:
            new = Pose(self.pose.x, self.pose.y, new.theta)  # bumped a wall: rotate only
        self.pose = new
        self.t_ms += self.dt_ms

        left, front, right = (self.averagers[n].add(self._sense(n)) for n in SENSOR_NAMES)
        was_done = self.controller.state == "DONE"
        override = self.override
        if override is None:
            self.motors = self.controller.decide(left, front, right, self.t_ms)
        else:
            self.motors = override
        line = format_line(Telemetry(self.t_ms, left, front, right, *self.motors))
        if override is None and self.controller.state == "DONE" and not was_done:
            line = f"EXIT reached at {self.t_ms} ms\n{line}"
        return line
