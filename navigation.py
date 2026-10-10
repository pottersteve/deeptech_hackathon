"""Drive the robot to a maze cell (e.g. back to START) through the cells it has already driven through.

The PC plans and steers; the robot just obeys motor commands sent over serial:

    STOP              stop both motors and stop wall following
    M,<left>,<right>  drive these motor powers (-255..255) until the next command
    GO                resume wall following

The simulator and ../maze_robot_nav/maze_robot_nav.ino understand these; maze_robot.ino ignores them.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque

from mapping import Pose

Cell = tuple[int, int]


def cell_of(x: float, y: float, cell_cm: float) -> Cell:
    """Maze cell containing a world point; cell (0, 0) is centred on the start."""
    return round(x / cell_cm), round(y / cell_cm)


def visited_graph(path: list[tuple[float, float]], cell_cm: float) -> dict[Cell, set[Cell]]:
    """Cells the driven path passed through, linked where it moved straight from one to the next."""
    graph: dict[Cell, set[Cell]] = defaultdict(set)
    prev = cell_of(*path[0], cell_cm)
    graph[prev]
    for x, y in path[1:]:
        c = cell_of(x, y, cell_cm)
        if abs(c[0] - prev[0]) + abs(c[1] - prev[1]) == 1:
            graph[prev].add(c)
            graph[c].add(prev)
        prev = c
    return graph


def nearest_visited(path: list[tuple[float, float]], x: float, y: float, cell_cm: float) -> Cell:
    """The visited cell closest to a world point: the only places a route can safely end."""
    graph = visited_graph(path, cell_cm)
    return min(graph, key=lambda c: math.hypot(c[0] * cell_cm - x, c[1] * cell_cm - y))


def plan_route(path: list[tuple[float, float]], here: Pose, cell_cm: float,
               goal: Cell = (0, 0)) -> list[tuple[float, float]]:
    """Waypoints from the robot's position to the centre of `goal` (a visited cell), through cell centres.

    The shortest route through the visited-cell graph skips every dead end the robot explored.
    Falls back to retracing the raw path backwards to where it came closest to the goal if the
    cell graph is broken (e.g. the pose estimate jumped a cell).
    """
    graph = visited_graph(path, cell_cm)
    start = cell_of(here.x, here.y, cell_cm)
    came_from = {start: start}
    todo = deque([start])
    while todo and goal not in came_from:
        c = todo.popleft()
        for n in graph[c]:
            if n not in came_from:
                came_from[n] = c
                todo.append(n)

    gx, gy = goal[0] * cell_cm, goal[1] * cell_cm
    if goal not in came_from:
        closest = min(range(len(path)), key=lambda i: math.hypot(path[i][0] - gx, path[i][1] - gy))
        route, last = [], None
        for x, y in reversed(path[closest:]):
            if last is None or math.hypot(x - last[0], y - last[1]) >= 5.0:
                route.append((x, y))
                last = (x, y)
        return route + [(gx, gy)]

    cells = [goal]
    while cells[-1] != start:
        cells.append(came_from[cells[-1]])
    return [(c[0] * cell_cm, c[1] * cell_cm) for c in reversed(cells)]


class DriveTo:
    """Steers along the planned waypoints: turn on the spot towards the next one, then drive at it."""

    REACHED_CM = 4.0        # waypoint counts as reached within this distance
    SPIN_ABOVE_DEG = 30.0   # heading error above this -> turn on the spot first
    STEER_GAIN = 6.0        # motor power taken off the inside wheel per degree of heading error

    def __init__(self, path: list[tuple[float, float]], here: Pose, cell_cm: float, goal: Cell = (0, 0)):
        self.goal = (goal[0] * cell_cm, goal[1] * cell_cm)
        self.waypoints = plan_route(path, here, cell_cm, goal)
        self.done = False
        self.spin = 0  # direction of the current turn on the spot (+1 left, -1 right, 0 none)

    def command(self, pose: Pose) -> tuple[int, int]:
        while self.waypoints and math.hypot(self.waypoints[0][0] - pose.x,
                                            self.waypoints[0][1] - pose.y) < self.REACHED_CM:
            self.waypoints.pop(0)
        if not self.waypoints:
            self.done = True
            return 0, 0

        tx, ty = self.waypoints[0]
        err = math.degrees(math.atan2(ty - pose.y, tx - pose.x) - pose.theta)
        err = (err + 180) % 360 - 180  # positive = target is to the left
        if abs(err) > self.SPIN_ABOVE_DEG:
            # Keep turning the way we started: with the target right behind, the sign of `err`
            # flips every step and the robot would just wiggle on the spot.
            if not self.spin:
                self.spin = 1 if err > 0 else -1
            s = int(min(255, max(90, abs(err) * 4)))  # slow down near the target heading
            return (-s, s) if self.spin > 0 else (s, -s)
        self.spin = 0
        turn = self.STEER_GAIN * err
        return int(max(0, min(255, 255 - turn))), int(max(0, min(255, 255 + turn)))
