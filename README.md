# Maze Mapper

Live map of the maze as the Arduino robot (`../maze_robot/maze_robot.ino`) explores it.

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python maze_mapper.py --sim                 # try it without hardware
python maze_mapper.py --log run1.log        # real robot (auto-detects the Uno R4 port)
python maze_mapper.py --port COM5           # if auto-detect fails
python maze_mapper.py --replay run1.log     # watch a recorded run again
```

Controls: Space start the robot (sends GO) · X stop it · wheel zoom · drag pan · F follow · R reset · Q/E rotate heading ±5° · S save PNG · L wall lines · G show sim's hidden maze · H (or the top-right button) stop & return home · P (or the button below it) go to a point · Esc quit

## How it works

The robot has no wheel encoders. The PC tracks its position from each telemetry line:

    S,<millis>,<left_cm>,<front_cm>,<right_cm>,<motor_left>,<motor_right>

**Motors predict, sonars correct.** The motor powers and timestamps give a rough guess
of each move (`--speed`, `--wheelbase`); that guess drifts - every turn adds a few degrees,
and a wrong `--speed` stretches every corridor. So each sample the guess is corrected
against the maze itself: every wall lies on the 35 cm grid (`--maze-cell`), so the echoes
should land on grid lines. A few headings around the guess are tried, the one most echoes
agree with wins, and the position is fine-tuned to fit (`PoseTracker` in mapping.py).
In the simulator with the motor model 5% off, this brings the robot back to within
~4 cm of START, where motors alone missed by 30-90 cm. START must be the centre of a cell.

The robot has three sensors: one straight ahead and two pointing 30° left and right
of it. Each distance is the mean of its last 3 valid pings, so the readings don't jitter.
Each reading then updates an occupancy grid: the beam cone up to the echo is
marked free, the arc at the echo distance is marked as wall.

The sketch hugs the left wall (in ~35 cm corridors a one-wheel pivot only fits
from there). It treats "both side sensors read short" as a wall ahead; the front
sensor only takes over when something is right in front of the nose.

## Wall lines

The raw sonar map is fuzzy, so walls are also drawn as solid lines snapped to the
maze grid (`--maze-cell`, default 35 cm): every cell edge with enough sonar hits along
it becomes a line. Cell (0, 0) is centred on START, so start the robot in the middle of
a cell, facing along a corridor. L turns the lines off to show the raw map.

## Return home / go to a point

The **STOP & RETURN HOME** button (or H) stops the robot and drives it back to START
along the shortest route through the cells it has already driven, skipping the dead ends
it explored. The route is drawn in teal; press again to cancel (the robot stays stopped).

**GO TO POINT** (or P), then click the map: the robot drives to that cell the same way.
The click snaps to the nearest cell the robot has already been in (previewed under the
cursor), since only those are known to be reachable. Esc or P again cancels.
The PC steers by sending serial commands, documented in `navigation.py`
(`STOP`, `M,<left>,<right>`, `GO`). The simulator understands them, and so does
`../maze_robot_nav/maze_robot_nav.ino` - the simulated robot as a real sketch (same wiring
and wall follower as the simulator); upload that one to use the buttons on the real robot.
`maze_robot.ino` ignores them.
It steers by the tracked position, so it's only as accurate as that estimate.

## Start and exit

The robot starts in a closed corner cell (marked START on the map, at the origin).
Following the left wall leads it to the exit gap in the outer wall. Once all three
sensors see nothing within 80 cm for 6 readings in a row (`EXIT_CLEAR`, `EXIT_CONFIRM`),
it stops and prints `EXIT reached`, and the mapper marks EXIT where it stopped.
Leave at least ~80 cm of empty floor beyond the exit, or lower `EXIT_CLEAR`
(it must stay well above your corridor width).
Keep the start closed: if the robot can see open space behind it at the start,
it may turn round and "exit" that way.

## Calibrating (the sonars correct small errors; the closer these are, the less they have to)

1. **Speed**: run both motors for 5 s, measure the distance, `--speed = cm / 5`.
2. **Wheelbase**: measure between the wheel centres, `--wheelbase` (default 9 cm, for a 10 cm robot).
3. **Turns**: pivot for `TURN_90_MS` in the sketch and adjust until it's 90°.
4. Side sensors point 30° out from straight ahead by default. If yours differ, pass `--side-angle`
   and change `SIDE_ANGLE_DEG` in the sketch. Sensor pins: left D2/D3, front D4/D12, right A0/A1 (trig/echo). Sensor mount positions are in `sensor_mounts()` (mapping.py).

Wall-follower constants live in the sketch, mirrored in `WallFollower` (simulator.py);
tune them in the sim first, then copy them over.
