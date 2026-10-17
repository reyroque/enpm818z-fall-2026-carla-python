# L11 hands-on: an expert drives, a clone learns

An expert keeps the AV in its lane for one lap of CARLA's Town01. A clone learns
to copy it from a recording (behavior cloning), then from the states it reaches
itself (DAgger). Needs `numpy` and `matplotlib`; no CARLA, no ROS, no GPU.

```bash
python3 l11_handson.py --part road     # 1. the lap, and the expert driving it
python3 l11_handson.py --part bc       # 2. behavior cloning: fit, weights, the clone drives
python3 l11_handson.py --part shift    # 3. why the clone drifts: its error grows off center
python3 l11_handson.py --part dagger   # 4. DAgger, iteration by iteration (slider)
python3 l11_handson.py --part eval     # 5. open loop against closed loop
```

Add `--snapshot out.png` to save a still instead of opening a window, or
`--html out.html` for a page (needs `pip install mpld3`).

| Option | What it changes |
|---|---|
| `--detector-noise M` | the noise on the offset the clone sees, in meters (default 0.35) |
| `--gust D` | the random push on the steering that every driver gets, in degrees (default 0.4) |
| `--seed S` | the expert's recorded lap and DAgger's laps (default 11, the one on the slides) |
| `--iterations N` | DAgger iterations; the first is behavior cloning (default 6) |

Parts `dagger` and `eval` drive 20 test laps for each policy: they take about 15 s.

## What each part shows

1. **road**: the lap from above, and the expert's offset from the lane center
   along the lap. The dashed lines at 1.0 m are where a wheel leaves the lane.
2. **bc**: the recorded pairs (offset the detector reports, steering the expert
   gave), the expert's rule and the clone's fitted rule; then the clone driving
   one test lap. The terminal prints the four weights next to the expert's.
3. **shift**: the clone's steering error at each step against how far it really
   is from the center, and where the expert and the clone spend their time.
4. **dagger**: the labeled data after each iteration, and how many of 20 test
   laps each iteration's clone finishes. Drag the slider.
5. **eval**: the open-loop error (on a recorded expert lap) and the laps
   finished (each one drives) for behavior cloning, DAgger and the expert.

## The example

`l11_road.npz` is one lap of Town01's outer loop, read from the town's OpenDRIVE
map: the lane centerline, its heading and curvature every 0.5 m, the lane width
(4.0 m), and every lane waypoint of the town for the plots. `l11_junctions.npz`
holds the five junctions on the lap, each with its two branches.

`l11common.py` has everything else:

- **the AV**: a kinematic bicycle in lane coordinates (distance along the lane,
  offset, heading error), 8 m/s, 20 steps per second;
- **the expert**: steering = the curve's steering 0.3 s ahead, minus 0.06 rad per
  meter of offset, minus 0.8 times the heading error. It reads the true state;
- **the clone**: steering = w1 x offset seen + w2 x heading seen + w3 x curvature
  + w4. It reads a lane detector: offset with noise 0.35 m, heading with noise
  0.3 deg;
- **behavior cloning**: least squares on the expert's recorded lap;
- **DAgger** (Ross, Gordon and Bagnell, 2011): the clone drives, the expert
  labels every state it reached, the pairs join the data, the clone is refit.

**Every value in `CHOSEN` is chosen for the example.** They were picked so that
the expert always finishes a lap and the effect shows; change them with the
options above and see what moves.

## Tests

```bash
python3 -m pytest -q tests
```
