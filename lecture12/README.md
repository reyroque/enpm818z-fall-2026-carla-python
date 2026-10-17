# L12 hands-on: test a toy stack on a pedestrian crossing

A small 2-D simulation of the final report's pedestrian-crossing scenario, on a
Town01 street. Needs `numpy` and `matplotlib`; no CARLA, no ROS, no GPU.

```bash
python3 l12_handson.py --part run      # 1. one concrete run, played from above
python3 l12_handson.py --part sweep    # 1. the pass/fail map of the logical scenario
python3 l12_handson.py --part sample   # 2. a budget of runs: grid, random, search
python3 l12_handson.py --part score    # 3. GP4-style numbers and the Leaderboard score
```

Add `--snapshot out.png` to save a still instead of opening a window, or
`--html out.html` (part `run` only) to save an animation you open in a browser.

| Option | Part | What it does |
|---|---|---|
| `--v0 V` | run | the AV's speed when the pedestrian steps off the curb, m/s (default 10) |
| `--d0 D` | run | the gap from the AV's front bumper to the crossing line then, m (default 36) |
| `--vped W` | run, sweep | the pedestrian's walking speed, m/s (default 1.4) |
| `--cell C` | run, sweep | the ODD cell: `dry_day`, `wet_day`, `dry_night`, `wet_night` |
| `--budget N` | sample | the number of runs (default 25) |
| `--seed S` | sample | the random seed (default 1) |

## The scenario

The street is a straight two-lane Town01 street: lanes 4.0 m wide, a 0.3 m
shoulder, a 4.0 m sidewalk, speed limit 25 mph (11.2 m/s). `town01_probe.py`
read these values from `Town01.xodr` with CARLA's map API, offline, and wrote
`town01.json`.

The AV drives along +x in the right-hand lane. A pedestrian steps off the far
curb and walks straight across at a steady speed: first the oncoming lane, then
the AV's lane. The **logical scenario** has three parameters:

| Parameter | Range | Why |
|---|---|---|
| `v0`, the AV's speed | 2 to 11.2 m/s | the top is the speed limit; the bottom is chosen |
| `d0`, the gap | 5 to 60 m | chosen |
| `vped`, the walking speed | 0.8 to 2.0 m/s | chosen: a slow walk to a jog |

A **concrete scenario** is one value of each. A run fails if the pedestrian (a
circle of radius 0.25 m) touches the AV (a 4.5 m by 1.9 m rectangle).

## The system under test: a toy stack

It is a stand-in, not a driving stack, and not GP4's: it shows how to test, not
how to drive. Every value is chosen.

- It sees the pedestrian when the crossing line is up to 30 m ahead (12 m at night).
- It calls the pedestrian a hazard only inside its own lane plus 0.5 m each side.
- 0.5 s after the first hazard it brakes at 6 m/s^2 (4 m/s^2 on a wet road) until
  it stops. Otherwise it holds its speed.

## Files

| File | What it is |
|---|---|
| `l12_sim.py` | the street, the toy stack, `simulate()` (many runs at once, 0.01 s steps), grid and random sampling, the boundary search, the zero-failure bound, Kalra and Paddock's miles, the Leaderboard 2.0 and 2.1 driving scores |
| `l12_handson.py` | the four parts above |
| `town01_probe.py`, `town01.json` | the street values, read from `Town01.xodr` (needs the `carla` client package, not a server) |
| `tests/` | `python3 -m pytest tests` |

The slides' figures and numbers come from these files, through
`gen_l12_figures.py` in the slides repository.
