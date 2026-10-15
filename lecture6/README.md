# L6 hands-on: where is the AV?

Five parts, one recorded CARLA drive. Needs `numpy`, `scipy` and `matplotlib`;
no CARLA, no ROS.

```bash
python3 l6_handson.py --part gnss   # 1. the GNSS fixes: scatter, bias and the outage
python3 l6_handson.py --part dr     # 2. dead reckoning: wheels + gyro, no GNSS
python3 l6_handson.py --part ekf    # 3. the fused filter: 8 runs, from results files
python3 l6_handson.py --part mcl    # 4. Monte Carlo localization on the map
python3 l6_handson.py --part icp    # 5. ICP scan matching, one iteration at a time
```

Add `--snapshot out.png` to save a still instead of opening a window.

| Option | Part | What it does |
|---|---|---|
| `--particles N` | mcl | number of particles (default 2000) |
| `--seed S` | mcl | random seed (default 818) |
| `--kidnap T EAST NORTH` | mcl | at T s, every particle jumps EAST and NORTH m, as if the AV had been moved without telling MCL |
| `--random K` | mcl | after each scan, K particles get random poses (any heading) near MCL's estimate (default 0) |
| `--random-radius R` | mcl | how near: within R m of MCL's estimate (default 20) |
| `--icp-start EAST NORTH DEG` | icp | how far off the truth ICP starts (default `1.5 -1.0 5`) |
| `--time T` | gnss | open (or snapshot) at T seconds instead of the start; the slides use `--time 61.55`, the middle of the outage |

## What each part shows

1. **gnss**: the GNSS fixes as they arrive. **Play** runs the drive from start
   to finish, one second per tick (about 13 s in all); press it again to
   pause. **Step: +1 s** moves one second, one fix, at a time; **Reset**
   goes back to the start. A fix every
   second, then none for 55 s. Four panels:
   - **the whole route**: the truth, the fixes so far, the GNSS-only path (the
     fixes joined in order) and the AV, drawn 3 times its size so it shows;
   - **a close-up** 40 m across that follows the AV, at true size, with a
     dashed circle of 1.5 m (the noise we chose) around the last fix;
   - **each fix's error against time**, east and north; the dashed lines are
     their means;
   - **the errors as points**, filling in as the fixes arrive (blue so far,
     gray still to come, the newest circled): their center is the bias, their
     spread the noise, next to the bias and the 1.5 m we chose.

   The two laps cover the same road, so on the map the gap is hidden: the
   other lap's fixes fill it. The gap shows in time.
2. **dr**: the pose from wheel speed and yaw rate alone through the outage,
   against the truth, from the true pose at the last fix. Three panels: the
   outage from above, the position error and the heading error, both against
   the distance since the last fix. Two check boxes hold the errors we chose:
   untick **gyro bias** to subtract the 0.001 rad/s from every gyro reading,
   untick **wheels 0.5 % fast** to divide every wheel reading by 1.005, and
   dead reckoning runs again. What stays is the sensors' noise and CARLA's own
   gyro, which reports a little more turning than the car makes in a corner.
3. **ekf**: the fused filter's error and its own 1 sigma, and its NIS. Tick
   **GNSS outage**, **wheels** and **map matching** to switch between eight
   runs. The runs were computed in advance and saved in `ekf_results/`; this
   folder has no filter code. You build the filter yourself in GP3.
4. **mcl**: 2000 particles start anywhere within 20 m of the first GNSS fix,
   with any heading. The plot shows MCL's error and the spread of its
   particles over the drive. With `--kidnap`, a dotted line marks the moment
   every particle jumps away from the AV.
5. **icp**: one LiDAR scan, started off the truth, matched to the map. Drag
   the slider to step through the iterations.

## The data

`record_drive.py` drove one route in CARLA twice. The **mapping pass** stitched
LiDAR scans at the true poses into `l6_map.npz`. The **localization pass** is
`l6_drive.csv` and `l6_scans.npz`. `l6_drive_meta.json` lists the town, the
route and every value below.

Frames: the map frame is in meters, `x` east and `y` north (CARLA's `y` is
flipped). The vehicle frame is `x` forward and `y` left.

| column | meaning |
|---|---|
| `t` | time (s), one row every 0.05 s |
| `x, y, theta` | where the AV **really** is. Ground truth: never read it inside a filter |
| `v` | the AV's **true** speed (m/s) |
| `gyro_z` | what the gyro **says** the yaw rate is (rad/s) |
| `wheel_v` | what the wheels **say** the speed is (m/s) |
| `gnss_x, gnss_y` | GNSS fix (m), once a second. Empty between fixes and during the outage |

CARLA's sensors are too clean, and CARLA has no wheel encoder. So the recorder
starts from the true motion and adds these errors. **Every value is chosen**:

| sensor | error |
|---|---|
| GNSS | noise σ 1.5 m in `x` and in `y`; bias 0.5 m east and 0.3 m south; no fixes for 400 m |
| wheels | 0.5% fast; noise σ 0.05 m/s |
| gyro | bias 0.001 rad/s; noise σ 0.01 rad/s |
| LiDAR | 10 scans per second; a flat slice at sensor height; one point per 20 cm square |

`l6_scans.npz` holds the scans in the vehicle frame. Read them with
`l6common.load_scans`.

## The code

| file | what it holds |
|---|---|
| `l6common.py` | loading the drive, dead reckoning, the absolute pose error (APE) |
| `icp.py` | the five-step rigid fit (SVD) and the ICP loop |
| `mcl.py` | the likelihood field, and predict, weigh and resample |
| `record_drive.py` | the recorder (needs CARLA; you do not need to run it) |
| `l6_handson.py` | the four parts |

Run the tests with `python3 -m pytest tests -q`.

## Exercises

Reference numbers below come from the shipped drive (CARLA Town10HD_Opt, two
laps, 916 m in 125.7 s; no GNSS from 250 m to 650 m along the route, 34.2 s to
89.0 s) and the default seed 818 unless a seed is named.

1. In `--part dr`, find the stretch where the error grows fastest. Explain it
   with the heading error, not the speed error.
   *Reference:* started from the true pose at the last fix before the outage,
   dead reckoning is 2.69 m off after 100 m, 3.56 m after 200 m, 8.24 m after
   300 m, at worst 9.03 m (at 329 m), and 7.84 m at the next fix, 401 m on.
   It grows fastest from 250 m to 300 m (4.59 m to 8.24 m), on the straight
   middle street, where its heading is about 5 degrees off.
2. In `--part ekf`, compare the APE of the eight runs. Which input lowers it
   most? During the outage, does the filter's 1 sigma grow as fast as its
   error?
   *Reference (APE RMSE, m):*

   | GNSS | wheels | map | APE RMSE | worst |
   |---|---|---|---|---|
   | outage | yes | no | 4.99 | 9.50 |
   | outage | yes | yes | 0.52 | 2.49 |
   | outage | no | no | 4.58 | 12.36 |
   | outage | no | yes | 27.66 | 65.62 |
   | always | yes | no | 1.28 | 2.62 |
   | always | yes | yes | 0.47 | 2.38 |
   | always | no | no | 1.97 | 5.93 |
   | always | no | yes | 1.38 | 6.81 |

   With wheels and no map, the filter's 1 sigma is 3.93 m at the outage's worst
   moment, where the real error is 9.50 m. The run "outage, no wheels, map"
   loses the map at 45 s and stays lost until GNSS returns: find out why in its
   NIS plot.
3. In `--part mcl`, run `--particles 200` with five seeds. How many of them
   find the car? Then try 2000 particles.
   *Reference (seeds 0 to 4, median error after the first 10 s):* 200
   particles: 0.29, 0.78, 0.29, 0.37, 0.70 m; 2000 particles: 0.28, 0.21,
   0.27, 0.20, 0.34 m. All ten runs find the car.
5. In `--part mcl`, lose the car on purpose: `--kidnap 40 0 10` moves every
   particle 10 m north at 40 s. Does MCL find the car again (seeds 0 to 4)?
   Then try `--kidnap 40 0 30`. Then add `--random 20` and `--random 100` to
   `--kidnap 40 0 10`: do random particles help?
   *Reference (seeds 0 to 4, seconds after the jump until the error is first
   under 1 m):*

   | Run | Seeds 0 to 4 |
   |---|---|
   | `--kidnap 40 0 10` | 21.6, 16.6, 16.6, 45.1, 10.6 |
   | `--kidnap 40 0 30` | never, 72.1, never, never, never |
   | `--kidnap 40 0 10 --random 20` | 4.6, never, 21.1, 6.1, never |
   | `--kidnap 40 0 10 --random 100` | 3.6, 11.1, 2.6, 1.6, 10.1 |

   A 10 m jump heals by itself: the particles spread while the car drives,
   and the scans pull them back. A 30 m jump mostly does not. Random
   particles near the estimate can speed the return, but with 20 per scan two
   runs never return: a random particle that fits a wrong place moves the
   estimate there, the next random particles land around it, and the estimate
   wanders off (seed 1 ends 29.8 m off, seed 4 73.4 m). After 50 s their
   median spread is 0.38 and 0.28 m while their median error is 101 and 64 m:
   a small spread does not mean MCL is right.
4. In `--part icp`, raise `--icp-start` until ICP stops landing on the walls.
   How far off can it start?
   *Reference (middle scan):* `1.5 -1 5`, `3 -2 10` and `4 -3 15` land within
   0.10 m of the truth; `6 -4 20` ends 7.09 m off.

## SLAM

`slam/` holds the code behind the two reading pages on SLAM: graph SLAM on
this drive's scans, and visual SLAM on the same route recorded with a stereo
camera. See `slam/README.md`.

## The AV drawing

`av_top.png` is the course's top view of the AV, the same drawing as on the
slides, front up. `l6common.draw_av` puts its base link (the middle of the rear
axle, 0.9 m in from the rear bumper) on the AV's position and turns it to the
AV's heading.

