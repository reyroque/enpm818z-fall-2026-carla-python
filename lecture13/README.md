# L13 hands-on: how long does the pipeline take to brake?

A timing model of the course pipeline (GP1 to GP4) and five views of it. Needs
`numpy` and `matplotlib`; no CARLA, no ROS, no GPU.

```bash
python3 l13_handson.py --part chain     # 1. one camera frame, step by step, through every node
python3 l13_handson.py --part launches  # 2. many launches: how the reaction time spreads
python3 l13_handson.py --part queue     # 3. a detector slower than the camera, queue depth 1 to 10
python3 l13_handson.py --part watchdog  # 4. the detector hangs: a timeout notices it
python3 l13_handson.py --part stop      # 5. reaction time and braking, in meters
```

Add `--snapshot out.png` to save a still instead of opening a window, or
`--html out.html` for a page (needs `mpld3`). `--times file.json` uses your own
detector times (see below).

## What the model is

`pipeline_sim.py` follows one camera frame at a time through the graph and records
when each node handles it. It models only times: no images, no boxes, no driving.

| Node | Runs | Time per call |
|---|---|---|
| `sensor_manager` | every 50 ms (GP1's `fixed_delta_seconds`); each tick captures a frame and applies the newest command | |
| `detector_node` | when a frame arrives; one at a time; its queue keeps the newest `depth` frames | **measured** (`detector_times.json`) |
| `fusion_node` | when detections arrive | 4 ms (chosen) |
| `behavior_node` | timer, 10 Hz (GP4: "e.g. 10 Hz") | 1 ms (chosen) |
| `controller_node` | timer, 20 Hz (chosen: GP4 gives no rate) | 1 ms (chosen) |
| `carla_vehicle_control` | when a command arrives | 1 ms (chosen) |

The camera takes 3 ms to deliver a frame (chosen). Each timer starts at a phase drawn
at random per launch, because a node starts when it starts.

**Reaction time** is the time from the stamp of the camera frame that first shows
something to the world tick at which CARLA applies a command based on that frame or a
newer one.

The model assumes things your pipeline may break: CARLA keeps up with its 50 ms tick,
each node has its own thread, and timers fire exactly once per period. The ROS 2 docs
do not promise the last one. So measure your own chain from a bag: copy the camera's
stamp into every message down the chain, and for each recorded message take the time
it was recorded minus its stamp.

## What each part shows

1. **chain**: one frame (YOLOv8s on the GPU at its median) through every node. Solid
   bars are computing, hatched bars are waiting. Move the two timer phases and pick a
   detector.
2. **launches**: the share of frames for each reaction time, over many launches with
   random timer phases. The longest value matches the worst case by hand: all the
   computing, plus one period of each timer, rounded up to the next tick.
3. **queue**: YOLOv8s on the CPU (about 75 ms per frame) while frames arrive every
   50 ms. Drag the queue depth: with 10, every detection is about half a second old;
   with 1, about a tenth of a second.
4. **watchdog**: the detector hangs at 20 s. The behavior node checks the age of the
   newest fused objects (now minus their stamp) at each tick. Drag the timeout: the
   title gives the false alarms in normal driving and how long the hang goes unnoticed.
5. **stop**: the distance from the frame to standing still, `v * t + v^2 / (2a)`,
   against a pedestrian ahead. Every slider value is yours to choose.

## The detector times

`detector_times.json` holds every call time measured on 2026-10-14 by
`measure_detector.py`: YOLOv8s and RT-DETR-L on an RTX 4060 Laptop GPU (600 calls
each) and YOLOv8s on 4 CPU threads (200 calls), on one 1280x720 CARLA frame, after 10
warm-up calls. Other jobs were running on the laptop (load average about 10).

To measure your own (needs `ultralytics` and the two GP2 models):

```bash
python3 measure_detector.py --image frame.png --weights-dir <folder with yolov8s.pt and rtdetr-l.pt> --out my_times.json
python3 l13_handson.py --part launches --times my_times.json
```

## Tests

```bash
python3 -m pytest -q tests
```

The tests check the queue (keep last drops the oldest), the slides' worked chain
(100 ms), the hand bound, the hang and the timeout, and that nothing here is GP code.
