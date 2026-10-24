# L7 hands-on: one Town01 trip, from the map file to a route

Five parts on one trip in CARLA's Town01. Needs `numpy` and `matplotlib`; no CARLA
server, no ROS, no GPU. The `map` part also needs CARLA's Python package,
`pip install carla==0.9.16`, which reads a map file without a server.

```bash
python3 l7_handson.py --part map      # 0. Town01's map file: zoom, search, click
python3 l7_handson.py --part graph    # 1. Town01's waypoint graph; scroll to zoom, drag to pan
python3 l7_handson.py --part costs    # 2. every edge's three costs, and the trip's route under each
python3 l7_handson.py --part search   # 3. Dijkstra and A*, replayed with a slider
python3 l7_handson.py --part replan   # 4. an obstacle on the route, and the new route
```

Add `--snapshot out.png` to save a still instead of opening a window, or
`--html out.html` for a web page (needs `pip install mpld3`).

The code on the slide "Read the map file yourself" is `read_map.py`. It prints the
slide's three counts (3266 waypoints, 160 topology segments, 36 traffic lights). Like
the `map` part, it needs CARLA's Python package and no server:

```bash
python3 read_map.py
```

`osm_to_map.py` turns an OpenStreetMap export into a CARLA map, the code on the slide
"From your .osm file to a map, in Python". With no argument it converts
`maps/umd_engineering.osm`, the campus around the Kim building (map data © OpenStreetMap
contributors, ODbL), and prints 314 roads, 46 intersections and 9144 waypoints at 2 m. Give
it your own export instead: openstreetmap.org saves it as `map.osm`, so run
`python3 osm_to_map.py ~/Downloads/map.osm` with the folder your browser saved it in. It centers the projection
on the file's own area and writes `my_area.xodr`, which you read with `carla.Map` as
Town01. The converter prints many warnings about map features it skips; they are normal.

**This folder has no search code.** The routes and the order in which Dijkstra and A*
took nodes from their open sets were computed in advance by a reference planner that
is not public, and saved in `l7_results/`. The script replays and draws them. GP4
Task 1 asks you to write A* on CARLA's waypoint graph yourself.

## The trip

- **Start:** on the road between intersections 278 and 306 (CARLA junction ids),
  40 m east of 306, driving west: node `start_node` in `l7_results/summary.json`.
- **Goal:** on the road between intersections 54 and 26, 40 m east of 54, driving
  east: node `goal_node`.
- 141.9 m apart in a straight line; the shortest route is 260.6 m.

## What each part shows

0. **map**: reads `maps/Town01.xodr`, CARLA's map file for Town01, with
   `carla.Map("Town01", text)` and no server. It prints what the file holds (122 roads,
   96 of them inside the 12 intersections, 36 traffic lights) and what CARLA makes of
   it (3266 waypoints at 2 m, 160 topology segments), and draws every waypoint. In the
   window:
   - **zoom** with the mouse wheel (at the cursor) or the Zoom in, Zoom out and Whole map
     buttons;
   - **search** in the box, then press Enter: `road 25` (or just `25`), `lane 25 -1`,
     `junction 332`, `segment 12` (one of the 160 topology segments, 0 to 159) or
     `waypoint 100`. The match turns pink, the view zooms to it, and the line below says
     what the file holds about it: a road's length, its ends and the line of
     `Town01.xodr` where it starts;
   - **x, y**: type a position in meters, x east and y north (`133.6, -129.4` is our
     start), then press Enter. CARLA's `Map.get_waypoint` finds the nearest lane point and
     the window prints its road, lane and distance along the road `s`. Use this rather
     than a click to get a position exactly: the drawn waypoints are 2 m apart;
   - **click** a waypoint to see its road, lane and distance along the road.

   Open `maps/Town01.xodr` in a text editor at the line it names.
1. **graph**: the real graph, 3107 nodes and 3143 edges. Seen whole, the nodes are 2 m
   apart and too close to tell apart, so the window shows the lanes in gray and, in
   amber, the 36 nodes with two successors: where a lane enters an intersection and the
   AV can go two ways. Scroll to zoom in at the mouse, and drag with the left button to
   pan: once the view is under 120 m across, every node is a dot and every edge an arrow,
   pointing the way the AV drives. A split's two successors usually sit at one point,
   because the two ways out of an intersection leave along the same line and part a few
   meters on: an amber ring marks every point that holds two nodes. Click a node (press
   and release without moving) to print its position, heading, road, lane and
   successors; for a split, also how far on its two ways part. A click on a ring lists
   both of its nodes, one for each way, and the split they follow.
2. **costs**: on the left, every edge of Town01, darker where one meter takes longer (the
   AV slows in a turn), with the 24 left-turn edges ringed. Scroll to zoom, drag to pan. Click an edge to print its
   three costs: its length, its travel time (length divided by speed), and its length
   plus 30 m if it is a left turn. Where a lane splits, two edges leave the same node
   along the same first meters; the click lists both. On the right, the same start and
   goal searched with each cost. Distance gives the shortest
   route (260.6 m, 4 turns). Travel time, with the AV slowed in each turn, gives a
   route 6.3 m longer with 2 turns and 3.3 s faster. Distance plus a penalty for each
   left turn gives the same route as travel time.
3. **search**: the nodes each search took from its open set, in order. Drag the
   slider to replay both searches side by side. Both find the same route; Dijkstra
   took 543 nodes, A* 196.
4. **replan**: an obstacle on the route's road north of intersection 332 (chosen),
   seen when the AV is 23 m before intersection 171. The edges within 3.0 m of it are
   closed (GP4's `obstacle_clearance`), and the route from the AV is planned again.

## Optional: the search in Rerun

`rerun_search.py` replays the same two searches as `--part search` in the
[Rerun](https://rerun.io) viewer, if you want to try it. Rerun is not needed for the
lecture: its package and viewer take several hundred megabytes.

```bash
pip install rerun-sdk
python3 rerun_search.py
```

Drag the time slider at the bottom, or press play, to watch both searches take their
nodes; the route appears when a search reaches G. Scroll to zoom and drag to pan in each
view. The plot under the maps shows how many nodes each search has taken by each step.
Rerun's 2D views point y down, so the script flips y: north stays up, and the axis shows
130 where the map has y = -130 m. `python3 rerun_search.py --save search.rrd` writes a
recording instead, which `rerun search.rrd` opens.

## The data

`maps/Town01.xodr` is CARLA 0.9.16's map file for Town01, unchanged. CARLA's assets
are distributed under the CC-BY license (CARLA's README); the file is from the CARLA
project, https://github.com/carla-simulator/carla.

`l7_town01_graph.npz` holds Town01's waypoint graph, read offline from `Town01.xodr`
with `carla.Map(name, xodr_text)`: a node every 2 m along every lane, sampled the way
CARLA's `GlobalRoutePlanner` samples it, and a directed edge from each node to the
next one along its lane.

| array | meaning |
|---|---|
| `node_xy` | position (m): x east, y north (CARLA's y flipped, as in L6) |
| `node_yaw_deg` | heading (degrees, counterclockwise from east) |
| `node_road`, `node_lane` | OpenDRIVE road and lane id |
| `node_junction`, `node_junction_id` | inside an intersection, and which one (-1: none) |
| `node_speed_limit` | m/s from the road's `<speed>` in the file; NaN inside intersections |
| `edge_from`, `edge_to`, `edge_len` | directed edges and their lengths (m) |
| `edge_lane_change` | a lane change edge (none in Town01: one lane each way) |
| `seg_turn` | for each topology segment through an intersection: `L`, `R` or `S` |

`l7_results/routes.npz` holds the routes (node indices), the two search orders, the
re-planned route, and the route's reference path with saved speeds (`ref_s`,
`ref_kappa`, `ref_cap`, `ref_fwd`, `ref_v`), which the lecture's route, path and
trajectory figure uses. `l7_results/summary.json` holds every number, the values we chose
(`chosen`), and the intersection graph the slides search by hand.

**Values we chose:** a left-turn penalty of 30 m and the obstacle's position; for the
saved speeds, lateral acceleration in a curve at most 2.0 m/s^2, acceleration
1.5 m/s^2, braking 2.0 m/s^2.
GP4's values: `obstacle_clearance` 3.0 m, `max_deviation` 2.0 m, `replan_frequency`
2 Hz.

## Tests

```bash
python3 -m pytest -q tests
```
