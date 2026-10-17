"""Guard: nothing public in lecture9 may solve GP4 Task 1, 2 or 3."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FORBIDDEN = [r"heapq", r"astar", r"dijkstra", r"priority.?queue", r"pure.?pursuit", r"behaviorfsm",
             r"lane_follow", r"obstacle_avoidance", r"emergency_stop", r"behavior_node", r"\brclpy\b",
             r"traffic.?light"]


def test_no_gp4_code_in_public_files():
    hits = []
    for f in HERE.rglob("*"):
        if f.suffix not in {".py", ".md", ".json"} or "tests" in f.parts:
            continue
        text = f.read_text(errors="ignore").lower()
        for pat in FORBIDDEN:
            if re.search(pat, text):
                hits.append(f"{f.name}: {pat}")
    assert not hits, hits
