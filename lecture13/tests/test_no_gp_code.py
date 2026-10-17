"""Guard: nothing public in lecture13 may solve a group project (GP3, GP4) or be a ROS node."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FORBIDDEN = [r"\bheapq\b", r"astar", r"dijkstra\(", r"pure_pursuit", r"BehaviorFSM", r"LANE_FOLLOW",
             r"\brclpy\b", r"jacobian", r"create_subscription", r"create_publisher"]


def test_no_gp_code_in_public_files():
    hits = []
    for f in HERE.rglob("*"):
        if f.suffix not in {".py", ".md"} or "tests" in f.parts:
            continue
        text = f.read_text(errors="ignore")
        for pat in FORBIDDEN:
            if re.search(pat, text, flags=re.IGNORECASE):
                hits.append(f"{f.name}: {pat}")
    assert not hits, hits
