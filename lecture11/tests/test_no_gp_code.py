"""Guard: nothing public in lecture11 may solve a group project (GP3 or GP4)."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FORBIDDEN = [r"heapq", r"astar", r"a_star", r"dijkstra\s*\(", r"pure_pursuit", r"lookahead_point",
             r"behaviorfsm", r"lane_follow", r"emergency_stop", r"\brclpy\b", r"jacobian"]


def test_no_gp_code_in_public_files():
    hits = []
    for f in HERE.rglob("*"):
        if f.suffix not in {".py", ".md", ".json"} or "tests" in f.parts:
            continue
        text = f.read_text(errors="ignore").lower()
        for pat in FORBIDDEN:
            if re.search(pat, text):
                hits.append(f"{f.name}: {pat}")
    assert not hits, hits
