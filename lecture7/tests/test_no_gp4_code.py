"""Guard: nothing public in lecture7 may solve GP4 Task 1 (or Tasks 2 and 3)."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FORBIDDEN = [r"\bheapq\b", r"astar", r"dijkstra\s*\(", r"priorityqueue", r"heappush", r"\.next\s*\(",
             r"get_next", r"pure_pursuit", r"behaviorfsm", r"lane_follow", r"\brclpy\b"]


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
