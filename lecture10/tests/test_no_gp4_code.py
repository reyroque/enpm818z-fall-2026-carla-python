"""Guard: nothing public in lecture10 may solve GP4 Task 2 (Pure Pursuit, the PID step).

The controller runs are precomputed by a private program; this folder only reads them.
"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FORBIDDEN = [r"pure_pursuit", r"def\s+compute", r"pidcontroller", r"\bintegral\s*\+=",
             r"_integral", r"sin\(\s*alpha", r"np\.sin\(a\)\s*/", r"lookahead_point",
             r"def\s+stanley", r"arctan2?\(\s*k\s*\*", r"\brclpy\b", r"astar", r"heapq"]


def test_no_gp4_controller_code_in_public_files():
    hits = []
    for f in HERE.rglob("*"):
        if f.suffix not in {".py", ".md", ".json"} or "tests" in f.parts:
            continue
        text = f.read_text(errors="ignore").lower()
        for pat in FORBIDDEN:
            if re.search(pat, text):
                hits.append(f"{f.name}: {pat}")
    assert not hits, hits
