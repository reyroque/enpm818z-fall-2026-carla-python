"""Guard: nothing public in lecture6 may solve GP3 Task 2 (or Task 1)."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
FORBIDDEN = [r"jacobian", r"\blat(itude)?\b", r"\blon(gitude)?\b", r"navsatfix", r"\brclpy\b",
             r"geodetic", r"project_to_image", r"frustum"]


def test_no_gp3_code_in_public_files():
    hits = []
    for f in HERE.rglob("*"):
        if f.suffix not in {".py", ".md", ".json"} or "tests" in f.parts:
            continue
        text = f.read_text(errors="ignore").lower()
        for pat in FORBIDDEN:
            if re.search(pat, text):
                hits.append(f"{f.name}: {pat}")
    assert not hits, hits
