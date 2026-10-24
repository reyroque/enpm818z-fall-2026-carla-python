import pytest


def test_rerun_search_writes_a_recording(tmp_path):
    pytest.importorskip("rerun")
    import rerun_search
    out = tmp_path / "search.rrd"
    rerun_search.main(["--save", str(out)])
    assert out.exists() and out.stat().st_size > 10000
