import json
from pathlib import Path

A = json.loads((Path(__file__).resolve().parents[1] / "docs" / "analysis.json").read_text())


def test_leakage_is_visible():
    m = A["models"]
    assert m["with_formula_inputs"]["r2"] > 0.99
    assert m["without_formula_inputs"]["r2"] < 0.1


def test_segments_cover_every_customer():
    assert sum(s["n"] for s in A["segments"]) == A["rows"]
