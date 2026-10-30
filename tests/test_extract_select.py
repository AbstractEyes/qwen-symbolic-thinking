"""The extraction's row selection and the picture bound (no network, no model)."""
from PIL import Image

from qwen_symbolic_thinking import extract


def _rows():
    out = []
    for i, (src, split) in enumerate([("coco", "pilot"), ("coco", "main"), ("coco", "main"), ("nga", "main"), ("trdg_words", "main"), ("docci", "main")]):
        out.append({"sample_id": f"qti-{i:06d}", "source": src, "split": split, "image": {"kind": "url", "urls": ["http://example.invalid/x.jpg"]}})
    out.append({"sample_id": "qti-000099", "source": "met", "split": "main", "image": {"kind": "pending", "resolver": "met", "id": "1"}})
    return out


def test_select_rows_filters_in_order():
    rows = _rows()
    assert [r["sample_id"] for r in extract.select_rows(rows)] == [f"qti-{i:06d}" for i in range(6)]        # the pending row is out
    assert len(extract.select_rows(rows, split="main")) == 5
    assert {r["source"] for r in extract.select_rows(rows, split="main", exclude=("trdg_words", "docci"))} == {"coco", "nga"}
    assert [r["source"] for r in extract.select_rows(rows, sources=("nga",))] == ["nga"]
    assert len(extract.select_rows(rows, per_source=1)) == 4
    assert len(extract.select_rows(rows, n=2)) == 2


def test_bounded_caps_the_longer_side():
    im = Image.new("RGB", (4000, 3000))
    assert max(extract.bounded(im).size) == 1024 and extract.bounded(im).size[0] == 1024
    small = Image.new("RGB", (640, 480))
    assert extract.bounded(small).size == (640, 480)
    tall = Image.new("RGB", (800, 2600))
    assert extract.bounded(tall, 1300).size == (400, 1300)
