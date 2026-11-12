"""tools.filter_anchors on a tiny general-row file: pictures of a named source keep only under an allowed own license, every other
item passes unchanged, and the summary counts both. No network."""
import json

from qwen_symbolic_thinking.tools import filter_anchors

ITEMS = [
    {"kind": "anchor", "id": "a1", "category": "picture-train", "source": "coco", "seed_id": "coco:1", "answer": "A dog."},
    {"kind": "anchor", "id": "a2", "category": "picture-train", "source": "coco", "seed_id": "coco:2", "answer": "A cat."},
    {"kind": "anchor", "id": "a3", "category": "picture-train", "source": "coco", "seed_id": "coco:3", "answer": "A bus."},
    {"kind": "anchor", "id": "a4", "category": "picture-train", "source": "localized_narratives",
     "seed_id": "localized_narratives:4", "answer": "Two boats."},
    {"kind": "anchor", "id": "a5", "category": "picture-train", "source": "docci", "seed_id": "docci:d1", "answer": "A tree."},
    {"kind": "anchor", "id": "a6", "category": "text-train", "source": None, "seed_id": None, "answer": "Paris."},
]
LICENSES = {"coco:1": "Attribution License", "coco:2": "Attribution-NonCommercial License",
            "localized_narratives:4": "Attribution License"}


def write(tmp_path):
    path = tmp_path / "anchors.json"
    path.write_text(json.dumps({"version": 1, "model": "m", "tag": "g", "split": "train", "items": ITEMS}), encoding="utf-8")
    return str(path)


def test_only_allowed_pictures_of_named_sources_stay(tmp_path):
    out_dir = tmp_path / "out"
    summary = filter_anchors.run(write(tmp_path), tag="g_by", own_license_sources=["coco", "localized_narratives"],
                                 picture_licenses=LICENSES, licenses=["Attribution License"], do_ship=False,
                                 out_dir=str(out_dir))
    kept = json.loads((out_dir / "anchors.json").read_text(encoding="utf-8"))
    assert [it["id"] for it in kept["items"]] == ["a1", "a4", "a5", "a6"]
    assert kept["items"][0] == ITEMS[0] and kept["model"] == "m" and kept["split"] == "train" and kept["tag"] == "g_by"
    assert summary["kept"] == 4 and summary["items"] == 6
    assert summary["dropped"] == {"coco: license": 1, "coco: picture's own license not given": 1}
    assert summary["kept_licenses"] == {"Attribution License": 2}
    assert summary["kept_by_category"] == {"picture-train": 3, "text-train": 1}
    assert json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))["kept"] == 4


def test_no_map_drops_every_picture_of_the_named_sources(tmp_path):
    summary = filter_anchors.run(write(tmp_path), tag="g_none", own_license_sources=["coco", "localized_narratives"],
                                 do_ship=False, out_dir=str(tmp_path / "o2"))
    assert summary["kept"] == 2 and summary["kept_by_source"] == {"docci": 1, "None": 1}
