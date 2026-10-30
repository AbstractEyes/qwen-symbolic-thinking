import json
import os

from qwen_symbolic_thinking import data


def _row(i, depth, seed_id=None):
    src = {"kind": "seeded", "seed_dataset": seed_id.split(":")[0], "seed_id": seed_id, "license": "CC BY 4.0"} if seed_id \
        else {"kind": "synthetic", "license": "Apache-2.0"}
    return {"id": f"art-{i:06d}", "depth": depth, "domain": "photograph", "genre": "portrait",
            "spec": {"person": {"count": "1"}}, "thoughts": [{"step": "see", "text": "One person."}],
            "wordings": {"caption": "One person."}, "source": src}


def _fixture(tmp_path):
    """24 picture rows (12 simple coco, 12 medium narratives), 12 drawn medium, 12 drawn deep; seed shards with hints."""
    rows_dir = tmp_path / "prod"
    rows_dir.mkdir()
    rows = []
    for i in range(12):
        rows.append(_row(i, "simple", f"coco:{i}"))
        rows.append(_row(100 + i, "medium", f"localized_narratives:{100 + i}"))
        rows.append(_row(200 + i, "medium"))
        rows.append(_row(300 + i, "deep"))
    with open(rows_dir / "batch_001.rows.jsonl", "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(r) + "\n" for r in rows)
    (tmp_path / "ACCEPTED.txt").write_text("prod 1\n", encoding="ascii")
    seeds = tmp_path / "seeds"
    for src in ("coco", "localized_narratives"):
        (seeds / src).mkdir(parents=True)
        with open(seeds / src / "x.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                sid = r["source"].get("seed_id") or ""
                if sid.startswith(src):
                    fh.write(json.dumps({"seed_id": sid, "hint": f"hint for {sid}"}) + "\n")
    return str(tmp_path / "ACCEPTED.txt"), str(seeds)


def test_image_urls_follow_the_coco_layout():
    assert data.image_urls("coco:397133") == ["http://images.cocodataset.org/val2017/000000397133.jpg",
                                              "http://images.cocodataset.org/train2017/000000397133.jpg"]


def test_build_screen_draws_both_kinds_and_examples(tmp_path):
    accepted, seeds = _fixture(tmp_path)
    out = data.build_screen(accepted, seeds, str(tmp_path / "out" / "screen.json"), n_images=8, n_specs=6, n_examples=2)
    kinds = [(it["kind"], it["depth"]) for it in out["items"]]
    assert kinds.count(("image", "simple")) == 4 and kinds.count(("image", "medium")) == 4
    assert kinds.count(("spec", "medium")) == 3 and kinds.count(("spec", "deep")) == 3
    assert all(it["hint"].startswith("hint for") and it["image_urls"] for it in out["items"] if it["kind"] == "image")
    assert all(not it["image_urls"] and it["seed_id"] is None for it in out["items"] if it["kind"] == "spec")
    assert set(out["examples"]) == {"simple", "medium"} and all(len(v) == 2 for v in out["examples"].values())
    picked = {it["id"] for it in out["items"]}
    assert not any(ex["id"] in picked for v in out["examples"].values() for ex in v)
    assert os.path.exists(tmp_path / "out" / "screen.json")


def test_build_sft_holds_out_the_screen_and_balances_the_depths(tmp_path):
    accepted, seeds = _fixture(tmp_path)
    screen_path = str(tmp_path / "out" / "screen.json")
    data.build_screen(accepted, seeds, screen_path, n_images=8, n_specs=6, n_examples=2)
    out = data.build_sft(accepted, seeds, screen_path, str(tmp_path / "out" / "sft.json"))
    held = set(out["held_out"])
    assert len(held) == 8 + 6 + 4
    assert not {it["id"] for it in out["items"]} & held
    pics = [it for it in out["items"] if it["kind"] == "image"]
    specs = [it for it in out["items"] if it["kind"] == "spec"]
    assert len(pics) == 24 - 8 - 4 and all(it["hint"] for it in pics)      # every picture row outside the screen and its examples
    depths = [it["depth"] for it in specs]
    assert depths.count("deep") == 6 and depths.count("medium") == 3 and "simple" not in depths   # 12 wanted 3:3:6; no drawn simple rows
    assert all(not it["image_urls"] for it in specs)
