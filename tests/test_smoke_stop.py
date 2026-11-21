"""The smoke that stops itself (queue: "stop_if" and "waits_for_runs"): a training read against its limits after it ends (its
steady pace from its own step log, its peak), its marker failed with the reason in plain words, and the runs that wait for it
held back while the rest of the lane goes on. The training is stubbed and the Hub is a dict; no GPU, no network."""
import copy
import json
import sys
import types

import pytest

from qwen_symbolic_thinking import queue, train

# the 2B v1.2 smoke's own numbers: 70 s at step 10 (the warm-up), 129 s at step 30: 2.95 s a step over steps 11-30
LOG = [{"step": 10, "spent_s": 70, "loss": 1.5}, {"step": 20, "spent_s": 99, "loss": 1.3}, {"step": 30, "spent_s": 129, "loss": 1.2}]
LIMITS = {"steady_s_per_step": 3.98, "peak_gb": 85, "after_step": 10}


def test_the_steady_pace_is_read_after_the_warm_up():
    assert queue.steady_pace(LOG, 10) == {"from_step": 10, "to_step": 30, "s_per_step": 2.95}
    assert queue.steady_pace(LOG, 20)["s_per_step"] == 3.0
    assert queue.steady_pace(LOG, 0) == {"from_step": 10, "to_step": 30, "s_per_step": 2.95}   # no record at step 0: the first one
    assert queue.steady_pace(LOG[:1], 10) is None and queue.steady_pace([], 10) is None
    assert queue.steady_pace(LOG, 30) is None                      # nothing logged past step 30


def test_the_verdict_against_the_limits():
    ok = queue.smoke_verdict(LOG, 33.3, LIMITS)
    assert ok["passed"] and ok["reason"] is None and ok["steady_s_per_step"] == 2.95 and ok["peak_gb"] == 33.3
    slow = queue.smoke_verdict(LOG, 33.3, {**LIMITS, "steady_s_per_step": 2.5})
    assert not slow["passed"] and "2.95 s a step (steps 11-30)" in slow["reason"] and "limit of 2.5 s a step" in slow["reason"]
    heavy = queue.smoke_verdict(LOG, 90.2, LIMITS)
    assert not heavy["passed"] and "90.2 GB, above its limit of 85 GB" in heavy["reason"]
    both = queue.smoke_verdict(LOG, 90.2, {**LIMITS, "steady_s_per_step": 2.5})
    assert "steady pace" in both["reason"] and "peak GPU memory" in both["reason"] and "do not start" in both["reason"]
    blind = queue.smoke_verdict(LOG[:1], 33.3, LIMITS)
    assert not blind["passed"] and "could not be read" in blind["reason"]
    assert queue.smoke_verdict(LOG, None, LIMITS)["passed"]       # no GPU, no peak: not held against it
    assert queue.smoke_verdict(LOG, 99.0, {"steady_s_per_step": 3.98, "after_step": 10})["passed"]   # no peak limit given


def _summary(log=LOG, peak=33.3):
    return {"steps": log[-1]["step"] if log else 0, "peak_gpu_gb": peak, "path_in_repo": "lora/x/r32/20261121_000000_S",
            "adapter_dir": "/tmp/a", "log": list(log), "env": {"gpu": "none"}}


def _smoke(name="S_smoke", limit=3.98):
    return {"name": name, "kind": "train", "train": {"model_id": "m", "max_steps": 30, "do_ship": False}, "screen": False,
            "stop_if": {**LIMITS, "steady_s_per_step": limit}}


def test_run_spec_fails_a_smoke_past_its_limits_and_keeps_its_summary(monkeypatch):
    monkeypatch.setattr(train, "run", lambda **kw: _summary())
    marks = []
    monkeypatch.setattr(queue, "_mark", lambda name, result, repo, out_dir: marks.append((name, result["status"])))
    bad = queue.run_spec(_smoke(limit=2.5), "lane", "owner/repo", "/tmp/runs")
    assert bad["status"] == "failed" and bad["error"] == bad["stop"]["reason"] and "2.95 s a step" in bad["error"]
    assert bad["train"]["steps"] == 30 and "log" not in bad["train"] and bad["stop"]["limits"]["steady_s_per_step"] == 2.5
    assert bad["finished"] and bad.get("screen") is None
    good = queue.run_spec(_smoke(), "lane", "owner/repo", "/tmp/runs")
    assert good["status"] == "done" and good["stop"]["passed"] and good["screen"] is None
    plain = queue.run_spec({**_smoke(), "stop_if": None}, "lane", "owner/repo", "/tmp/runs")
    assert plain["status"] == "done" and "stop" not in plain                 # a spec without stop_if: as before
    assert marks == [("S_smoke", "started")] * 3


class Hub:
    """The project repo as a dict: the lane's queue, the markers, other files."""

    def __init__(self, lane: str, specs: list, markers: dict | None = None, files=()):
        self.lane, self.specs, self.markers, self.files = lane, specs, dict(markers or {}), set(files)

    def fresh(self, path, repo):
        if path == f"runs/queue_{self.lane}.json":
            return copy.deepcopy(self.specs)
        if path.startswith("runs/done/"):
            return copy.deepcopy(self.markers.get(path[len("runs/done/"):-len(".json")]))
        return None

    def repo_files(self, repo):
        return {f"runs/done/{n}.json" for n in self.markers} | self.files

    def mark(self, name, payload, repo, out_dir):
        self.markers[name] = json.loads(json.dumps(payload, default=str))


@pytest.fixture
def lane(monkeypatch):
    """A lane on an in-memory Hub: run(specs, markers, summaries) -> (the names run, the Hub, the trainings run). summaries:
    tag -> the stubbed training's summary (the default: the 2B smoke's numbers)."""
    fake = types.ModuleType("qwen_symbolic_thinking.tools.fake_read")
    fake.run = lambda **kw: {"ok": True}
    monkeypatch.setitem(sys.modules, "qwen_symbolic_thinking.tools.fake_read", fake)
    monkeypatch.setattr(queue, "runnable_here", lambda spec: True)

    def run(specs, markers=None, summaries=None, raises=(), wait_hours=12.0):
        hub, trained = Hub("L", specs, markers), []

        def fake_train(**kw):
            trained.append(kw["tag"])
            if kw["tag"] in raises:
                raise RuntimeError(f"{kw['tag']} broke")
            return _summary(**(summaries or {}).get(kw["tag"], {}))
        monkeypatch.setattr(train, "run", fake_train)
        monkeypatch.setattr(queue, "_fresh", hub.fresh)
        monkeypatch.setattr(queue, "_repo_files", hub.repo_files)
        monkeypatch.setattr(queue, "_mark", hub.mark)
        ran = queue.run_queue("L", "owner/repo", out_dir="/tmp/runs", idle_minutes=0, poll_s=0, wait_hours=wait_hours)
        return ran, hub, trained
    return run


def _training(name="V12_train", waits=("S_smoke",)):
    spec = {"name": name, "kind": "train", "train": {"model_id": "m"}, "screen": False}
    if waits:
        spec["waits_for_runs"] = list(waits)
    return spec


READ = {"name": "X_other", "kind": "tool", "tool": {"module": "fake_read", "args": {}}}


def test_a_failed_smoke_holds_its_training_back_and_the_lane_goes_on(lane, capsys):
    ran, hub, trained = lane([_smoke(limit=2.5), _training(), READ])
    assert ran == ["S_smoke", "X_other"] and trained == ["S_smoke"]
    assert hub.markers["S_smoke"]["status"] == "failed" and "2.95 s a step" in hub.markers["S_smoke"]["error"]
    assert "V12_train" not in hub.markers                              # held back: never started, never marked
    out = capsys.readouterr().out
    assert "FAILED S_smoke: stopped by its limits" in out and "V12_train is held back and does not start" in out


def test_a_passing_smoke_lets_its_training_start(lane):
    ran, hub, trained = lane([_smoke(), _training(), READ])
    assert ran == ["S_smoke", "V12_train", "X_other"] and trained == ["S_smoke", "V12_train"]
    assert hub.markers["S_smoke"]["stop"]["passed"] and hub.markers["V12_train"]["status"] == "done"


def test_a_heavy_smoke_fails_on_its_peak(lane):
    ran, hub, _ = lane([_smoke(), _training()], summaries={"S_smoke": {"peak": 88.0}})
    assert ran == ["S_smoke"] and "88.0 GB, above its limit of 85 GB" in hub.markers["S_smoke"]["error"]


def test_a_smoke_that_failed_on_another_machine_still_holds_and_a_done_one_releases(lane):
    failed = {"S_smoke": {"status": "failed", "error": "stopped by its limits: ..."}}
    ran, hub, trained = lane([_smoke(), _training(), READ], markers=failed)
    assert ran == ["X_other"] and trained == [] and "V12_train" not in hub.markers
    ran, hub, trained = lane([_smoke(), _training()], markers={"S_smoke": {"status": "done"}})
    assert ran == ["V12_train"] and trained == ["V12_train"]


def test_a_training_waits_while_its_smoke_has_not_finished(lane, capsys):
    ran, hub, trained = lane([_training(waits=("S_elsewhere",)), READ], markers={}, wait_hours=0.0)
    assert ran == [] and trained == []                     # it waits (and the lane runs nothing past it), then leaves
    out = capsys.readouterr().out
    assert "V12_train waits for" in out and "S_elsewhere" in out and "none yet" in out
    ran, _, _ = lane([_training(waits=("S_elsewhere",)), READ], markers={"S_elsewhere": {"status": "started"}}, wait_hours=0.0)
    assert ran == []


def test_without_the_new_keys_a_failed_run_does_not_stop_the_next(lane):
    ran, hub, trained = lane([{**_smoke(), "stop_if": None}, _training(waits=()), READ], raises=("S_smoke",))
    assert ran == ["S_smoke", "V12_train", "X_other"] and trained == ["S_smoke", "V12_train"]
    assert hub.markers["S_smoke"]["status"] == "failed" and hub.markers["V12_train"]["status"] == "done"


def test_run_states_read_this_lanes_results_first_and_fetch_only_existing_markers():
    asked = []

    def read(path, repo):
        asked.append(path)
        return {"status": "failed"}
    spec = {"name": "T", "waits_for_runs": ["A", "B", "C"]}
    states = queue.run_states(spec, {"B": "runs/done/B.json", "C": "runs/done/C.json"}, "o/r", {"C": "done"}, read=read)
    assert states == {"A": None, "B": "failed", "C": "done"} and asked == ["runs/done/B.json"]
    assert queue.run_states({"name": "U"}, {}, "o/r") == {}
