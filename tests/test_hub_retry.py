"""The Hub's passing faults are waited out: the rate limit (429), server faults and dropped connections are retried with
growing waits; a missing file is not; the lane's polls, its markers and its uploads go through the same waits."""
import pytest

from qwen_symbolic_thinking import hub, queue


class FakeResponse:
    def __init__(self, status, retry_after=None):
        self.status_code = status
        self.headers = {} if retry_after is None else {"Retry-After": str(retry_after)}


class HubFault(Exception):
    def __init__(self, status, retry_after=None):
        super().__init__(f"status {status}")
        self.response = FakeResponse(status, retry_after)


def flaky(faults, value="ok"):
    """A call that raises the given faults in turn, then returns value; .calls counts the calls."""
    state = {"calls": 0}

    def call():
        state["calls"] += 1
        if faults:
            raise faults.pop(0)
        return value
    call.state = state
    return call


def test_transient_reads_the_status_and_the_client():
    assert hub.transient(HubFault(429)) and hub.transient(HubFault(503))
    assert not hub.transient(HubFault(404)) and not hub.transient(HubFault(401))
    assert hub.transient(ConnectionError("reset")) and hub.transient(TimeoutError("slow"))
    assert not hub.transient(ValueError("bad"))
    client_error = type("ReadTimeout", (Exception,), {"__module__": "httpx"})
    assert hub.transient(client_error("timed out"))


def test_the_waits_grow_to_a_cap_and_honor_the_server():
    assert [hub.wait_for(k) for k in range(6)] == [15, 30, 60, 120, 120, 120]
    assert hub.wait_for(0, HubFault(429, retry_after=45)) == 45      # the server's ask wins when it is longer
    assert hub.wait_for(3, HubFault(429, retry_after=19)) == 120
    assert sum(hub.wait_for(k) for k in range(7)) > 300                # the default tries span the 300 s window


def test_with_retries_waits_out_a_rate_limit_then_returns():
    waits = []
    call = flaky([HubFault(429, retry_after=19), HubFault(502)])
    assert hub.with_retries(call, "test", sleep=waits.append) == "ok"
    assert call.state["calls"] == 3 and waits == [19, 30]


def test_with_retries_raises_a_missing_file_at_once_and_the_last_fault_after_its_tries():
    waits = []
    with pytest.raises(HubFault):
        hub.with_retries(flaky([HubFault(404)]), "test", sleep=waits.append)
    assert waits == []
    call = flaky([HubFault(429)] * 5)
    with pytest.raises(HubFault):
        hub.with_retries(call, "test", tries=3, sleep=waits.append)
    assert call.state["calls"] == 3 and len(waits) == 2


def test_ship_retries_then_returns_the_url_or_gives_up(monkeypatch, tmp_path):
    import huggingface_hub
    outcomes = []

    class FakeApi:
        def __init__(self, token=None):
            pass

        def upload_folder(self, **kw):
            o = outcomes.pop(0)
            if isinstance(o, Exception):
                raise o
            return type("Info", (), {"commit_url": o})()

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    waits = []
    outcomes[:] = [HubFault(429), HubFault(500), "https://hub/commit/1"]
    assert hub.ship(str(tmp_path), "x", "m", repo="o/r", sleep=waits.append) == "https://hub/commit/1"
    assert waits == [15, 30]
    waits.clear()
    outcomes[:] = [HubFault(429)] * 3
    assert hub.ship(str(tmp_path), "x", "m", repo="o/r", retries=3, sleep=waits.append) is None
    assert len(waits) == 2 and not outcomes


def test_the_repo_listing_waits_out_a_rate_limit(monkeypatch):
    import huggingface_hub
    faults = [HubFault(429, retry_after=1)]

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo):
            if faults:
                raise faults.pop(0)
            return ["runs/done/a.json", "sft/x.json"]

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    monkeypatch.setattr(hub.time, "sleep", lambda s: None)
    assert queue._repo_files("o/r") == {"runs/done/a.json", "sft/x.json"}
    assert not faults


def test_a_marker_that_cannot_be_written_never_stops_the_lane(monkeypatch, tmp_path):
    import huggingface_hub

    class FakeApi:
        def __init__(self, token=None):
            pass

        def upload_file(self, **kw):
            raise HubFault(403)

    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    queue._mark("run_a", {"status": "done"}, "o/r", str(tmp_path))      # no exception
    assert (tmp_path / "run_a.json").exists()


def test_the_lane_keeps_polling_through_a_hub_that_does_not_answer(monkeypatch):
    calls = {"n": 0}

    def files(repo):
        calls["n"] += 1
        if calls["n"] == 1:
            raise HubFault(429)
        return set()

    monkeypatch.setattr(queue, "_fresh", lambda path, repo: [])
    monkeypatch.setattr(queue, "_repo_files", files)
    monkeypatch.setattr(queue.time, "sleep", lambda s: None)
    assert queue.run_queue("lane_t", repo="o/r", idle_minutes=0, poll_s=0) == []
    assert calls["n"] >= 2
