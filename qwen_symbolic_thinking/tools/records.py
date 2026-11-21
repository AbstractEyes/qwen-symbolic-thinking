"""Where the record-only reads (tools.bars, tools.text_names, tools.keep_text_names, tools.lettering_ask,
tools.quote_precision; the command `python -m qwen_symbolic_thinking.reads`) find what a lane's runs wrote: the project repo
on the Hub, read-only (HubRecords), or a local folder that holds the same paths (FolderRecords: a copy taken earlier, or a
test's small records).

The paths are the project repo's own: runs/done/<run>.json (a run's marker: its status, its spec and what it wrote), an
extraction's rows and summary (extractions/<stamp>_<run>/rows.jsonl, summary.json), a preservation read's rows
(keep/<stamp>_<run>/rows.jsonl) and a screen's results (screens/<stamp>_<run>/results.jsonl). A marker is read fresh once per
Records (it changes while its run goes); every other file once. The Hub's passing faults are waited out (hub.with_retries).
The token is the package's own (hub.token(): HF_TOKEN, or the Colab secret) and is never printed. Nothing is written to any
repo: the downloads go to a folder of their own, removed when the Records closes.

A manifest is named the way extract.load_manifest names one: "repo:<path>" (a file of the project repo),
"hub:<owner>/<name>/<path>" (a file of a public dataset repo, kept in the Hub cache), or a local path."""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

from ..hub import token as hub_token
from ..hub import with_retries


class RecordError(RuntimeError):
    """A record a read needs that is missing, or a run whose marker does not say done."""


class Records:
    """The common part: JSON and JSON-lines files, the runs that have a marker, a run's marker (fresh, once)."""

    def __init__(self, repo: str | None = None):
        self.repo = repo                  # the repo these records are (an extraction's adapter in it is labelled "project")
        self._markers, self._done = {}, None

    def local(self, path: str, fresh: bool = False) -> str:          # a file of the records, as a local path
        raise NotImplementedError

    def listing(self, folder: str, recursive: bool = True) -> list:  # the files under a folder, as paths of the records
        raise NotImplementedError

    def json(self, path: str, fresh: bool = False):
        with open(self.local(path, fresh), encoding="utf-8") as fh:
            return json.load(fh)

    def jsonl(self, path: str) -> list:
        with open(self.local(path), encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]

    def runs(self) -> set:
        """The runs that have a marker (runs/done/<run>.json), listed once."""
        if self._done is None:
            self._done = {p.rsplit("/", 1)[-1][:-len(".json")] for p in self.listing("runs/done", recursive=False)
                          if p.endswith(".json")}
        return self._done

    def marker(self, run: str) -> dict:
        """A run's marker, read fresh the first time it is asked for."""
        if run not in self._markers:
            self._markers[run] = self.json(f"runs/done/{run}.json", fresh=True)
        return self._markers[run]

    def done(self, run: str) -> dict:
        """A run's marker, which must say done."""
        m = self.marker(run)
        if m.get("status") != "done":
            raise RecordError(f"{run}: its marker says {m.get('status')!r}, not done")
        return m

    def close(self) -> None:
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class HubRecords(Records):
    """The records of a Hub repo (the project repo, a model repo by default), read with the package's token."""

    def __init__(self, repo: str, repo_type: str = "model", tok: str | None = None, workdir: str | None = None):
        super().__init__(repo)
        self.repo_type = repo_type
        self._token = tok if tok is not None else hub_token()
        self._own = workdir is None
        self.dir = workdir or tempfile.mkdtemp(prefix="qst_reads_")

    def local(self, path: str, fresh: bool = False) -> str:
        from huggingface_hub import hf_hub_download
        return with_retries(lambda: hf_hub_download(self.repo, path, repo_type=self.repo_type, token=self._token,
                                                    local_dir=self.dir, force_download=fresh), f"read {path}")

    def listing(self, folder: str, recursive: bool = True) -> list:
        from huggingface_hub import HfApi
        from huggingface_hub.hf_api import RepoFile
        api = HfApi(token=self._token)
        return with_retries(lambda: sorted(x.path for x in api.list_repo_tree(self.repo, path_in_repo=folder.strip("/"),
                                                                              recursive=recursive, repo_type=self.repo_type)
                                           if isinstance(x, RepoFile)), f"list {folder}")

    def close(self) -> None:
        if self._own:
            shutil.rmtree(self.dir, ignore_errors=True)


class FolderRecords(Records):
    """The records of a local folder that holds the project repo's paths (repo: the repo they stand for, if any)."""

    def __init__(self, root: str, repo: str | None = None):
        super().__init__(repo)
        self.root = os.path.abspath(root)

    def _path(self, path: str) -> str:
        return os.path.join(self.root, *path.strip("/").split("/"))

    def local(self, path: str, fresh: bool = False) -> str:
        p = self._path(path)
        if not os.path.isfile(p):
            raise FileNotFoundError(f"{path} is not under {self.root}")
        return p

    def listing(self, folder: str, recursive: bool = True) -> list:
        base, out = self._path(folder), []
        if not os.path.isdir(base):
            return out
        if recursive:
            for where, _, files in os.walk(base):
                out += [os.path.relpath(os.path.join(where, f), self.root).replace(os.sep, "/") for f in files]
        else:
            out = [f"{folder.strip('/')}/{f}" for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))]
        return sorted(out)


def open_records(project_repo: str | None, folder: str | None = None) -> Records:
    """The records a command reads: a local folder of the project repo's paths (folder), else the project repo on the Hub."""
    if folder:
        return FolderRecords(folder, project_repo)
    if not project_repo:
        raise SystemExit("name the project repo (--project-repo owner/name) or a local folder of its records (--records)")
    return HubRecords(project_repo)


def utf8_stdout() -> None:
    """Print the tables as UTF-8 whatever the console's code page (a quoted row may hold any letter); a stream that cannot
    switch is left as it is."""
    try:
        if (getattr(sys.stdout, "encoding", "") or "").lower().replace("-", "") != "utf8":
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError, OSError):
        pass


def add_source(parser) -> None:
    """The options every read's command takes for where its records are."""
    parser.add_argument("--project-repo", help="the Hub repo the runs wrote to (owner/name), read with HF_TOKEN, read-only")
    parser.add_argument("--records", help="a local folder holding the project repo's paths, read instead of the Hub")


def manifest_file(spec: str, records: Records) -> str:
    """A manifest as a local file: "repo:<path>" from the records, "hub:<owner>/<name>/<path>" from a public dataset repo
    (the Hub cache), anything else a local path."""
    if spec.startswith("repo:"):
        return records.local(spec[len("repo:"):])
    if spec.startswith("hub:"):
        from huggingface_hub import hf_hub_download
        parts = spec[len("hub:"):].split("/")
        return with_retries(lambda: hf_hub_download("/".join(parts[:2]), "/".join(parts[2:]), repo_type="dataset",
                                                     token=hub_token()), f"read {spec}")
    if not os.path.isfile(spec):
        raise FileNotFoundError(f"no manifest at {spec}")
    return spec


def manifest_rows(specs: str | list, records: Records) -> list:
    """The rows of one manifest, or of several (a list, or one string joined by commas), in order."""
    if isinstance(specs, str):
        specs = [s for s in specs.split(",") if s.strip()]
    rows = []
    for spec in specs:
        with open(manifest_file(spec.strip(), records), encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    return rows


def write(stem: str, data, lines: list, ensure_ascii: bool = True, txt: bool = True, md: bool = True) -> list:
    """A read's outputs: <stem>.json (data, indent 1), <stem>.md (the table) and <stem>.txt (the table as printed); data None
    writes no JSON. Every file is UTF-8 with LF line ends. Returns the paths written."""
    os.makedirs(os.path.dirname(os.path.abspath(stem)) or ".", exist_ok=True)
    text, out = "\n".join(lines) + "\n", []
    if data is not None:
        with open(stem + ".json", "w", encoding="utf-8", newline="\n") as fh:
            json.dump(data, fh, indent=1, ensure_ascii=ensure_ascii)
        out.append(stem + ".json")
    for ext, on in ((".md", md), (".txt", txt)):
        if on:
            with open(stem + ext, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
            out.append(stem + ext)
    return out
