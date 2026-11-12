"""Resuming a long training on a fresh machine.

A rented runtime ends at a fixed age (Colab's at about 24 hours), so a training of near a day does not fit in one. With resume
on (train.run(resume=True), or "resume": true in a queue spec's "train" keys), the run saves its whole training state to the
project repo under resume/<tag>/ on a wall-clock schedule: every `every_h` hours, and once more when the machine's age passes
`before_h`. A run of the same tag that finds a saved state continues from it on any machine, along the path the uninterrupted run
would have taken:

    the adapter, the optimizer's moments, the learning-rate schedule, the step and the random state: the Trainer's own
        checkpoint, read back by Trainer.train(resume_from_checkpoint=...);
    the order of the training examples: the Trainer skips the batches already trained, in the same seeded order;
    the order of the general rows (keep.PairedCollator): replayed from its seed for the batches already trained;
    the collators' counts, the progress log, the keeping terms' running sums, the loss sum and the time spent: resume.json.

A save happens only at a log step, so the loss log reads the same with or without a resume. The resumed run keeps the first run's
stamp, so its step ships and its final adapter land in the same folder. A run whose settings or training examples differ from the
saved state's stops before it trains. When the training ends and its adapter has shipped, resume.json alone stays, marked
finished with the run's summary, so a run of the same tag after that returns the summary and the shipped adapter without
training again. A LoRA only (a full fine-tune's state would be the whole model and its optimizer, many times larger).

The data loader collates one batch ahead of the one in training (and the Trainer fetches a step's batches together), so a state
taken by reading the collators when the save happens would hold a batch that never trained; the counts are kept batch by batch
(Counts) and the general rows replayed, both up to the batches trained (consumed)."""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import shutil
import time
from datetime import datetime, timezone

STATE = "resume.json"      # the run's own part of a saved state, beside the Trainer's checkpoint files
ROOT = "resume"            # resume/<tag>/ in the project repo
SKIP_FILES = ("README.md", "training_args.bin")   # in a checkpoint folder and not needed to resume (PEFT's card, a pickle)


# ---------------------------------------------------------------------------------------------------------- counting
def consumed(global_step: int, n_examples: int, batch: int, grad_accum: int) -> int:
    """The micro-batches trained after global_step optimizer steps. An epoch has ceil(n_examples / batch) micro-batches in
    ceil(that / grad_accum) steps, and only its last step may take fewer than grad_accum (the count the Trainer itself uses
    when it skips the batches already trained)."""
    per_epoch = math.ceil(n_examples / batch)
    steps = math.ceil(per_epoch / grad_accum)
    epochs, rest = divmod(int(global_step), steps)
    return epochs * per_epoch + rest * grad_accum


class Counts:
    """A collator's counts, batch by batch: the examples collated, those longer than max_len (their answer's tail cut) and the
    longest example's tokens. Kept per batch so a resumed run takes them back exactly as they stood after the batches that
    trained."""

    def __init__(self):
        self.trace = []

    def add(self, n: int, cut: int, longest: int) -> None:
        self.trace.append((int(n), int(cut), int(longest)))

    def clear(self) -> None:
        self.trace = []

    def upto(self, k: int) -> list:
        """The counts of the first k batches, as saved."""
        if len(self.trace) < k:
            raise ValueError(f"{k} batches trained but {len(self.trace)} collated")
        return [list(x) for x in self.trace[:k]]

    def restore(self, trace) -> None:
        self.trace = [tuple(int(v) for v in x) for x in trace]

    @property
    def collated(self) -> int:
        return sum(x[0] for x in self.trace)

    @property
    def cut(self) -> int:
        return sum(x[1] for x in self.trace)

    @property
    def longest(self) -> int:
        return max((x[2] for x in self.trace), default=0)


def fingerprint(pairs) -> str:
    """A short hash of training examples in their training order ((item, form) pairs; the item's id and the form), so a resumed
    run is checked to train on the same examples in the same order."""
    h = hashlib.sha256()
    for it, form in pairs:
        h.update(json.dumps([it["id"], form]).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()[:16]


def normalized(settings: dict) -> dict:
    """Settings as JSON reads them back (tuples become lists), so a saved copy compares equal to the live one."""
    return json.loads(json.dumps(settings, default=str))


def check_settings(saved: dict, settings: dict) -> None:
    """Stop a resumed run whose settings differ from the saved state's (the keys that define the run: the model, the inputs,
    the recipe, the keeping terms, the seed)."""
    want, have = normalized(saved), normalized(settings)
    differ = sorted(k for k in set(want) | set(have) if want.get(k) != have.get(k))
    if differ:
        raise ValueError("the saved training state was made with other settings: "
                         + "; ".join(f"{k}: saved {want.get(k)!r}, now {have.get(k)!r}" for k in differ))


def check_examples(saved: dict, examples: str, general_rows: str | None, n_examples: int) -> None:
    """Stop a resumed run whose training examples (or general rows) are not the saved run's, in the same order: a picture that
    did not load on this machine leaves its item out, and the run would no longer be the same run."""
    if saved.get("n_examples") != n_examples or saved.get("examples") != examples:
        raise ValueError(f"the training examples differ from the saved run's ({n_examples} now, {saved.get('n_examples')} saved; "
                         f"fingerprint {examples} now, {saved.get('examples')} saved): a picture that loaded before did not "
                         f"load here, or the inputs changed")
    if saved.get("general_rows") != general_rows:
        raise ValueError(f"the general rows differ from the saved run's (fingerprint {general_rows} now, "
                         f"{saved.get('general_rows')} saved)")


# ---------------------------------------------------------------------------------------------------------- when
def machine_age_h() -> float | None:
    """The machine's age in hours (Linux /proc/uptime; on a Colab runtime, the time since the runtime started), or None."""
    try:
        with open("/proc/uptime", encoding="ascii") as fh:
            return float(fh.read().split()[0]) / 3600.0
    except (OSError, ValueError, IndexError):
        return None


class Schedule:
    """When a save is due: at a log step, every_h hours after the last save (or after the training began), and once at the
    first log step after the machine's age passes before_h (a runtime that ends at a fixed age; a machine already past it when
    the training begins does not tell its age that way, so it has no such save). Never at the run's last step (the final
    adapter ships then). A save that fails is tried again retry_min minutes later. at_steps: saves at these steps instead of
    by the clock (tests)."""

    def __init__(self, every_h: float = 2.0, before_h: float = 23.0, retry_min: float = 15.0, clock=time.time,
                 age=machine_age_h, at_steps=None):
        self.every_s, self.before_h, self.retry_s = float(every_h) * 3600.0, before_h, float(retry_min) * 60.0
        self.clock, self.age = clock, age
        self.at_steps = set(at_steps) if at_steps is not None else None
        self.last = None
        self.late_left = False

    def start(self) -> None:
        self.last = self.clock()
        age = self.age() if self.before_h is not None else None
        self.late_left = age is not None and age < self.before_h

    def due(self, step: int, logging_steps: int, max_steps: int) -> str | None:
        """Why a save is due at this step ("schedule", "age", "step"), or None."""
        if step >= max_steps or not logging_steps or step % logging_steps:
            return None
        if self.at_steps is not None:
            return "step" if step in self.at_steps else None
        if self.last is None:
            self.start()
        if self.clock() - self.last >= self.every_s:
            return "schedule"
        if self.late_left:
            age = self.age()
            if age is not None and age >= self.before_h:
                return "age"
        return None

    def saved(self, why: str) -> None:
        self.last = self.clock()
        if why == "age":
            self.late_left = False

    def failed(self) -> None:
        """The next try retry_min minutes from now (not at the next log step: a Hub that is down would stall the run)."""
        self.last = self.clock() - self.every_s + self.retry_s


# ---------------------------------------------------------------------------------------------------------- where
class FolderStore:
    """Saved states in a local folder, <root>/<tag>/ (tests; a machine whose disk outlives the run)."""

    def __init__(self, root: str):
        self.root = root

    def put(self, folder: str, tag: str, message: str = "") -> None:
        dst = os.path.join(self.root, tag)
        tmp = dst + ".new"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(folder, tmp, ignore=shutil.ignore_patterns(*SKIP_FILES))
        shutil.rmtree(dst, ignore_errors=True)
        os.replace(tmp, dst)

    def get(self, tag: str, dest: str) -> str | None:
        src = os.path.join(self.root, tag)
        if not os.path.isfile(os.path.join(src, STATE)):
            return None
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest)
        return dest

    def finish(self, tag: str, record: dict) -> None:
        dst = os.path.join(self.root, tag)
        shutil.rmtree(dst, ignore_errors=True)
        os.makedirs(dst)
        with open(os.path.join(dst, STATE), "w", encoding="utf-8") as fh:
            json.dump(record, fh, indent=1, default=str)


class HubStore:
    """Saved states in the project repo, under resume/<tag>/ (for the runs of a private twin, the twin). A save replaces the
    folder's files in one commit; the Hub's passing faults are waited out (hub.with_retries)."""

    def __init__(self, repo: str):
        self.repo = repo

    def _api(self):
        from huggingface_hub import HfApi
        from .hub import token
        return HfApi(token=token())

    def put(self, folder: str, tag: str, message: str = "") -> None:
        from .hub import with_retries
        api = self._api()
        with_retries(lambda: api.upload_folder(folder_path=folder, path_in_repo=f"{ROOT}/{tag}", repo_id=self.repo,
                                               repo_type="model", ignore_patterns=list(SKIP_FILES), delete_patterns=["*"],
                                               commit_message=message or f"{tag}: the training state"),
                     f"save the training state of {tag}")

    def get(self, tag: str, dest: str) -> str | None:
        from huggingface_hub import snapshot_download
        from .hub import token, with_retries
        api = self._api()
        if not with_retries(lambda: api.file_exists(self.repo, f"{ROOT}/{tag}/{STATE}", repo_type="model"),
                            f"look for the training state of {tag}"):
            return None
        shutil.rmtree(dest, ignore_errors=True)
        with_retries(lambda: snapshot_download(self.repo, repo_type="model", allow_patterns=[f"{ROOT}/{tag}/*"], local_dir=dest,
                                               token=token()),
                     f"read the training state of {tag}")
        return os.path.join(dest, ROOT, tag)

    def finish(self, tag: str, record: dict) -> None:
        from huggingface_hub import CommitOperationAdd, CommitOperationDelete
        from .hub import with_retries
        api = self._api()
        prefix = f"{ROOT}/{tag}/"
        files = [f for f in with_retries(lambda: api.list_repo_files(self.repo, repo_type="model"), "list the repo's files")
                 if f.startswith(prefix) and f != prefix + STATE]
        data = json.dumps(record, indent=1, default=str).encode("utf-8")
        ops = [CommitOperationDelete(path_in_repo=f) for f in files] + [CommitOperationAdd(path_in_repo=prefix + STATE,
                                                                                           path_or_fileobj=data)]
        with_retries(lambda: api.create_commit(self.repo, operations=ops, repo_type="model",
                                               commit_message=f"{tag}: training finished; the saved state is no longer needed"),
                     f"mark the training state of {tag} finished")


# ---------------------------------------------------------------------------------------------------------- read back
def load(store, tag: str, dest: str) -> dict | None:
    """The saved state of a run, or None when it has none: resume.json's record, with "dir" = the local folder holding the
    checkpoint files (Trainer.train(resume_from_checkpoint=dir))."""
    folder = store.get(tag, dest)
    if folder is None:
        return None
    with open(os.path.join(folder, STATE), encoding="utf-8") as fh:
        record = json.load(fh)
    record["dir"] = folder
    return record


def drop_local_checkpoints(output_dir: str) -> None:
    """The Trainer's checkpoint folders on this machine (saved to the store; not needed here once read back)."""
    for folder in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        shutil.rmtree(folder, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------- the callback
def saver(store, tag: str, schedule: Schedule, record):
    """The callback that saves the training state when the schedule says. record(state): the run's own part of the state
    (resume.json), read at the save, after the Trainer has written its checkpoint. A save that cannot be written is reported
    and tried again later (Schedule.failed); it never stops the run. .saves lists the saves made."""
    from transformers import TrainerCallback

    class Saver(TrainerCallback):
        def __init__(self):
            self.why, self.saves = None, []

        def on_train_begin(self, args, state, control, **kw):
            schedule.start()

        def on_step_end(self, args, state, control, **kw):
            self.why = schedule.due(state.global_step, args.logging_steps, state.max_steps)
            if self.why:
                control.should_save = True
            return control

        def on_save(self, args, state, control, **kw):
            if not self.why:
                return
            why, self.why = self.why, None
            folder = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            t0 = time.time()
            try:
                rec = record(state)
                with open(os.path.join(folder, STATE), "w", encoding="utf-8") as fh:
                    json.dump(rec, fh, indent=1, default=str)
                store.put(folder, tag, f"{tag}: the training state at step {state.global_step}")
            except Exception as e:      # a save that fails never stops the run; the next try comes later
                schedule.failed()
                print(f"  the training state at step {state.global_step} was NOT saved ({type(e).__name__}: {str(e)[:200]}); "
                      f"trying again in {schedule.retry_s / 60:.0f} min", flush=True)
            else:
                schedule.saved(why)
                self.saves.append({"step": state.global_step, "why": why, "s": round(time.time() - t0, 1),
                                   "at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
                print(f"  training state saved at step {state.global_step} ({why}; {time.time() - t0:.0f} s)", flush=True)
            finally:
                drop_local_checkpoints(args.output_dir)

    return Saver()
