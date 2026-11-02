"""Readers for a trained arm's runs on the Hub: the medium term on the held-out probe (`probe_read`) and on the museum pictures
(`museum_read`, `museum_mismatch`), and a screen's conflicts (`screen_conflicts`). Each is a command:

    python -m qwen_symbolic_thinking.tools.probe_read P_probe_med4
    python -m qwen_symbolic_thinking.tools.museum_read P_museum_med4
    python -m qwen_symbolic_thinking.tools.museum_mismatch P_museum_med4
    python -m qwen_symbolic_thinking.tools.screen_conflicts 20261031_131844_G_med4_2ep

`mix` builds the mixes of several lookers' rows over the same pictures (the agreement pick, the field picks, the kind vote, the
caption and prompt pools) and reads them beside every looker; `picture_probe` measures how much a model's answer depends on the
picture.

Files live under a data root (`--data`, else the QST_DATA variable, else `./data`): `runs/done/<run>.json` (the run's marker),
`extractions/<stamp>_<run>/rows.jsonl` (its rows), `screens/<stamp>/results.jsonl` and `summary.json` (a screen), and the manifest.
Each is fetched from the Hub when missing (HF_TOKEN in the environment while a repo is private); `--offline` reads what is there."""
