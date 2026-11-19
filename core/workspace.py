from __future__ import annotations

import json
from pathlib import Path


WORKSPACE_VERSION = 1


def save_workspace(path: str | Path, datasets: list[dict], plots: list[dict]) -> None:
    payload = {"version": WORKSPACE_VERSION, "datasets": datasets, "plots": plots}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_workspace(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("version") != WORKSPACE_VERSION:
        raise ValueError("Unsupported workspace version.")
    if not isinstance(payload.get("datasets"), list) or not isinstance(payload.get("plots"), list):
        raise ValueError("Invalid workspace structure.")
    return payload
