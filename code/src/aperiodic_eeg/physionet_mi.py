from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


RUN_TASKS = {
    1: "baseline_eyes_open",
    2: "baseline_eyes_closed",
    3: "executed_left_right_fist",
    4: "imagined_left_right_fist",
    5: "executed_fists_feet",
    6: "imagined_fists_feet",
    7: "executed_left_right_fist",
    8: "imagined_left_right_fist",
    9: "executed_fists_feet",
    10: "imagined_fists_feet",
    11: "executed_left_right_fist",
    12: "imagined_left_right_fist",
    13: "executed_fists_feet",
    14: "imagined_fists_feet",
}

TASK_RUNS = {
    "imagined_fists": [4, 8, 12],
    "executed_fists": [3, 7, 11],
    "imagined_fists_feet": [6, 10, 14],
    "executed_fists_feet": [5, 9, 13],
    "all_imagery": [4, 6, 8, 10, 12, 14],
    "all_execution": [3, 5, 7, 9, 11, 13],
    "all_tasks": list(range(3, 15)),
}


@dataclass(frozen=True)
class MIRecording:
    path: Path
    subject: str
    run: int

    @property
    def recording(self) -> str:
        return self.path.stem

    @property
    def task_family(self) -> str:
        return RUN_TASKS[self.run]


def parse_recording_path(path: Path) -> MIRecording:
    match = re.fullmatch(r"(S\d{3})R(\d{2})", path.stem)
    if match is None:
        raise ValueError(f"Unexpected PhysioNet MI recording name: {path.name}")
    return MIRecording(path=path, subject=match.group(1), run=int(match.group(2)))


def discover_recordings(
    data_root: Path | str,
    task: str = "imagined_fists",
    max_subjects: int | None = None,
    max_recordings: int | None = None,
) -> list[MIRecording]:
    root = Path(data_root)
    if task not in TASK_RUNS:
        raise ValueError(f"Unknown task {task!r}. Choices: {sorted(TASK_RUNS)}")
    allowed_runs = set(TASK_RUNS[task])
    recordings = [
        parse_recording_path(path)
        for path in root.glob("S*/S*R*.edf")
        if parse_recording_path(path).run in allowed_runs
    ]
    recordings = sorted(recordings, key=lambda item: (item.subject, item.run))
    if max_subjects is not None:
        keep_subjects = {subject for subject in sorted({r.subject for r in recordings})[:max_subjects]}
        recordings = [recording for recording in recordings if recording.subject in keep_subjects]
    if max_recordings is not None:
        recordings = recordings[:max_recordings]
    return recordings


def normalize_event_code(description: str) -> str | None:
    text = description.strip().upper()
    for code in ("T0", "T1", "T2"):
        if code in text:
            return code
    return None


def condition_for_event(run: int, event_code: str) -> str | None:
    task_family = RUN_TASKS[run]
    if event_code == "T0":
        return "rest"
    if task_family.endswith("left_right_fist"):
        if event_code == "T1":
            return "left_fist"
        if event_code == "T2":
            return "right_fist"
    if task_family.endswith("fists_feet"):
        if event_code == "T1":
            return "both_fists"
        if event_code == "T2":
            return "both_feet"
    return None


def resolve_channel_names(requested: Iterable[str], available: Iterable[str]) -> list[str]:
    available_names = list(available)
    if list(requested) == ["all"]:
        return available_names

    resolved: list[str] = []
    for channel in requested:
        if channel in available_names:
            resolved.append(channel)
            continue

        target = channel.lower().rstrip(".")
        exactish = [name for name in available_names if name.lower().rstrip(".") == target]
        if len(exactish) == 1:
            resolved.append(exactish[0])
            continue

        contains = [name for name in available_names if target in name.lower().rstrip(".")]
        if len(contains) == 1:
            resolved.append(contains[0])
            continue

        raise ValueError(
            f"Could not resolve channel {channel!r}. Available channels: {available_names}"
        )
    return resolved
