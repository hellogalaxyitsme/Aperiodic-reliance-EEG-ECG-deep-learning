from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


TUAB_STANDARD_CHANNELS = [
    "FP1",
    "FP2",
    "F3",
    "F4",
    "C3",
    "C4",
    "P3",
    "P4",
    "O1",
    "O2",
    "F7",
    "F8",
    "T3",
    "T4",
    "T5",
    "T6",
    "A1",
    "A2",
    "FZ",
    "CZ",
    "PZ",
]


def normalize_channel_name(name: str) -> str:
    """Normalize TUAB EDF channel labels to compact referential names."""
    cleaned = name.strip().upper()
    cleaned = re.sub(r"^EEG\s+", "", cleaned)
    cleaned = re.sub(r"-(REF|LE|AVG|AV|A1|A2)$", "", cleaned)
    cleaned = cleaned.replace(" ", "")
    return cleaned


def resolve_tuab_channels(
    requested: Iterable[str],
    available: Iterable[str],
) -> list[str]:
    """Resolve compact TUAB channel names against EDF labels."""
    available_names = list(available)
    normalized_to_original: dict[str, list[str]] = {}
    for original in available_names:
        normalized_to_original.setdefault(normalize_channel_name(original), []).append(original)

    resolved: list[str] = []
    missing: list[str] = []
    ambiguous: dict[str, list[str]] = {}
    for channel in requested:
        key = normalize_channel_name(channel)
        matches = normalized_to_original.get(key, [])
        if not matches:
            missing.append(channel)
        elif len(matches) > 1:
            ambiguous[channel] = matches
        else:
            resolved.append(matches[0])

    if missing or ambiguous:
        parts: list[str] = []
        if missing:
            parts.append(f"missing={missing}")
        if ambiguous:
            parts.append(f"ambiguous={ambiguous}")
        parts.append(f"available={available_names}")
        raise ValueError("Could not resolve TUAB channels: " + "; ".join(parts))

    return resolved


def local_edf_path(data_root: Path, remote_rel_path: str) -> Path:
    return data_root / "edf" / remote_rel_path


def tuab_recording_id(row: dict[str, str]) -> str:
    path = Path(row["remote_rel_path"])
    return "/".join(path.parts[-4:])


def binary_label(label: str) -> int:
    normalized = label.strip().lower()
    if normalized == "normal":
        return 0
    if normalized == "abnormal":
        return 1
    raise ValueError(f"Unexpected TUAB label: {label!r}")
