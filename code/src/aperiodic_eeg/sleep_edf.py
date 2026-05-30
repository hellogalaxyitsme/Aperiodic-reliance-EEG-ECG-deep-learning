from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class RecordingPair:
    """A Sleep-EDF PSG file and its matching hypnogram annotation file."""

    key: str
    psg_path: Path
    hypnogram_path: Path

    @property
    def subject_code(self) -> str:
        return self.key[:5]

    @property
    def night_code(self) -> str:
        return self.key[5]


def recording_key(path: Path) -> str:
    """Return the shared SC recording key, for example SC4001."""
    return path.name[:6]


def discover_recording_pairs(data_root: Path | str) -> list[RecordingPair]:
    root = Path(data_root)
    psg_by_key = {recording_key(path): path for path in root.glob("SC*PSG.edf")}
    hyp_by_key = {recording_key(path): path for path in root.glob("SC*Hypnogram.edf")}

    pairs: list[RecordingPair] = []
    for key in sorted(psg_by_key):
        hypnogram_path = hyp_by_key.get(key)
        if hypnogram_path is None:
            continue
        pairs.append(
            RecordingPair(
                key=key,
                psg_path=psg_by_key[key],
                hypnogram_path=hypnogram_path,
            )
        )
    return pairs


def find_missing_pairs(data_root: Path | str) -> dict[str, list[str]]:
    root = Path(data_root)
    psg_keys = {recording_key(path) for path in root.glob("SC*PSG.edf")}
    hyp_keys = {recording_key(path) for path in root.glob("SC*Hypnogram.edf")}
    return {
        "missing_hypnograms": sorted(psg_keys - hyp_keys),
        "missing_psg": sorted(hyp_keys - psg_keys),
    }


def find_partial_downloads(data_root: Path | str) -> list[Path]:
    root = Path(data_root)
    return sorted(root.glob("*.aria2"))


def total_size_bytes(paths: Iterable[Path]) -> int:
    return sum(path.stat().st_size for path in paths)


STAGE_MAP = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
}

EXCLUDED_STAGE_DESCRIPTIONS = {"Movement time", "Sleep stage ?"}


def normalize_stage(description: str) -> str | None:
    """Map Sleep-EDF annotation text to the analysis-stage label."""
    description = description.strip()
    if description in EXCLUDED_STAGE_DESCRIPTIONS:
        return None
    return STAGE_MAP.get(description)


def resolve_channel_name(requested: str, available: Iterable[str]) -> str:
    """Resolve compact channel names like Fpz-Cz against EDF channel labels."""
    available_names = list(available)
    if requested in available_names:
        return requested

    requested_lower = requested.lower()
    suffix_matches = [
        name for name in available_names if name.lower().endswith(requested_lower)
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]

    contains_matches = [
        name for name in available_names if requested_lower in name.lower()
    ]
    if len(contains_matches) == 1:
        return contains_matches[0]

    raise ValueError(
        f"Could not resolve channel {requested!r}. Available channels: {available_names}"
    )


def trim_wake_epochs(
    rows: list[dict[str, object]],
    wake_trim_minutes: float | None,
    epoch_seconds: float,
) -> list[dict[str, object]]:
    """Keep wake epochs only near sleep, following common Sleep-EDF practice."""
    if wake_trim_minutes is None or wake_trim_minutes < 0:
        return rows

    non_wake_onsets = [float(row["onset_sec"]) for row in rows if row["stage"] != "W"]
    if not non_wake_onsets:
        return rows

    margin_sec = wake_trim_minutes * 60.0
    keep_start = min(non_wake_onsets) - margin_sec
    keep_end = max(non_wake_onsets) + epoch_seconds + margin_sec

    return [
        row
        for row in rows
        if keep_start <= float(row["onset_sec"]) < keep_end
    ]
