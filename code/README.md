# Code Overview

The `code/` directory contains the reusable package and executable scripts for
the aperiodic spectral audit.

```text
src/aperiodic_eeg/       Dataset and spectral-processing utilities
scripts/                 Command-line entry points
configs/                 Small configuration files
requirements.txt         Script-level Python dependencies
pyproject.toml           Package metadata for editable installs
```

Most scripts expose command-line arguments through `--help`. Paths default to
the repository layout where possible, but large datasets and checkpoints should
be supplied explicitly for reproducible runs on a new system.

## Script Groups

- `download_*`: dataset download helpers for public datasets and authorized
  restricted datasets.
- `make_*` and `extract_*`: dataset manifests, epoch/trial caches and spectral
  feature extraction.
- `fit_*`: SpecParam or IRASA spectral decomposition.
- `run_*`: single audit jobs for PSD, raw neural or foundation-model models.
- `launch_*`: sequential multi-seed experiment launchers.
- `aggregate_*`: subject-level or hierarchical bootstrap aggregation.
- `analyze_*` and `compare_*`: diagnostic analyses and decomposition checks.

See [`../docs/REPRODUCIBILITY.md`](../docs/REPRODUCIBILITY.md) for the
recommended run order.
