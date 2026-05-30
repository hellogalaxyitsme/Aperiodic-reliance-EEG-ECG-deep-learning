# Scripts

The scripts are grouped by prefix:

- `download_*`: dataset download helpers.
- `make_*` and `extract_*`: manifests, epochs/trials and feature caches.
- `fit_*`: SpecParam or IRASA decomposition.
- `run_*`: individual PSD, neural or foundation-model audits.
- `launch_*`: sequential multi-seed or remote-style experiment launchers.
- `aggregate_*`: bootstrap aggregation across subjects/seeds.
- `generate_*` and `plot_*`: manuscript tables and figures.

Most scripts provide `--help` for arguments. The launch scripts preserve the
paths/resource settings used in the original H200 runs and may need path edits on
another machine.

