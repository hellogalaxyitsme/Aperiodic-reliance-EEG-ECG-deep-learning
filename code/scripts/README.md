# Scripts

Scripts are grouped by prefix:

- `download_*`: dataset download helpers.
- `make_*` and `extract_*`: manifests, epochs/trials and feature caches.
- `fit_*`: SpecParam or IRASA decomposition.
- `run_*`: individual PSD, neural or foundation-model audits.
- `launch_*`: sequential multi-seed experiment launchers.
- `aggregate_*`: bootstrap aggregation across subjects and seeds.
- `analyze_*`, `compare_*` and `summarize_*`: diagnostics and result summaries.

Most scripts provide `--help` for arguments. Launch scripts are convenience
wrappers around the underlying Python entry points and may need path or resource
settings adjusted for a new compute environment.
