# Result tables

Aggregate result tables and statistical-test outputs (see
[`docs/RESULTS_INDEX.md`](../../docs/RESULTS_INDEX.md)).

## Corrected results

For the analyses below, the values reported in the article are in
[`results/revision/`](../revision/), which supersedes the listed files in this
folder ([`docs/REVISION_ANALYSES.md`](../../docs/REVISION_ANALYSES.md)).

| Analysis | Values reported in the article | Superseded file in this folder |
|---|---|---|
| BENDR on TUAB (corrected input scaling) | [`bendr_corrected.csv`](../revision/bendr_corrected.csv) | BENDR rows of `tuab_full_foundation_multiseed_subject_bootstrap.*` |
| TUAB conventional neural models, full cohort (pooled balanced accuracy) | [`tuab_conventional_full_pooled_ba.csv`](../revision/tuab_conventional_full_pooled_ba.csv) | `tuab_full_multiseed_neural_subject_bootstrap.*` |
| TUAB conventional neural models, age/sex-matched cohort (pooled balanced accuracy) | [`tuab_conventional_matched_pooled_ba.csv`](../revision/tuab_conventional_matched_pooled_ba.csv) | `tuab_full_age_matched_multiseed_neural_subject_bootstrap.*` |
| PTB-XL PSD ridge baseline, unmatched (pooled balanced accuracy) | [`ptbxl_psd_ridge_pooled_ba.csv`](../revision/ptbxl_psd_ridge_pooled_ba.csv) | – |
| Formal tests (Benjamini-Hochberg FDR) | [`formal_tests_primary.csv`](../revision/formal_tests_primary.csv), [`formal_tests_sham.csv`](../revision/formal_tests_sham.csv) | `formal_hypothesis_tests*` |

The other foundation models, the PTB-XL neural models, the TUAB PSD ridge
analyses, and the Sleep-EDF and PhysioNet MI results in this folder are the
values reported in the article.
