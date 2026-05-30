# Supplementary Information

This folder contains the generated Supplementary Information package for the NMI manuscript draft.

Generated files:

- `supplementary_information.tex`: main SI text with Supplementary Tables 1-7 and Supplementary Notes 1-3.
- `supplementary_table_1_complete_results.*`: normalized table of all available domain x task x model x condition results with confidence intervals.
- `supplementary_table_2_specparam_fit_quality.*`: SpecParam fit-quality summary by dataset, including full-TUAB fixed-mode fits.
- `supplementary_table_3_irasa_ridge_intervention.*`: Sleep-EDF IRASA downstream ridge intervention results.
- `supplementary_table_4_tuab_age_matching.*`: Full-TUAB demographic metadata and age/sex-matching details.
- `supplementary_table_5_physionet_mi_psd_matrix.*`: PhysioNet MI PSD ridge train-by-test intervention matrix.
- `supplementary_table_6_foundation_model_config.*`: Full-TUAB foundation-model checkpoint and preprocessing configurations.
- `supplementary_table_7_preprocessing_details.*`: Dataset-level preprocessing, channel, filtering and exclusion details.
- `supplementary_note_2_simulation_results.*`: synthetic validation results used in Supplementary Note 2.

Regenerate with:

```bash
python3 code/scripts/generate_nmi_supplementary_information.py
```

The generator reads only result CSV/JSON artifacts and does not touch raw datasets.
