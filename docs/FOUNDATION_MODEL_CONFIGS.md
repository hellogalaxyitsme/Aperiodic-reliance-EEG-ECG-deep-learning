# Foundation-Model Configuration Summary

The TUAB foundation-model audit fine-tunes a binary classification head on the
official TUAB training split and evaluates the official TUAB evaluation split
under raw, sham, aperiodic-shaped and flattened interventions. Interventions are
applied after model-specific preprocessing and immediately before the forward
pass.

All aggregate summaries use three seeds and hierarchical seed/subject bootstrap
aggregation with 10,000 resamples.

## Models

- BIOT: PREST-16 pretrained encoder, 10 s windows at 200 Hz, 16-channel bipolar
  TUAB montage.
- LaBraM: `labram-base.pth`, 23-channel referential TUAB cache, 10 s windows at
  200 Hz.
- EEGPT: Braindecode EEGPT checkpoint, 23-channel referential TUAB cache, 10 s
  windows at 200 Hz.
- CBraMod: Braindecode CBraMod pretrained checkpoint, 16-channel bipolar
  longitudinal montage, official input scaling.
- REVE-base: `brain-bzh/reve-base` encoder, supported-channel subset from the
  referential TUAB cache, z-score and clipping as implemented in the runner.
- EEGMamba: `weighting666/EEGMamba` checkpoint with the official EEGMamba code,
  16-channel bipolar montage and official input scaling.
- BENDR: Braindecode InterpolatedBENDR encoder, referential TUAB cache, clipped
  and scaled inputs as implemented in the runner.

Model-specific command-line defaults and checkpoint arguments are defined in the
corresponding `run_tuab_*_intervention.py` and
`launch_tuab_full_*_intervention.sh` scripts.
