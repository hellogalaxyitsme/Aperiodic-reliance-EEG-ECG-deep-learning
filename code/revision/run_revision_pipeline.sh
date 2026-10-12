#!/usr/bin/env bash
# Revision analyses: experiments, aggregation, and checks (see docs/REVISION_ANALYSES.md).
#
# Required environment variables:
#   RESULTS_DIR       results root of the original analyses (tuab_full_v3_0_1/, ptbxl_1f_demo/,
#                     sleep_edf_full/, ...), containing the cached arrays and prediction files
#   REVISION_DIR      output root for the revision analyses
#   BENDR_CHECKPOINT  braindecode/braindecode-bendr model.safetensors (Hugging Face snapshot)
# Optional: STAGES (default "fits experiments aggregate checks"), NB (bootstrap resamples, default 10000).
set -euo pipefail
: "${RESULTS_DIR:?}" "${REVISION_DIR:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
SCRIPTS=$(cd "$HERE/../scripts" && pwd)
M=$RESULTS_DIR
R=$REVISION_DIR
T=$M/tuab_full_v3_0_1/preprocess_20s_100hz
P=$M/ptbxl_1f_demo
SP=$R/results/ptbxl_specparam/ptbxl_test_specparam_fixed.npz
TSP=$R/results/tuab_full/specparam_fixed_20s_eval.npz
A=$R/results/aggregates
BC=$A/ba_correction
NB=${NB:-10000}
STAGES=${STAGES:-"fits experiments aggregate checks"}
cd "$HERE"
mkdir -p "$A" "$BC"
has() { [[ " $STAGES " == *" $1 "* ]]; }

# ---------------------------------------------------------------- 1. SpecParam fits
if has fits; then
  # TUAB evaluation windows (fixed mode, EEG settings) from the multitaper PSD cache
  python3 fit_tuab_specparam_eval.py --psd-npz "$T/psd_20s_multitaper.npz" --psd-index-csv "$T/psd_20s_multitaper_index.csv" \
    --raw-index-csv "$T/raw_epochs_20s_100hz_index.csv" --output-npz "$TSP" --n-jobs 20
  # PTB-XL test fold (fixed mode with peaks, EEG settings)
  python3 fit_ptbxl_specparam_test.py --psd-npz "$P/ptbxl_records100_normal_abnormal_psd_fixed.npz" --output-npz "$SP" --n-jobs 24
fi

# ---------------------------------------------------------------- 2. Experiments (GPU)
if has experiments; then
  for s in 42 43 44; do
    # envelope-estimator sensitivity, TUAB EEGNet (Table S5)
    python3 tuab_fit_procedure_sensitivity.py --raw-npz "$T/raw_epochs_20s_100hz.npz" --index-csv "$T/raw_epochs_20s_100hz_index.csv" \
      --specparam-eval-npz "$TSP" --model eegnet --seed $s --output-dir "$R/results/tuab_fit_sensitivity/eegnet_seed$s"
    # PTB-XL envelope estimators (Table S6)
    for m in resnet1d_wang inception1d xresnet1d101; do
      python3 ptbxl_revision_interventions.py --raw-npz "$P/ptbxl_records100_normal_abnormal_raw.npz" \
        --psd-npz "$P/ptbxl_records100_normal_abnormal_psd_fixed.npz" --specparam-npz "$SP" \
        --model $m --seed $s --epochs 35 --batch-size 192 --output-dir "$R/results/ptbxl_revision_v2/${m}_seed$s"
    done
    # training-time mitigation (Table S8)
    python3 ptbxl_flataug_mitigation.py --raw-npz "$P/ptbxl_records100_normal_abnormal_raw.npz" \
      --psd-npz "$P/ptbxl_records100_normal_abnormal_psd_fixed.npz" --specparam-npz "$SP" --model resnet1d_wang --seed $s \
      --aug-prob 0.5 --output-dir "$R/results/mitigation_v2/ptbxl_resnet1d_wang_flataug_p0.5_seed$s"
    python3 tuab_flataug_mitigation.py --raw-npz "$T/raw_epochs_20s_100hz.npz" --index-csv "$T/raw_epochs_20s_100hz_index.csv" \
      --specparam-eval-npz "$TSP" --model eegnet --seed $s --aug-prob 0.5 \
      --output-dir "$R/results/mitigation/tuab_eegnet_flataug_p0.5_seed$s"
  done
  # median-beat morphology baseline (Table S6)
  python3 ptbxl_morphology_baseline.py --raw-npz "$P/ptbxl_records100_normal_abnormal_raw.npz" \
    --psd-npz "$P/ptbxl_records100_normal_abnormal_psd_fixed.npz" --output-dir "$R/results/ptbxl_revision/morphology_baseline"
  # corrected BENDR evaluation (original hyper-parameters and seeds)
  : "${BENDR_CHECKPOINT:?}"
  for s in 20260525 20260526 20260527; do
    python3 run_tuab_bendr_intervention_fixed.py --cache-npz "$M/tuab_full_v3_0_1/labram_10s_200hz_cache.npz" --cache-format npy \
      --output-dir "$R/results/bendr_fixed/bendr_seed$s" --checkpoint-path "$BENDR_CHECKPOINT" \
      --pretrained-repo braindecode/braindecode-bendr --encoder-only --epochs 30 --batch-size 128 \
      --intervention-batch-size 256 --n-bootstrap $NB --seed $s --device cuda --input-scale 1e-6 --clip-uv 500.0
  done
  # waveform changes (Table S7, Fig. S5) and model-free ECG envelope diagnostics
  python3 waveform_distortion_audit.py --output-dir "$R/results/waveform_distortion/part_noTUAB" --tuab-specparam-eval "$TSP" \
    --datasets sleep_edf,physionet_mi,ptbxl
  python3 waveform_distortion_audit.py --output-dir "$R/results/waveform_distortion/part_TUAB" --tuab-specparam-eval "$TSP" --datasets tuab
  python3 ptbxl_envelope_removal_diagnostics.py --raw-npz "$P/ptbxl_records100_normal_abnormal_raw.npz" \
    --psd-npz "$P/ptbxl_records100_normal_abnormal_psd_fixed.npz" --specparam-npz "$SP" \
    --output-csv "$R/results/ptbxl_revision_v2/envelope_removal_diagnostics.csv"
fi

# ---------------------------------------------------------------- 3. Aggregation
if has aggregate; then
  TT=$M/tuab_full_v3_0_1
  # corrected BENDR and the formal test families
  ins=(); for s in 20260525 20260526 20260527; do ins+=(--input "BENDR:$s=$R/results/bendr_fixed/bendr_seed$s/tuab_bendr_intervention_predictions.csv"); done
  python3 aggregate_revision_bootstrap.py --scheme pooled_fm --n-bootstrap $NB "${ins[@]}" \
    --conditions raw_eeg,phase_sham,phase_aperiodic,phase_flattened --output-csv "$A/r3_2_bendr_fixed.csv"
  python3 update_formal_tests_bendr.py "$HERE/../../results/tables" "$A/r3_2_bendr_fixed.csv" "$A"
  # TUAB conventional models, pooled balanced accuracy (original foundation-model evaluator)
  for cohort in full matched; do
    ins=()
    for m in eegnet shallow_fbcsp deep4; do for s in 42 43 44; do
      if [[ $cohort == full ]]; then f=$TT/multiseed_neural/${m}_seed$s/tuab_full_${m}_intervention_predictions.csv
      else f=$TT/age_matched/multiseed_neural/${m}_seed$s/tuab_full_age_matched_${m}_intervention_predictions.csv; fi
      ins+=(--input "$m:$s=$f")
    done; done
    python3 "$SCRIPTS/aggregate_foundation_multiseed_predictions.py" "${ins[@]}" --n-bootstrap $NB \
      --output-csv "$BC/tuab_${cohort}_neural_pooled_fm_evaluator.csv" --output-md "$BC/tuab_${cohort}_neural_pooled_fm_evaluator.md"
  done
  python3 update_formal_tests_ba.py "$A" "$BC"
  # PTB-XL unmatched PSD ridge, pooled patient bootstrap
  python3 - <<PY
import pandas as pd
d = pd.read_csv("$P/psd_interventions/ptbxl_psd_intervention_predictions.csv")
d = d[d.train_input == "full_log_psd"].copy(); d["seed"] = 0
d.to_csv("$BC/ptbxl_psd_ridge_full_trained_predictions.csv", index=False)
PY
  python3 aggregate_revision_bootstrap.py --scheme pooled_ptbxl --n-bootstrap $NB --baseline full_log_psd \
    --conditions full_log_psd,aperiodic_spectrum,flattened_log_psd \
    --input "ptbxl_psd_ridge:0=$BC/ptbxl_psd_ridge_full_trained_predictions.csv" --output-csv "$BC/ptbxl_psd_ridge_pooled.csv"
  # envelope-estimator sensitivity (Table S5)
  ii=(); for s in 42 43 44; do ii+=(--input "eegnet:$s=$R/results/tuab_fit_sensitivity/eegnet_seed$s/predictions.csv"); done
  python3 aggregate_revision_bootstrap.py --scheme pooled_fm --n-bootstrap $NB "${ii[@]}" \
    --compare phase_flattened,lin_flattened --compare phase_aperiodic,lin_aperiodic \
    --output-csv "$BC/r1_1_tuab_eegnet_fit_sensitivity_pooled.csv"
  # 250-Hz evaluation subset (subjects with sfreq_original == 250 in the official evaluation split)
  python3 - <<PY
import pandas as pd
i = pd.read_csv("$T/raw_epochs_20s_100hz_index.csv", usecols=["subject", "official_split", "sfreq_original"])
s = i[(i.official_split == "eval") & (i.sfreq_original == 250.0)].subject.unique()
open("$A/tuab_eval_subjects_250hz.txt", "w").write("\n".join(map(str, s)))
PY
  for m in eegnet shallow_fbcsp deep4; do
    ii=(); for s in 42 43 44; do ii+=(--input "$m:$s=$TT/multiseed_neural/${m}_seed$s/tuab_full_${m}_intervention_predictions.csv"); done
    python3 aggregate_revision_bootstrap.py --scheme pooled_fm --n-bootstrap $NB "${ii[@]}" \
      --conditions raw_eeg,phase_sham,phase_aperiodic,phase_flattened --include-units "$A/tuab_eval_subjects_250hz.txt" \
      --output-csv "$BC/r1_4_tuab_${m}_250hz_only_pooled.csv"
  done
  # PTB-XL envelope estimators and morphology baseline (Table S6)
  for m in resnet1d_wang inception1d xresnet1d101; do
    ii=(); for s in 42 43 44; do ii+=(--input "$m:$s=$R/results/ptbxl_revision_v2/${m}_seed$s/predictions.csv"); done
    python3 aggregate_revision_bootstrap.py --scheme pooled_ptbxl --n-bootstrap $NB "${ii[@]}" \
      --compare phase_flattened,flat_harmonic_masked --compare phase_flattened,flat_specparam \
      --compare phase_flattened,flat_filterbank --output-csv "$BC/r3_1_ptbxl_${m}_v2.csv"
  done
  python3 aggregate_revision_bootstrap.py --scheme pooled_ptbxl --n-bootstrap $NB \
    --input "morphology_gbm:0=$R/results/ptbxl_revision/morphology_baseline/predictions.csv" \
    --output-csv "$A/r3_1_ptbxl_morphology_baseline.csv"
  # training-time mitigation, paired by seed (Table S8)
  b=(); a=(); for s in 42 43 44; do
    b+=(--base "$s=$R/results/tuab_fit_sensitivity/eegnet_seed$s/predictions.csv")
    a+=(--aug "$s=$R/results/mitigation/tuab_eegnet_flataug_p0.5_seed$s/predictions.csv"); done
  python3 paired_mitigation_bootstrap.py --label tuab_eegnet --scheme pooled_fm --n-bootstrap $NB "${b[@]}" "${a[@]}" \
    --conditions raw_eeg,phase_sham,phase_aperiodic,phase_flattened,lin_aperiodic,lin_flattened \
    --output-csv "$BC/r3_4_mitigation_tuab_eegnet_pooled.csv"
  b=(); a=(); for s in 42 43 44; do
    b+=(--base "$s=$R/results/ptbxl_revision_v2/resnet1d_wang_seed$s/predictions.csv")
    a+=(--aug "$s=$R/results/mitigation_v2/ptbxl_resnet1d_wang_flataug_p0.5_seed$s/predictions.csv"); done
  python3 paired_mitigation_bootstrap.py --label ptbxl_resnet1d_wang --scheme pooled_ptbxl --n-bootstrap $NB "${b[@]}" "${a[@]}" \
    --conditions raw_eeg,phase_sham,phase_aperiodic,phase_flattened,flat_harmonic_masked,flat_specparam,flat_filterbank \
    --output-csv "$BC/r3_4_mitigation_ptbxl_resnet1d_wang_v2.csv"
  # Sleep-EDF / PhysioNet MI binary tasks: subject-wise and pooled BA (Table S9)
  O=$R/results/ba_correction_binary
  python3 reconstruct_binary_counts.py "$M" "$O"
  for f in "$O"/pred_*_seed42.csv; do
    tag=$(basename "$f" _seed42.csv); tag=${tag#pred_}
    if [[ $tag == sleep_MLP* ]]; then cs=full_log_psd,aperiodic_spectrum,flattened_log_psd; bl=full_log_psd
    else cs=raw_eeg,phase_sham,phase_aperiodic,phase_flattened; bl=raw_eeg; fi
    ins=(); for s in 42 43 44; do ins+=(--input "$tag:$s=$O/pred_${tag}_seed$s.csv"); done
    python3 aggregate_revision_bootstrap.py --scheme subject_mean --n-bootstrap $NB --baseline $bl --conditions $cs "${ins[@]}" \
      --output-csv "$O/agg_${tag}_subject_mean.csv"
    # one subject draw shared across sampled seeds
    python3 aggregate_revision_bootstrap.py --scheme pooled_ptbxl --n-bootstrap $NB --baseline $bl --conditions $cs "${ins[@]}" \
      --output-csv "$O/agg_${tag}_pooled_shared.csv"
  done
fi

# ---------------------------------------------------------------- 4. Checks
if has checks; then
  # evaluator validation: independent implementation, scikit-learn, constant/perfect predictions
  python3 validate_ba_evaluators.py "$M" "$BC" "$SCRIPTS" | tee "$BC/validation_report.txt"
  # TUAB with one subject draw shared across seeds; complete formal families re-tested
  python3 shared_subject_check.py "$R" "$M" "$A" "$BC/shared_subject_check"
  # window-level sham integrity, including the corrected BENDR runs
  extra=(); for s in 20260525 20260526 20260527; do extra+=(--extra "TUAB|foundation|$R/results/bendr_fixed/bendr_seed$s/tuab_bendr_intervention_predictions.csv"); done
  RESULTS_DIR=$M python3 sham_integrity_from_predictions.py "${extra[@]}" --output-csv "$R/results/sham_integrity/sham_integrity_all.csv"
  RESULTS_DIR=$M REVISION_DIR=$R python3 fm_sham_dc_check.py
fi
echo "done: $STAGES"
