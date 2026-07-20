# Revision Analyses

This directory is reserved for post-hoc analyses added during manuscript revision.
The scripts here are not required for running the four OMC-bench tasks in `main/Scripts/`.
They document reviewer-requested checks and should use stable, reviewer-linked names.

## Naming Convention

Use the pattern:

```text
rx_comment_topic_action.py
```

where `rx` identifies the reviewer/comment group and `topic` describes the analysis.
Keep generated figures, tables, and large intermediate outputs outside the repository or in a separately documented release archive.

## Planned Analysis Scripts

| Filename | Purpose |
|---|---|
| `r1_major_1_high_error_metadata_diagnostic.py` | Relate cross-model high-error consensus cases to atom count and common space groups. |
| `r1_major_2_uma_overlap_stratified_metrics.py` | Compare UMA-md metrics on OMC25-overlapping, non-overlapping, and all OMC-bench entries. |
| `r1_minor_2_task4_kpe_practical_stats.py` | Compute Task 4 exact discordant-pair counts, top-1 correct counts, and FRA from raw ranking outputs. |
| `r1_major_5_mace_mpa_omc_finetuning_summary.py` | Summarize off-the-shelf and OMC-fine-tuned MACE-MPA-0 validation and OMC-bench metrics. |
| `r2_major_8_atombit_embedding_ood_diagnostic.py` | Compare Task 1 and OMC-train samples in the AtomBit-OMC-l learned representation. |
| `r2_major_9_task1_force_stress_rmse.py` | Compute Task 1 force/stress RMSE alongside MAE for all evaluated MLIPs. |
| `r2_major_10_atombit_ablation_summary.py` | Summarize AtomBit gate/channel/layer ablations and parameter-cost tradeoffs. |
| `r1_major_6_r3_minor_3_gate_saliency_diagnostics.py` | Analyze gate-weight saliency, contact classes, and edge-deletion diagnostics. |
| `r3_major_2_md_stability_validation.py` | Analyze finite-temperature MD stability trajectories and energy-drift metrics. |

## Reproducibility Notes

- Scripts should take explicit `--input` and `--output` arguments rather than hard-coded personal paths.
- Keep model checkpoints and datasets referenced by path or DOI/release ID, not duplicated inside this directory.
- If a script depends on licensed software such as the CSD Python API, state that requirement at the top of the script and provide an open-source fallback only when scientifically equivalent.
