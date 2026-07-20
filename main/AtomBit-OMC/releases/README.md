# AtomBit-OMC Release Assets

This directory defines the expected layout for the AtomBit-OMC checkpoints used
by `main/Scripts/Calculator_defs.json`.

```text
releases/
  meta_e0_data_OMC_r6_single.pt
  AtomBit-OMC-s/
    model_epoch_15.pt
    model_config.json
  AtomBit-OMC-l/
    model_epoch_15.pt
    model_config.json
```

`AtomBit-OMC-s` is the H64 model trained on the full OMC training dataset for
15 epochs. `AtomBit-OMC-l` is the H128 model trained on the full OMC training
dataset for 15 epochs. Both use `cutoff=6.0 Å`, two message-passing layers,
`num_rbf=10`, scalar/vector/rank-2 tensor channels, and the physics-informed
gate.

Large checkpoint files may be distributed through Git LFS, GitHub Releases, or
an external model archive. If checkpoints are not stored directly in Git, place
or symlink them to the paths above before running the benchmark.

## Included Files

| File | Size | SHA256 |
|---|---:|---|
| `AtomBit-OMC-s/model_epoch_15.pt` | 7,804,111 bytes | `0a85106a1a32cc4984e035e4bf1893318c54db0eb96c2162525d5ecf8a52e402` |
| `AtomBit-OMC-l/model_epoch_15.pt` | 30,472,463 bytes | `c2a22dc512b53252ec653ce33f35b48f15f84a4f77dca4f0e21467481fbb019c` |
| `meta_e0_data_OMC_r6_single.pt` | 2,187 bytes | `feaabcc547f1dcddf0afd10c54e85a6041193541b3ea5825fa875ee804f99fdf` |
