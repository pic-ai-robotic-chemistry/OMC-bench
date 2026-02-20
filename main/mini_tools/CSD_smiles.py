from ccdc import io
import pandas as pd

# =========================
# 1. Read REF codes
# =========================
df_ref = pd.read_csv("/Users/Desktop/refcodes.csv")

refcodes = (
    df_ref["REF_code"]
    .astype(str)
    .str.strip()
    .tolist()
)

csd_reader = io.EntryReader("CSD")

rows = []

# =========================
# 2. Loop over refcodes
# =========================
for code in refcodes:
    try:
        entry = csd_reader.entry(code)

        if entry is None:
            print(f"Not found in local CSD: {code}")
            continue

        mol = entry.molecule

        # ====== Get SMILES for all components and deduplicate ======
        smiles_set = set()

        for comp in mol.components:
            try:
                s = comp.smiles
                if s:  # Guard against empty strings
                    smiles_set.add(s)
            except Exception:
                continue

        # Convert set to sorted list (stable column order)
        smiles_list = sorted(list(smiles_set))

        # Assemble one row: REF_CODE + SMILES_1, SMILES_2, ...
        row = {"REF_CODE": code}

        for i, s in enumerate(smiles_list):
            row[f"SMILES_{i+1}"] = s

        rows.append(row)

    except Exception as e:
        print(f"Skip {code}: {e}")

# =========================
# 3. Save result
# =========================
out_df = pd.DataFrame(rows).fillna("")

print(out_df.head())

out_path = "/Users/Desktop/ccdc_refcode_all_smiles.csv"
out_df.to_csv(out_path, index=False)

print(f"Results written to: {out_path}")
