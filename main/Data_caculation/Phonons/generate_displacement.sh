#!/usr/bin/env bash
set -euo pipefail

# POSCAR file in the current directory and minimum target lattice length (Å)
POSCAR="POSCAR"
min_axis=12

# Check whether the POSCAR file exists
if [[ ! -f "$POSCAR" ]]; then
    echo "Error: '$POSCAR' not found in $(pwd)" >&2
    exit 1
fi

# --- Read lattice vectors (lines 3–5 in POSCAR) and compute supercell multipliers ---
read n1 n2 n3 < <(
  awk -v min_axis="$min_axis" '
    NR>=3 && NR<=5 {
      # Compute length of lattice vector
      len = sqrt($1*$1 + $2*$2 + $3*$3)

      # Determine required scaling factor to reach min_axis
      f = min_axis / len
      i = int(f)

      # Round up if necessary
      if (f > i) i++

      # Ensure at least 1× replication
      if (i < 1) i = 1

      dims[NR-2] = i
    }
    END {
      if (length(dims) != 3) {
        # If three lattice vectors are not properly read,
        # fall back to 1×1×1 supercell
        printf("1 1 1\n")
      } else {
        printf("%d %d %d\n", dims[1], dims[2], dims[3])
      }
    }
  ' "$POSCAR"
)

echo "→ Using supercell dimensions: ${n1}×${n2}×${n3}"

# --- Run phonopy to generate displaced supercells ---
phonopy -d --dim="${n1} ${n2} ${n3}" --pa auto -c "$POSCAR"