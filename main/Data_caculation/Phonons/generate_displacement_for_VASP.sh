#!/usr/bin/env bash
set -euo pipefail

# ==========================================================
# Configuration
# ==========================================================
POSCAR="POSCAR"
min_axis=12   # minimum lattice length (Å)

# ==========================================================
# Check POSCAR
# ==========================================================
if [[ ! -f "$POSCAR" ]]; then
    echo "Error: POSCAR not found in current directory."
    exit 1
fi

# ==========================================================
# Compute supercell dimensions from lattice vectors
# (Lines 3–5 of POSCAR)
# ==========================================================
dims=$(awk -v min_axis="$min_axis" '
    NR>=3 && NR<=5 {
        # Compute lattice vector length
        len = sqrt($1*$1 + $2*$2 + $3*$3)

        # Required scaling factor
        f = min_axis / len
        i = int(f)

        # Round up if not integer
        if (f > i) {
            i = i + 1
        }

        # Ensure at least 1× replication
        if (i < 1) {
            i = 1
        }

        dims[NR-2] = i
    }
    END {
        if (length(dims) != 3) {
            printf("1 1 1\n")
        } else {
            printf("%d %d %d\n", dims[1], dims[2], dims[3])
        }
    }
' "$POSCAR")

echo "Supercell dimensions: $dims"

# ==========================================================
# Generate displaced supercells with phonopy
# ==========================================================
phonopy -d --dim="$dims"

# ==========================================================
# Check required VASP input files
# ==========================================================
if [[ ! -f "INCAR" || ! -f "POTCAR" ]]; then
    echo "Error: INCAR or POTCAR file missing."
    exit 1
fi

if [[ ! -f "submit.sh" ]]; then
    echo "Warning: submit.sh not found. Continuing without it."
fi

# ==========================================================
# Collect generated POSCAR-xxx files
# ==========================================================
poscar_files=(POSCAR-???)

if [[ ${#poscar_files[@]} -eq 0 ]]; then
    echo "Error: No POSCAR-xxx files found."
    exit 1
fi

# ==========================================================
# Prepare VASP calculation folders
# ==========================================================
for poscar_file in "${poscar_files[@]}"; do
    index=${poscar_file##*-}
    folder_name="disp-${index}"

    mkdir -p "$folder_name"

    cp "$poscar_file" "${folder_name}/POSCAR"
    cp INCAR POTCAR "$folder_name/"

    if [[ -f "submit.sh" ]]; then
        cp submit.sh "$folder_name/"
    fi
done

echo "All displaced structure folders have been created and prepared."
