#!/usr/bin/env python3
import os
import argparse
import json
import pandas as pd
from joblib import Parallel, delayed

# Import torch if used by the calculator
try:
    import torch
except ImportError:
    torch = None

from ase.io import read, write
from ase.optimize import BFGS
from Calculator_factory import CalculatorFactory
# Ensure FrechetCellFilter is correctly imported if needed
from ase.filters import FrechetCellFilter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch optimize CIFs in parallel using joblib with dynamic GPU mapping"
    )
    parser.add_argument('--input_dir', required=True,
                        help="Directory containing .cif files")
    parser.add_argument('--output_dir', default='../results',
                        help="Directory for output results")
    parser.add_argument('--model_name', required=True,
                        help="Model name defined in Calculator_defs.json")
    parser.add_argument('--config_json', default='Calculator_defs.json',
                        help="Path to calculator_defs.json")
    parser.add_argument('--fmax', type=float, default=0.01,
                        help="BFGS convergence force threshold")
    parser.add_argument('--max_steps', type=int, default=3000,
                        help="Maximum number of BFGS steps")
    parser.add_argument('--ref_energy_csv', default=None,
                        help="Reference energy CSV file for comparison")
    parser.add_argument('--compare', choices=['none', 'energy'],
                        default='none', help="Whether to compare with reference energy")
    parser.add_argument('--gpus', nargs='+', type=int, required=True,
                        help="List of GPUs to use, e.g., --gpus 0 1 2 3")
    parser.add_argument('--n_jobs', type=int, default=None,
                        help="joblib n_jobs, recommended to be larger than GPU count for higher concurrency")
    return parser.parse_args()


def load_ref_energies(csv_path):
    df = pd.read_csv(csv_path)
    return {row["name"]: row for _, row in df.iterrows()}


def optimize_one(idx, cif_file, args, ref_infos):
    """
    Optimize a single CIF, dynamically selecting GPU:
    gpu_id = args.gpus[idx % len(args.gpus)]
    """
    name = os.path.splitext(cif_file)[0]
    # Individual result JSON
    indiv_dir = os.path.join(args.output_dir, 'individual_results')
    os.makedirs(indiv_dir, exist_ok=True)
    json_path = os.path.join(indiv_dir, f"{name}.json")
    if os.path.isfile(json_path):
        print(f"[SKIP] {name} exists, skipping.")
        return json.load(open(json_path))

    # Determine GPU
    gpu_list = args.gpus
    gpu_id = gpu_list[idx % len(gpu_list)]

    # Limit OpenMP threads
    os.environ['OMP_NUM_THREADS'] = '1'
    # Bind device
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    if torch and torch.cuda.is_available():
        torch.cuda.set_device(0)

    # Load calculator for each task
    calc = CalculatorFactory.from_config(args.model_name, args.config_json)

    result = {
        "name": name,
        "converged": False,
        "steps": None,
        "energy_eV_cell": None,
        "energy_kjmol": None,
        "ref_energy": None,
        "delta_energy": None,
        "polymorph": None,
        "n_mol": None
    }

    try:
        atoms = read(os.path.join(args.input_dir, cif_file))
        atoms.calc = calc
        sf = FrechetCellFilter(atoms)
        opt = BFGS(sf, logfile=None)
        opt.run(fmax=args.fmax, steps=args.max_steps)
        result['converged'] = True
        result['steps'] = getattr(opt, 'nsteps', None)

        # Write structure
        xyz_dir = os.path.join(args.output_dir, 'optimized_xyz')
        os.makedirs(xyz_dir, exist_ok=True)
        write(os.path.join(xyz_dir, f"{name}_opt.extxyz"), atoms)

        # Energy calculation
        e = atoms.get_potential_energy()
        result['energy_eV_cell'] = e
        if args.compare == 'energy' and args.ref_energy_csv:
            info = ref_infos.get(name)
            if info:
                n_mol = info['n_mol']
                ref_e = info['ref_energy']
                polymorph = info['polymorph']
                ek = e * 96.485 / n_mol
                result.update({
                    'energy_kjmol': ek,
                    'ref_energy': ref_e,
                    'delta_energy': ek - ref_e,
                    'n_mol': n_mol,
                    'polymorph': polymorph
                })
    except Exception as ex:
        print(f"[GPU {gpu_id}] {name} failed: {ex}")

    # Write JSON
    with open(json_path, 'w') as jf:
        json.dump(result, jf, indent=2)
    return result


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'optimized_xyz'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'individual_results'), exist_ok=True)

    cif_files = sorted(f for f in os.listdir(args.input_dir) if f.endswith('.cif'))
    if not cif_files:
        print('No .cif files found in input directory, exiting.')
        return

    ref_infos = {}
    if args.compare == 'energy' and args.ref_energy_csv:
        ref_infos = load_ref_energies(args.ref_energy_csv)

    # Task list with global index
    tasks = list(enumerate(cif_files))
    n_jobs = args.n_jobs or len(tasks)

    # Execute in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(optimize_one)(idx, cif, args, ref_infos)
        for idx, cif in tasks
    )

    # Aggregate/Summary
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(args.output_dir, 'results.csv'), index=False)

    # Summarize JSON by polymorph
    if args.compare == 'energy' and args.ref_energy_csv:
        summary = {}
        for r in results:
            if r.get('converged') and r.get('polymorph'):
                summary.setdefault(r['polymorph'], []).append({
                    'name': r['name'], 'energy_kjmol': r['energy_kjmol']
                })
        for p in summary:
            summary[p] = sorted(summary[p], key=lambda x: x['energy_kjmol'])
        with open(os.path.join(args.output_dir, 'summary_by_polymorph.json'), 'w') as sf:
            json.dump(summary, sf, indent=2)

    fails = [r['name'] for r in results if not r['converged']]
    if fails:
        with open(os.path.join(args.output_dir, 'failed_list.txt'), 'w') as f:
            for n in fails:
                f.write(n + '\n')

    print('Done. Results in', args.output_dir)