"""
create_dummy_data.py
=====================
Genera dataset dummy compatibili con la pipeline pod-transformer, per test
rapidi end-to-end senza dover usare i dati reali (pesanti).

Shape generata: [n_cells, nf, n_timesteps] (stessa convenzione del dataset
reale, nessuna assunzione di griglia nx/nz).

Uso tipico:
    python3 create_dummy_data.py --output-dir ./dummy_data
    python3 create_dummy_data.py --output-dir ./dummy_data --reference train.json

Con --reference (un train.json o test.json reale) il numero di timesteps
e di feature (nf) viene copiato dal primo dataset reale referenziato nel
JSON, cosi' i dati dummy hanno lo stesso numero di timestep del dataset
reale ma un numero di celle molto piu' piccolo (quindi un peso molto
inferiore), utili per validare rapidamente l'intera pipeline.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def _infer_from_reference(reference_path: str):
    with open(reference_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    dataset_cfg = cfg.get("dataset", cfg)
    data_paths = dataset_cfg["data_paths"]
    arr = np.load(data_paths[0], mmap_mode="r")  # (n_cells, nf, nt) — mmap: non carica tutto in RAM
    if arr.ndim != 3:
        raise ValueError(f"Dataset di riferimento {data_paths[0]} non ha shape [n_cells, nf, n_timesteps]")
    _, nf, nt = arr.shape
    n_datasets = len(data_paths)
    return nf, nt, n_datasets


def main():
    parser = argparse.ArgumentParser(description="Genera dataset dummy [n_cells, nf, n_timesteps] per debug pipeline")
    parser.add_argument("--output-dir", default="./dummy_data", help="Cartella di output")
    parser.add_argument("--reference", default=None,
                         help="train.json/test.json reale da cui copiare nf e n_timesteps")
    parser.add_argument("--nf", type=int, default=11, help="Numero di feature (ignorato se --reference)")
    parser.add_argument("--n-cells", type=int, default=64,
                         help="Numero di celle spaziali (molto minore del dataset reale)")
    parser.add_argument("--n-timesteps-train", type=int, default=4001,
                         help="Timestep per i dataset di training (ignorato se --reference)")
    parser.add_argument("--n-timesteps-test", type=int, default=2001,
                         help="Timestep per i dataset di test (ignorato se --reference)")
    parser.add_argument("--n-train-sets", type=int, default=2, help="Numero di dataset di training da generare")
    parser.add_argument("--n-test-sets", type=int, default=6, help="Numero di dataset di test da generare")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)

    nf = args.nf
    n_timesteps_train = args.n_timesteps_train
    n_timesteps_test = args.n_timesteps_test
    if args.reference is not None:
        nf, n_timesteps_train, n_ref_sets = _infer_from_reference(args.reference)
        n_timesteps_test = max(1, n_timesteps_train // 2)
        print(f"Parametri copiati da {args.reference}: nf={nf}, n_timesteps_train={n_timesteps_train}")

    n_cells = args.n_cells
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {}
    for i in range(1, args.n_train_sets + 1):
        files[f"dummy_train_{i}.npy"] = (n_cells, nf, n_timesteps_train)
    for i in range(1, args.n_test_sets + 1):
        files[f"dummy_test_{i}.npy"] = (n_cells, nf, n_timesteps_test)

    for filename, shape in files.items():
        data = rng.randn(*shape).astype(np.float32)
        path = output_dir / filename
        np.save(path, data)
        size_mb = data.nbytes / (1024 ** 2)
        print(f"Creato {path} — shape: {data.shape}, dtype: {data.dtype}, {size_mb:.2f} MB")

        # forcing scalare phi allineato nel tempo
        nt = shape[-1]
        phi = rng.randn(nt).astype(np.float32)
        phi_path = output_dir / f"phi_{filename}"
        np.save(phi_path, phi)
        print(f"Creato {phi_path} — shape: {phi.shape}")

    # griglia dummy: (n_cells, 3), nessuna assunzione di struttura nz/nx
    grid = rng.rand(n_cells, 3).astype(np.float32)
    grid_path = output_dir / "grid.npy"
    np.save(grid_path, grid)
    print(f"Creato {grid_path} — shape: {grid.shape}")

    print(f"\nFatto! Dati dummy in: {output_dir}")
    print(f"n_cells={n_cells} (molto minore del dataset reale) -> peso ridotto per test rapidi della pipeline.")


if __name__ == "__main__":
    main()
