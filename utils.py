"""
utils.py
========
Tutte le utility condivise dalla pipeline (train.py / test.py / HPO.py).
Nessuna logica di architettura qui: quella sta esclusivamente in models.py.

Sezioni:
  - Riproducibilita' (seed, init pesi)
  - GPU / device resolution (1..N GPU sullo stesso nodo)
  - Scaling dati (wrapper openmeasure ROM)
  - Dataset latente (sliding windows sullo spazio latente, gia' proiettato)
  - CV / split windows (contiguous folds e random split per cv_folds=0)
  - Config / CLI: parsing argomenti, file "experiments.txt", override ricorsivi
  - Checkpoint: ricerca miglior checkpoint, validazione, plot loss trend
  - Metriche: R2, NRMSE
  - HPO -> JSON: merge dei migliori iperparametri in train.json / test.json
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Riproducibilita'
# ─────────────────────────────────────────────────────────────────────────────

def seed_everything(seed: int = 42, deterministic: bool = True) -> int:
    """Fissa i seed per Python, NumPy, PyTorch (e opzionalmente rende
    deterministiche le operazioni cuDNN)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        np.random.seed(seed)
    except Exception:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            try:
                torch.use_deterministic_algorithms(True)
            except Exception:
                pass
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass

    return seed


def init_weights(module) -> None:
    """Inizializzazione deterministica dei pesi (chiamare dopo seed_everything
    e dopo la costruzione del modello)."""
    import torch.nn as nn

    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(module.weight, nonlinearity="leaky_relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


# ─────────────────────────────────────────────────────────────────────────────
# GPU / device resolution — 1..N GPU sullo stesso nodo (default fino a 4)
# ─────────────────────────────────────────────────────────────────────────────

MAX_GPUS_DEFAULT = 4


def resolve_devices(requested_num_gpus: Optional[int] = None):
    """
    Determina la lista di device torch da usare.

    requested_num_gpus:
        None      -> usa 1 GPU se disponibile, altrimenti CPU
        0         -> forza CPU
        N (1..4)  -> usa min(N, GPU disponibili) GPU

    Ritorna (primary_device, device_list). device_list ha len>1 solo se sono
    state richieste ed effettivamente disponibili piu' GPU: in quel caso il
    chiamante puo' avvolgere il modulo con nn.DataParallel(device_ids=...).
    Non assume mai che siano disponibili esattamente 4 GPU.
    """
    import torch

    n_available = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if requested_num_gpus is None:
        n_use = 1 if n_available > 0 else 0
    else:
        n_use = max(0, int(requested_num_gpus))

    n_use = min(n_use, n_available, MAX_GPUS_DEFAULT)

    if n_use == 0:
        return torch.device("cpu"), [torch.device("cpu")]

    device_list = [torch.device(f"cuda:{i}") for i in range(n_use)]
    return device_list[0], device_list


def maybe_data_parallel(module, device_list):
    """Sposta `module` sul device primario e lo avvolge in nn.DataParallel
    se sono state risolte piu' GPU. Con 1 GPU (o CPU) si comporta come un
    semplice `.to(device)`."""
    import torch.nn as nn

    module = module.to(device_list[0])
    if len(device_list) > 1:
        module = nn.DataParallel(module, device_ids=[d.index for d in device_list])
    return module


def unwrap_module(module):
    """Ritorna il modulo "vero" anche se avvolto in nn.DataParallel
    (utile per accedere ad attributi custom o per lo state_dict)."""
    return module.module if hasattr(module, "module") else module


# ─────────────────────────────────────────────────────────────────────────────
# Scaling dati (wrapper attorno a openmeasure ROM)
# ─────────────────────────────────────────────────────────────────────────────

def scale_train_tensor(data_train_tensor: np.ndarray, grid: np.ndarray):
    """
    data_train_tensor : (Nt, Nf, n_cells)
    grid               : (n_cells, 3)
    """
    from openmeasure.sparse_sensing import ROM

    Nt, Nf, n_cells = data_train_tensor.shape
    data_train_mat = data_train_tensor.reshape(Nt, Nf * n_cells).T
    rom = ROM(data_train_mat, Nf, grid)
    data_train_mat_scaled = rom.scale_data(scale_type="range")
    data_train_scaled = data_train_mat_scaled.T.reshape(Nt, Nf, n_cells)
    return data_train_scaled, rom


def scale_test_tensor(data_test_tensor: np.ndarray, rom):
    """data_test_tensor : (Nt, Nf, n_cells)"""
    Nt, Nf, n_cells = data_test_tensor.shape
    data_test_mat = data_test_tensor.reshape(Nt, Nf * n_cells).T
    data_test_mat_scaled = (data_test_mat - rom.X_cnt) / rom.X_scl
    return data_test_mat_scaled.T.reshape(Nt, Nf, n_cells)


def rescale_back_output(output_tensor: np.ndarray, rom):
    """output_tensor : (Nt, Nf, n_cells)"""
    Nt, Nf, n_cells = output_tensor.shape
    output_mat = output_tensor.reshape(Nt, Nf * n_cells).T
    output_mat_rescaled = rom.X_scl * output_mat + rom.X_cnt
    return output_mat_rescaled.T.reshape(Nt, Nf, n_cells)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset latente — sliding windows estratte direttamente dallo spazio
# latente gia' proiettato (nessuna re-encoding per epoca/batch).
# ─────────────────────────────────────────────────────────────────────────────

class LatentWindowDataset:
    """
    Dataset map-style di finestre scorrevoli su coordinate latenti gia'
    calcolate (POD applicata una sola volta, upstream).

    latents : lista di array (Nt_i, r)  — uno per dataset/file di origine
    phis    : lista di array (Nt_i,)    — forcing scalare allineato nel tempo
    past_len, future_len -> lunghezza totale finestra = past_len + future_len

    Ogni elemento restituito: (z_window, phi_window) come tensori float32
        z_window   : (T, r)
        phi_window : (T,)

    Essendo i dati gia' in spazio latente (poche decine di float per
    snapshot) l'intero dataset di finestre e' minuscolo rispetto ai dati
    fisici originali: non serve un IterableDataset con worker sharding,
    un Dataset map-style e' sufficiente ed e' piu' efficiente/veloce.
    """

    def __init__(self, latents: List[np.ndarray], phis: List[np.ndarray],
                 past_len: int, future_len: int, indices: Optional[List[np.ndarray]] = None):
        import torch

        self._torch = torch
        self.past_len = past_len
        self.future_len = future_len
        self.total = past_len + future_len

        self._z = []
        self._phi = []
        self._offsets = []  # (dataset_idx, local_start_idx) per ogni sample globale

        for i, (z, phi) in enumerate(zip(latents, phis)):
            if phi.shape[0] != z.shape[0]:
                raise ValueError(f"Forcing length {phi.shape[0]} != Nt latente {z.shape[0]}")
            self._z.append(torch.as_tensor(z, dtype=torch.float32))
            self._phi.append(torch.as_tensor(phi, dtype=torch.float32))

            if indices is not None:
                valid = [idx for idx in indices[i] if idx + self.total <= z.shape[0]]
            else:
                n_samples = z.shape[0] - self.total + 1
                if n_samples <= 0:
                    raise ValueError(
                        f"Dataset latente troppo corto (Nt={z.shape[0]}) per "
                        f"past_len={past_len}, future_len={future_len}"
                    )
                valid = list(range(n_samples))

            for start in valid:
                self._offsets.append((i, int(start)))

    def __len__(self):
        return len(self._offsets)

    def __getitem__(self, idx):
        ds_idx, start = self._offsets[idx]
        z_window = self._z[ds_idx][start:start + self.total]
        phi_window = self._phi[ds_idx][start:start + self.total]
        return z_window, phi_window

    def subset(self, indices_per_dataset: List[np.ndarray]) -> "LatentWindowDataset":
        """Ricostruisce un nuovo LatentWindowDataset limitato a specifici
        indici di finestra per ciascun dataset sorgente (usato per CV / split)."""
        z_np = [t.numpy() for t in self._z]
        phi_np = [t.numpy() for t in self._phi]
        return LatentWindowDataset(z_np, phi_np, self.past_len, self.future_len,
                                    indices=indices_per_dataset)


# ─────────────────────────────────────────────────────────────────────────────
# CV / split delle finestre
# ─────────────────────────────────────────────────────────────────────────────

def build_contiguous_folds(n_samples: int, cv_folds: int):
    """
    Walk-forward validation split per dati temporali (nessun data leakage).
    Fold k (1-indexed): train = chunk_1..chunk_k, val = chunk_{k+1}.
    """
    if cv_folds < 2:
        return []

    n_chunks = cv_folds + 1
    if n_chunks > n_samples:
        raise ValueError(f"cv_folds={cv_folds} richiede almeno {n_chunks} campioni, ma n_samples={n_samples}")

    indices = np.arange(n_samples)
    chunk_sizes = np.full(n_chunks, n_samples // n_chunks, dtype=int)
    chunk_sizes[: n_samples % n_chunks] += 1
    boundaries = np.concatenate([[0], np.cumsum(chunk_sizes)])

    folds = []
    for k in range(cv_folds):
        train_end = boundaries[k + 1]
        val_start = boundaries[k + 1]
        val_end = boundaries[k + 2]
        folds.append((indices[:train_end], indices[val_start:val_end]))
    return folds


def build_embargo_kfold_splits(n_samples: int, cv_folds: int, window_total: int, embargo_steps: int = 0):
    """Purged Block K-Fold CV con Embargo.

    Divide n_samples in cv_folds blocchi contigui nel tempo. Per il fold i:
    il blocco i e' validation, i restanti cv_folds-1 blocchi sono training.
    Dal training vengono rimossi (purge + embargo):
      - tutte le finestre che si sovrappongono temporalmente al blocco di
        validation (purge: window_total-1 step prima del blocco)
      - un margine aggiuntivo di embargo_steps timestep sia subito prima
        sia subito dopo il blocco di validation (buffer di sicurezza contro
        correlazione seriale residua).
    Ritorna una lista di (train_idx, val_idx), indici di inizio finestra
    (snapshot-level, filtrati poi da LatentWindowDataset)."""
    if cv_folds < 2:
        raise ValueError(f"cv_folds deve essere >= 2, ricevuto {cv_folds}")

    chunk_sizes = np.full(cv_folds, n_samples // cv_folds, dtype=int)
    chunk_sizes[: n_samples % cv_folds] += 1
    boundaries = np.concatenate([[0], np.cumsum(chunk_sizes)])

    all_starts = np.arange(max(0, n_samples - window_total + 1))
    folds = []
    for i in range(cv_folds):
        v_start, v_end = int(boundaries[i]), int(boundaries[i + 1])
        val_idx = all_starts[(all_starts >= v_start) & (all_starts < v_end)]

        purge_left = window_total - 1
        excl_start = max(0, v_start - purge_left - embargo_steps)
        excl_end = min(n_samples, v_end + embargo_steps)
        train_idx = all_starts[(all_starts < excl_start) | (all_starts >= excl_end)]

        folds.append((train_idx, val_idx))
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Config / CLI comuni a train.py, test.py, HPO.py
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_MODEL_NAME = "pod-transformer"


def build_arg_parser(default_config: str, description: str = "") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=default_config, help="File di configurazione JSON")
    parser.add_argument("--experiments", default=None, help="File CSV con una riga per esperimento")
    parser.add_argument("--experiment-index", type=int, default=None,
                         help="Esegue solo la riga 0-based indicata di --experiments")
    parser.add_argument("--model", default=None,
                         help="Nome del modello (models.MODEL_REGISTRY). "
                              "Se omesso, viene letto da settings['model'] o dal default.")
    parser.add_argument("--gpus", type=int, default=None,
                         help="Numero di GPU da usare sullo stesso nodo (1..4). "
                              "Se omesso, usa settings['hardware']['num_gpus'] oppure 1 GPU.")
    return parser


def resolve_model_name(args, settings: Dict[str, Any]) -> str:
    return args.model or settings.get("model", DEFAULT_MODEL_NAME)


def resolve_num_gpus(args, settings: Dict[str, Any]) -> Optional[int]:
    if args.gpus is not None:
        return args.gpus
    return settings.get("hardware", {}).get("num_gpus")


def load_json(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(payload: Dict[str, Any], path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _parse_experiment_value(value: str):
    value = value.strip()
    lower = value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_experiments(experiments_path) -> List[Dict[str, Any]]:
    path = Path(experiments_path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(line for line in f if line.strip() and not line.lstrip().startswith("#")))

    if not rows:
        raise ValueError(f"Nessun esperimento trovato in {path}")

    headers = [h.strip() for h in rows[0]]
    if any(not h for h in headers):
        raise ValueError(f"Nome di iperparametro vuoto nell'header di {path}")

    experiments = []
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) != len(headers):
            raise ValueError(f"Riga {row_number} non valida in {path}: attesi {len(headers)} valori, trovati {len(row)}")
        experiments.append({h: _parse_experiment_value(v) for h, v in zip(headers, row)})
    return experiments


def _override_key(node, key, value) -> int:
    matches = 0
    if isinstance(node, dict):
        for child_key, child_value in node.items():
            if child_key == key:
                node[child_key] = value
                matches += 1
            else:
                matches += _override_key(child_value, key, value)
    elif isinstance(node, list):
        for item in node:
            matches += _override_key(item, key, value)
    return matches


def apply_overrides(base_settings: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    settings = deepcopy(base_settings)
    overrides = dict(overrides)
    trial_id = overrides.pop("trial_id", None)
    missing = []
    for key, value in overrides.items():
        if _override_key(settings, key, value) == 0:
            missing.append(key)
    if missing:
        raise KeyError("Iperparametri non trovati nel JSON di base: " + ", ".join(missing))

    # Il suffisso di output e' derivato DIRETTAMENTE dagli override applicati
    # in questa riga (non da una ricostruzione euristica di "architecture"/
    # "inference" nel JSON, che puo' non essere presente — es. test.json ora
    # deriva l'architettura dal checkpoint, non duplica quei campi).
    suffix = experiment_suffix(overrides, trial_id=trial_id)
    update_output_paths(settings, suffix=suffix)
    return settings


def experiment_suffix(overrides: Dict[str, Any], trial_id=None) -> str:
    """Costruisce un suffisso di cartella leggibile e univoco direttamente
    dagli iperparametri sovrascritti in questa riga di experiments.txt (o dal
    trial_id, per le righe generate da HPO)."""
    parts = []
    if trial_id is not None:
        parts.append(f"HPO_{trial_id}")
    for key, value in overrides.items():
        val_str = str(value).strip().replace(" ", "").replace(".", "p")
        parts.append(f"{key}{val_str}")
    return "_".join(parts) if parts else "exp"


def _replace_named_path(path_value, prefix, suffix) -> str:
    path = Path(path_value)
    return str(path.with_name(f"{prefix}_{suffix}"))


def update_output_paths(settings: Dict[str, Any], suffix: str) -> Dict[str, Any]:
    """Rende univoca la cartella di output per questo esperimento, sostituendo
    il nome dell'ultima componente del path con `<prefix>_<suffix>`. Chiamata
    SOLO quando si sta eseguendo un file --experiments (righe multiple che
    altrimenti scriverebbero tutte nella stessa cartella): un singolo run
    senza --experiments usa i path del JSON esattamente come scritti."""
    output = settings.get("output")
    if not isinstance(output, dict):
        return settings

    if "models_dir" in output:
        output["models_dir"] = _replace_named_path(output["models_dir"], "models", suffix)
    if "test_dir" in output:
        output["test_dir"] = _replace_named_path(output["test_dir"], "test", suffix)
    return settings


def iter_settings(config_path, experiments_path=None, experiment_index=None):
    """Yield (experiment_index, overrides, settings).

    Senza --experiments: una sola iterazione, settings = JSON cosi' com'e'
    (nessuna riscrittura dei path di output).
    Con --experiments: una iterazione per riga del CSV (o solo quella
    selezionata da --experiment-index), con path di output resi univoci per
    evitare che righe diverse si sovrascrivano a vicenda."""
    base_settings = load_json(config_path)
    if not experiments_path:
        yield None, {}, base_settings
        return

    experiments = load_experiments(experiments_path)
    selected = list(enumerate(experiments))
    if experiment_index is not None:
        if experiment_index < 0 or experiment_index >= len(experiments):
            raise IndexError(f"experiment-index {experiment_index} fuori range [0, {len(experiments) - 1}]")
        selected = [(experiment_index, experiments[experiment_index])]

    for index, overrides in selected:
        yield index, overrides, apply_overrides(base_settings, overrides)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint: ricerca / validazione / plot loss trend
# ─────────────────────────────────────────────────────────────────────────────

def _checkpoint_pattern(checkpoint_label: str):
    return re.compile(re.escape(checkpoint_label) + r"_(?P<epoch>\d+)_(?P<loss>\d+\.\d+)\.pt")


def find_best_checkpoint(models_dir, checkpoint_label: str, plot_start_epoch: int, output_dir):
    """Trova il checkpoint con loss minima e salva il plot del loss trend.
    Ritorna (best_file: Path, min_loss: float)."""
    models_path = Path(models_dir)
    if not models_path.exists():
        raise FileNotFoundError(f"models_dir non esiste: {models_path}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    pattern = _checkpoint_pattern(checkpoint_label)
    epochs, losses, files = [], [], []
    for f in models_path.glob(f"{checkpoint_label}_*_*.pt"):
        m = pattern.match(f.name)
        if not m:
            continue
        try:
            epoch, loss = int(m.group("epoch")), float(m.group("loss"))
        except ValueError:
            continue
        if not math.isfinite(loss):
            continue
        epochs.append(epoch)
        losses.append(loss)
        files.append(f)

    if not epochs:
        raise FileNotFoundError(f"Nessun checkpoint '{checkpoint_label}_{{epoch}}_{{loss}}.pt' trovato in {models_path}")

    data = sorted(zip(epochs, losses, files), key=lambda x: x[0])
    epochs, losses, files = map(list, zip(*data))

    min_loss = min(losses)
    best_file = files[losses.index(min_loss)]

    filtered = [(e, l) for e, l in zip(epochs, losses) if e >= plot_start_epoch]
    plot_loss_trend(filtered, Path(output_dir) / "loss_trend.png", title="Loss trend (test)")

    return best_file, min_loss


def plot_loss_trend(epoch_loss_pairs, out_path, title="Training loss"):
    """Plot dettagliato del trend delle loss: loss principale + eventuali
    metriche non pesate (reconstruction/latent/mass) se presenti nella history."""
    import matplotlib.pyplot as plt

    if not epoch_loss_pairs:
        return

    if isinstance(epoch_loss_pairs[0], dict):
        epochs = [e["epoch"] for e in epoch_loss_pairs]
        train_loss = [e["total"] for e in epoch_loss_pairs]
        val_loss = [e.get("val") for e in epoch_loss_pairs]
        has_val = any(v is not None for v in val_loss)

        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
        axs[0].plot(epochs, train_loss, label="train", linewidth=1.3)
        if has_val:
            axs[0].plot(epochs, [v for v in val_loss], label="val", linewidth=1.3)
        axs[0].set_yscale("log")
        axs[0].set_xlabel("Epoch")
        axs[0].set_ylabel("Loss")
        axs[0].set_title(title)
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)

        components = ["val_reconstruction_pod", "val_latent_pred"]
        plotted_any = False
        for comp in components:
            xs = [e["epoch"] for e in epoch_loss_pairs if e.get(comp) is not None]
            ys = [e[comp] for e in epoch_loss_pairs if e.get(comp) is not None]
            if xs:
                axs[1].plot(xs, ys, marker="o", markersize=3, label=comp.replace("val_", ""))
                plotted_any = True
        axs[1].set_yscale("log")
        axs[1].set_xlabel("Epoch")
        axs[1].set_ylabel("Unweighted loss")
        axs[1].set_title("Unweighted validation losses (last epochs)")
        axs[1].grid(True, alpha=0.3)
        if plotted_any:
            axs[1].legend()
        else:
            axs[1].text(0.5, 0.5, "no data", ha="center", va="center", transform=axs[1].transAxes)

        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
    else:
        epochs, losses = zip(*epoch_loss_pairs)
        plt.figure()
        plt.plot(epochs, losses)
        plt.yscale("log")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Loss trend plot saved to: {out_path}")


def cleanup_checkpoints(models_dir, checkpoint_label: str, keep_best: int = 20, last_fraction: float = 0.25):
    """Mantiene solo i migliori `keep_best` checkpoint nell'ultimo
    `last_fraction` delle epoche, elimina gli altri. Comportamento invariato
    rispetto alla versione originale."""
    models_path = Path(models_dir)
    pattern = _checkpoint_pattern(checkpoint_label)

    epochs, losses, files = [], [], []
    for f in models_path.glob(f"{checkpoint_label}_*_*.pt"):
        m = pattern.match(f.name)
        if not m:
            continue
        try:
            epoch, loss = int(m.group("epoch")), float(m.group("loss"))
        except ValueError:
            continue
        if not math.isfinite(loss):
            continue
        epochs.append(epoch)
        losses.append(loss)
        files.append(f)

    if not epochs:
        print("WARNING: nessun checkpoint trovato per il cleanup.")
        return

    max_epoch = max(epochs)
    threshold_epoch = int(max_epoch * (1.0 - last_fraction))
    candidates = [i for i, e in enumerate(epochs) if e >= threshold_epoch]
    if not candidates:
        print("WARNING: nessun checkpoint nell'ultima frazione di epoche. Nessuna eliminazione.")
        return

    candidates_sorted = sorted(candidates, key=lambda i: losses[i])
    best_indices = set(candidates_sorted[:keep_best])

    n_deleted = 0
    for i, f in enumerate(files):
        if i not in best_indices:
            f.unlink()
            n_deleted += 1

    print(f"Checkpoint cleanup: mantenuti {len(best_indices)} migliori (epoch >= {threshold_epoch}), eliminati {n_deleted}.")


# ─────────────────────────────────────────────────────────────────────────────
# Metriche
# ─────────────────────────────────────────────────────────────────────────────

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def nrmse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    norm = np.max(y_true) - np.min(y_true)
    return rmse / norm if norm > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# HPO -> JSON: propaga i migliori iperparametri in train.json / test.json
# ─────────────────────────────────────────────────────────────────────────────

def apply_best_hpo_params(best_params: Dict[str, Any], train_json_path, test_json_path) -> None:
    """
    Aggiorna train.json e test.json con i migliori iperparametri trovati
    dall'HPO. Scrive solo le chiavi effettivamente presenti in best_params,
    usando lo stesso meccanismo di override annidato usato per experiments.txt
    (niente duplicazione di logica).
    """
    train_settings = load_json(train_json_path)
    for key, value in best_params.items():
        _override_key(train_settings, key, value)
    save_json(train_settings, train_json_path)
    print(f"[HPO] train.json aggiornato: {train_json_path}")

    test_path = Path(test_json_path)
    if test_path.exists():
        test_settings = load_json(test_path)
        # test.json contiene solo i parametri di dataset/inferenza: propaga
        # solo le chiavi di architettura pertinenti (rank_POD escluso: non
        # ottimizzato dall'HPO).
        for key, value in best_params.items():
            _override_key(test_settings, key, value)
        save_json(test_settings, test_path)
        print(f"[HPO] test.json aggiornato: {test_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Memoria
# ─────────────────────────────────────────────────────────────────────────────

def free_memory(*objects) -> None:
    """Cancella esplicitamente gli oggetti passati, forza il garbage
    collector e libera la cache CUDA (se disponibile)."""
    for obj in objects:
        del obj
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass