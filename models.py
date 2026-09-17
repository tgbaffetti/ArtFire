"""
models.py
=========
Esclusivamente logica di architettura/modelli. Nessuna orchestrazione I/O,
nessun parsing di config o CLI: quello vive in train.py / test.py / HPO.py.
Tutte le utility condivise (seed, scaling, dataset, checkpoint, metriche)
vivono in utils.py.

Per aggiungere un nuovo modello (es. "vae-transformer"):
    1. Implementare una classe con l'interfaccia descritta in
       `_ModelInterface` qui sotto.
    2. Registrarla con `@register_model("vae-transformer")`.
    3. train.py / test.py / HPO.py la useranno automaticamente selezionando
       "model": "vae-transformer" (o --model vae-transformer) nella config,
       senza bisogno di alcuna modifica a train.py/test.py/HPO.py.
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.utils.extmath import randomized_svd

Device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Model registry — permette a train.py/test.py/HPO.py di essere model-agnostic
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY: Dict[str, type] = {}


def register_model(name: str):
    def _decorator(cls):
        MODEL_REGISTRY[name] = cls
        cls.model_name = name
        return cls
    return _decorator


def get_model_class(name: str) -> type:
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f"Modello '{name}' non registrato. Modelli disponibili: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[name]


class _ModelInterface:
    """Contratto informale richiesto da train.py / test.py / HPO.py.
    Non e' una vera ABC per restare leggero, ma ogni nuovo modello deve
    implementare questi metodi con la stessa firma:

        classmethod from_config(arch_cfg, training_cfg, device) -> istanza
        prepare_data(cls, dataset_cfg, device) -> dict di dati pronti
        set_datasets(self, train_dataset, val_dataset=None)
        set_optimizer(self, lr, ...)
        train(self, ...) -> (elapsed_time, epoch_loss_history)
        evaluate(self, dataset, ...) -> float
        evaluate_unweighted_losses(self, dataset, ...) -> dict
        predict_next(self, ...) -> np.ndarray
        save_checkpoint(self, path, epoch, loss, settings, extra=None)
        classmethod load_checkpoint(path, device) -> (istanza, checkpoint_dict)
        hpo_stages -> lista di nomi di "stadi" ottimizzabili dall'HPO
                      (per pod-transformer: ["transformer"])
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks generici (riutilizzabili da future architetture, es. VAE)
# ─────────────────────────────────────────────────────────────────────────────

class Encoder(nn.Module):
    """Encoder convoluzionale generico 2D, riusabile da future architetture
    (es. un futuro vae-transformer). Non e' usato da pod-transformer."""

    def __init__(self, n_features, channels, pool_mode="avg",
                 activation=nn.LeakyReLU, padding_mode="circular"):
        super().__init__()
        cs = [n_features] + channels
        layers = []
        for i in range(len(cs) - 1):
            if pool_mode == "stride":
                layers.append(nn.Conv2d(cs[i], cs[i + 1], kernel_size=4, stride=2,
                                         padding=1, padding_mode=padding_mode))
            else:
                layers.append(nn.Conv2d(cs[i], cs[i + 1], kernel_size=3, stride=1,
                                         padding=1, padding_mode=padding_mode))
            layers.append(nn.BatchNorm2d(cs[i + 1]))
            layers.append(activation())
            if pool_mode == "avg":
                layers.append(nn.AvgPool2d(2))
            elif pool_mode == "max":
                layers.append(nn.MaxPool2d(2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """Decoder convoluzionale generico 2D, riusabile da future architetture.
    Non e' usato da pod-transformer."""

    def __init__(self, n_features, channels, up_mode="bilinear",
                 activation=nn.LeakyReLU, padding_mode="circular"):
        super().__init__()
        cs = channels + [n_features]
        layers = []
        for i in range(len(cs) - 1):
            is_last = i == len(cs) - 2
            act = nn.Identity if is_last else activation
            if up_mode == "stride":
                layers.append(nn.ConvTranspose2d(cs[i], cs[i + 1], kernel_size=4, stride=2, padding=1))
            else:
                layers.append(nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False))
                layers.append(nn.Conv2d(cs[i], cs[i + 1], kernel_size=3, stride=1,
                                         padding=1, padding_mode=padding_mode))
            if not is_last:
                layers.append(nn.BatchNorm2d(cs[i + 1]))
            layers.append(act())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# TransformerSequence — Transformer su sequenza di token scalari (POD coeffs)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerSequence(nn.Module):
    """
    Transformer con sequenza di token scalari.

    Sequenza input:
        [phi_{t-n+1}, a_{t-n+1,1}, ..., a_{t-n+1,r},
         ...
         phi_t,       a_{t,1},     ..., a_{t,r},
         phi_{t+1},
         PRED_1, PRED_2, ..., PRED_r]

    Lunghezza sequenza: L = n_past*(r+1) + 1 + r

    Token types (type embedding):
        0 = phi
        1 = coefficiente POD modo k (k=1..r)
        2 = PRED

    Temporal segments (segment embedding):
        0 = passato   1 = futuro (incluso phi_{t+1})

    Output:
        delta_a : (B, r) — residuo predetto per ogni modo POD

    Dropout (leggero, di default disattivato se non specificato):
        - `dropout` passato a nn.MultiheadAttention (attention dropout)
        - `dropout` applicato anche sull'embedding di input (token latenti)
          prima dei layer transformer, per ridurre overfitting/memorizzazione.
    """

    TOKEN_PHI = 0
    TOKEN_POD = 1
    TOKEN_PRED = 2
    SEG_PAST = 0
    SEG_FUTURE = 1

    def __init__(self, rank_pod, n_past, n_layers=4, embed_dim=128, num_heads=8,
                 hidden_dim=256, dropout: float = 0.0):
        super().__init__()

        self.rank_pod = rank_pod
        self.n_past = n_past
        self.embed_dim = embed_dim
        self.dropout_p = dropout
        self.L = n_past * (rank_pod + 1) + 1 + rank_pod

        self.in_proj_phi = nn.Linear(1, embed_dim)
        self.in_proj_pod = nn.Linear(1, embed_dim)
        self.in_proj_pred = nn.Linear(1, embed_dim)

        self.type_embed = nn.Embedding(3, embed_dim)
        self.pos_embed = nn.Embedding(self.L, embed_dim)
        self.seg_embed = nn.Embedding(2, embed_dim)

        self.pred_token = nn.Parameter(torch.zeros(rank_pod, 1))
        self.input_dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            nn.ModuleDict({
                "mha": nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                              batch_first=True, dropout=dropout),
                "ff": nn.Sequential(
                    nn.Linear(embed_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, embed_dim),
                ),
                "norm1": nn.LayerNorm(embed_dim),
                "norm2": nn.LayerNorm(embed_dim),
            })
            for _ in range(n_layers)
        ])

        self.out_proj = nn.Linear(embed_dim, 1)

    def _build_type_ids(self, device):
        ids = []
        for _ in range(self.n_past):
            ids.append(self.TOKEN_PHI)
            ids.extend([self.TOKEN_POD] * self.rank_pod)
        ids.append(self.TOKEN_PHI)
        ids.extend([self.TOKEN_PRED] * self.rank_pod)
        return torch.tensor(ids, dtype=torch.long, device=device)

    def _build_seg_ids(self, device):
        n_past_tokens = self.n_past * (self.rank_pod + 1)
        ids = [self.SEG_PAST] * n_past_tokens + [self.SEG_FUTURE] * (1 + self.rank_pod)
        return torch.tensor(ids, dtype=torch.long, device=device)

    def forward(self, a_history, phi_context, return_attn=False):
        """
        a_history   : lista di n_past tensori (B, r) — coordinate POD passate
        phi_context : (B, n_past+1)                  — phi da t-n_past+1 a t+1
        """
        B = a_history[0].shape[0]
        device = a_history[0].device

        embedded = []
        for i in range(self.n_past):
            embedded.append(self.in_proj_phi(phi_context[:, i:i + 1]))
            for k in range(self.rank_pod):
                embedded.append(self.in_proj_pod(a_history[i][:, k:k + 1]))
        embedded.append(self.in_proj_phi(phi_context[:, self.n_past:self.n_past + 1]))
        x_past_emb = torch.stack(embedded, dim=1)  # (B, L-r, embed_dim)

        pred_emb = self.in_proj_pred(self.pred_token.unsqueeze(0).expand(B, -1, -1))  # (B, r, embed_dim)

        x = torch.cat([x_past_emb, pred_emb], dim=1)  # (B, L, embed_dim)

        pos_ids = torch.arange(self.L, device=device)
        type_ids = self._build_type_ids(device)
        seg_ids = self._build_seg_ids(device)

        x = (x + self.type_embed(type_ids).unsqueeze(0)
               + self.pos_embed(pos_ids).unsqueeze(0)
               + self.seg_embed(seg_ids).unsqueeze(0))
        x = self.input_dropout(x)

        attn_weights_all = []
        for layer in self.layers:
            if return_attn:
                attn_out, attn_w = layer["mha"](x, x, x, need_weights=True, average_attn_weights=True)
                attn_weights_all.append(attn_w)
            else:
                attn_out, _ = layer["mha"](x, x, x, need_weights=False)
            x = layer["norm1"](x + attn_out)
            x_ff = layer["ff"](x)
            x = layer["norm2"](x + x_ff)

        pred_out = x[:, -self.rank_pod:, :]
        delta_a = self.out_proj(pred_out).squeeze(-1)

        if return_attn:
            return delta_a, attn_weights_all
        return delta_a


# ─────────────────────────────────────────────────────────────────────────────
# CustomLoss — reconstruction / latent prediction / mass conservation
# ─────────────────────────────────────────────────────────────────────────────

class CustomLoss:
    """
    Termini disponibili:
      - reconstruction     : MSE(decode(encode(x)), x)
      - latent_pred         : MSE(z_pred, z_true)
      - mass_conservation   : errore di conservazione della massa (richiede rom)

    Spazio fisico atteso: (B, Nf, n_cells) — nessuna assunzione di griglia
    strutturata (Nz, Nx): il dominio spaziale e' unicamente n_cells.
    """

    def __init__(self, weight_reconstruction: float = 1.0, weight_latent_pred: float = 1.0,
                 weight_mass_conservation_transformer: float = 0.0,
                 weight_mass_conservation_cae: float = 0.0,
                 base_criterion=nn.MSELoss(), rom=None, device=None):
        self.weight_reconstruction = weight_reconstruction
        self.weight_latent_pred = weight_latent_pred
        self.weight_mass_conservation_transformer = weight_mass_conservation_transformer
        self.weight_mass_conservation_cae = weight_mass_conservation_cae
        self.criterion = base_criterion
        self.rom = rom
        self.device = device

    def _to_torch(self, arr):
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def _rescale(self, x_scaled):
        """x_scaled : (B, Nf, n_cells) -> spazio fisico (differenziabile)."""
        if self.rom is None:
            raise ValueError("rom=None: passare l'oggetto rom a CustomLoss per usare mass_conservation_loss.")
        B, Nf, n_cells = x_scaled.shape
        X_scl = self._to_torch(self.rom.X_scl).reshape(Nf, n_cells).unsqueeze(0)
        X_cnt = self._to_torch(self.rom.X_cnt).reshape(Nf, n_cells).unsqueeze(0)
        return X_scl * x_scaled + X_cnt

    def reconstruction_loss(self, x_pred_scaled, x_true_scaled):
        return self.criterion(x_pred_scaled, x_true_scaled)

    def latent_prediction_loss(self, z_pred, z_true):
        return self.criterion(z_pred, z_true)

    def mass_conservation_loss(self, x_pred_scaled):
        """x_pred_scaled : (B, Nf, n_cells). Richiede rom != None.
        NOTA: assume che le prime 5 feature siano ("rho","p","U1","U3","T")
        da escludere dalla somma di massa; se il dataset cambia, aggiornare
        l'indice di slicing qui sotto."""
        x_pred_physical = self._rescale(x_pred_scaled)
        B, Nf, n_cells = x_pred_physical.shape
        total_mass_pred = x_pred_physical[:, 5:, :].sum(dim=1)
        total_mass_true = torch.ones(B, n_cells, dtype=total_mass_pred.dtype, device=total_mass_pred.device)
        return self.criterion(total_mass_pred, total_mass_true)

    def __call__(self, *, x_rec=None, x_ref=None, z_pred=None, z_true=None,
                 x_phys_pred=None, source="transformer"):
        device = z_pred.device if z_pred is not None else (x_rec.device if x_rec is not None else torch.device("cpu"))
        total = torch.tensor(0.0, device=device)
        self.last = {}

        if self.weight_reconstruction > 0 and x_rec is not None:
            l1 = self.reconstruction_loss(x_rec, x_ref)
            total = total + self.weight_reconstruction * l1
            self.last["reconstruction"] = l1.item()

        if self.weight_latent_pred > 0 and z_pred is not None and z_true is not None:
            l2 = self.latent_prediction_loss(z_pred, z_true)
            total = total + self.weight_latent_pred * l2
            self.last["latent_pred"] = l2.item()

        if source == "transformer":
            mass_weight = self.weight_mass_conservation_transformer
            if mass_weight > 0 and x_phys_pred is not None:
                l3 = self.mass_conservation_loss(x_phys_pred)
                total = total + mass_weight * l3
                self.last["mass_conservation"] = l3.item()

        if source == "cae":
            mass_weight = self.weight_mass_conservation_cae
            if mass_weight > 0 and x_rec is not None:
                l3 = self.mass_conservation_loss(x_rec)
                total = total + mass_weight * l3
                self.last["mass_conservation"] = l3.item()

        self.last["total"] = total.item()
        return total


# ─────────────────────────────────────────────────────────────────────────────
# PODReducer — riduzione POD (SVD randomizzata economy)
# ─────────────────────────────────────────────────────────────────────────────

class PODReducer:
    """
    Riduzione POD su una matrice di snapshot.

    Convenzioni dimensionali (dataset unificato [n_cells, Nf, Nt]):
        - snapshot fisico   : (Nt, Nf, n_cells)
        - matrice snapshot  : X (D, Nt)  con D = Nf * n_cells
        - proiettore        : U_r (D, r)
        - coordinate ridotte: At_r (r, Nt)

    Calcola una sola volta all'inizio del training/HPO (`fit`), poi
    encode/decode sono semplici proiezioni lineari (economiche).
    """

    def __init__(self, rank: int):
        self.rank = rank
        self.U_r: Optional[np.ndarray] = None
        self.mean: Optional[np.ndarray] = None
        self.reconstruction_error: Optional[float] = None  # MSE(decode(encode(X)), X), calcolato una volta in fit()

    @property
    def effective_rank(self) -> int:
        """Rank effettivo dopo il fit (puo' essere < rank richiesto se
        limitato da min(D, Nt))."""
        return self.U_r.shape[1] if self.U_r is not None else self.rank

    def fit(self, snapshots: np.ndarray) -> np.ndarray:
        """snapshots : (Nt, Nf, n_cells) — dati di training gia' scalati.
        Ritorna At_r : (r, Nt)."""
        Nt, Nf, n_cells = snapshots.shape
        D = Nf * n_cells

        X = snapshots.reshape(Nt, D).T  # (D, Nt)
        self.mean = X.mean(axis=1, keepdims=True)
        X_c = X - self.mean

        r = min(self.rank, min(D, Nt))
        U_r, S_r, Vt_r = randomized_svd(X_c, n_components=r, n_oversamples=20, n_iter=7, random_state=42)

        self.U_r = U_r
        At_r = S_r[:, None] * Vt_r

        # Errore di ricostruzione POD: dipende solo dal rank, calcolato una
        # sola volta qui (riusato ovunque, niente ricalcoli ripetuti).
        X_rec = U_r @ (U_r.T @ X_c)
        self.reconstruction_error = float(np.mean((X_c - X_rec) ** 2))

        return At_r

    def encode_np(self, x: np.ndarray) -> np.ndarray:
        """x : (Nf, n_cells) oppure (N, Nf, n_cells) -> (r,) oppure (r, N)."""
        assert self.U_r is not None, "PODReducer non ancora fittato."
        single = x.ndim == 2
        if single:
            x = x[np.newaxis]
        N = x.shape[0]
        D = int(np.prod(x.shape[1:]))
        X = x.reshape(N, D).T
        X_c = X - self.mean
        At = self.U_r.T @ X_c
        return At[:, 0] if single else At

    def to_torch(self, device):
        if not hasattr(self, "_cached_device") or self._cached_device != str(device):
            self._U_r_t = torch.tensor(self.U_r, dtype=torch.float32, device=device)
            self._mean_t = torch.tensor(self.mean.flatten(), dtype=torch.float32, device=device)
            self._cached_device = str(device)

    def encode_torch(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, Nf, n_cells) -> (B, r)."""
        assert self.U_r is not None
        B = x.shape[0]
        D = int(np.prod(x.shape[1:]))
        self.to_torch(x.device)
        x_flat = x.reshape(B, D)
        x_c = x_flat - self._mean_t.unsqueeze(0)
        return x_c @ self._U_r_t

    def decode_np(self, a: np.ndarray, shape_out) -> np.ndarray:
        """a : (r,) oppure (r, N); shape_out=(Nf, n_cells)."""
        assert self.U_r is not None
        single = a.ndim == 1
        if single:
            a = a[:, np.newaxis]
        X_c = self.U_r @ a
        X = X_c + self.mean
        N = X.shape[1]
        out = X.T.reshape(N, *shape_out)
        return out[0] if single else out

    def decode_torch(self, a: torch.Tensor, shape_out) -> torch.Tensor:
        """a : (B, r); shape_out=(Nf, n_cells) -> (B, Nf, n_cells)."""
        assert self.U_r is not None
        B = a.shape[0]
        self.to_torch(a.device)
        X_c = a @ self._U_r_t.T
        X = X_c + self._mean_t.unsqueeze(0)
        return X.reshape(B, *shape_out)


def _load_scale_and_concat(dataset_cfg: Dict[str, Any]):
    """Helper condiviso da ogni modello che ha bisogno dei dati fisici
    scalati e concatenati (POD, AE, ...): carica i dataset grezzi
    [n_cells, Nf, Nt], valida le shape, carica/scala il forcing phi, scala
    i dati fisici (ROM) e li concatena. Non fa alcuna riduzione dimensionale
    (POD o AE): quella e' responsabilita' di ciascun modello.

    Ritorna: scaled_concat (Nt_tot, Nf, n_cells), rom, phi_scaled_list,
    phi_mean, phi_std, nf_ref, n_cells_ref, nts (lista lunghezze per dataset).
    """
    import utils

    data_paths = dataset_cfg["data_paths"]
    phi_paths = dataset_cfg["phi_paths"]
    grid_path = dataset_cfg["grid_path"]
    if len(data_paths) != len(phi_paths):
        raise ValueError("data_paths e phi_paths devono avere la stessa lunghezza")

    raw_list, phi_list, nts = [], [], []
    nf_ref = n_cells_ref = None
    for data_path, phi_path in zip(data_paths, phi_paths):
        arr = np.load(data_path)  # (n_cells, Nf, Nt)
        if arr.ndim != 3:
            raise ValueError(f"ERRORE: shape inattesa {arr.shape} per {data_path}, attese 3 dimensioni [n_cells, Nf, Nt]")
        n_cells, nf, nt = arr.shape
        if nf_ref is None:
            nf_ref, n_cells_ref = nf, n_cells
        elif (nf, n_cells) != (nf_ref, n_cells_ref):
            raise ValueError(
                f"ERRORE: {data_path} ha (Nf,n_cells)=({nf},{n_cells}), atteso ({nf_ref},{n_cells_ref})"
            )

        phi = np.load(phi_path).astype(np.float32)
        if phi.shape != (nt,):
            raise ValueError(f"ERRORE: shape phi inattesa {phi.shape}, attesa {(nt,)} per {phi_path}")

        data_txf = np.transpose(arr, (2, 1, 0)).astype(np.float32)  # (Nt, Nf, n_cells)
        print(f"Dataset caricato: {Path(data_path).name} -> [n_cells={n_cells}, Nf={nf}, Nt={nt}]")

        raw_list.append(data_txf)
        phi_list.append(phi)
        nts.append(nt)

    grid = np.load(grid_path)
    if grid.shape[0] != n_cells_ref:
        raise ValueError(f"ERRORE: grid shape {grid.shape} incompatibile con n_cells={n_cells_ref}")

    phi_concat = np.concatenate(phi_list, axis=0)
    phi_mean = float(phi_concat.mean())
    phi_std = float(phi_concat.std())
    if phi_std <= 0.0:
        raise ValueError("ERRORE: phi_std deve essere positivo")
    phi_scaled_list = [(phi - phi_mean) / phi_std for phi in phi_list]

    data_concat = np.concatenate(raw_list, axis=0)
    print(f"Scaling dati ({sum(nts)} snapshot totali, {len(nts)} dataset)...")
    scaled_concat, rom = utils.scale_train_tensor(data_concat, grid)
    utils.free_memory(data_concat, raw_list)

    return scaled_concat, rom, phi_scaled_list, phi_mean, phi_std, nf_ref, n_cells_ref, nts


# ─────────────────────────────────────────────────────────────────────────────
# PODTransformerModel — modello "pod-transformer" (Transformer nello spazio
# latente POD, con forcing scalare phi)
# ─────────────────────────────────────────────────────────────────────────────

@register_model("pod-transformer")
class PODTransformerModel:
    """
    Modello POD + Transformer per next-step forecasting con forcing scalare.

    Il Transformer opera ESCLUSIVAMENTE su coordinate latenti gia' proiettate
    (nessun encode/decode ripetuto per batch): le sliding windows vengono
    costruite direttamente sull'array latente pre-calcolato (vedi
    utils.LatentWindowDataset), la proiezione POD e' one-shot.

    Training iterativo (rollout): dato un contesto di n_past coordinate
    latenti passate, il Transformer predice un delta che, sommato all'ultima
    coordinata nota, produce lo step successivo; la finestra scorre in avanti
    (si rimuove il primo elemento, si aggiunge la predizione) e si ripete per
    `rollout_steps` iterazioni. La loss totale e' la somma (pesata da
    `rollout_discount`, di default 1.0 = somma semplice) delle loss delle
    singole predizioni. L'autograd propaga correttamente attraverso l'intero
    rollout (nessun detach delle predizioni intermedie).
    """

    def __init__(self, reducer: PODReducer, transformer: nn.Module, device=Device):
        self.reducer = reducer
        self.device = device
        self._device_list = [device]
        self.transformer = transformer.to(device)
        self.snapshot_shape: Optional[Tuple[int, int]] = None  # (Nf, n_cells)
        self.arch_cfg: Dict[str, Any] = {}
        self.training_cfg: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Costruzione da config (interfaccia richiesta da train/test/HPO)
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, arch_cfg: Dict[str, Any], training_cfg: Dict[str, Any],
                     device_list, latent_dim: Optional[int] = None) -> "PODTransformerModel":
        rank = latent_dim if latent_dim is not None else arch_cfg["rank_POD"]
        n_past_eff = training_cfg["n_past"]

        transformer = TransformerSequence(
            rank_pod=rank,
            n_past=n_past_eff,
            n_layers=arch_cfg["transformer_layers"],
            embed_dim=arch_cfg["transformer_embed_dim"],
            num_heads=arch_cfg["transformer_heads"],
            hidden_dim=arch_cfg["transformer_hidden_dim"],
            dropout=float(training_cfg.get("dropout", 0.0)),
        )

        from utils import init_weights, maybe_data_parallel
        transformer.apply(init_weights)
        transformer = maybe_data_parallel(transformer, device_list)

        reducer = PODReducer(rank=rank)
        model = cls(reducer=reducer, transformer=transformer, device=device_list[0])
        model._device_list = device_list
        model.arch_cfg = dict(arch_cfg)
        model.training_cfg = dict(training_cfg)
        return model

    @property
    def _raw_transformer(self) -> nn.Module:
        """Il modulo transformer "vero", anche se avvolto in DataParallel."""
        from utils import unwrap_module
        return unwrap_module(self.transformer)

    # ------------------------------------------------------------------
    # Preparazione dati: caricamento, scaling, POD fit + proiezione one-shot
    # ------------------------------------------------------------------
    @classmethod
    def prepare_data(cls, dataset_cfg: Dict[str, Any], latent_dim: int, device) -> Dict[str, Any]:
        """
        Carica i dataset grezzi, li scala (ROM), fitta la POD UNA VOLTA sola
        sull'intero dataset concatenato e proietta l'intero dataset nello
        spazio latente UNA VOLTA sola. Ritorna un dizionario pronto per
        costruire le sliding windows in spazio latente (nessun dato fisico
        resta in memoria dopo la chiamata, se non richiesto da chiamante).

        Shape dataset attesa: [n_cells, Nf, Nt] su disco -> qui trasposta
        internamente a (Nt, Nf, n_cells) per comodita' di indicizzazione
        temporale.
        """
        import utils

        scaled_concat, rom, phi_scaled_list, phi_mean, phi_std, nf_ref, n_cells_ref, nts = \
            _load_scale_and_concat(dataset_cfg)

        rank = min(latent_dim, nf_ref * n_cells_ref, sum(nts))
        pod = PODReducer(rank=rank)
        print(f"Fitting POD (rank={rank}) sull'intero dataset di training...")
        pod.fit(scaled_concat)
        print(f"POD fit completo. Rank effettivo: {pod.U_r.shape[1]}. "
              f"Errore di ricostruzione POD (MSE): {pod.reconstruction_error:.6e}")

        print("Proiezione one-shot dell'intero dataset nello spazio latente...")
        latent_full = pod.encode_np(scaled_concat)  # (r, Nt_tot)
        latent_full = latent_full.T.astype(np.float32)  # (Nt_tot, r)

        split_points = np.cumsum(nts)[:-1]
        latent_list = np.split(latent_full, split_points, axis=0)

        utils.free_memory(scaled_concat)

        return {
            "reducer": pod,
            "rom": rom,
            "latent_list": latent_list,
            "phi_scaled_list": phi_scaled_list,
            "phi_mean": phi_mean,
            "phi_std": phi_std,
            "snapshot_shape": (nf_ref, n_cells_ref),
            "nts": nts,
        }

    # ------------------------------------------------------------------
    # Dataset / optimizer
    # ------------------------------------------------------------------
    def set_datasets(self, train_dataset, val_dataset=None):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.has_val = val_dataset is not None

    def set_optimizer(self, lr=1e-3):
        self.optimizer = optim.Adam(self.transformer.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Predizione one-step nello spazio latente
    # ------------------------------------------------------------------
    def _predict_next_latent(self, latent_history: List[torch.Tensor], n_past: int,
                              phi_context: torch.Tensor) -> torch.Tensor:
        a_history = latent_history[-n_past:]
        delta = self.transformer(a_history, phi_context)
        return latent_history[-1] + delta

    # ------------------------------------------------------------------
    # Loss di rollout iterativo (train + eval condividono la stessa logica)
    # ------------------------------------------------------------------
    def _compute_rollout_loss(self, z_seq: torch.Tensor, phi_seq: torch.Tensor, criterion,
                               n_past: int, rollout_steps: int, rollout_discount: float = 1.0) -> torch.Tensor:
        """
        z_seq   : (B, T, r)  T = n_past + rollout_steps  — finestra latente
        phi_seq : (B, T)
        Ritorna la somma (pesata da rollout_discount**step) delle loss di
        ciascuno degli `rollout_steps` step predetti autoregressivamente.
        """
        latent_history = [z_seq[:, i] for i in range(n_past)]
        total_loss = torch.tensor(0.0, device=self.device)

        for step in range(rollout_steps):
            phi_context = phi_seq[:, step:step + n_past + 1]
            z_pred = self._predict_next_latent(latent_history, n_past, phi_context)
            z_true = z_seq[:, n_past + step]
            weight = rollout_discount ** step

            step_loss = criterion(z_pred, z_true)
            total_loss = total_loss + weight * step_loss
            latent_history.append(z_pred)  # nessun detach: autograd propaga lungo tutto il rollout

        return total_loss

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def train(self, epochs: int, batch_size: int, n_past: int, rollout_steps: int = 1,
               rollout_discount: float = 1.0,
               checkpoint_dir=None, save_checkpoint_each_epoch: bool = True,
               checkpoint_label: str = "model", stage_name: str = "transformer",
               settings: Optional[Dict[str, Any]] = None, scheduler_cfg: Optional[Dict[str, Any]] = None,
               extra_checkpoint_data: Optional[Dict[str, Any]] = None,
               epoch_callback=None):
        from torch.utils.data import DataLoader
        import utils

        loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()

        checkpoint_path = None
        if checkpoint_dir is not None:
            checkpoint_path = Path(checkpoint_dir)
            checkpoint_path.mkdir(parents=True, exist_ok=True)

        scheduler = None
        if scheduler_cfg:
            warmup_epochs = int(scheduler_cfg.get("warmup_epochs", 0))
            initial_lr_divisor = float(scheduler_cfg.get("initial_lr_divisor", 100.0))
            decay_every_epochs = int(scheduler_cfg.get("decay_every_epochs", 20))
            decay_gamma = float(scheduler_cfg.get("decay_gamma", 0.8))
            start_factor = 1.0 / max(initial_lr_divisor, 1e-9)

            def lr_lambda(epoch_idx):
                if warmup_epochs > 0 and epoch_idx < warmup_epochs:
                    progress = (epoch_idx + 1) / warmup_epochs
                    return start_factor + (1.0 - start_factor) * progress
                decay_steps = 0 if epoch_idx < warmup_epochs else (epoch_idx - warmup_epochs) // decay_every_epochs
                return decay_gamma ** decay_steps

            scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

        t0 = time.perf_counter()
        history: List[Dict[str, Any]] = []

        for epoch in range(epochs):
            self.transformer.train()
            epoch_losses = []
            e0 = time.perf_counter()

            for z_seq, phi_seq in loader:
                z_seq = z_seq.to(self.device)
                phi_seq = phi_seq.to(self.device)
                if not torch.isfinite(z_seq).all():
                    raise ValueError("Dati di training latenti contengono NaN/Inf.")

                self.optimizer.zero_grad()
                loss = self._compute_rollout_loss(z_seq, phi_seq, criterion, n_past,
                                                    rollout_steps, rollout_discount)
                if not torch.isfinite(loss):
                    raise RuntimeError("Loss non finita incontrata durante il training. Interruzione.")
                loss.backward()
                self.optimizer.step()
                epoch_losses.append(float(loss.item()))

            epoch_loss = float(np.mean(epoch_losses))
            epoch_duration = time.perf_counter() - e0
            record = {"epoch": epoch + 1, "total": epoch_loss, "val": None}
            print(f"[{stage_name}] Epoch {epoch + 1}/{epochs} - loss: {epoch_loss:.6f} - time: {epoch_duration:.2f}s")

            if self.has_val:
                val_loss = self.evaluate(self.val_dataset, batch_size, n_past, rollout_steps, rollout_discount)
                record["val"] = val_loss
                print(f"[{stage_name}] Epoch {epoch + 1}/{epochs} - val_loss: {val_loss:.6f}")

                n_avg_window = max(1, epochs // 20)
                if epoch >= epochs - n_avg_window:
                    unweighted = self.evaluate_unweighted_losses(self.val_dataset, batch_size, n_past, rollout_steps)
                    record["val_reconstruction_pod"] = unweighted["reconstruction_pod"]
                    record["val_latent_pred"] = unweighted["latent_pred"]
                else:
                    record["val_reconstruction_pod"] = None
                    record["val_latent_pred"] = None

            history.append(record)

            if not math.isfinite(epoch_loss):
                raise RuntimeError(f"Epoch {epoch + 1} ha prodotto loss non finita. Training interrotto.")

            if save_checkpoint_each_epoch and checkpoint_path is not None:
                ckpt_file = checkpoint_path / f"{checkpoint_label}_{epoch + 1:03d}_{epoch_loss:.6f}.pt"
                self.save_checkpoint(ckpt_file, epoch=epoch + 1, loss=epoch_loss,
                                      settings=settings, extra=extra_checkpoint_data)

            if scheduler is not None:
                scheduler.step()

            if epoch_callback is not None:
                epoch_callback(epoch, record.get("val", epoch_loss))

        total_time = time.perf_counter() - t0
        return total_time, history

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate(self, dataset, batch_size: int, n_past: int, rollout_steps: int = 1,
                 rollout_discount: float = 1.0) -> float:
        from torch.utils.data import DataLoader
        criterion = nn.MSELoss()
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        losses = []
        self.transformer.eval()
        with torch.no_grad():
            for z_seq, phi_seq in loader:
                z_seq, phi_seq = z_seq.to(self.device), phi_seq.to(self.device)
                loss = self._compute_rollout_loss(z_seq, phi_seq, criterion, n_past,
                                                    rollout_steps, rollout_discount)
                losses.append(float(loss.item()))
        return float(np.mean(losses)) if losses else float("nan")

    def evaluate_unweighted_losses(self, dataset, batch_size: int, n_past: int,
                                    rollout_steps: int = 1) -> Dict[str, Optional[float]]:
        """
        - reconstruction_pod : errore di ricostruzione della riduzione dimensionale
          (POD o AE), costante (dipende solo dal rank/latent_dim), riusato da
          self.reducer.reconstruction_error — nessun decode/encode ripetuto.
        - latent_pred         : MSE(z_pred, z_true) nello spazio latente
        """
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        mse = nn.MSELoss()
        lat_vals = []

        self.transformer.eval()
        with torch.no_grad():
            for z_seq, phi_seq in loader:
                z_seq, phi_seq = z_seq.to(self.device), phi_seq.to(self.device)
                latent_history = [z_seq[:, i] for i in range(n_past)]
                for step in range(rollout_steps):
                    phi_context = phi_seq[:, step:step + n_past + 1]
                    z_pred = self._predict_next_latent(latent_history, n_past, phi_context)
                    z_true = z_seq[:, n_past + step]
                    lat_vals.append(float(mse(z_pred, z_true).item()))
                    latent_history.append(z_pred)

        return {
            "reconstruction_pod": self.reducer.reconstruction_error,
            "latent_pred": float(np.mean(lat_vals)) if lat_vals else None,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_next(self, x_context: np.ndarray, phi_context: np.ndarray,
                      n_past: int = 1) -> np.ndarray:
        """
        x_context   : (n_past, Nf, n_cells) — snapshot FISICI (gia' scalati) di contesto
        phi_context : (n_past+1,)
        returns     : (Nf, n_cells) — snapshot predetto (spazio scalato)
        """
        assert self.snapshot_shape is not None
        self.transformer.eval()
        phi_context = np.asarray(phi_context, dtype=np.float32)
        if phi_context.shape[0] < n_past + 1:
            raise ValueError(f"phi_context deve contenere almeno {n_past + 1} valori")

        with torch.no_grad():
            if x_context.ndim == 2:
                x_context = x_context[np.newaxis]
            latent_history = [
                self.reducer.encode_torch(torch.tensor(x_context[i:i + 1], dtype=torch.float32, device=self.device))
                for i in range(n_past)
            ]
            phi_t = torch.tensor(phi_context[:n_past + 1][np.newaxis, :], dtype=torch.float32, device=self.device)
            z_next = self._predict_next_latent(latent_history, n_past, phi_t)
            z_next_np = z_next.squeeze(0).cpu().numpy()

        return self.reducer.decode_np(z_next_np, self.snapshot_shape)

    # ------------------------------------------------------------------
    # Checkpointing (model-agnostic per train.py/test.py: firma comune)
    # ------------------------------------------------------------------
    def extra_checkpoint_payload(self) -> Dict[str, Any]:
        """Stato specifico del reducer di questo modello, da includere nel
        checkpoint. Ogni modello registrato in MODEL_REGISTRY implementa
        questo metodo con il proprio contenuto (train.py/test.py/HPO.py non
        hanno bisogno di sapere cosa contiene)."""
        return {
            "pod_U_r": self.reducer.U_r,
            "pod_mean": self.reducer.mean,
            "pod_rank": self.reducer.effective_rank,
            "pod_reconstruction_error": self.reducer.reconstruction_error,
        }

    def save_checkpoint(self, path, epoch: int, loss: float, settings=None, extra: Optional[Dict[str, Any]] = None):
        payload = {
            "model_name": self.model_name,
            "transformer": self._raw_transformer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "settings": settings,
            "architecture": self.arch_cfg,
            "training": self.training_cfg,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    @classmethod
    def load_checkpoint(cls, path, device_list) -> Tuple["PODTransformerModel", Dict[str, Any]]:
        checkpoint = torch.load(path, map_location=device_list[0], weights_only=False)
        required = ["transformer", "pod_U_r", "pod_mean", "pod_rank", "phi_mean", "phi_std",
                    "architecture", "training"]
        missing = [k for k in required if k not in checkpoint]
        if missing:
            raise KeyError(
                f"Checkpoint '{path}' manca delle chiavi richieste: {missing}. "
                f"Probabilmente salvato con una versione precedente di train.py. Ri-allenare."
            )

        model = cls.from_config(checkpoint["architecture"], checkpoint["training"], device_list,
                                 latent_dim=int(checkpoint["pod_rank"]))
        model._raw_transformer.load_state_dict(checkpoint["transformer"])

        model.reducer.U_r = checkpoint["pod_U_r"]
        model.reducer.mean = checkpoint["pod_mean"]
        model.reducer.reconstruction_error = checkpoint.get("pod_reconstruction_error")
        if checkpoint.get("snapshot_shape") is not None:
            model.snapshot_shape = tuple(checkpoint["snapshot_shape"])

        return model, checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# AETransformerModel — modello "ae-transformer" (Transformer nello spazio
# latente di un autoencoder fully-connected, al posto della POD analitica)
# ─────────────────────────────────────────────────────────────────────────────

class FCAutoencoder(nn.Module):
    """Autoencoder fully-connected: input (Nf*n_cells,) -> latente (r,) ->
    ricostruzione (Nf*n_cells,). Building block usato da ae-transformer."""

    def __init__(self, input_dim: int, latent_dim: int, hidden_dims=(256, 128),
                 dropout: float = 0.0, activation=nn.ReLU):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = list(hidden_dims)

        enc_dims = [input_dim] + self.hidden_dims
        enc_layers = []
        for i in range(len(enc_dims) - 1):
            enc_layers += [nn.Linear(enc_dims[i], enc_dims[i + 1]), activation(), nn.Dropout(dropout)]
        enc_layers.append(nn.Linear(enc_dims[-1], latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        dec_dims = [latent_dim] + list(reversed(self.hidden_dims))
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers += [nn.Linear(dec_dims[i], dec_dims[i + 1]), activation(), nn.Dropout(dropout)]
        dec_layers.append(nn.Linear(dec_dims[-1], input_dim))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x_flat):
        return self.encoder(x_flat)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x_flat):
        z = self.encode(x_flat)
        return self.decode(z), z


class AEReducer:
    """Adapter che espone la STESSA interfaccia di PODReducer
    (encode_np/encode_torch/decode_np/decode_torch/effective_rank/
    reconstruction_error) ma basata su un FCAutoencoder allenato invece che
    su una SVD analitica. Grazie a questa interfaccia comune,
    AETransformerModel riusa integralmente la logica di rollout/training di
    PODTransformerModel senza duplicarla."""

    def __init__(self, autoencoder: Optional[FCAutoencoder], latent_dim: int, device):
        self.ae = autoencoder
        self.latent_dim = latent_dim
        self.device = device
        self.reconstruction_error: Optional[float] = None

    @property
    def effective_rank(self) -> int:
        return self.latent_dim

    def encode_torch(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, Nf, n_cells) -> (B, r)."""
        B = x.shape[0]
        return self.ae.encode(x.reshape(B, -1))

    def decode_torch(self, a: torch.Tensor, shape_out) -> torch.Tensor:
        """a : (B, r) -> (B, Nf, n_cells)."""
        flat = self.ae.decode(a)
        return flat.reshape(a.shape[0], *shape_out)

    def encode_np(self, x: np.ndarray) -> np.ndarray:
        """x : (Nf, n_cells) oppure (N, Nf, n_cells) -> (r,) oppure (r, N)
        (stessa convenzione di PODReducer.encode_np)."""
        single = x.ndim == 2
        arr = x[np.newaxis] if single else x
        t = torch.tensor(arr, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            z = self.encode_torch(t).cpu().numpy()  # (N, r)
        return z[0] if single else z.T

    def decode_np(self, a: np.ndarray, shape_out) -> np.ndarray:
        """a : (r,) oppure (r, N) -> (Nf, n_cells) oppure (N, Nf, n_cells)."""
        single = a.ndim == 1
        arr = a[:, np.newaxis] if single else a  # (r, N)
        t = torch.tensor(arr.T, dtype=torch.float32, device=self.device)  # (N, r)
        with torch.no_grad():
            x = self.decode_torch(t, shape_out).cpu().numpy()
        return x[0] if single else x


@register_model("ae-transformer")
class AETransformerModel(PODTransformerModel):
    """
    Variante di pod-transformer in cui la riduzione dimensionale e' un
    autoencoder fully-connected allenato (invece della SVD/POD analitica).

    Riusa INTERAMENTE la logica di rollout iterativo, training, valutazione
    e inferenza di PODTransformerModel (che opera solo attraverso
    l'interfaccia generica self.reducer.*, mai attraverso dettagli
    POD-specifici): qui si ridefinisce solo cio' che e' specifico della
    riduzione AE:
      - from_config       : costruisce il Transformer come per pod-transformer,
                             + un reducer AE "vuoto" (i pesi dell'AE si
                             conoscono solo dopo prepare_data o load_checkpoint)
      - prepare_data       : carica/scala i dati (helper condiviso), ALLENA
                             l'autoencoder (loss di ricostruzione, epoche/lr
                             proprie), poi proietta l'intero dataset nel suo
                             spazio latente una sola volta (come la POD)
      - extra_checkpoint_payload / load_checkpoint : salvano/ricaricano i
                             pesi dell'AE invece di U_r/mean

    E' l'esempio concreto di "come aggiungere un nuovo modello senza
    toccare train.py/test.py/HPO.py": questa classe e' l'unica cosa nuova,
    piu' le chiavi di configurazione in train.json/HPO.json (vedi README).
    """

    @classmethod
    def from_config(cls, arch_cfg: Dict[str, Any], training_cfg: Dict[str, Any],
                     device_list, latent_dim: Optional[int] = None) -> "AETransformerModel":
        latent = latent_dim if latent_dim is not None else arch_cfg["latent_dim"]
        n_past_eff = training_cfg["n_past"]

        transformer = TransformerSequence(
            rank_pod=latent,
            n_past=n_past_eff,
            n_layers=arch_cfg["transformer_layers"],
            embed_dim=arch_cfg["transformer_embed_dim"],
            num_heads=arch_cfg["transformer_heads"],
            hidden_dim=arch_cfg["transformer_hidden_dim"],
            dropout=float(training_cfg.get("dropout", 0.0)),
        )
        from utils import init_weights, maybe_data_parallel
        transformer.apply(init_weights)
        transformer = maybe_data_parallel(transformer, device_list)

        # Reducer "vuoto": l'autoencoder viene creato/allenato in
        # prepare_data (serve conoscere Nf*n_cells) oppure ricostruito da
        # checkpoint in load_checkpoint.
        reducer = AEReducer(autoencoder=None, latent_dim=latent, device=device_list[0])

        model = cls(reducer=reducer, transformer=transformer, device=device_list[0])
        model._device_list = device_list
        model.arch_cfg = dict(arch_cfg)
        model.training_cfg = dict(training_cfg)
        return model

    @classmethod
    def prepare_data(cls, dataset_cfg: Dict[str, Any], latent_dim: int, device) -> Dict[str, Any]:
        """Come PODTransformerModel.prepare_data, ma la riduzione e' un
        autoencoder allenato invece della SVD. Iperparametri di training
        dell'AE letti da dataset_cfg['ae_training'] (vedi README/JSON):
        ae_epochs, ae_lr, ae_batch_size, ae_hidden_dims, ae_dropout."""
        import utils

        scaled_concat, rom, phi_scaled_list, phi_mean, phi_std, nf_ref, n_cells_ref, nts = \
            _load_scale_and_concat(dataset_cfg)

        ae_cfg = dataset_cfg.get("ae_training", {})
        ae_epochs = int(ae_cfg.get("ae_epochs", 50))
        ae_lr = float(ae_cfg.get("ae_lr", 1e-3))
        ae_batch_size = int(ae_cfg.get("ae_batch_size", 64))
        hidden_dims = tuple(ae_cfg.get("ae_hidden_dims", [256, 128]))
        ae_dropout = float(ae_cfg.get("ae_dropout", 0.0))

        input_dim = nf_ref * n_cells_ref
        ae = FCAutoencoder(input_dim, latent_dim, hidden_dims=hidden_dims, dropout=ae_dropout).to(device)
        ae.apply(utils.init_weights)

        X = torch.tensor(scaled_concat.reshape(scaled_concat.shape[0], -1), dtype=torch.float32, device=device)
        optimizer = optim.Adam(ae.parameters(), lr=ae_lr)
        mse = nn.MSELoss()
        n_total = X.shape[0]

        print(f"Training autoencoder FC (latent_dim={latent_dim}, hidden_dims={hidden_dims}, "
              f"epochs={ae_epochs}, lr={ae_lr})...")
        ae.train()
        for epoch in range(ae_epochs):
            perm = torch.randperm(n_total, device=device)
            epoch_losses = []
            for i in range(0, n_total, ae_batch_size):
                idx = perm[i:i + ae_batch_size]
                xb = X[idx]
                optimizer.zero_grad()
                xb_rec, _ = ae(xb)
                loss = mse(xb_rec, xb)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.item()))
            if (epoch + 1) % max(1, ae_epochs // 10) == 0 or epoch == ae_epochs - 1:
                print(f"  [autoencoder] epoch {epoch + 1}/{ae_epochs} - recon_loss: {np.mean(epoch_losses):.6e}")

        ae.eval()
        with torch.no_grad():
            X_rec, latent_full_t = ae(X)
            reconstruction_error = float(mse(X_rec, X).item())
        print(f"Autoencoder pre-training completo. Errore di ricostruzione (MSE): {reconstruction_error:.6e}")

        reducer = AEReducer(autoencoder=ae, latent_dim=latent_dim, device=device)
        reducer.reconstruction_error = reconstruction_error

        print("Proiezione one-shot dell'intero dataset nello spazio latente (AE)...")
        latent_full = latent_full_t.detach().cpu().numpy().astype(np.float32)  # (Nt_tot, r)

        split_points = np.cumsum(nts)[:-1]
        latent_list = np.split(latent_full, split_points, axis=0)

        utils.free_memory(scaled_concat, X)

        return {
            "reducer": reducer,
            "rom": rom,
            "latent_list": latent_list,
            "phi_scaled_list": phi_scaled_list,
            "phi_mean": phi_mean,
            "phi_std": phi_std,
            "snapshot_shape": (nf_ref, n_cells_ref),
            "nts": nts,
        }

    def extra_checkpoint_payload(self) -> Dict[str, Any]:
        return {
            "ae_state_dict": self.reducer.ae.state_dict(),
            "ae_input_dim": self.reducer.ae.input_dim,
            "ae_hidden_dims": self.reducer.ae.hidden_dims,
            "ae_latent_dim": self.reducer.latent_dim,
            "ae_reconstruction_error": self.reducer.reconstruction_error,
        }

    @classmethod
    def load_checkpoint(cls, path, device_list) -> Tuple["AETransformerModel", Dict[str, Any]]:
        checkpoint = torch.load(path, map_location=device_list[0], weights_only=False)
        required = ["transformer", "ae_state_dict", "ae_input_dim", "ae_hidden_dims", "ae_latent_dim",
                    "phi_mean", "phi_std", "architecture", "training"]
        missing = [k for k in required if k not in checkpoint]
        if missing:
            raise KeyError(f"Checkpoint '{path}' manca delle chiavi richieste: {missing}. Ri-allenare.")

        model = cls.from_config(checkpoint["architecture"], checkpoint["training"], device_list,
                                 latent_dim=int(checkpoint["ae_latent_dim"]))
        model._raw_transformer.load_state_dict(checkpoint["transformer"])

        ae = FCAutoencoder(int(checkpoint["ae_input_dim"]), int(checkpoint["ae_latent_dim"]),
                            hidden_dims=tuple(checkpoint["ae_hidden_dims"])).to(device_list[0])
        ae.load_state_dict(checkpoint["ae_state_dict"])
        model.reducer.ae = ae
        model.reducer.reconstruction_error = checkpoint.get("ae_reconstruction_error")

        if checkpoint.get("snapshot_shape") is not None:
            model.snapshot_shape = tuple(checkpoint["snapshot_shape"])

        return model, checkpoint


# ─────────────────────────────────────────────────────────────────────────────
# Baseline models condivisi con ArtFire_Baseline (opzionale — non richiesto
# dalla pipeline pod-transformer). Import protetto: se il pacchetto esterno
# ArtFire_Baseline non e' disponibile nell'ambiente, non blocca l'uso di
# models.py per la pipeline principale.
# ─────────────────────────────────────────────────────────────────────────────

try:
    from ArtFire_Baseline.models import (  # noqa: F401
        UNetBaseline,
        FNO3DBaseline,
        LSTMBaseline,
        BASELINE_REGISTRY,
        build_baseline,
        rollout,
    )
except ImportError:
    pass