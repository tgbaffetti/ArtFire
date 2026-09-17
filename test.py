"""
test.py
=======
Orchestrazione del test. Nessuna definizione di architettura qui (vedi
models.py) e nessuna utility condivisa (vedi utils.py).

CLI:
    python3 test.py --config test.json
    python3 test.py --config test.json --experiments experiments.txt

test.py e' model-agnostic: architettura e iperparametri di training
(n_past, rollout usato in training, ecc.) sono letti DIRETTAMENTE dal
checkpoint prodotto da train.py, non da test.json — questo elimina il
rischio di disallineamento tra architettura allenata e architettura
ricostruita in test (bug presente nella versione originale).
test.json contiene solo i parametri che riguardano davvero il test
(dataset di test, output). Il rollout di test copre SEMPRE l'intero
dataset di test disponibile (nessun N_teststeps configurabile): si parte
dai primi n_past snapshot noti e si predice fino alla fine del file.

IMPORTANTE: le metriche (R2/NRMSE per feature e per modo POD, evoluzione
RMSE/errore/R2 nel tempo, mappa di attenzione) sono le stesse della
versione originale. I soli plot rimossi sono quelli che richiedevano
esplicitamente una griglia 2D strutturata (Nz, Nx) per il rendering
spaziale (comparison_grid, GIF dei campi) — incompatibili con la nuova
shape dataset [n_cells, Nf, Nt] che non assume alcuna struttura di griglia.
"""

import os
import pickle
from pathlib import Path

import numpy as np
import torch

import models
import utils


def run_test(settings, model_name, device_list, config_label="test.json",
             experiment_index=None, overrides=None):
    print(f'Lettura file "{config_label}" ...')
    if experiment_index is not None:
        print(f"Esperimento #{experiment_index}: {overrides}")
    print(f"Modello: {model_name} | device: {[str(d) for d in device_list]}")

    ModelClass = models.get_model_class(model_name)

    dataset_cfg = settings["dataset"]
    output_cfg = settings["output"]

    data_paths = dataset_cfg["data_paths"]
    phi_paths = dataset_cfg["phi_paths"]
    features = dataset_cfg["features"]
    if len(data_paths) != len(phi_paths):
        raise ValueError("ERROR: data_paths e phi_paths devono avere la stessa lunghezza")

    models_dir = output_cfg["models_dir"]
    test_dir = output_cfg["test_dir"]
    plot_start_epoch = output_cfg["plot_start_epoch"]

    # ── Checkpoint: architettura, n_past, POD/AE -> tutto dal checkpoint ──
    checkpoint_path, checkpoint_loss = utils.find_best_checkpoint(
        models_dir, checkpoint_label=model_name.replace("-", "_"), plot_start_epoch=plot_start_epoch,
        output_dir=Path(test_dir),
    )
    print(f"\nMiglior checkpoint: {checkpoint_path.name} (loss={checkpoint_loss:.6f})")

    model, checkpoint = ModelClass.load_checkpoint(checkpoint_path, device_list)
    n_past = int(checkpoint["training"]["n_past"])
    phi_mean = float(checkpoint["phi_mean"])
    phi_std = float(checkpoint["phi_std"])
    if phi_std <= 0.0:
        raise ValueError("ERROR: phi_std deve essere positivo")

    rank_pod = model.reducer.effective_rank
    experiment_label = f"exp_rank{rank_pod}_past{n_past}"
    base_output_dir = Path(test_dir) / experiment_label
    base_output_dir.mkdir(parents=True, exist_ok=True)

    rom_path = Path(models_dir) / "rom.pkl"
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM non trovato: {rom_path}. Ri-eseguire il training.")
    with open(rom_path, "rb") as f:
        rom = pickle.load(f)
    print(f"ROM caricato da: {rom_path}")

    nf_ref, n_cells_ref = model.snapshot_shape

    for data_path, phi_path in zip(data_paths, phi_paths):
        dataset_name = Path(data_path).stem
        output_dir = base_output_dir / dataset_name
        output_dir.mkdir(parents=True, exist_ok=True)

        arr = np.load(data_path)  # (n_cells, Nf, Nt)
        if arr.ndim != 3 or arr.shape[:2] != (n_cells_ref, nf_ref):
            raise ValueError(
                f"ERROR: shape inattesa {arr.shape} per {data_path}, attesa (n_cells={n_cells_ref}, Nf={nf_ref}, Nt)"
            )
        nt = arr.shape[2]
        dataset = np.transpose(arr, (2, 1, 0)).astype(np.float32)  # (Nt, Nf, n_cells)

        phi = np.load(phi_path).astype(np.float32)
        if phi.shape != (nt,):
            raise ValueError(f"ERROR: shape phi inattesa {phi.shape}, attesa {(nt,)} per {phi_path}")
        print(f"\nDataset di test: {Path(data_path).name} -> [n_cells={n_cells_ref}, Nf={nf_ref}, Nt={nt}]")

        print("Scaling dati di test ...")
        test_snapshots_scaled = utils.scale_test_tensor(dataset, rom)
        test_phi_scaled = (phi - phi_mean) / phi_std

        # Rollout autoregressivo sull'INTERO dataset di test disponibile
        # (nessun N_teststeps configurabile: si predice sempre l'intero
        # dataset, a partire dai primi n_past snapshot noti).
        required_eval = nt - n_past
        if required_eval < 1:
            raise ValueError(
                f"ERROR: dataset {Path(data_path).name} troppo corto. "
                f"Nt={nt}, richiesti almeno {n_past + 1}"
            )
        n_training = n_past - 1

        print(f"n_past: {n_past} | step di valutazione (intero dataset di test): {required_eval}")

        print("\nAvvio test ...")
        import time
        t0 = time.time()

        generated, targets = [], []
        history = [test_snapshots_scaled[i] for i in range(n_past)]
        for step in range(required_eval):
            x_context = np.stack(history[-n_past:], axis=0)
            phi_context = test_phi_scaled[step: step + n_past + 1]
            x_next = model.predict_next(x_context, phi_context, n_past=n_past)
            generated.append(x_next)
            history.append(x_next)
            targets.append(dataset[n_past + step])

        print(f"Fine test. Tempo: {time.time() - t0:.2f}s")

        generated_array_scaled = np.stack(generated, axis=0).astype(np.float32)
        generated_array = utils.rescale_back_output(generated_array_scaled, rom)
        target_array = np.stack(targets, axis=0).astype(np.float32)
        print(f"----> shape array generato: {generated_array.shape}")

        output_file = output_dir / "predicted_data.npy"
        np.save(output_file, generated_array)
        print(f"Rollout salvato in: {output_file}")

        _compute_and_plot_metrics(
            generated_array, generated_array_scaled, target_array, model, rom,
            features, output_dir, n_training,
        )

        utils.free_memory(dataset, test_snapshots_scaled, generated, targets,
                           generated_array_scaled, generated_array, target_array)

    _plot_attention_map(model, base_output_dir=base_output_dir, rank_pod=rank_pod, n_past=n_past,
                         last_data_path=data_paths[-1], last_phi_path=phi_paths[-1],
                         phi_mean=phi_mean, phi_std=phi_std, rom=rom)

    print("\nTest completato con successo!\n")


def _compute_and_plot_metrics(generated_array, generated_array_scaled, target_array,
                               model, rom, features, output_dir, n_training):
    import matplotlib.pyplot as plt

    nf = generated_array.shape[1]

    print("\n" + "=" * 60)
    print("METRICHE GLOBALI")
    print("=" * 60)

    print("\n[1] Metriche per feature (tutti gli step, tutte le celle):")
    print(f"  {'Feature':<10} {'R2':>10} {'NRMSE':>10}")
    print(f"  {'-' * 32}")
    r2_features, nrmse_features = [], []
    for f_idx in range(nf):
        y_true = target_array[:, f_idx, :].ravel()
        y_pred = generated_array[:, f_idx, :].ravel()
        r2 = utils.r2_score(y_true, y_pred)
        nrmse = utils.nrmse_score(y_true, y_pred)
        r2_features.append(r2)
        nrmse_features.append(nrmse)
        print(f"  {features[f_idx]:<10} {r2:>10.6f} {nrmse:>10.6f}")

    print("\n[2] Metriche per modo POD (coefficienti nel tempo):")
    print(f"  {'Mode':<10} {'R2':>10} {'NRMSE':>10}")
    print(f"  {'-' * 32}")
    target_scaled_metrics = utils.scale_test_tensor(target_array, rom)
    z_true = model.reducer.encode_np(target_scaled_metrics)
    z_pred = model.reducer.encode_np(generated_array_scaled)

    r2_modes, nrmse_modes = [], []
    for mode_idx in range(model.reducer.effective_rank):
        y_true, y_pred = z_true[mode_idx, :], z_pred[mode_idx, :]
        r2 = utils.r2_score(y_true, y_pred)
        nrmse = utils.nrmse_score(y_true, y_pred)
        r2_modes.append(r2)
        nrmse_modes.append(nrmse)
        print(f"  {'mode_' + str(mode_idx + 1):<10} {r2:>10.6f} {nrmse:>10.6f}")

    print("\n[3] Medie:")
    print(f"  Mean R2    (features) : {np.mean(r2_features):.6f}")
    print(f"  Mean R2    (POD modes): {np.mean(r2_modes):.6f}")
    print(f"  Mean NRMSE (features) : {np.mean(nrmse_features):.6f}")
    print(f"  Mean NRMSE (POD modes): {np.mean(nrmse_modes):.6f}")
    print("=" * 60 + "\n")

    # ── Evoluzione temporale (RMSE / errore relativo / R2) per feature ──
    # Nota: queste metriche sono aggregate sulla dimensione delle celle
    # (n_cells) e NON richiedono alcuna struttura di griglia 2D: sono
    # identiche concettualmente alla versione originale.
    eval_steps = generated_array.shape[0]
    for f in range(nf):
        plot_dir = os.path.join(output_dir, f"{features[f]}")
        os.makedirs(plot_dir, exist_ok=True)

        rmse = np.zeros(eval_steps)
        err = np.zeros(eval_steps)
        r2 = np.zeros(eval_steps)
        for t in range(eval_steps):
            ref = target_array[t, f, :]
            diff_t = ref - generated_array[t, f, :]
            err[t] = np.linalg.norm(diff_t) / np.linalg.norm(ref)
            rmse[t] = np.sqrt(np.mean(diff_t ** 2))
            ss_res = np.sum(diff_t ** 2)
            ss_tot = np.sum((ref - np.mean(ref)) ** 2)
            r2[t] = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(eval_steps), rmse, "-o", markersize=3)
        plt.xlabel("Timestep"); plt.ylabel("RMSE")
        plt.title(f"RMSE evolution - feature {features[f]}")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "RMSE_evolution.png")); plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(eval_steps), 100 * err, "-o", markersize=3)
        plt.xlabel("Timestep"); plt.ylabel("error (%)")
        plt.title(f"% error - feature {features[f]}")
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "Err_evolution.png")); plt.close()

        plt.figure(figsize=(8, 4))
        plt.plot(np.arange(eval_steps), r2, "-o", markersize=3, color="green")
        plt.xlabel("Timestep"); plt.ylabel("R\u00b2")
        plt.title(f"R\u00b2 evolution - feature {features[f]}")
        plt.ylim(-0.1, 1.05)
        plt.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
        plt.grid(True); plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "R2_evolution.png")); plt.close()

    print(
        "NOTA: i plot spaziali 2D (comparison_grid, GIF dei campi) della versione "
        "originale richiedevano una griglia strutturata (Nz, Nx) e sono stati omessi: "
        "la nuova shape dataset [n_cells, Nf, Nt] non assume alcuna struttura di griglia."
    )


def _plot_attention_map(model, base_output_dir, rank_pod, n_past,
                         last_data_path, last_phi_path, phi_mean, phi_std, rom):
    """Estrae e plotta la mappa di attenzione dell'ultimo layer (e la media
    su tutti i layer) usando il primo contesto disponibile dell'ultimo
    dataset di test. Non dipende dalla struttura spaziale."""
    import matplotlib.pyplot as plt

    arr = np.load(last_data_path)
    dataset = np.transpose(arr, (2, 1, 0)).astype(np.float32)
    phi = np.load(last_phi_path).astype(np.float32)
    test_snapshots_scaled = utils.scale_test_tensor(dataset, rom)
    test_phi_scaled = (phi - phi_mean) / phi_std

    print("\nEstrazione mappa di attenzione ...")
    x_sample = torch.tensor(test_snapshots_scaled[0:n_past], dtype=torch.float32).unsqueeze(0).to(model.device)
    phi_sample = torch.tensor(test_phi_scaled[0:n_past + 1], dtype=torch.float32).unsqueeze(0).to(model.device)

    latent_history_attn = [model.reducer.encode_torch(x_sample[:, i]) for i in range(n_past)]

    model.transformer.eval()
    with torch.no_grad():
        _, attn_all_layers = model._raw_transformer(latent_history_attn, phi_sample, return_attn=True)

    attn_map = attn_all_layers[-1].squeeze(0).cpu().numpy()
    attn_map_avg = torch.stack(attn_all_layers, dim=0).mean(dim=0).squeeze(0).cpu().numpy()

    labels_full = []
    for i in range(n_past):
        labels_full.append(f"\u03c6{i}")
        labels_full.extend([f"a{i},{k + 1}" for k in range(rank_pod)])
    labels_full.append("\u03c6_{t+1}")
    labels_full.extend([f"P{k + 1}" for k in range(rank_pod)])

    L_total = len(labels_full)
    assert L_total == attn_map.shape[0], f"Mismatch: labels={L_total}, attn_map={attn_map.shape[0]}"

    def plot_attn_full(matrix, title, out_path):
        L = matrix.shape[0]
        fig_size = max(8, L * 0.35)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        im = ax.imshow(matrix, cmap="jet", vmin=matrix.min(), vmax=matrix.max(),
                        interpolation="nearest", aspect="equal")
        ax.set_xticks(range(L)); ax.set_xticklabels(labels_full, fontsize=5, rotation=90)
        ax.set_yticks(range(L)); ax.set_yticklabels(labels_full, fontsize=5)
        ax.set_xlabel("Key token (attended to)", fontsize=11)
        ax.set_ylabel("Query token", fontsize=11)
        ax.set_title(title, fontsize=12)

        for t in range(1, n_past):
            sep = t * (rank_pod + 1)
            ax.axvline(sep - 0.5, color="white", linewidth=0.6, alpha=0.5)
            ax.axhline(sep - 0.5, color="white", linewidth=0.6, alpha=0.5)

        sep_phi_future = n_past * (rank_pod + 1)
        ax.axvline(sep_phi_future - 0.5, color="white", linewidth=1.2, alpha=0.8)
        ax.axhline(sep_phi_future - 0.5, color="white", linewidth=1.2, alpha=0.8)

        sep_pred = sep_phi_future + 1
        ax.axvline(sep_pred - 0.5, color="white", linewidth=1.8, alpha=1.0)
        ax.axhline(sep_pred - 0.5, color="white", linewidth=1.8, alpha=1.0)

        plt.colorbar(im, ax=ax, label="attention weight", shrink=0.6)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"Salvato: {out_path}")

    attn_dir = base_output_dir / "attention"
    attn_dir.mkdir(parents=True, exist_ok=True)
    plot_attn_full(attn_map, "Self-attention weights \u2014 ultimo layer", attn_dir / "attention_map.png")
    plot_attn_full(attn_map_avg, "Self-attention weights \u2014 media su tutti i layer", attn_dir / "attention_map_avg.png")


def main():
    parser = utils.build_arg_parser("test.json", description="Test POD+Transformer (model-agnostic)")
    args = parser.parse_args()

    for experiment_index, overrides, settings in utils.iter_settings(
        args.config, args.experiments, args.experiment_index
    ):
        model_name = utils.resolve_model_name(args, settings)
        num_gpus = utils.resolve_num_gpus(args, settings)
        _, device_list = utils.resolve_devices(num_gpus)

        run_test(
            settings,
            model_name=model_name,
            device_list=device_list,
            config_label=args.config,
            experiment_index=experiment_index,
            overrides=overrides,
        )


if __name__ == "__main__":
    main()