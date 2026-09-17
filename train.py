"""
train.py
========
Orchestrazione del training. Nessuna definizione di architettura qui
(vedi models.py) e nessuna utility condivisa (vedi utils.py).

CLI:
    python3 train.py --config train.json
    python3 train.py --config train.json --model pod-transformer
    python3 train.py --config train.json --experiments experiments.txt
    python3 train.py --config train.json --experiments experiments.txt --experiment-index 3
    python3 train.py --config train.json --gpus 2

Il training e' completamente model-agnostic: il modello da allenare e'
selezionato tramite --model (o settings["model"] nel JSON) e risolto via
models.MODEL_REGISTRY. Aggiungere un nuovo modello in models.py non
richiede alcuna modifica a questo file.
"""

from pathlib import Path
import pickle

import numpy as np
import torch

import models
import utils


def run_training(settings, model_name, device_list, config_label="train.json",
                  experiment_index=None, overrides=None):
    utils.seed_everything(42)

    print(f'Lettura file "{config_label}" ...')
    if experiment_index is not None:
        print(f"Esperimento #{experiment_index}: {overrides}")
    print(f"Modello: {model_name} | device: {[str(d) for d in device_list]}")

    ModelClass = models.get_model_class(model_name)

    dataset_cfg = settings["dataset"]
    training_cfg = settings["training"]
    architecture_cfg = settings["architecture"]
    output_cfg = settings["output"]
    models_dir = output_cfg["models_dir"]

    # ── Preparazione dati: caricamento, scaling, POD fit + proiezione one-shot ──
    latent_dim = architecture_cfg.get("rank_POD", architecture_cfg.get("latent_dim"))
    prepared = ModelClass.prepare_data(dataset_cfg, latent_dim, device_list[0])
    reducer = prepared["reducer"]
    rom = prepared["rom"]
    latent_list = prepared["latent_list"]
    phi_scaled_list = prepared["phi_scaled_list"]
    snapshot_shape = prepared["snapshot_shape"]

    rom_path = Path(models_dir) / "rom.pkl"
    rom_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rom_path, "wb") as f:
        pickle.dump(rom, f, protocol=4)
    print(f"ROM salvato in: {rom_path}")

    n_past = training_cfg["n_past"]
    rollout_steps = int(training_cfg.get("rollout_steps", 1))
    rollout_discount = float(training_cfg.get("rollout_discount", 1.0))
    past_len = n_past  # finestra = n_past passati + rollout_steps futuri; n_past>1 => autoregressivo, senza bisogno di un flag dedicato

    print(f"\nn_past: {n_past} | rollout_steps: {rollout_steps} (discount={rollout_discount})")

    train_dataset = utils.LatentWindowDataset(latent_list, phi_scaled_list,
                                               past_len=past_len, future_len=rollout_steps)

    # ── Modello ──────────────────────────────────────────────────────────────
    model = ModelClass.from_config(architecture_cfg, training_cfg, device_list, latent_dim=reducer.effective_rank)
    model.reducer = reducer
    model.snapshot_shape = snapshot_shape

    model.set_datasets(train_dataset)
    model.set_optimizer(lr=training_cfg["lr"])

    extra_checkpoint_data = {
        "phi_mean": prepared["phi_mean"],
        "phi_std": prepared["phi_std"],
        "snapshot_shape": snapshot_shape,
    }
    extra_checkpoint_data.update(model.extra_checkpoint_payload())

    total_time, history = model.train(
        epochs=training_cfg["epochs_transformer"],
        batch_size=training_cfg["batch_size"],
        n_past=past_len,
        rollout_steps=rollout_steps,
        rollout_discount=rollout_discount,
        checkpoint_dir=models_dir,
        save_checkpoint_each_epoch=training_cfg.get("save_checkpoint_each_epoch", False),
        checkpoint_label=model_name.replace("-", "_"),
        stage_name="transformer",
        settings=settings,
        scheduler_cfg=training_cfg.get("scheduler"),
        extra_checkpoint_data=extra_checkpoint_data,
    )

    utils.plot_loss_trend(history, Path(models_dir) / "loss_trend.png", title=f"Training loss ({model_name})")
    print(f"\nTraining completato in {total_time:.1f}s. Modelli salvati in: {models_dir}")

    utils.free_memory(model, train_dataset, latent_list, phi_scaled_list, reducer, prepared)
    print("Memoria liberata per il prossimo esperimento.\n")


def main():
    parser = utils.build_arg_parser("train.json", description="Training POD+Transformer (model-agnostic)")
    args = parser.parse_args()

    for experiment_index, overrides, settings in utils.iter_settings(
        args.config, args.experiments, args.experiment_index
    ):
        model_name = utils.resolve_model_name(args, settings)
        num_gpus = utils.resolve_num_gpus(args, settings)
        _, device_list = utils.resolve_devices(num_gpus)

        run_training(
            settings,
            model_name=model_name,
            device_list=device_list,
            config_label=args.config,
            experiment_index=experiment_index,
            overrides=overrides,
        )


if __name__ == "__main__":
    main()