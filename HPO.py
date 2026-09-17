"""
HPO.py
======
Orchestrazione HPO (Optuna). Nessuna definizione di architettura qui (vedi
models.py) e nessuna utility condivisa (vedi utils.py).

CLI:
    python3 HPO.py --config HPO.json
    python3 HPO.py --config HPO.json --model pod-transformer

Design "a stadi", generico ed estendibile a future architetture (es. un
futuro "vae-transformer" che ottimizza prima il VAE, poi il Transformer,
oppure entrambi insieme): HPO.json definisce un dizionario "stages", ognuno
con il proprio spazio di iperparametri; HPO.py itera sugli stadi nell'ordine
in cui compaiono nel JSON, senza alcuna logica specifica per un modello.
Per "pod-transformer" esiste un solo stadio, "transformer": la POD NON viene
ottimizzata (rank fissato da train.json, calcolata una sola volta e
riutilizzata per tutti i trial).

Objective: media della loss di validazione sulle ultime epoch (mantiene
esattamente la definizione originale), mediata anche tra i fold se
cv_folds > 0. cv_folds e cv_walking vivono in HPO.json (usati solo qui,
mai in train.json):
  - cv_folds == 0        -> nessuna CV, split random 70% training / 30% validation
  - cv_folds >= 2, cv_walking=true  -> walk-forward CV (rispetta l'ordine temporale)
  - cv_folds >= 2, cv_walking=false -> k-fold "classico" (shuffle, ignora l'ordine temporale)
Include pruning (MedianPruner) per interrompere presto i trial non
promettenti.

Al termine, i migliori iperparametri aggiornano automaticamente train.json
e test.json.
"""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import torch
from torch.utils.data import DataLoader, Subset

import models
import utils

OOM_SENTINEL = 1e12


class ConfigError(ValueError):
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _is_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg


def _is_non_finite_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return "non-finite" in msg or "nan" in msg or "inf" in msg


def _validate_hpo_config(cfg: Dict[str, Any]) -> None:
    for key in ["epochs", "stages", "output"]:
        if key not in cfg:
            raise ConfigError(f"Chiave richiesta mancante in HPO config: '{key}'")
    if int(cfg["epochs"].get("transformer", 0)) <= 0:
        raise ConfigError("epochs.transformer deve essere > 0")
    if int(cfg.get("cv_folds", 0)) == 1:
        print("WARNING: cv_folds=1 non ha senso per una cross-validation; trattato come cv_folds=0 (split random 70/30).")


def _sample_float(trial, name: str, key: str, spec: Dict[str, Any]) -> float:
    lo, hi = float(spec["min"]), float(spec["max"])
    if lo == hi:
        return lo
    return trial.suggest_float(name, lo, hi, log=bool(spec.get("log", False)))


def _sample_int(trial, name: str, key: str, spec: Dict[str, Any]) -> int:
    if "choices" in spec:
        choices = [int(v) for v in spec["choices"]]
        return choices[0] if len(choices) == 1 else int(trial.suggest_categorical(name, choices))
    lo, hi = int(spec["min"]), int(spec["max"])
    if lo == hi:
        return lo
    return trial.suggest_int(name, lo, hi)


# ─────────────────────────────────────────────────────────────────────────────
# Split train/val delle finestre latenti (CV walk-forward oppure random 70/30)
# ─────────────────────────────────────────────────────────────────────────────

def _build_train_val_splits(latent_list, phi_list, past_len, future_len, cv_folds: int,
                             cv_walking: bool = True, seed: int = 42):
    """
    cv_folds == 0 (o == 1, degenere) -> split RANDOM 70/30 a livello di finestra (no CV)
    cv_folds >= 2, cv_walking=True    -> walk-forward CV per dataset (rispetta l'ordine temporale)
    cv_folds >= 2, cv_walking=False   -> k-fold "classico": finestre mischiate e divise in
                                          cv_folds blocchi, ignorando l'ordine temporale/il
                                          dataset di provenienza
    Ritorna una lista di (train_dataset, val_dataset).
    """
    n_datasets = len(latent_list)

    if cv_folds < 2:
        full_ds = utils.LatentWindowDataset(latent_list, phi_list, past_len, future_len)
        train_idx, val_idx = utils.random_window_split(len(full_ds), val_fraction=0.3, seed=seed)
        return [(Subset(full_ds, train_idx), Subset(full_ds, val_idx))]

    if not cv_walking:
        full_ds = utils.LatentWindowDataset(latent_list, phi_list, past_len, future_len)
        kfolds = utils.build_classic_kfold_splits(len(full_ds), cv_folds, seed=seed)
        return [(Subset(full_ds, tr), Subset(full_ds, val)) for tr, val in kfolds]

    per_dataset_folds = [utils.build_contiguous_folds(len(latent_list[i]), cv_folds) for i in range(n_datasets)]
    if any(len(f) == 0 for f in per_dataset_folds):
        raise ValueError(
            f"cv_folds={cv_folds} (cv_walking=true) richiede almeno {cv_folds + 1} snapshot per "
            f"ciascun dataset di training; usa cv_walking:false oppure riduci cv_folds."
        )
    folds = []
    for fold_id in range(cv_folds):
        tr_idx = [per_dataset_folds[i][fold_id][0] for i in range(n_datasets)]
        val_idx = [per_dataset_folds[i][fold_id][1] for i in range(n_datasets)]
        train_ds = utils.LatentWindowDataset(latent_list, phi_list, past_len, future_len, indices=tr_idx)
        val_ds = utils.LatentWindowDataset(latent_list, phi_list, past_len, future_len, indices=val_idx)
        folds.append((train_ds, val_ds))
    return folds


def _measure_avg_inference_time(model, dataset, batch_size: int, n_past: int) -> float:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    durations: List[float] = []
    model.transformer.eval()
    with torch.no_grad():
        for z_seq, phi_seq in loader:
            z_seq, phi_seq = z_seq.to(model.device), phi_seq.to(model.device)
            latent_history = [z_seq[:, i] for i in range(n_past)]
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model._predict_next_latent(latent_history, n_past, phi_seq[:, 0:n_past + 1])
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            durations.append(time.perf_counter() - t0)
            break  # una sola batch e' sufficiente per una stima di timing
    return float(np.mean(durations)) if durations else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Training di un singolo fold (usato sia per CV sia per lo split 70/30)
# ─────────────────────────────────────────────────────────────────────────────

def _train_single_fold(ModelClass, settings, reducer, snapshot_shape, rank, train_ds, val_ds,
                        epochs, device_list, report_fn=None):
    training_cfg = settings["training"]
    past_len = training_cfg["n_past"]
    rollout_steps = int(training_cfg.get("rollout_steps", 1))
    rollout_discount = float(training_cfg.get("rollout_discount", 1.0))

    model = ModelClass.from_config(settings["architecture"], training_cfg, device_list, latent_dim=rank)
    model.reducer = reducer
    model.snapshot_shape = snapshot_shape
    model.set_datasets(train_ds, val_ds)
    model.set_optimizer(lr=training_cfg["lr"])

    t0 = time.perf_counter()
    _, history = model.train(
        epochs=epochs,
        batch_size=training_cfg["batch_size"],
        n_past=past_len,
        rollout_steps=rollout_steps,
        rollout_discount=rollout_discount,
        checkpoint_dir=None,
        save_checkpoint_each_epoch=False,
        checkpoint_label="hpo",
        stage_name="hpo_transformer",
        settings=settings,
        scheduler_cfg=training_cfg.get("scheduler"),
        epoch_callback=report_fn,
    )
    training_time = time.perf_counter() - t0

    n_avg = max(1, len(history) // 20)
    last_n = history[-n_avg:]

    def _mean(key):
        vals = [e[key] for e in last_n if e.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    avg_val = _mean("val")
    unweighted = {
        "reconstruction_pod": _mean("val_reconstruction_pod"),
        "latent_pred": _mean("val_latent_pred"),
    }
    inference_time = _measure_avg_inference_time(model, train_ds, training_cfg["batch_size"], past_len)

    utils.free_memory(model)
    return avg_val, unweighted, training_time, inference_time


def _train_for_trial(ModelClass, settings, reducer, snapshot_shape, rank, latent_list, phi_list,
                      epochs, cv_folds, cv_walking, device_list, trial=None, seed=42):
    training_cfg = settings["training"]
    past_len = training_cfg["n_past"]
    rollout_steps = int(training_cfg.get("rollout_steps", 1))

    splits = _build_train_val_splits(latent_list, phi_list, past_len, rollout_steps, cv_folds,
                                      cv_walking=cv_walking, seed=seed)

    all_avg_val, all_unweighted, all_times, all_infer = [], [], [], []
    for fold_id, (train_ds, val_ds) in enumerate(splits):
        report_fn = None
        if trial is not None:
            def report_fn(epoch, val_loss, _fold_id=fold_id):
                step = _fold_id * epochs + epoch
                trial.report(val_loss, step)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        avg_val, unweighted, t_time, inf_time = _train_single_fold(
            ModelClass, settings, reducer, snapshot_shape, rank, train_ds, val_ds,
            epochs, device_list, report_fn=report_fn,
        )
        all_avg_val.append(avg_val)
        all_unweighted.append(unweighted)
        all_times.append(t_time)
        all_infer.append(inf_time)

    def _mean_key(k):
        vals = [d[k] for d in all_unweighted if d.get(k) is not None]
        return float(np.mean(vals)) if vals else None

    final_losses = {
        "reconstruction_pod": _mean_key("reconstruction_pod"),
        "latent_pred": _mean_key("latent_pred"),
    }
    return float(np.mean(all_avg_val)), final_losses, float(np.sum(all_times)), float(np.mean(all_infer))


# ─────────────────────────────────────────────────────────────────────────────
# Stadio di ottimizzazione generico (uno per ogni chiave di HPO.json["stages"])
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(stage_name: str, stage_cfg: Dict[str, Any], base_settings: Dict[str, Any],
              epochs_cfg: Dict[str, int], cv_folds: int, cv_walking: bool,
              reducer, snapshot_shape, latent_list, phi_list, rom, device_list, ModelClass,
              pruning_cfg: Dict[str, Any], storage: Optional[str] = None,
              study_name: Optional[str] = None, n_trials_override: Optional[int] = None):
    hp = stage_cfg["hp"]
    n_trials = int(n_trials_override) if n_trials_override is not None else int(stage_cfg["n_trials"])
    rank = reducer.effective_rank

    experiments: List[Dict[str, Any]] = []

    def objective(trial: "optuna.Trial") -> float:
        settings = __import__("copy").deepcopy(base_settings)
        tr = settings["training"]
        ar = settings["architecture"]

        applied: Dict[str, Any] = {}

        def sample_f(json_key, spec_key):
            v = _sample_float(trial, f"{stage_name}_{spec_key}", spec_key, hp[spec_key])
            applied[json_key] = v
            return v

        def sample_i(json_key, spec_key):
            v = _sample_int(trial, f"{stage_name}_{spec_key}", spec_key, hp[spec_key])
            applied[json_key] = v
            return v

        tr["lr"] = sample_f("lr", "lr")
        ar["transformer_heads"] = sample_i("transformer_heads", "transformer_heads")
        ar["transformer_layers"] = sample_i("transformer_layers", "transformer_layers")

        if "dropout" in hp:
            tr["dropout"] = sample_f("dropout", "dropout")

        sched = tr.setdefault("scheduler", {})
        if "scheduler_warmup_epochs" in hp:
            sched["warmup_epochs"] = sample_i("warmup_epochs", "scheduler_warmup_epochs")
        if "scheduler_initial_lr_divisor" in hp:
            sched["initial_lr_divisor"] = sample_f("initial_lr_divisor", "scheduler_initial_lr_divisor")
        if "scheduler_decay_every_epochs" in hp:
            sched["decay_every_epochs"] = sample_i("decay_every_epochs", "scheduler_decay_every_epochs")
        if "scheduler_decay_gamma" in hp:
            sched["decay_gamma"] = sample_f("decay_gamma", "scheduler_decay_gamma")

        # n_past e rollout_steps sono sempre campionati se presenti in hp:
        # n_past>1 rende la finestra "autoregressiva" e rollout_steps>1
        # attiva il rollout iterativo, senza bisogno di flag booleani dedicati.
        if "n_past" in hp:
            tr["n_past"] = sample_i("n_past", "n_past")
        if "rollout_steps" in hp:
            tr["rollout_steps"] = sample_i("rollout_steps", "rollout_steps")
        if "rollout_discount" in hp:
            tr["rollout_discount"] = sample_f("rollout_discount", "rollout_discount")

        print(f"\n{'=' * 60}\n[HPO/{stage_name}] Trial {trial.number}", flush=True)
        for k, v in applied.items():
            print(f"  {k:<40}: {v}", flush=True)
        print("=" * 60, flush=True)

        status = "ok"
        avg_val = None
        try:
            avg_val, unweighted, training_time, inference_time = _train_for_trial(
                ModelClass, settings, reducer, snapshot_shape, rank, latent_list, phi_list,
                epochs_cfg["transformer"], cv_folds, cv_walking, device_list, trial=trial,
            )
            print(f"[HPO/{stage_name} trial {trial.number}] avg_val={avg_val:.6f} "
                  f"training_time={training_time:.2f}s inference_time={inference_time:.6f}s", flush=True)
        except optuna.TrialPruned:
            raise
        except RuntimeError as exc:
            status = ("non_finite" if _is_non_finite_error(exc) else
                       "oom" if _is_oom_error(exc) else "runtime_error")
            unweighted = {"reconstruction_pod": None, "latent_pred": None}
            avg_val = OOM_SENTINEL
            training_time = inference_time = None
            utils.free_memory()
            if status == "runtime_error":
                raise

        experiments.append({
            "trial": trial.number,
            "status": status,
            "stage": stage_name,
            "applied_params": applied,
            "unweighted_losses": unweighted,
            "avg_last_val_loss": avg_val,
            "training_time_s": training_time,
            "inference_time_s": inference_time,
        })
        return float(avg_val)

    pruner = optuna.pruners.NopPruner()
    if pruning_cfg.get("enabled", True):
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=int(pruning_cfg.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruning_cfg.get("n_warmup_steps", 10)),
            interval_steps=int(pruning_cfg.get("interval_steps", 1)),
        )

    if storage is not None:
        # Storage condiviso (es. sqlite:///path/study.db): permette a piu'
        # processi paralleli (1 per GPU, vedi RUN_HPO.sh) di contribuire
        # trial allo stesso study concorrentemente.
        study = optuna.create_study(direction="minimize", pruner=pruner, storage=storage,
                                     study_name=study_name or f"pod_transformer_{stage_name}",
                                     load_if_exists=True)
    else:
        study = optuna.create_study(direction="minimize", pruner=pruner)
    study.optimize(objective, n_trials=n_trials)

    best_record = next((e for e in experiments if e["trial"] == study.best_trial.number), None)
    best_applied_params = best_record["applied_params"] if best_record else {}

    return study, experiments, best_applied_params


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    utils.seed_everything(42)

    parser = utils.build_arg_parser("HPO.json", description="HPO POD+Transformer (model-agnostic, a stadi)")
    parser.add_argument("--train-config", default="train.json",
                         help="File train.json di base (fornisce dataset/architettura/cv_folds non ottimizzati)")
    parser.add_argument("--test-config", default="test.json",
                         help="File test.json da aggiornare con i migliori iperparametri")
    parser.add_argument("--storage", default=None,
                         help="Storage Optuna condiviso (es. sqlite:///path/study.db) per far "
                              "contribuire piu' processi paralleli (1 per GPU) allo stesso study.")
    parser.add_argument("--study-name", default=None, help="Nome base dello study Optuna (con --storage).")
    parser.add_argument("--n-trials-per-worker", type=int, default=None,
                         help="Numero di trial da eseguire in QUESTO processo (sovrascrive n_trials "
                              "della singola stage-config); usato per dividere il totale tra worker paralleli.")
    parser.add_argument("--skip-apply", action="store_true",
                         help="Non scrivere risultati finali ne' aggiornare train.json/test.json "
                              "(da usare nei worker paralleli: solo un processo 'coordinatore' finalizza).")
    args = parser.parse_args()

    hpo_cfg = utils.load_json(args.config)
    _validate_hpo_config(hpo_cfg)
    base_settings = utils.load_json(args.train_config)

    model_name = args.model or hpo_cfg.get("model", base_settings.get("model", utils.DEFAULT_MODEL_NAME))
    num_gpus = utils.resolve_num_gpus(args, hpo_cfg)
    _, device_list = utils.resolve_devices(num_gpus)
    print(f"Modello: {model_name} | device: {[str(d) for d in device_list]}")

    ModelClass = models.get_model_class(model_name)

    dataset_cfg = base_settings["dataset"]
    architecture_cfg = base_settings["architecture"]

    # ── Dati: caricati e proiettati in latente UNA VOLTA SOLA (POD non ottimizzata) ──
    prepared = ModelClass.prepare_data(dataset_cfg, architecture_cfg["rank_POD"], device_list[0])
    reducer = prepared["reducer"]
    rom = prepared["rom"]
    latent_list = prepared["latent_list"]
    phi_scaled_list = prepared["phi_scaled_list"]
    snapshot_shape = prepared["snapshot_shape"]
    print(f"[HPO] rank_POD fissato = {reducer.effective_rank} (da train.json, non ottimizzato). "
          f"Errore di ricostruzione POD: {reducer.reconstruction_error:.6e}")

    # cv_folds/cv_walking vivono SOLO in HPO.json (non in train.json: usati
    # esclusivamente qui per la validazione degli iperparametri).
    cv_folds = int(hpo_cfg.get("cv_folds", 0))
    cv_walking = bool(hpo_cfg.get("cv_walking", True))
    if cv_folds < 2:
        cv_desc = "random split 70/30 (no CV)"
    elif cv_walking:
        cv_desc = "walk-forward CV (rispetta l'ordine temporale)"
    else:
        cv_desc = "k-fold classico (shuffle)"
    print(f"[HPO] cv_folds: {cv_folds} | cv_walking: {cv_walking} -> {cv_desc}")

    epochs_cfg = {"transformer": int(hpo_cfg["epochs"].get("transformer", 100))}
    pruning_cfg = hpo_cfg.get("pruning", {})

    all_results: Dict[str, Any] = {}
    best_params_overall: Dict[str, Any] = {}

    for stage_name, stage_cfg in hpo_cfg["stages"].items():
        print(f"\n{'#' * 60}\n[HPO] Stadio '{stage_name}'\n{'#' * 60}", flush=True)
        study, experiments, best_applied_params = run_stage(
            stage_name, stage_cfg, base_settings, epochs_cfg, cv_folds, cv_walking,
            reducer, snapshot_shape, latent_list, phi_scaled_list, rom, device_list, ModelClass,
            pruning_cfg, storage=args.storage, study_name=args.study_name,
            n_trials_override=args.n_trials_per_worker,
        )
        # I parametri migliori di questo stadio vengono accumulati e
        # applicati in blocco a fine funzione (dopo tutti gli stadi): utile
        # per architetture multi-stadio future in cui uno stadio successivo
        # dipende dai risultati di quello precedente.
        best_params_overall.update(best_applied_params)

        all_results[stage_name] = {
            "best_trial": {"number": study.best_trial.number, "value": study.best_trial.value,
                            "params": best_applied_params},
            "trials": experiments,
        }

    if args.skip_apply:
        print("\n--skip-apply: questo worker non scrive risultati finali ne' aggiorna train.json/test.json "
              "(demandato al processo coordinatore).")
        utils.free_memory(reducer, latent_list, phi_scaled_list, prepared)
        return

    # ── Salvataggio risultati completi ──────────────────────────────────────
    out_cfg = hpo_cfg["output"]
    output_dir = Path(out_cfg["directory"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / out_cfg["filename"]
    utils.save_json({
        "model": model_name,
        "config_used": str(args.config),
        "cv_folds": cv_folds,
        "cv_walking": cv_walking,
        "epochs": hpo_cfg["epochs"],
        "results_by_stage": all_results,
    }, output_path)
    print(f"\nRisultati HPO salvati in: {output_path}")

    # ── Propaga i migliori iperparametri (di tutti gli stadi) a train.json/test.json ──
    utils.apply_best_hpo_params(best_params_overall, args.train_config, args.test_config)

    utils.free_memory(reducer, latent_list, phi_scaled_list, prepared)


if __name__ == "__main__":
    main()