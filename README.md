# POD/AE + Transformer — guida ai test

Questa guida elenca i comandi per verificare che l'intera pipeline
(`train.py` → `test.py` → `HPO.py`) funzioni, in 4 scenari:

| # | GPU | Dati |
|---|-----|------|
| 1 | 1   | dummy |
| 2 | 1   | reali |
| 3 | 2   | dummy |
| 4 | 2   | reali |

e come aggiungere il nuovo modello `ae-transformer`.

Tutti i comandi vanno lanciati dalla cartella del progetto (dove stanno
`train.py`, `test.py`, `HPO.py`, `models.py`, `utils.py`).

Prerequisiti: `pip install torch optuna scikit-learn matplotlib openmeasure`
(oppure l'ambiente conda/venv gia' predisposto sul cluster).

---

## 0. Generare dati dummy (se non li hai gia')

```bash
python3 create_dummy_data.py --output-dir dummy_data --n-cells 64 --nf 11 --n-timesteps-train 2001 --n-timesteps-test 4001 --n-train-sets 6 --n-test-sets 2
```

Con `--reference train.json` copia automaticamente `nf` e il numero di
timestep dal dataset reale referenziato in quel JSON (utile per avere un
dummy "realistico" ma leggero):

```bash
python3 create_dummy_data.py --output-dir dummy_data --reference train.json
```

Genera `dummy_train_*.npy`, `dummy_test_*.npy`, `phi_*.npy`, `grid.npy`.

Poi crea un `train_dummy.json` / `test_dummy.json` / `HPO_dummy.json`
copiando quelli reali e sostituendo `dataset.data_paths`, `dataset.phi_paths`,
`dataset.grid_path`, `dataset.features` (in test.json) con i path dei file
dummy appena generati. Riduci anche `training.epochs_transformer` (es. 5) e
`architecture.rank_POD`/`latent_dim` (es. 4) per un giro rapido.

---

## 1. Caso 1 GPU — dati dummy

**Training (run singolo):**
```bash
python3 train.py --config train_dummy.json --gpus 1
```
Verifica: log `Modello: pod-transformer | device: ['cuda:0']`, POD fit,
`ROM salvato in: ...`, epoche con `loss:`/`val_loss:` decrescente,
`loss_trend.png` e `<label>_<epoch>_<loss>.pt` dentro `output.models_dir`.

**Training con esperimenti (griglia parametri):**
```bash
python3 train.py --config train_dummy.json --experiments experiments.txt --gpus 1
python3 train.py --config train_dummy.json --experiments experiments.txt --experiment-index 2 --gpus 1
```
Verifica: una cartella `models_<override>` per ogni riga di `experiments.txt`
(vedi log `Esperimento #N: {...}`), ciascuna con i propri checkpoint.

**Test:**
```bash
python3 test.py --config test_dummy.json --gpus 1
```
Verifica: `Miglior checkpoint: ...`, metriche R2/NRMSE per feature e per
modo stampate a schermo, `T_t<N>_f<Nf>_c<n_cells>.npy` salvato,
`RMSE_evolution.png`/`Err_evolution.png`/`R2_evolution.png` per feature,
`attention_map.png`/`attention_map_avg.png`.

**HPO:**
```bash
python3 HPO.py --config HPO_dummy.json --train-config train_dummy.json --test-config test_dummy.json --gpus 1
```
Verifica: un blocco `[HPO/transformer] Trial N` per trial, eventuali
`Trial N pruned.`, `Risultati HPO salvati in: ...`, e infine
`train.json aggiornato` / `test.json aggiornato` — riapri
`train_dummy.json` e controlla che `lr`/`transformer_heads`/ecc. siano
cambiati rispetto a prima.

**Verifica di coerenza end-to-end (il punto piu' importante):**
```bash
python3 train.py --config train_dummy.json --gpus 1   # riallena con gli iperparametri appena aggiornati da HPO
python3 test.py --config test_dummy.json --gpus 1     # testa il modello appena riallenato
```
Se entrambi girano senza errori di shape/state_dict, il giro
HPO → train.json/test.json → train → test e' coerente.

---

## 2. Caso 1 GPU — dati reali

Identico al caso 1, ma con `train.json`/`test.json`/`HPO.json` reali (quelli
gia' pronti nel repository, che puntano ai path reali su
`/globalsc/ulb/...`):

```bash
python3 train.py --config train.json --gpus 1
python3 train.py --config train.json --experiments experiments.txt --gpus 1
python3 test.py --config test.json --gpus 1
python3 HPO.py --config HPO.json --gpus 1
```

Su cluster SLURM, equivalenti via script (chiamano automaticamente gli stessi comandi sopra):
```bash
NUM_GPUS=1 CONFIG=train.json EXPERIMENTS=experiments.txt sbatch RUN_train.sh
NUM_GPUS=1 CONFIG=test.json sbatch RUN_test.sh
NUM_GPUS=1 CONFIG=HPO.json sbatch RUN_HPO.sh
```

---

## 3. Caso 2 GPU — dati dummy

Ci sono DUE modi distinti di usare 2 GPU (vedi risposta precedente):

**A) Un solo training diviso su 2 GPU (`nn.DataParallel`, stesso batch spezzato):**
```bash
python3 train.py --config train_dummy.json --gpus 2
```
Verifica: log `device: ['cuda:0', 'cuda:1']`. Comportamento numerico
identico (nessun cambio di loss atteso), ma piu' veloce su batch grandi.
Funziona anche se sul nodo c'e' una sola GPU: si adatta da solo, non crasha
(vedi `utils.resolve_devices`).

**B) Esperimenti indipendenti, uno per GPU (parallelismo "imbarazzante"):**
```bash
CUDA_VISIBLE_DEVICES=0 python3 train.py --config train_dummy.json --experiments experiments.txt --experiment-index 0 --gpus 1 &
CUDA_VISIBLE_DEVICES=1 python3 train.py --config train_dummy.json --experiments experiments.txt --experiment-index 1 --gpus 1 &
wait
```
oppure, automaticamente per tutte le righe:
```bash
NUM_GPUS=2 CONFIG=train_dummy.json EXPERIMENTS=experiments.txt bash RUN_train.sh
```
Verifica: due (o piu') training in esecuzione in parallelo (`nvidia-smi`
mostra entrambe le GPU occupate), ciascuno con la propria cartella
`models_<override>`.

**Test con 2 GPU** (poco utile in pratica — un singolo rollout autoregressivo
non parallelizza bene, ma il meccanismo e' identico):
```bash
python3 test.py --config test_dummy.json --gpus 2
```

**HPO con 2 GPU (2 worker sullo stesso Optuna study, storage condiviso):**
```bash
NUM_GPUS=2 CONFIG=HPO_dummy.json TRAIN_CONFIG=train_dummy.json TEST_CONFIG=test_dummy.json bash RUN_HPO.sh
```
Oppure manualmente (equivalente a quello che fa lo script):
```bash
STORAGE="sqlite:///hpo_storage/study_dummy.db"
CUDA_VISIBLE_DEVICES=1 python3 HPO.py --config HPO_dummy.json --train-config train_dummy.json --test-config test_dummy.json \
    --storage "$STORAGE" --study-name pod_dummy --n-trials-per-worker 5 --gpus 1 --skip-apply &
CUDA_VISIBLE_DEVICES=0 python3 HPO.py --config HPO_dummy.json --train-config train_dummy.json --test-config test_dummy.json \
    --storage "$STORAGE" --study-name pod_dummy --n-trials-per-worker 5 --gpus 1
wait
```
Verifica: i trial dei due processi si intrecciano nel log (numeri di trial
crescenti condivisi tra i due, non ripetuti); solo il processo SENZA
`--skip-apply` scrive i risultati finali e aggiorna train.json/test.json.

---

## 4. Caso 2 GPU — dati reali

Stessi comandi della sezione 3, con `train.json`/`test.json`/`HPO.json`
reali al posto di quelli `_dummy`:
```bash
python3 train.py --config train.json --gpus 2
NUM_GPUS=2 CONFIG=train.json EXPERIMENTS=experiments.txt sbatch RUN_train.sh
NUM_GPUS=2 CONFIG=HPO.json sbatch RUN_HPO.sh
```

---

## 5. Checklist rapida "funziona sicuro"

1. `create_dummy_data.py` gira senza errori e produce file piccoli (poche MB).
2. `train.py` (dummy, 1 GPU, poche epoche) produce checkpoint + `loss_trend.png`
   e la loss scende nelle prime epoche.
3. `test.py` sullo stesso `models_dir` trova il checkpoint, produce metriche
   e plot senza `Traceback`.
4. `HPO.py` (dummy, 2 trial, 2 epoche) completa, produce `hpo_results.json`
   e aggiorna `train.json`/`test.json`.
5. Ri-lanciare `train.py`/`test.py` sui JSON appena aggiornati da HPO non da'
   errori (architettura coerente, presa dal checkpoint in test.py — vedi
   `models.PODTransformerModel.load_checkpoint`).
6. Ripetere 2-5 con `--gpus 2` (o su un nodo con 2 GPU reali): stessi
   risultati, nessun errore, `device: ['cuda:0', 'cuda:1']` nei log.
7. Ripetere tutto con `train.json`/`test.json`/`HPO.json` reali al posto dei
   `_dummy` (stessa procedura, solo path/dimensioni diverse).

---

## 6. Aggiungere `ae-transformer` (autoencoder fully-connected + Transformer)

E' gia' implementato in `models.py` (classe `AETransformerModel`, registrata
come `"ae-transformer"`). Non serve **nessuna modifica** a
`train.py`/`test.py`/`HPO.py`: e' proprio la dimostrazione che il design
model-agnostic funziona (l'ho verificato con un giro train→test completo).

Riusa integralmente la logica di rollout/training/valutazione di
`PODTransformerModel` (stessa classe base): l'unica differenza e' che la
riduzione dimensionale non e' una SVD analitica (POD) ma un autoencoder
fully-connected allenato con la sua loss di ricostruzione, prima di
proiettare l'intero dataset nello spazio latente una sola volta (stesso
principio "one-shot" della POD).

### Cosa aggiungere in `train.json`

```jsonc
{
  "model": "ae-transformer",          // era "pod-transformer"
  "dataset": {
    "data_paths": [...],
    "grid_path": "...",
    "phi_paths": [...],
    "ae_training": {                   // NUOVO blocco, solo per ae-transformer
      "ae_epochs": 100,
      "ae_lr": 0.001,
      "ae_batch_size": 64,
      "ae_hidden_dims": [256, 128],    // layer nascosti encoder (decoder = specchiati)
      "ae_dropout": 0.05
    }
  },
  "architecture": {
    "latent_dim": 10,                  // sostituisce "rank_POD" (che resta solo per pod-transformer)
    "transformer_embed_dim": 64,
    "transformer_layers": 2,
    "transformer_heads": 8,
    "transformer_hidden_dim": 256
  },
  "training": { /* invariato: lr, dropout, n_past, rollout_steps, ecc. */ }
}
```

`train.py` legge il rank/latent-dim con
`architecture.get("rank_POD", architecture.get("latent_dim"))`, quindi basta
usare `latent_dim` al posto di `rank_POD` e il resto del file (training,
scheduler, physics_constraints, loss_weights, hardware, output) resta
identico nella struttura.

### Cosa aggiungere in `test.json`

**Niente di nuovo.** Come per pod-transformer, `test.json` non contiene
architettura: tutto (incluso `ae_hidden_dims`/`latent_dim`) viene letto dal
checkpoint. Basta cambiare `"model": "ae-transformer"`.

### Cosa aggiungere in `HPO.json`

```jsonc
{
  "model": "ae-transformer",
  "stages": {
    "transformer": { /* identico a pod-transformer: ottimizza SOLO il Transformer */ }
  }
}
```

Nella versione attuale l'HPO ottimizza solo lo stadio `"transformer"` anche
per `ae-transformer` (l'autoencoder viene allenato una volta con gli
iperparametri fissi di `dataset.ae_training`, esattamente come la POD ha
`rank_POD` fisso). Il design a stadi in `HPO.py` (`run_stage`, chiamato per
ogni chiave di `hpo_cfg["stages"]`) e' gia' pronto per un vero stadio
`"autoencoder"` (es. per ottimizzare `ae_hidden_dims`/`ae_lr`/`ae_dropout`
prima del Transformer): serve solo aggiungere una seconda chiave in
`"stages"` e una piccola funzione `objective` per quello stadio (stessa
forma di quella per `"transformer"`, ma che allena solo l'AE e usa la sua
reconstruction loss come objective) — non l'ho implementata perche' non
richiesta esplicitamente, ma l'impalcatura e' gia' quella pensata per
supportarlo senza toccare `train.py`/`test.py`.

### Comando di prova rapido (dati dummy)
```bash
python3 create_dummy_data.py --output-dir dummy_data --n-cells 40 --nf 5 \
    --n-timesteps-train 60 --n-timesteps-test 40
# poi train_dummy.json/test_dummy.json con "model": "ae-transformer",
# "architecture.latent_dim" al posto di "rank_POD", "dataset.ae_training" aggiunto
python3 train.py --config train_dummy.json --gpus 1
python3 test.py --config test_dummy.json --gpus 1
```
Verifica: log `Training autoencoder FC (latent_dim=..., ...)` con
`recon_loss` decrescente PRIMA dell'addestramento del Transformer, poi il
resto identico al caso pod-transformer. I checkpoint si chiamano
`ae_transformer_<epoch>_<loss>.pt` (il nome deriva automaticamente da
`--model`/`settings["model"]`, non e' piu' hardcoded a "pod_model").