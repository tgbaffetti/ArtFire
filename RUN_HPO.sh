#!/bin/bash
#SBATCH --job-name=POD_HPO_multiGPU
#SBATCH --output=/globalsc/ulb/atm/baffetti/research/Transformer/ArtFire_files/AAA_newT_logs_POD_OOM_HPO/POD_OOM_%j.out
#SBATCH --error=/globalsc/ulb/atm/baffetti/research/Transformer/ArtFire_files/AAA_newT_logs_POD_OOM_HPO/POD_OOM_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tommaso.baffetti@ulb.be
#SBATCH --partition=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=5-00:00:00

# Numero di GPU sullo stesso nodo da usare per l'HPO (1..4). Ogni GPU esegue
# un proprio processo worker che contribuisce trial allo STESSO Optuna study
# (storage sqlite condiviso): non assume mai che siano disponibili
# esattamente 4 GPU. Sovrascrivibile: `NUM_GPUS=2 sbatch RUN_HPO.sh`.
NUM_GPUS="${NUM_GPUS:-4}"
CONFIG="${CONFIG:-HPO.json}"
TRAIN_CONFIG="${TRAIN_CONFIG:-train.json}"
TEST_CONFIG="${TEST_CONFIG:-test.json}"

echo "Start:"; date
module load Python/3.11.3
source /globalsc/ulb/atm/baffetti/envs/artfire/bin/activate

STUDY_NAME="pod_transformer_hpo_$$"
STORAGE_DIR="$(dirname "$TRAIN_CONFIG")/hpo_storage"
mkdir -p "$STORAGE_DIR"
STORAGE="sqlite:///${STORAGE_DIR}/${STUDY_NAME}.db"

N_TRIALS_TOTAL=$(python3 -c "import json;print(json.load(open('$CONFIG'))['stages']['transformer']['n_trials'])")
N_TRIALS_PER_WORKER=$(( (N_TRIALS_TOTAL + NUM_GPUS - 1) / NUM_GPUS ))
echo "GPU: $NUM_GPUS — trial totali: $N_TRIALS_TOTAL — trial per worker: $N_TRIALS_PER_WORKER"

# Worker 1..NUM_GPUS-1: contribuiscono trial ma non scrivono i risultati finali
for ((gpu = 1; gpu < NUM_GPUS; gpu++)); do
    CUDA_VISIBLE_DEVICES=$gpu python3 -u HPO.py --config "$CONFIG" \
        --train-config "$TRAIN_CONFIG" --test-config "$TEST_CONFIG" \
        --storage "$STORAGE" --study-name "$STUDY_NAME" \
        --n-trials-per-worker "$N_TRIALS_PER_WORKER" --gpus 1 --skip-apply &
done

# Worker 0 (coordinatore): contribuisce trial e infine salva i risultati e
# aggiorna train.json/test.json con i migliori iperparametri.
CUDA_VISIBLE_DEVICES=0 python3 -u HPO.py --config "$CONFIG" \
    --train-config "$TRAIN_CONFIG" --test-config "$TEST_CONFIG" \
    --storage "$STORAGE" --study-name "$STUDY_NAME" \
    --n-trials-per-worker "$N_TRIALS_PER_WORKER" --gpus 1

wait
echo "End:"; date
