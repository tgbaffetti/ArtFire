#!/bin/bash
#SBATCH --job-name=train_POD_Transformer
#SBATCH --output=../logs_POD_Transformer_train/train_%j.out
#SBATCH --error=../logs_POD_Transformer_train/train_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tommaso.baffetti@ulb.be
#SBATCH --partition=batch
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

# Numero di GPU sullo stesso nodo da usare (1..4). Non assume che siano
# sempre disponibili 4 GPU: se ne sono richieste/disponibili meno, si usano
# quelle. Sovrascrivibile: `NUM_GPUS=2 sbatch RUN_train.sh`
NUM_GPUS="${NUM_GPUS:-4}"
CONFIG="${CONFIG:-train.json}"
EXPERIMENTS="${EXPERIMENTS:-experiments.txt}"

echo "Start:"; date
module load Python/3.11.3
source /globalsc/ulb/atm/baffetti/envs/artfire/bin/activate

if [ -f "$EXPERIMENTS" ]; then
    # Esperimenti indipendenti (una riga di experiments.txt = un training):
    # parallelizzazione "embarrassingly parallel", 1 esperimento per GPU,
    # fino a NUM_GPUS esperimenti in parallelo sullo stesso nodo.
    N_EXPERIMENTS=$(($(wc -l < "$EXPERIMENTS") - 1))
    echo "Esperimenti trovati: $N_EXPERIMENTS — GPU disponibili: $NUM_GPUS"

    idx=0
    while [ "$idx" -lt "$N_EXPERIMENTS" ]; do
        gpu=$((idx % NUM_GPUS))
        echo "Lancio esperimento #$idx su GPU $gpu"
        CUDA_VISIBLE_DEVICES=$gpu python3 -u train.py --config "$CONFIG" \
            --experiments "$EXPERIMENTS" --experiment-index "$idx" --gpus 1 &

        # Non piu' di NUM_GPUS processi contemporanei
        if [ $(( (idx + 1) % NUM_GPUS )) -eq 0 ]; then
            wait
        fi
        idx=$((idx + 1))
    done
    wait
else
    # Nessun file esperimenti: un singolo training, eventualmente distribuito
    # su piu' GPU con nn.DataParallel (vedi utils.resolve_devices).
    echo "Training singolo su $NUM_GPUS GPU (DataParallel se > 1)"
    python3 -u train.py --config "$CONFIG" --gpus "$NUM_GPUS"
fi

echo "End:"; date
