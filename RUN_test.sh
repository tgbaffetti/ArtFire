#!/bin/bash
#SBATCH --job-name=test_POD_Transformer
#SBATCH --output=../logs_POD_Transformer_test/test_%j.out
#SBATCH --error=../logs_POD_Transformer_test/test_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tommaso.baffetti@ulb.be
#SBATCH --partition=batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=00:30:00

# Il test e' tipicamente un singolo rollout autoregressivo: non beneficia di
# DataParallel su piu' GPU quanto il training. Di default usa 1 GPU;
# sovrascrivibile con `NUM_GPUS=2 sbatch RUN_test.sh`.
NUM_GPUS="${NUM_GPUS:-1}"
CONFIG="${CONFIG:-test.json}"
EXPERIMENTS="${EXPERIMENTS:-}"

echo "Start:"; date
module load Python/3.11.3
source /globalsc/ulb/atm/baffetti/envs/artfire/bin/activate

if [ -n "$EXPERIMENTS" ] && [ -f "$EXPERIMENTS" ]; then
    python3 -u test.py --config "$CONFIG" --experiments "$EXPERIMENTS" --gpus "$NUM_GPUS"
else
    python3 -u test.py --config "$CONFIG" --gpus "$NUM_GPUS"
fi

echo "End:"; date
