#!/bin/bash -e
#SBATCH --job-name=train_2615
#SBATCH --output=logs/train_2615_%j.log
#SBATCH --error=logs/train_2615_%j.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:rtx_a6000:1
#SBATCH --cpus-per-task=24
#SBATCH --mem=120G
#SBATCH --partition=gpunodes
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=pellement@cs.toronto.edu



set -x

echo "Running on node: $SLURM_JOB_NODELIST"

SCRATCH_BASE="/w/nobackup/385/scratch-space"
if [ ! -d "$SCRATCH_BASE" ]; then
  echo "ERROR: $SCRATCH_BASE does not exist."
  exit 1
fi

SCRATCH_DIR=$(ls -1 $SCRATCH_BASE | grep expires | sort | tail -n 1)
if [ -z "$SCRATCH_DIR" ]; then
  echo "ERROR: No expiry directories found in $SCRATCH_BASE"
  exit 1
fi


SCRATCH_PATH="$SCRATCH_BASE/$SCRATCH_DIR/$USER"
export XDG_CACHE_HOME="$SCRATCH_PATH/.cache"
export PIP_CACHE_DIR="$SCRATCH_PATH/.cache/pip"
export TMPDIR="$SCRATCH_PATH/tmp"
export WANDB_CACHE_DIR="$SCRATCH_PATH/.wandb"
export WANDB_DIR="$SCRATCH_PATH/.wandb"
mkdir -p "$SCRATCH_PATH"

echo "Using scratch path: $SCRATCH_PATH"

# Copy SAeUron repo to scratch
if [ ! -d ~/CSC2615Proj/ ]; then
  echo "ERROR: ~/CSC2615Proj does not exist!"
  exit 1
fi
rsync -a --exclude='.git' ~/CSC2615Proj/ "$SCRATCH_PATH/CSC2615Proj/"

# Copy activations dataset (from previous job's scratch path)
# If activations already in a scratch path (no home quota), just use from there:

cd "$SCRATCH_PATH"

# Prepare venv and requirements
python3 -m venv venv
source venv/bin/activate
pip install --upgrade numpy transformers datasets torch evaluate scikit-learn accelerate

# Train SAE for the collected hookpoint
echo "Training."
cd CSC2615Proj/fair_bio
python baseline.py

ls -lr "$SCRATCH_PATH/CSC2615Proj"

echo "Training complete"