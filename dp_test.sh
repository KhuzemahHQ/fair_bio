#!/bin/bash -e
#SBATCH --job-name=dp_test
#SBATCH --output=logs/dp_test_%j.log
#SBATCH --error=logs/dp_test_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpunodes
#SBATCH --gres=gpu:1         
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
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

# Copy repo to scratch
if [ ! -d ~/CSC2615Proj/ ]; then
  echo "ERROR: ~/CSC2615Proj does not exist!"
  exit 1
fi
rsync -a --exclude='.git' ~/CSC2615Proj/ "$SCRATCH_PATH/CSC2615Proj/"

cd "$SCRATCH_PATH"

# Prepare venv and requirements (you can trim this if you already have an env)
python3 -m venv venv
source venv/bin/activate
pip install --upgrade numpy transformers datasets torch evaluate scikit-learn accelerate pandas matplotlib seaborn tqdm

echo "Running demographic parity test."
cd CSC2615Proj/fair_bio

# Make sure data directory exists in this repo layout; adjust if needed
python dp_test.py

ls -lr "$SCRATCH_PATH/CSC2615Proj"

echo "dp_test complete"
