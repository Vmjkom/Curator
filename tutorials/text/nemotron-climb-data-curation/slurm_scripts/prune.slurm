#!/bin/bash
#SBATCH --job-name=climb-prune
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --account=OELLM_prod2026
#SBATCH --output=logs/%x-42446631.log

set -euo pipefail

CONTAINER="/leonardo_work/OELLM_prod2026/container_images/nemo_curator_26.04.sif"
export SINGULARITY_BINDPATH="/leonardo_work/OELLM_Catalog,/leonardo_work/OELLM_prod2026,${SCRATCH},${FAST}"

VENV_PATH="/leonardo_work/OELLM_prod2026/users/vkomulai/Curator/.venv"
# per-file counts and total (counts non-empty lines)

export SINGULARITYENV_HF_HOME=$HF_HOME
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

INPUT_DIR="/leonardo_scratch/large/userexternal/vkomulai/climb-embed-42446631"
OUTPUT_DIR="$INPUT_DIR/pruned_clusters"
SNAPSHOT="/leonardo_work/OELLM_prod2026/huggingface/hub/models--nvidia--nemotron-climb-fasttext-classifiers/snapshots/5a8e0e082d0d64f34281b6f6ff2555a158c36496"

FASTTEXT_MODEL_PATHS=(
  "$SNAPSHOT/best_model_advertisement.bin"
  "$SNAPSHOT/best_model_cultural_value.bin"
  "$SNAPSHOT/best_model_educational_value.bin"
  "$SNAPSHOT/best_model_informational_value.bin"
  "$SNAPSHOT/best_model_quality.bin"
)

FASTTEXT_SCORE_FIELDS=(
  advertisement_score
  cultural_value_score
  educational_value_score
  informational_value_score
  quality_score
)

FASTTEXT_PRUNING_THRESHOLDS=(2.0 1.0 1.0 1.0 1.0)

mkdir -p "$OUTPUT_DIR"

srun singularity exec --nv \
  --bind /leonardo_work/OELLM_prod2026/users/vkomulai/Curator:/opt/Curator \
  "$CONTAINER" bash -lc "
    source /opt/venv/bin/activate
    export PYTHONPATH=/opt/Curator:$VENV_PATH/lib/python3.12/site-packages:\$PYTHONPATH
    cd /opt/Curator/tutorials/text/nemotron-climb-data-curation

    exec python 3_prune.py \
      --input-path "$INPUT_DIR/clusters" \
      --output-path "$OUTPUT_DIR" \
      --fasttext-model-paths ${FASTTEXT_MODEL_PATHS[*]} \
      --score-fields ${FASTTEXT_SCORE_FIELDS[*]} \
      --text-field text \
      --pruning-thresholds ${FASTTEXT_PRUNING_THRESHOLDS[*]} \
      --centroids-path "$INPUT_DIR/centroids" \
      --merge-threshold 1.5
  "