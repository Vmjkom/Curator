#!/bin/bash
#SBATCH --job-name=climb-prune-k1000
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --account=OELLM_prod2026
#SBATCH --output=logs/%x-42446631.log

set -euo pipefail

CONTAINER="/leonardo_work/OELLM_prod2026/container_images/nemo_curator_CLIMB_26.04.sif"
export SINGULARITY_BINDPATH="/leonardo_work/OELLM_Catalog,/leonardo_work/OELLM_prod2026,${SCRATCH},${FAST}"

export SINGULARITYENV_HF_HOME=$HF_HOME
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

INPUT_DIR="/leonardo_scratch/large/userexternal/vkomulai/climb-embed-42446631/1000/clusters"
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
    source /opt/venv/env.sh
    exec python 3_prune.py \
      --input-path "$INPUT_DIR" \
      --output-path "$OUTPUT_DIR" \
      --fasttext-model-paths ${FASTTEXT_MODEL_PATHS[*]} \
      --score-fields ${FASTTEXT_SCORE_FIELDS[*]} \
      --text-field text \
      --pruning-thresholds ${FASTTEXT_PRUNING_THRESHOLDS[*]} \
      --centroids-path /leonardo_scratch/large/userexternal/vkomulai/climb-embed-42446631/1000/centroids \
      --merge-threshold 1.5 \
      --metrics-dir metrics/ \
      --num-cpus $SLURM_CPUS_PER_TASK \
  "

sacct -X -j $SLURM_JOBID --format=JobID,JobName%30,State,Elapsed,ReqTRES%70
