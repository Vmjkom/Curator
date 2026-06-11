#!/bin/bash
#SBATCH --job-name=climb-embed
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:2
#SBATCH --mem=256G
#SBATCH --qos=normal
#SBATCH --time=06:00:00
#SBATCH --account=OELLM_prod2026
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

CONTAINER="/leonardo_work/OELLM_prod2026/container_images/nemo_curator_CLIMB_26.04.sif"
export SINGULARITY_BINDPATH="/leonardo_work/OELLM_Catalog,/leonardo_work/OELLM_prod2026,${SCRATCH},${FAST}"

VENV_PATH="/leonardo_work/OELLM_prod2026/users/vkomulai/Curator/.venv"
# per-file counts and total (counts non-empty lines)

export SINGULARITYENV_HF_HOME=$HF_HOME
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK

SHARD_DIR="/leonardo_scratch/fast/OELLM_prod2026/vkomulai/sharded"
OUTPUT_DIR="/leonardo_scratch/large/userexternal/vkomulai/$SLURM_JOB_NAME-$SLURM_JOB_ID"
LOG_FILE=$SLURM_JOB_NAME-RAY.log
mkdir -p $SHARD_DIR $OUTPUT_DIR

srun singularity exec --nv \
  --bind /leonardo_work/OELLM_prod2026/users/vkomulai/Curator:/opt/Curator \
  "$CONTAINER" bash -c "
    source /opt/venv/bin/activate && \
    export PYTHONPATH=/opt/Curator:${VENV_PATH}/lib/python3.12/site-packages:\$PYTHONPATH && \
    cd /opt/Curator/tutorials/text/nemotron-climb-data-curation && \
    python 1_embed.py \
      --embedding-model NovaSearch/stella_en_400M_v5 \
      --input-path $SHARD_DIR/ \
      --input-filetype jsonl \
      --output-path $OUTPUT_DIR/embeddings \
      --text-field text \
      --id-field _curator_climb_id \
      --use-sentence-transformer \
      --output-filetype parquet \
      --metrics-dir $OUTPUT_DIR/metrics \
      --num-gpus 2 \
      --num-cpus 32 \
"

END_TIME=$(date +%s)
START_TIME=$(date -d "$(squeue -j $SLURM_JOB_ID -h -o '%S' 2>/dev/null || sacct -j $SLURM_JOB_ID -n -o Start | head -1)" +%s 2>/dev/null)

sacct -j $SLURM_JOB_ID \
  --format=JobID,JobName,Partition,ReqTRES,AllocTRES,Elapsed,CPUTime,MaxRSS,AveRSS,State \
  --units=G

# GPU hours estimate
ELAPSED=$(sacct -j $SLURM_JOB_ID -n -o Elapsed | head -1 | tr -d ' ')
echo "Elapsed: $ELAPSED"
