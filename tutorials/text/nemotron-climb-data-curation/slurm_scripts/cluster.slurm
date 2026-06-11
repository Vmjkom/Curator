#!/bin/bash
#SBATCH --job-name=climb-cluster
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=50G
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --account=OELLM_prod2026
#SBATCH --output=logs/%x-%j.log

set -euo pipefail

CONTAINER="/leonardo_work/OELLM_prod2026/container_images/nemo_curator_CLIMB_26.04.sif"
export SINGULARITY_BINDPATH="/leonardo_work/OELLM_Catalog,/leonardo_work/OELLM_prod2026,${SCRATCH},${FAST}"

export SINGULARITYENV_HF_HOME=$HF_HOME
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export SINGULARITYENV_RAY_DASHBOARD_HOST=0.0.0.0

K=${K:-1000}
INPUT_DIR="/leonardo_scratch/large/userexternal/vkomulai/climb-embed-42446631"
OUTPUT_DIR="$INPUT_DIR/$K"
LOG_FILE=$OUTPUT_DIR/$SLURM_JOB_NAME.log
mkdir -p $OUTPUT_DIR

srun singularity exec --nv \
  --bind /leonardo_work/OELLM_prod2026/users/vkomulai/Curator:/opt/Curator \
    "$CONTAINER" bash -lc "
    source /opt/venv/env.sh
    python 2_cluster.py \
    --input-path $INPUT_DIR/embeddings \
    --output-path $OUTPUT_DIR/clusters \
    --text-field 'text' \
    --id-field '_curator_climb_id' \
    --n-clusters $K \
    --embedding-dim 3072 \
    --centroids-path $OUTPUT_DIR/centroids \
    --fit-data-fraction 0.9 \
    --num-cpus $SLURM_CPUS_PER_TASK \
    --num-gpus 4 \
" 2>&1 | tee -a $LOG_FILE

sacct -X -j $SLURM_JOBID --format=JobID,JobName%30,State,Elapsed,ReqTRES%70
