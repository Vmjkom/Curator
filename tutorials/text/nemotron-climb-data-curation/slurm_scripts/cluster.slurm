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

CONTAINER="/leonardo_work/OELLM_prod2026/container_images/nemo_curator_26.04.sif"
export SINGULARITY_BINDPATH="/leonardo_work/OELLM_Catalog,/leonardo_work/OELLM_prod2026,${SCRATCH},${FAST}"

VENV_PATH="/leonardo_work/OELLM_prod2026/users/vkomulai/Curator/.venv"

export SINGULARITYENV_HF_HOME=$HF_HOME
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export SINGULARITYENV_RAY_DASHBOARD_HOST=0.0.0.0

# RAY_PORT_BROADCAST_DIR needs to be in a path that's visible to all nodes. defaults to /tmp, which is node specific
if [ $SLURM_NNODES -gt 1 ]; then
  export RAY_PORT_BROADCAST_DIR="$FAST/$USER/$SLURM_JOB_NAME-$SLURM_JOB_ID/ray_ports"
fi
export RAY_TMPDIR=$TMPDIR/$SLURM_JOB_NODEID

INPUT_DIR="/leonardo_scratch/large/userexternal/vkomulai/climb-embed-42446631/embeddings"
OUTPUT_DIR="$INPUT_DIR"
LOG_FILE=$OUTPUT_DIR/$SLURM_JOB_NAME.log
mkdir -p $OUTPUT_DIR

srun singularity exec --nv \
  --bind /leonardo_work/OELLM_prod2026/users/vkomulai/Curator:/opt/Curator \
  "$CONTAINER" bash -c "
    source /opt/venv/bin/activate && \
    export PYTHONPATH=/opt/Curator:${VENV_PATH}/lib/python3.12/site-packages:\$PYTHONPATH && \
    cd /opt/Curator/tutorials/text/nemotron-climb-data-curation && \
    python 2_cluster.py \
    --input-path $INPUT_DIR \
    --output-path $OUTPUT_DIR/clusters \
    --text-field 'text' \
    --id-field '_curator_climb_id' \
    --embedding-dim 3072 \
    --centroids-path $OUTPUT_DIR/centroids \
    --verbose \
    --fit-data-fraction 0.9 \
    --num-cpus $SLURM_CPUS_PER_TASK \
    --num-gpus 4 \
" 2>&1 | tee -a $LOG_FILE

END_TIME=$(date +%s)
START_TIME=$(date -d "$(squeue -j $SLURM_JOB_ID -h -o '%S' 2>/dev/null || sacct -j $SLURM_JOB_ID -n -o Start | head -1)" +%s 2>/dev/null)

sacct -j $SLURM_JOB_ID \
  --format=JobID,JobName,Partition,ReqTRES,AllocTRES,Elapsed,CPUTime,MaxRSS,AveRSS,State \
  --units=G
