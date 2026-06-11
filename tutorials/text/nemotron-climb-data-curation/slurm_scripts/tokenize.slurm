#!/bin/bash
#SBATCH --job-name=climb-tokenize
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --qos=boost_qos_dbg
#SBATCH --gres=gpu:4
#SBATCH --time=00:30:00
#SBATCH --account=OELLM_prod2026
#SBATCH --output=logs/%x-42446631.log

#set -euo pipefail
module purge

CONTAINER="/leonardo_work/OELLM_prod2026/container_images/nemo_curator_CLIMB_26.04.sif"
export SINGULARITY_BINDPATH="/leonardo_work/OELLM_Catalog,/leonardo_work/OELLM_prod2026,${SCRATCH},${FAST}"

export SINGULARITYENV_HF_HOME=$HF_HOME
export SINGULARITYENV_HF_HUB_OFFLINE=1
export SINGULARITYENV_NUMEXPR_MAX_THREADS=$SLURM_CPUS_PER_TASK
export SINGULARITYENV_LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu/nvidia":$LD_LIBRARY_PATH
K=${K:-1000}

INPUT_DIR="/leonardo_scratch/large/userexternal/vkomulai/climb-embed-42446631/$K"
OUTPUT_DIR="$INPUT_DIR/tokenized_docs"
mkdir -p "$OUTPUT_DIR"

srun singularity exec --nv \
  --bind /leonardo_work/OELLM_prod2026/users/vkomulai/Curator:/opt/Curator \
  "$CONTAINER" bash -lc "
    source /opt/venv/env.sh
    exec python 4_tokenize.py \
    --input-path $INPUT_DIR/clusters/pruned_clusters \
    --output-path $OUTPUT_DIR \
    --text-field "text" \
    --tokenizer-model "/leonardo_work/OELLM_prod2026/huggingface/hub/models--meta-llama--Llama-2-7b/snapshots/69656aac4cb47911a639f5890ff35b41ceb82e98" \
    --append-eod
  "

sacct -X -j $SLURM_JOBID --format=JobID,JobName%30,State,Elapsed,ReqTRES%70
