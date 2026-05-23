#!/bin/bash
CONTAINER="/leonardo_scratch/large/userexternal/vkomulai/nemo_curator_26.04.sif"
export SINGULARITY_BINDPATH="/leonardo_work/OELLM_Catalog,/leonardo_work/OELLM_prod2026,${SCRATCH},${TMPDIR},${FAST}"

VENV_PATH="/leonardo_work/OELLM_prod2026/users/vkomulai/Curator/.venv"

export SINGULARITYENV_HF_HOME=$HF_HOME
export SINGULARITYENV_HF_HUB_OFFLINE=1

SHARD_DIR="/leonardo_scratch/fast/OELLM_prod2026/vkomulai/sharded"
OUTPUT_DIR="/leonardo_scratch/fast/OELLM_prod2026/vkomulai/embeddings"
mkdir -p $SHARD_DIR $OUTPUT_DIR

singularity exec --nv --bind /leonardo_work/OELLM_prod2026/users/vkomulai/Curator:/opt/Curator "$CONTAINER" bash -c "
  source /opt/venv/bin/activate && \
  export PYTHONPATH=/opt/Curator:${VENV_PATH}/lib/python3.12/site-packages:\$PYTHONPATH && \
  cd /opt/Curator/tutorials/text/nemotron-climb-data-curation && \
  python 1_embed.py \
    --embedding-model NovaSearch/stella_en_400M_v5 \
    --input-path $SHARD_DIR \
    --input-filetype jsonl \
    --output-path $OUTPUT_DIR \
    --text-field text \
    --id-field _curator_climb_id \
    --use-sentence-transformer \
    --output-filetype jsonl
"