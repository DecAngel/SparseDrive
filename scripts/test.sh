#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ENV_FILE="$SCRIPT_DIR/.env"
if [ -f $FILE ]; then
  export $(cat "$ENV_FILE" | xargs)
fi


CONFIG_FILE=projects/configs/sparsedrive_small_stage2.py
CKPT_FILE=ckpt/sparsedrive_stage2.pth

bash ./tools/dist_test.sh \
    "$CONFIG_FILE" \
    "$CKPT_FILE" \
    "$GPU_NUM" \
    --deterministic \
    --eval bbox
    # --result_file ./work_dirs/sparsedrive_small_stage2/results.pkl
