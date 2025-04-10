#!/usr/bin/env bash



#bash ./tools/dist_train.sh \
#   projects/configs/sparsedrive_small_stage1.py \
#   "$GPU_NUM" \
#   --deterministic && \
#bash ./tools/dist_train.sh \
#   projects/configs/sparsedrive_small_stage2.py \
#   "$GPU_NUM" \
#   --deterministic

bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_tiny_stage1.py \
   "$GPU_NUM" \
   --deterministic && \
bash ./tools/dist_train.sh \
   projects/configs/sparsedrive_tiny_stage2.py \
   "$GPU_NUM" \
   --deterministic
