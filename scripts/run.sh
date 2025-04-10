#!/usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")

ENV_FILE="$ROOT_DIR/.env"
if [ -f $ENV_FILE ]; then
  export $(cat "$ENV_FILE" | xargs)
fi

CONFIG_DIR="$ROOT_DIR/$CONFIG_DIR"
WORK_DIR="$ROOT_DIR/$WORK_DIR"

cd "$ROOT_DIR" || {
    echo "Cannot cd to $ROOT_DIR"
    exit 1
}
export PYTHONPATH="$SCRIPT_DIR/..":$PYTHONPATH

# 使用方法说明函数
usage() {
    echo "Usage: $0 {train|test|visualize}"
    exit 1
}

# 检查参数数量
if [ $# -ne 1 ]; then
    usage
fi

# 获取动作参数
ACTION=$1

case $ACTION in
    train1)
        echo "Training Stage 1"
        export CHECKPOINT=${CHECKPOINT:-""}
        python3 -m torch.distributed.launch --nproc_per_node="$GPU_NUM" --master_port="$PORT" \
        "${ROOT_DIR}/tools/train.py" "${CONFIG_DIR}/${STAGE_1}.py" --launcher pytorch --deterministic
        ;;
    train2)
        echo "Training Stage 2"
        export CHECKPOINT=${CHECKPOINT:-"${WORK_DIR}/${STAGE_1}/latest.pth"}
        python3 -m torch.distributed.launch --nproc_per_node="$GPU_NUM" --master_port="$PORT" \
        "${ROOT_DIR}/tools/train.py" "${CONFIG_DIR}/${STAGE_2}.py" --launcher pytorch --deterministic
        ;;
    train)
        echo "Training Stage 1&2"
        CHECKPOINT=${CHECKPOINT:-""} python3 -m torch.distributed.launch --nproc_per_node="$GPU_NUM" --master_port="$PORT" \
        "${ROOT_DIR}/tools/train.py" "${CONFIG_DIR}/${STAGE_1}.py" --launcher pytorch --deterministic && \
        CHECKPOINT="${WORK_DIR}/${STAGE_1}/latest.pth" python3 -m torch.distributed.launch --nproc_per_node="$GPU_NUM" --master_port="$PORT" \
        "${ROOT_DIR}/tools/train.py" "${CONFIG_DIR}/${STAGE_2}.py" --launcher pytorch --deterministic
        ;;
    test)
        echo "Testing"
        export CHECKPOINT=${CHECKPOINT:-"${WORK_DIR}/${STAGE_2}/latest.pth"}
        python3 -m torch.distributed.launch --nproc_per_node="$GPU_NUM" --master_port="$PORT" \
        "${ROOT_DIR}/tools/test.py" "${CONFIG_DIR}/${STAGE_2}.py" "$CHECKPOINT" --launcher pytorch --deterministic --eval bbox
        ;;
    visualize)
        echo "Visualizing"
        export CHECKPOINT=${CHECKPOINT:-"${WORK_DIR}/${STAGE_2}/latest.pth"}
        python3 "${ROOT_DIR}/tools/visualization/visualize.py" \
	      "${CONFIG_DIR}/${STAGE_2}.py" \
	      --result-path "${WORK_DIR}/${STAGE_2}/results.pkl"
        ;;
    *)
        echo "未知操作 '$ACTION'"
        bash "$@"
        usage
        ;;
esac
