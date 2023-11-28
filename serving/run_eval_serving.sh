#! /usr/bin/bash

set -ex

CURDIR=$(cd "$(dirname "$0")"; pwd)
cd $CURDIR

LLAVA_ROOT=/ML-A100/home/peter/Multimodal-LLaVA/
# LLAVA_ROOT=/ML-A100/home/peter/Multimodal-LLaVA
export PYTHONPATH="$LLAVA_ROOT:$PYTHONPATH"

python ./eval_serving.py \
    --host 0.0.0.0 \
    --port 8080 \
    --model-path /ML-A100/home/peter/checkpoints/llava-33b_1epoch_newdata_1120_nopi_gpt4all_ocr1123_fix4-chat >> serving.log 2>&1 \
