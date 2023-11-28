#!/bin/bash


export PYTHONPATH="$PYTHONPATH:/ML-A100/home/peter/multimodal_arch/:/ML-A100/home/peter/Multimodal-LLaVA/"

data_path=/ML-A100/home/peter/data/free_chat/badcase.json
image_folder=/ML-A100/home/peter/data/free_chat

checkpoint=/ML-A100/home/peter/checkpoints/llava-33b_1epoch_newdata_1120_nopi_gpt4all_ocr1123_fix4-chat

# checkpoint=/ML-A100/home/peter/Multimodal-LLaVA/scripts/checkpoints/llava-33b_2epoch_newdata_1120_nopi_gpt4all_ocr1123_fix2-chat


python interactive_chat.py \
    --checkpoint $checkpoint \
    --data_path $data_path \
    --image_folder $image_folder \
    --conv-mode cllava_v2 \
    --use_ocr True
