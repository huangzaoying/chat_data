import argparse
import itertools
import json
import copy
import os
import time
import random

import torch
from tqdm import tqdm
from PIL import Image
from llava.mm_utils import expand2square

from llava.train.train import add_speaker_and_signal
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from third_party.ocr import paddle_ocr

def construct_prompt(conversations):
    assert len(conversations) > 0


    i = len(conversations) - 1
    while conversations[i]['from'] != 'gpt':
        i -= 1
        conversations.pop()
    gt = conversations[-1]['value']
    conversations.pop()

    conv = conv_templates[args.conv_mode].copy()
    header = f"{conv.system}\n"
    prompt = add_speaker_and_signal(header, copy.deepcopy(conversations)) + conv.roles[1] + ": "
    conversations.append({'from': 'gpt_groundtruth', 'value': gt})
    return prompt


def single_inference(image_tensor, tokenizer, model, args, conversations, ocr_prompt):
    prompt = construct_prompt(conversations)
    prompt = paddle_ocr.append_ocr_prompt_to_prompt(ocr_prompt, prompt)
    print(prompt)

    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    conv = conv_templates[args.conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    if image_tensor is not None:
        image_tensor=image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda()

    model.to(dtype=torch.bfloat16)
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            num_beams=args.num_beams,
            max_new_tokens=1024,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
    outputs = outputs.strip()
    outputs = outputs.split(stop_str)[0]
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    outputs = outputs.strip()

    conversations.append({'from': 'gpt', 'value': outputs})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default='')
    parser.add_argument('--data_path', type=str, default='')
    parser.add_argument('--image_folder', type=str, default='')
    parser.add_argument("--conv-mode", type=str, default="cllava_v2")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--use_ocr", type=bool, default=False)
    args = parser.parse_args()

    # load llava model
    disable_torch_init()
    model_path = os.path.expanduser(args.checkpoint)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)

    print(f"----- Start to process dataset: {args.image_folder} -----\n")

    # open the test file
    test_file = open(args.data_path, 'r')
    data = json.load(test_file)
    model.to(dtype=torch.bfloat16)

    ocr = None
    if args.use_ocr:
        ocr = paddle_ocr.load_paddleocr()

    # generate results
    for sample in data:
        ocr_prompt = ""
        if 'image' in sample and sample['image'] != "":
            img_path = os.path.join(args.image_folder, sample['image'])
            if ocr is not None:
                ocr_result = paddle_ocr.ocr_img(ocr, img_path)
                ocr_prompt = paddle_ocr.extract_text_prompt(ocr_result)

            image = Image.open(img_path).convert('RGB')
            if getattr(model.config, "image_aspect_ratio", None) == 'pad':
                image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        else:
            image_tensor = None
        single_inference(image_tensor, tokenizer, model, args, sample['conversations'], ocr_prompt)
        formatted_json = json.dumps(sample, indent=2, ensure_ascii=False)
        print(formatted_json)

    with open("result.json", 'w') as fin:
        json.dump(data, fin, ensure_ascii=False, indent=2)
