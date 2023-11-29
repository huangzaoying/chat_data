import os
import argparse
import logging
from threading import Lock

import copy
import fastapi
from fastapi import Request
import uvicorn
import torch
import requests

from pydantic import BaseModel
from typing import Dict, List, Optional
from PIL import Image
import uuid

from llava.mm_utils import expand2square
from llava.train.train import add_speaker_and_signal
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.model.builder import load_pretrained_model
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from third_party.ocr import paddle_ocr

logger = logging.getLogger(__name__)

app = fastapi.FastAPI()
conv_mode = None
tokenizer = None
model = None
ocr = None
image_processor = None
lock = Lock()

# 记录一下 输入=key, 推理使用=value
chat_role_dict = {
    "assistant": "gpt",
    "user": "human"
}


class ChatImageCompletionRequest(BaseModel):
    image_path: Optional[str]
    messages: List[Dict[str, str]]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    num_beams: Optional[int] = 1
    max_tokens: Optional[int] = 256


class ChatImageCompletionResponse(BaseModel):
    message: Dict[str, str]
    finish_reason: str


def parse_inputs():
    parser = argparse.ArgumentParser(description='llava inference serving demo')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='IP address to listen')
    parser.add_argument('--port', type=int, default=8080, help='port to listen')
    parser.add_argument('--model-path', type=str, required=True, help='local model path')
    parser.add_argument("--conv-mode", type=str, default="cllava_v1")
    args = parser.parse_args()
    return args


def construct_prompt(conversations):
    assert len(conversations) > 0

    serving_conv = copy.deepcopy(conversations)
    for i, c in enumerate(serving_conv):
        if c['role'] == 'assistant':
            c['from'] = 'gpt'
        elif c['role'] == 'user':
            c['from'] = 'human'
        else:
            raise Exception(f"Wrong role input {c['role']}")
        c['value'] = conversations[i]['content']

    conv = conv_templates[args.conv_mode].copy()

    header = f"{conv.system}\n\n"
    prompt = add_speaker_and_signal(header, serving_conv) + conv.roles[1] + ": "
    return prompt


def get_input_ids(request, prompt):
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX,
                                      return_tensors='pt').unsqueeze(0).cuda()
    return input_ids


def get_image_tensor_and_ocr(request):
    '''
    从本地 或者 aliyun oss拉取图片
    '''
    if request.image_path == "" or request.image_path == None:
      return None, ""
    elif request.image_path.startswith("https") or request.image_path.startswith("http"):
        logger.debug(f"load image from remote address {request.image_path}")
        image_raw = requests.get(request.image_path, stream=True).raw
        new_image_path = os.path.basename(request.image_path)
        new_image_path = f'{uuid.uuid4()}-{new_image_path}'  # in case of concurrent requests.
        with open(new_image_path, 'wb') as file:
            print(f"receive image {new_image_path}\n")
            file.write(image_raw.read())
        assert os.path.getsize(new_image_path) > 1024  ## file is larger than 1kb, not corrupted.
        try:
            image = Image.open(new_image_path)
            ocr_result = paddle_ocr.ocr_img(ocr, new_image_path)
        except Exception as e:
            print(e)
            raise
        finally:
            os.remove(new_image_path)
    else:
        logger.debug(f"load image from local path {request.image_path}")
        image = Image.open(request.image_path)
        ocr_result = paddle_ocr.ocr_img(ocr, request.image_path)

    ocr_prompt = paddle_ocr.extract_text_prompt(ocr_result)
    image=image.convert('RGB')
    if getattr(model.config, "image_aspect_ratio", None) == 'pad':
        image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))

    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
    return image_tensor, ocr_prompt


def get_inference_outputs(request, input_ids, image_tensor):
    conv = conv_templates[conv_mode].copy()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    stopping_criteria = KeywordsStoppingCriteria([stop_str], tokenizer, input_ids)
    model.to(dtype=torch.bfloat16)
    if image_tensor is not None:
      image_tensor = image_tensor.unsqueeze(0).to(dtype=torch.bfloat16).cuda()
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True,
            temperature=request.temperature,
            top_p=request.top_p,
            num_beams=request.num_beams,
            max_new_tokens=request.max_tokens,
            use_cache=True,
            stopping_criteria=[stopping_criteria])

    input_token_len = input_ids.shape[1]
    #n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    #if n_diff_input_output > 0:
    #    logger.warn(f'{n_diff_input_output} output_ids are not the same as the input_ids')
    new_output_ids = output_ids[:, input_token_len:]
    finish_reason = 'length' if len(new_output_ids[0]) == request.max_tokens else 'stop'
    outputs = tokenizer.batch_decode(new_output_ids, skip_special_tokens=True)[0].strip()
    outputs = outputs.split(stop_str)[0]
    if outputs.endswith(stop_str):
        outputs = outputs[:-len(stop_str)]
    return outputs.strip(), finish_reason


@app.post('/v1/chat/image_completions')
async def create_chat_image_completions(request: ChatImageCompletionRequest,
                                  raw_request: Request):
    assert tokenizer is not None, 'tokenizer is None'

    lock.acquire()
    response = None
    try:
        logger.debug(f'request.image_path = {request.image_path}')
        logger.debug(f'request.messages = {request.messages}')
        logger.debug(f'request.max_tokens = {request.max_tokens}')

        image_tensor, ocr_prompt = get_image_tensor_and_ocr(request)

        prompt = construct_prompt(request.messages)
        prompt = paddle_ocr.append_ocr_prompt_to_prompt(ocr_prompt, prompt)
        print(f"\nprompt:{prompt}\n")
        logger.debug(f'next_role=ASSISTANT\nprompt={prompt}\n')
        input_ids = get_input_ids(request, prompt)
        outputs, finish_reason = get_inference_outputs(request, input_ids, image_tensor)
        logger.debug(f'outputs={outputs}')

        response = ChatImageCompletionResponse(
            message={'role': 'assistant', 'content': outputs},
            finish_reason=finish_reason)
    except Exception as e:
        response = ChatImageCompletionResponse(
            message={'role': 'assistant', 'content': ""},
            finish_reason="error")
    finally:
        lock.release()
    return response


def main(args):
    global tokenizer, model, image_processor, conv_mode, ocr
    conv_mode = args.conv_mode
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)

    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name)
    ocr = paddle_ocr.load_paddleocr()
    uvicorn.run(app,
                host=args.host,
                port=args.port,
                log_level='info',
                timeout_keep_alive=10)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='[%(asctime)s]-[%(levelname)s]-[%(filename)s:%(lineno)d] : %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S')
    args = parse_inputs()
    main(args)
