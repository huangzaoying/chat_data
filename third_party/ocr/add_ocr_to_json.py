import json
import os
import argparse
from pathlib import Path

from llava.eval.ocr import paddle_ocr
from llava.constants import DEFAULT_IMAGE_TOKEN

#######Process json file and add ocr for all images in a image_ocr json key
#######

# Set up argument parsing
parser = argparse.ArgumentParser(description='Process some images.')
parser.add_argument('--input', help='The input JSON file with chat data', required=True)
parser.add_argument('--output', help='The output JSON file with OCR results', required=True)
parser.add_argument('--image_dir', help='', required=True)

# Parse arguments
args = parser.parse_args()

ocr = paddle_ocr.load_paddleocr()

# Read the original JSON file
with open(args.input, 'r', encoding='utf-8') as file:
    data = json.load(file)

# Iterate over each entry and perform OCR on the image_path
for entry in data:
    # Prepend the base directory to the image path
    image_path = os.path.join(args.image_dir, entry['image'])
    
    # Ensure the image file exists before attempting OCR
    for conv in entry['conversations']:
        if DEFAULT_IMAGE_TOKEN in conv['value']:
            if os.path.exists(image_path):
                ocr_result = paddle_ocr.ocr_img(ocr, image_path)
                ocr_prompt = paddle_ocr.extract_text_prompt(ocr_result)
                conv['value'] = paddle_ocr.append_ocr_prompt_to_prompt(ocr_prompt, conv['value'])
            else:
                raise Exception(f"{image_path} not found\n")
            break

# Save the updated data to a new JSON file
with open(args.output, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)
