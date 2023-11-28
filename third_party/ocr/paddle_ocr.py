from paddleocr import PaddleOCR, draw_ocr
from llava.constants import DEFAULT_IMAGE_TOKEN

# Paddleocr目前支持的多语言语种可以通过修改lang参数进行切换
# 例如`ch`, `en`, `fr`, `german`, `korean`, `japan`
def load_paddleocr():
    ocr = PaddleOCR(use_angle_cls=True, lang="ch", use_gpu=False)  # need to run only once to download and load model into memory
    return ocr


def ocr_img(ocr, img_path):
    result = ocr.ocr(img_path, cls=True)
    return result

def extract_text_prompt(ocr_result):
    if not ocr_result:
        return ""

    contain_text = False
    text = "<text>"
    for idx in range(len(ocr_result)):
        res = ocr_result[idx]
        if not res:
            continue
        for line in res:
            if len(line) < 2 or len(line[-1]) < 2 or line[-1][1] < 0.5:
                continue
            contain_text = True
            text += "[" + line[-1][0] + "]"
    if not contain_text:
        return ""

    text += "</text>"
    return text


def visualize_ocr(img_path, result, output='result.jpg'):
    from PIL import Image
    result = result[0]
    image = Image.open(img_path).convert('RGB')
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    scores = [line[1][1] for line in result]
    im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')  # TODO
    im_show = Image.fromarray(im_show)
    im_show.save(output)


def append_ocr_prompt_to_prompt(ocr_prompt, prompt):
    if ocr_prompt != "" and DEFAULT_IMAGE_TOKEN in prompt:
        insert_position = prompt.find(DEFAULT_IMAGE_TOKEN) + len(DEFAULT_IMAGE_TOKEN)
        return prompt[:insert_position] + ocr_prompt + prompt[insert_position:]
    return prompt