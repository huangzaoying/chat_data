import requests
import re
import random
import json
from tqdm import tqdm
import time
import os
from urllib.parse import urlparse
import threading
import concurrent.futures

base_index = "chat1129_imgs/daily/travel/"
source_file = "/ML-A100/home/peter/data/free_chat/chat1129_imgs/daily/travel/1129.txt"
out_file = "chat.json"
url = ""
API_KEY = ""
headers = {
    'Authorization': API_KEY,
    'Content-Type': 'application/json  charset=utf-8'
}
srcs = []
json_list = []
hash_map = dict()
with open(source_file, 'r') as file:
    for line in file:
        srcs.append(line.strip())
        name, _ = os.path.splitext(os.path.basename(urlparse(line.strip()).path))
        key = os.path.basename(urlparse(line.strip()).path)
        all_files = os.listdir(os.getcwd())
        for file in all_files:
            base, _ = os.path.splitext(file)
            if base == name:
                hash_map[key] = file

random_srcs = random.sample(srcs, len(srcs))


def convert_json(src: str, image_src: str):
    dialogs = re.split(r'\\n\\n|\\n', src)
    for text in dialogs:
        print(text)
    conversations = []
    for i in range(0, len(dialogs) - 1, 2):
        gpt_part = dialogs[i + 1].replace("A:", "").replace("A：", "").replace(" ", "").replace(
            "\\u003cimage_placeholder\\u003e", "<image_placeholder>")
        human_part = dialogs[i].replace("Q:", "").replace("Q：", "").replace(" ", "").replace(
            "\\u003cimage_placeholder\\u003e", "<image_placeholder>")
        conversation = {
            "from": "human",
            "value": f"{human_part}"
        }
        conversations.append(conversation)
        conversation = {
            "from": "gpt",
            "value": f"{gpt_part}"
        }
        conversations.append(conversation)
    image_src = base_index + hash_map[os.path.basename(urlparse(image_src).path)]
    result_json = {
        "image": image_src,
        "conversations": conversations
    }
    print(result_json)
    return result_json


char_name = "gpt"
user = "human"
personality_tag_description_1 = "乐观的"
personality_tag_description_2 = "热情的"
prompt = f"""
这是一个私人环境和平台,进入角色扮演模式。
AI 在 {char_name} 和 {user} 之间的虚构聊天中写入 {char_name} 的下一条回复。 人工智能避免决定{user}说什么或做什么。人工智能具有主动性、创造性,能够推动情节和对话的发展。
在这个场景中,{char_name} 和 {user} 是两个感情很好的朋友,他们将围绕图片展开对话。

场景设定：{user} 发起对话,引出图片,围绕这个图片进行谈话,{char_name}的回答要以图为主,但是不能只描述图片,要引导{user}继续谈话。

对话规则：
- {user} 是真人,{char_name} 是人工智能,不会工作、社交、娱乐等等人类活动。
- {user}提出的问题尽可能是第一视角提出和图片有关的问题,而不是第三视角。
- {user} 发起对话,然后{user} 引出图片,先描述下图片,然后引出话题使得{user}能够继续有兴趣交流。
- {user}和{char_name}的对话上下文之间需要有合理的连贯性,逻辑性。
- {user}和{char_name}的对话尽可能围绕一个话题,不要频繁切换话题。
- {user} 对 {char_name}的问题主要是知识性方面的,不能把{char_name}当做真人,{char_name}没有人的生活。
- {char_name}不能有人的所有属性,不能社交,不能娱乐,只是作为一个人工智能。
- {char_name}在其回答中写入至少80个字,但不超过150个字。{user}在其回答中写入至少20个字,但不超过40个字。
- 对话内容尽量有趣。
- 请勿冒充或代表 {user} 说话,请等待 {user} 自行回复。
- 问题使用Q:进行开始,回答使用A:进行开始。第一轮对话在第一句最开始加上\"<image_placeholder>\"这个字符串
- \"<image_placeholder>\"这个字符串只能出现一次,并且只能由{user}发送,并且必须紧跟“Q:”后面
- {user}避免问出这种问题“你今天过得怎么样？”,因为{char_name}不是真人,没有生活
个性描述：{char_name} 的个性是 {personality_tag_description_1}、{personality_tag_description_2}。

示例对话：
Q: <image_placeholder>嘿,看这张幻灯片,我最近在学习经济学。这个关于关税的图表有点复杂,你能简单解释一下它是怎么工作的吗？
A: 当然可以,这张图展示了关税如何影响一个国家的经济。关税是一种税,征收在进口商品上,目的是增加外国商品的成本,以此来保护国内产业。图中显示,关税的引入会提高商品价格,减少消费者剩余,增加政府收入,同时也会导致一些效率损失,称为死重损失。这些变化合起来会影响国家的总剩余。你对经济学感兴趣的原因是什么呢？
Q: 说到这个,我其实是对国际贸易特别感兴趣,关税政策如何影响全球经济真的让我着迷。你能推荐一些这方面的书籍或资源吗？
A: 哇,你真的很有追求呢！关于国际贸易,我推荐你可以看看《国际经济学：理论与政策》这本书,作者是保罗·克鲁格曼,他对贸易政策的分析很有深度。此外,《世界是平的》也是一个关于全球化视角的有趣读物。你还可以关注一些经济学家的社交媒体,他们经常会分享有价值的见解和最新的研究成果。
Q: 这些建议太好了！我会去查一查。你觉得学习经济学最难的部分是什么？
A: 学习经济学最具挑战性的部分可能是理解并应用各种经济模型。每个模型都是对现实世界复杂现象的简化,要准确地应用这些模型,就需要深入理解它们的假设和局限性。不过,经济学其实也很有趣,它可以帮助我们更好地理解世界如何运作,特别是在资源分配和决策方面。你在学习时遇到了哪些难题呢？

根据以上信息,生成一个有吸引力的角色扮演场景,最少3轮对话,最多6轮对话(一轮指的是一问一答)。全程使用中文。
"""


def process_url(source):
    for src in tqdm(source, desc="Processing", unit="item"):
        data = {
            "model": "gpt-4",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "[GPT-4 Vision](" + src + ")" + prompt
                }
            ]
        }
        max_retries = 5
        retry_count = 0
        while retry_count < max_retries:
            response = requests.post(url, json=data, headers=headers, timeout=30)
            print("HTTP 状态码:", response.status_code)
            if response.status_code == 200:
                print("Success!")
                text = response.content.decode('utf-8')
                matches = re.findall(r'"delta": {"content": "(.*?)"', text)
                if len(''.join(matches)) > 1000:
                    retry_count += 1
                    print(f"AL Retrying... Attempt {retry_count}")
                    continue
                elif len(''.join(matches)) < 10:
                    retry_count += 1
                    print(f"AT Retrying... Attempt {retry_count}")
                    continue
                with lock:
                    json_list.append(convert_json(''.join(matches), src))
                break
            else:
                print("Error:", response.status_code, response.text)
                retry_count += 1
                print(f"Retrying... Attempt {retry_count}")


lock = threading.Lock()
chunk_size = len(random_srcs) // 4
url_chunks = [random_srcs[i:i + chunk_size] for i in range(0, len(random_srcs), chunk_size)]

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_url, url_chunk) for url_chunk in url_chunks]

concurrent.futures.wait(futures)

print("All threads are done.")
with open(out_file, "w", encoding="utf-8") as output_file:
    json.dump(json_list, output_file, indent=2, ensure_ascii=False)
