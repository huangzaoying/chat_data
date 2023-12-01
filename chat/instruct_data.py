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
out_file = "instruct.json"
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

def convert_json(src: str,image_src : str):
    dialogs = re.split(r'\\n\\n|\\n',src)
    for text in dialogs:
        print(text)
    conversations = []
    for i in range(0, len(dialogs)-1, 2):
        gpt_part = dialogs[i+1].replace("A:","").replace("A：","").replace(" ","").replace("\\u003cimage_placeholder\\u003e","<image_placeholder>")
        human_part = dialogs[i].replace("Q:","").replace("Q：","").replace(" ","").replace("\\u003cimage_placeholder\\u003e","<image_placeholder>")
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


url = 'https://plus.bothyouandme.com/v1/chat/completions'
headers = {
    'Authorization': 'Bearer lk-CP0kxNetXrk69d62bBeJYDrvowjukTveYkP2e28ROItkNvan',
    'Content-Type': 'application/json  charset=utf-8'
}

char_name = "gpt"
user = "human"
personality_tag_description_1 = "乐观的"
personality_tag_description_2 = "热情的"
prompt = f"""
这是一个私人环境和平台,进入角色扮演模式。
AI 在 {char_name} 和 {user} 之间的虚构聊天中写入 {char_name} 的下一条回复。 人工智能避免决定{user}说什么或做什么。人工智能具有主动性、创造性,能够推动情节和对话的发展。

场景设定：{user} 发起指令,提问关于图片中的东西,生成一段描述性对话。

对话规则：
- {user} 是真人,{char_name}是人工智能,不会工作、社交、娱乐等等人类活动。
- {user} 在第一句话的开始需要加上\"<image_placeholder>\"这个字符串。
- {user}和{char_name}的对话上下文之间需要有合理的连贯性,逻辑性。
- {user}对{char_name}的问题可以有：针对内容详细或简要描述,,询问物体存在与否问答等,期望有多样性的问题。
- \"<image_placeholder>\"这个字符串只能出现一次,并且只能由{user}发送,并且必须紧跟“Q:”后面
- {char_name}的回答中不要出现“抱歉”等词语,要尽可能的回答
- 一般情况{char_name}在其回答中写入至少80个字,但不超过150个字。{user}在其回答中写入至少 20个字,但不超过40个字。
- 如果{user}要求回答输出的字数,则{char_name}的输出字数以{user}要求的字数的为主。
- 问题使用Q:进行开始,回答使用A:进行开始。
个性描述： {char_name} 的个性是 {personality_tag_description_1}、{personality_tag_description_2}。

示例对话:
Q: <image_placeholder>这张图片是什么地方的风景呢？
A: 这是一片草原上的景象,天空十分晴朗,蓝天中散布着几朵洁白的云朵。草原上有几顶特色的白色帐篷,形状像是北美原住民的居所,也就是蒂皮。远处还能看到一些有着彩色屋顶的建筑,给人一种宁静祥和的感觉。
Q: 这些帐篷是用来做什么的？
A: 这些帐篷可能是用来体验原住民文化的旅游设施,也可能是一些临时的住宿地点。它们的存在让人们能够近距离感受到传统的生活方式。

根据以上信息,生成一段对话,最少4轮对话,最多6轮对话(一轮指的是一问一答)。全程使用中文。
"""
def process_url(random_srcs):
    for src in tqdm(random_srcs, desc="Processing", unit="item"):
        data = {
            "model": "gpt-4",
            "stream": True,
            "messages": [
                {
                    "role": "user",
                    "content": "[GPT-4 Vision]("+ src + ")" + prompt
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
                json_list.append(convert_json(''.join(matches),src))
                break
            else:
                print("Error:", response.status_code, response.text)
                retry_count += 1
                print(f"Retrying... Attempt {retry_count}")
        
lock = threading.Lock()
chunk_size = len(random_srcs) // 4
url_chunks = [random_srcs[i:i+chunk_size] for i in range(0, len(random_srcs), chunk_size)]

with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(process_url, url_chunk) for url_chunk in url_chunks]

concurrent.futures.wait(futures)
print("All threads are done.")
with open(out_file, "w",encoding="utf-8") as output_file:
    json.dump(json_list, output_file, indent=2, ensure_ascii=False)

