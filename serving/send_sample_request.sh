#! /usr/bin/bash

set -ex

CURDIR=$(cd "$(dirname "$0")"; pwd)
cd $CURDIR

one_run=$(cat << 'EOF'
{
        "model": "llava_image_chat",
        "image_path" : "/ML-A100/home/peter/multimodal_arch/serving/cat.jpg",
        "messages":
        [
          {
            "role" : "user",
            "content" : "<image_placeholder>"
          }
        ],
        "max_tokens" : 256
}
EOF
)

remote_image_run=$(cat << 'EOF'
{
        "model": "llava_image_chat",
        "image_path" : "/ML-A100/home/peter/multimodal_arch/serving/cat.jpg",
        "messages":
        [
          {
            "role" : "user",
            "content" : "<image_placeholder>"
          }
        ],
        "max_tokens" : 256
}
EOF
)

multi_run=$(cat << 'EOF'
{
        "model": "llava_image_chat",
        "image_path" : "/ML-A100/home/peter/multimodal_arch/serving/cat.jpg",
        "messages":
        [
          {
            "role" : "user",
            "content" : "<image_placeholder>"
          },
          {
            "role" : "assistant",
            "content" : "噢，这只很可爱，你喜欢猫吗？我真的很喜欢，觉得猫好像很有智慧，也很有个性。"
          },
          {
            "role" : "user",
            "content" : "我超喜欢猫。帮我看看这只猫的品种是啥？"
          }
        ],
        "max_tokens" : 512
}
EOF
)

echo "-----remote_image_run-------"
curl http://localhost:8080/v1/chat/image_completions \
    -X POST \
    -H 'Content-Type: application/json' \
    -d "$remote_image_run"

echo "-----multi_run-------"
curl http://localhost:8080/v1/chat/image_completions \
    -X POST \
    -H 'Content-Type: application/json' \
    -d "$multi_run"
