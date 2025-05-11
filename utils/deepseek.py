import json
from openai import OpenAI


def translate(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    client = OpenAI(api_key="sk-4e676ab53c044699bd4d14bd22cabf8f", base_url="https://api.deepseek.com")

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你是学术图像造假检测结果翻译助手，请将用户发送的jsonl文件内容中\"outputs\"替换为翻译后的中文，给出修改后的jsonl文件内容，不要返回其他多余的内容。"},
            {"role": "user", "content": content},
        ],
        stream=False
    )

    # print(response.choices[0].message.content)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(response.choices[0].message.content)

