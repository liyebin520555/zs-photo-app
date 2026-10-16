# -*- coding: utf-8 -*-
"""调用本地 Ollama 模型生成代码文件。用法: python gen.py <prompt文件> <输出py文件> [模型名]"""
import sys, json, re, urllib.request

prompt = open(sys.argv[1], encoding='utf-8').read()
out = sys.argv[2]
model = sys.argv[3] if len(sys.argv) > 3 else 'qwen3.8-27b-abliterated:latest'

req = urllib.request.Request(
    'http://127.0.0.1:11434/api/chat',
    data=json.dumps({
        'model': model,
        'messages': [{'role': 'user', 'content': prompt}],
        'stream': False,
        'think': False,
        'options': {'temperature': 0.2, 'num_ctx': 8192, 'top_p': 0.8}
    }).encode())
resp = json.loads(urllib.request.urlopen(req, timeout=7200).read())
content = resp['message']['content']
m = re.search(r'```(?:python)?\s*\n(.*)```', content, re.S)
code = m.group(1) if m else content
open(out, 'w', encoding='utf-8').write(code.strip() + '\n')
print('WRITTEN', out, len(code), 'chars')
