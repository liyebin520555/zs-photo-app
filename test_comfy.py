# -*- coding: utf-8 -*-
"""comfy_client 端到端联调测试"""
import os, sys, time, traceback
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
from core import comfy_client as cc

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_comfy_out")
os.makedirs(OUT, exist_ok=True)

print("== 1. 探测服务 ==")
print("server:", cc.find_server())
print("resolve_models:", cc.resolve_models(force=True))
print("report:", cc.server_models_report())
print()

print("== 2. 准备输入图 ==")
src = r"F:/bfcomfyui/ComfyUI/input/0233.jpg"
img = cv2.imread(src)
if img is None:
    print("!! 读不到", src); sys.exit(1)
print("input:", img.shape)
cv2.imwrite(os.path.join(OUT, "00_input.png"), img)

print()
print("== 3. 真实生成测试（换正装）==")
t0 = time.time()
try:
    out = cc.edit_image(
        img,
        "把人物身上的衣服换成合体的深色商务西装搭配白衬衫，肩线与身体自然贴合，保持人脸、发型和背景完全不变",
        progress=lambda m: print("   ", m, flush=True),
        timeout=900,
    )
    dt = time.time() - t0
    print(f"OK  {out.shape}  用时 {dt:.1f}s")
    cv2.imwrite(os.path.join(OUT, "01_suit.jpg"), out)
except Exception as e:
    print("FAIL:", type(e).__name__, e)
    traceback.print_exc()

print("\n输出目录:", OUT)
