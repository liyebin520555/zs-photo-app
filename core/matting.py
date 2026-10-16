# -*- coding: utf-8 -*-
"""
core/matting.py
人像抠图与换背景模块
"""

import numpy as np
import cv2
from PIL import Image
from rembg import remove, new_session

# 全局单例 session，避免重复加载模型
_rembg_session = None


def load_rembg_session():
    """
    单例方式创建 rembg 的 session（u2net 模型），避免每次调用重新加载模型。
    
    Returns:
        session: rembg session 对象
    """
    global _rembg_session
    if _rembg_session is None:
        _rembg_session = new_session('u2net')
    return _rembg_session


def remove_background(bgr):
    """
    移除背景，返回 RGBA numpy 图像。
    
    Args:
        bgr: BGR 格式的 numpy 图像 (H, W, 3)
        
    Returns:
        rgba: RGBA 格式的 numpy 图像 (H, W, 4)，尺寸与缩放后的输出一致
    """
    # 获取原始尺寸
    h, w = bgr.shape[:2]
    
    # 缩放到最长边 ≤1600 提速
    max_side = max(h, w)
    if max_side > 1600:
        scale = 1600.0 / max_side
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized_bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        resized_bgr = bgr.copy()
    
    # 转换为 PIL Image (RGB)
    bgr_pil = Image.fromarray(cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB))
    
    # 使用 rembg 抠图
    session = load_rembg_session()
    rgba_pil = remove(bgr_pil, session=session)

    # 转换为 numpy RGBA（rembg 返回 RGB 顺序，转回 BGR 保持全库一致）
    rgba = np.array(rgba_pil)[:, :, [2, 1, 0, 3]]

    # 关键：rembg 内部会缩放图像，必须还原到输入尺寸，
    # 否则调用方的人脸框等坐标全部对不上（表现为证件照人脸溢出画面）
    if rgba.shape[0] != h or rgba.shape[1] != w:
        rgba = cv2.resize(rgba, (w, h), interpolation=cv2.INTER_AREA)

    return rgba


def change_background(rgba, bg_bgr):
    """
    将 RGBA 人像合成到纯色背景上，返回 BGR 图像。
    
    Args:
        rgba: RGBA 格式的 numpy 图像 (H, W, 4)
        bg_bgr: 背景颜色，BGR 格式的 numpy 数组 (3,) 或标量
        
    Returns:
        result: BGR 格式的 numpy 图像 (H, W, 3)
    """
    # 分离通道
    b, g, r, a = cv2.split(rgba)
    
    # 对 alpha 通道做轻度腐蚀 (3x3)
    kernel_erode = np.ones((3, 3), np.uint8)
    a_eroded = cv2.erode(a, kernel_erode, iterations=1)
    
    # 高斯模糊 (3x3) 实现边缘羽化
    a_blurred = cv2.GaussianBlur(a_eroded, (3, 3), 0)
    
    # 归一化 alpha 到 [0, 1]
    alpha = a_blurred.astype(np.float32) / 255.0
    
    # 构建背景图像
    if isinstance(bg_bgr, np.ndarray):
        if bg_bgr.ndim == 1 and len(bg_bgr) == 3:
            # 纯色背景
            h, w = rgba.shape[:2]
            bg = np.zeros((h, w, 3), dtype=np.uint8)
            bg[:, :, 0] = bg_bgr[0]
            bg[:, :, 1] = bg_bgr[1]
            bg[:, :, 2] = bg_bgr[2]
        else:
            # 背景图像
            bg = bg_bgr.copy()
    else:
        h, w = rgba.shape[:2]
        bg = np.zeros((h, w, 3), dtype=np.uint8)
        bg[:, :, 0] = bg_bgr[0] if len(bg_bgr) > 0 else 0
        bg[:, :, 1] = bg_bgr[1] if len(bg_bgr) > 1 else 0
        bg[:, :, 2] = bg_bgr[2] if len(bg_bgr) > 2 else 0
    
    # Alpha 混合
    # result = alpha * fg + (1 - alpha) * bg
    alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
    
    fg = rgba[:, :, :3].astype(np.float32)
    bg_float = bg.astype(np.float32)
    
    result = alpha_3ch * fg + (1 - alpha_3ch) * bg_float
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result
