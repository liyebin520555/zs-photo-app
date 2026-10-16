# -*- coding: utf-8 -*-
"""
core/beautify.py - 人像美颜处理模块
所有函数输入输出均为 BGR numpy 图像（uint8）
"""

import cv2
import numpy as np


def smooth_skin(bgr: np.ndarray, strength: int = 40) -> np.ndarray:
    """
    磨皮：在 YCrCb 色彩空间提取肤色掩码，对肤色区域进行双边滤波磨皮。

    Args:
        bgr: 输入 BGR 图像 (H, W, 3), uint8
        strength: 磨皮强度 0-100

    Returns:
        处理后的 BGR 图像
    """
    if strength <= 0:
        return bgr.copy()

    # 转换为 YCrCb 色彩空间
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCR_CB)

    # 提取肤色掩码：Y 在 [135, 255], Cr 在 [85, 180], Cb 在 [0, 135]
    mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))

    # 对原图进行双边滤波磨皮
    blurred = cv2.bilateralFilter(bgr, d=9, sigmaColor=75, sigmaSpace=75)

    # 掩码羽化，避免肤色边界生硬
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    alpha = (mask.astype(np.float32) / 255.0) * (strength / 100.0)
    alpha = alpha[:, :, None]

    # 只在肤色区域按强度混合磨皮结果
    result = (blurred.astype(np.float32) * alpha
              + bgr.astype(np.float32) * (1 - alpha))
    return np.clip(result, 0, 255).astype(np.uint8)


def whiten(bgr: np.ndarray, amount: int = 20) -> np.ndarray:
    """
    美白：在肤色掩码区域提亮亮度。

    Args:
        bgr: 输入 BGR 图像 (H, W, 3), uint8
        amount: 美白强度 0-100

    Returns:
        处理后的 BGR 图像
    """
    if amount <= 0:
        return bgr.copy()

    # 转换为 YCrCb 色彩空间
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCR_CB)

    # 提取肤色掩码
    mask = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))

    # 只对亮度通道 Y 提亮，避免色度偏移
    y_ch, cr_ch, cb_ch = cv2.split(ycrcb)
    y_ch = cv2.convertScaleAbs(y_ch, alpha=1.0, beta=amount * 0.6)
    whitened_ycrcb = cv2.merge([y_ch, cr_ch, cb_ch])
    whitened_bgr = cv2.cvtColor(whitened_ycrcb, cv2.COLOR_YCR_CB2BGR)

    # 掩码羽化后按区域混合
    mask = cv2.GaussianBlur(mask, (15, 15), 0)
    alpha = (mask.astype(np.float32) / 255.0) * (amount / 100.0)
    alpha = alpha[:, :, None]
    result = (whitened_bgr.astype(np.float32) * alpha
              + bgr.astype(np.float32) * (1 - alpha))
    return np.clip(result, 0, 255).astype(np.uint8)


def sharpen(bgr: np.ndarray, amount: int = 30) -> np.ndarray:
    """
    USM 锐化：高斯模糊后按 amount/100 比例做 addWeighted。

    Args:
        bgr: 输入 BGR 图像 (H, W, 3), uint8
        amount: 锐化强度 0-100

    Returns:
        处理后的 BGR 图像
    """
    if amount <= 0:
        return bgr.copy()

    # 高斯模糊（5x5）
    blurred = cv2.GaussianBlur(bgr, (5, 5), 0)

    # USM 锐化：result = 原图 * (1 + a) - 模糊图 * a
    alpha = amount / 100.0
    result = cv2.addWeighted(bgr, 1 + alpha, blurred, -alpha, 0)

    return result


def beautify_pipeline(bgr: np.ndarray, smooth: int = 40, white: int = 20, sharp: int = 30) -> np.ndarray:
    """
    美颜流水线：依次执行磨皮、美白、锐化。

    Args:
        bgr: 输入 BGR 图像 (H, W, 3), uint8
        smooth: 磨皮强度 0-100
        white: 美白强度 0-100
        sharp: 锐化强度 0-100

    Returns:
        处理后的 BGR 图像
    """
    result = bgr.copy()
    result = smooth_skin(result, strength=smooth)
    result = whiten(result, amount=white)
    result = sharpen(result, amount=sharp)
    return result
