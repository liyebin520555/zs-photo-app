# -*- coding: utf-8 -*-
"""
文档扫描校正模块
提供文档轮廓检测、透视变换矫正和图像增强功能。
"""

import cv2
import numpy as np


def find_document_contour(bgr: np.ndarray) -> np.ndarray | None:
    """
    查找文档轮廓并返回四顶点（左上、右上、右下、左下排序）。
    
    处理流程：
        灰度化 → 高斯模糊 → Canny 边缘检测 → 膨胀 → findContours → 
        筛选最大轮廓 → approxPolyDP 逼近四边形 → 面积校验 → 凸性校验 → 顶点排序
    
    Args:
        bgr: BGR 格式的图像 numpy 数组。
    
    Returns:
        按左上、右上、右下、左下顺序排列的 4 个顶点的 numpy 数组；
        若未找到符合条件的轮廓，返回 None。
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)
    kernel = np.ones((3, 3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=3)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    max_contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(max_contour)
    img_area = bgr.shape[0] * bgr.shape[1]

    if area < 0.15 * img_area:
        return None

    approx = cv2.approxPolyDP(max_contour, 0.02 * cv2.arcLength(max_contour, True), True)

    if len(approx) != 4:
        return None

    if not cv2.isContourConvex(approx):
        return None

    pts = approx.reshape(4, 2).astype(np.float32)

    # 标准顶点排序：左上角 x+y 最小，右下角 x+y 最大，
    # 右上角 x-y 最大，左下角 x-y 最小
    s = pts.sum(axis=1)
    d = (pts[:, 0] - pts[:, 1])
    tl_pt = pts[np.argmin(s)]
    br_pt = pts[np.argmax(s)]
    tr_pt = pts[np.argmax(d)]
    bl_pt = pts[np.argmin(d)]

    return np.array([tl_pt, tr_pt, br_pt, bl_pt], dtype=np.float32)


def four_point_transform(bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    按给定四点做透视变换矫正，目标宽高取四点的最大边长。
    
    Args:
        bgr: BGR 格式的图像 numpy 数组。
        pts: 按左上、右上、右下、左下顺序排列的 4 个顶点的 numpy 数组。
    
    Returns:
        矫正后的图像 numpy 数组。
    """
    pts = np.float32(pts)
    tl, tr, br, bl = pts

    # 目标宽高取对应边的最大长度
    width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(bgr, matrix, (width, height),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT,
                               borderValue=(0, 0, 0))


def enhance_document(bgr: np.ndarray, mode: str) -> np.ndarray:
    """
    对图像进行增强处理。
    
    Args:
        bgr: BGR 格式的图像 numpy 数组。
        mode: 增强模式，可选值为 'original'、'enhanced'、'gray'、'bw'。
    
    Returns:
        处理后的图像 numpy 数组。
    """
    if mode == 'original':
        return bgr

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    if mode == 'enhanced':
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        return cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)

    if mode == 'gray':
        return gray

    if mode == 'bw':
        # 自适应阈值二值化，适合文档文字提取
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, blockSize=25, C=15)
        kernel = np.ones((3, 3), np.uint8)
        denoised = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        return denoised

    raise ValueError(f"Unknown mode: {mode}")
