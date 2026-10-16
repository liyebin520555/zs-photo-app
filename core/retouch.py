# -*- coding: utf-8 -*-
"""core/retouch.py — 基于 mediapipe 人脸关键点的局部精修（瘦脸/大眼/微笑/牙齿美白/换发色）"""

import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'models', 'face_landmarker.task')

_landmarker = None


def _get_landmarker():
    """FaceLandmarker 单例"""
    global _landmarker
    if _landmarker is None:
        options = vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            num_faces=1,
            running_mode=vision.RunningMode.IMAGE)
        _landmarker = vision.FaceLandmarker.create_from_options(options)
    return _landmarker


def get_landmarks(bgr):
    """返回 (N,2) 像素坐标 float32 数组；无人脸返回 None"""
    if bgr is None or bgr.ndim != 3:
        return None
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = _get_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None
    h, w = bgr.shape[:2]
    return np.array([(p.x * w, p.y * h) for p in result.face_landmarks[0]],
                    dtype=np.float32)


def _warp(bgr, center, radius, shift_fn):
    """
    局部形变。
    shift_fn(dx, dy, d) 返回**视觉位移向量** (vx, vy)：该处内容在画面上移动多少。
    内部按 map(p) = p - v * w 计算采样坐标（w 为高斯衰减），即内容向 v 方向平移。
    """
    h, w_img = bgr.shape[:2]
    cx, cy = float(center[0]), float(center[1])
    r = float(max(radius, 1.0))

    ys, xs = np.indices((h, w_img), dtype=np.float32)
    dx = xs - cx
    dy = ys - cy
    d = np.sqrt(dx * dx + dy * dy)

    falloff = np.exp(-(d / r) ** 2 * 2.5)
    vx, vy = shift_fn(dx, dy, d)
    vx = np.broadcast_to(np.asarray(vx, dtype=np.float32), d.shape)
    vy = np.broadcast_to(np.asarray(vy, dtype=np.float32), d.shape)

    map_x = xs - vx * falloff
    map_y = ys - vy * falloff
    return cv2.remap(bgr, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def level_head(bgr, max_angle=15.0):
    """自动摆正头部：以双眼连线水平为基准旋转矫正。

    :param bgr: BGR 图像
    :param max_angle: 最大矫正角度（度）
    :return: (矫正后图像, 实际矫正角度)
    """
    pts = get_landmarks(bgr)
    if pts is None or len(pts) < 363:
        return bgr.copy(), 0.0

    left = (pts[33] + pts[133]) / 2.0    # 左眼中心
    right = (pts[362] + pts[263]) / 2.0  # 右眼中心
    dx = float(right[0] - left[0])
    dy = float(right[1] - left[1])
    if abs(dx) < 1e-3:
        return bgr.copy(), 0.0

    angle = float(np.degrees(np.arctan2(dy, dx)))
    angle = float(np.clip(angle, -max_angle, max_angle))
    if abs(angle) < 0.3:
        return bgr.copy(), 0.0

    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    out = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    # 验证方向是否正确：矫正后残余角度应比原来小
    pts2 = get_landmarks(out)
    if pts2 is not None and len(pts2) >= 363:
        l2 = (pts2[33] + pts2[133]) / 2.0
        r2 = (pts2[362] + pts2[263]) / 2.0
        residual = abs(np.degrees(np.arctan2(r2[1] - l2[1], r2[0] - l2[0])))
        if residual > abs(angle):
            # 方向反了，反向再转
            M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -angle, 1.0)
            out = cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REFLECT)
            return out, -angle
    return out, angle


def slim_face(bgr, amount=0):
    """瘦脸：左右脸颊局部内容向脸中轴水平移动"""
    if amount <= 0:
        return bgr.copy()
    pts = get_landmarks(bgr)
    if pts is None or len(pts) < 455:
        return bgr.copy()

    left_cheek, right_cheek, nose = pts[234], pts[454], pts[1]
    face_w = float(np.linalg.norm(left_cheek - right_cheek))
    if face_w <= 1:
        return bgr.copy()

    radius = 0.40 * face_w
    shift = amount / 100.0 * 0.055 * face_w          # 最大视觉位移
    sign_l = 1.0 if nose[0] >= left_cheek[0] else -1.0
    sign_r = -1.0 if nose[0] >= left_cheek[0] else 1.0

    out = _warp(bgr, left_cheek, radius,
                lambda dx, dy, d: (np.full_like(d, sign_l * shift), np.zeros_like(d)))
    out = _warp(out, right_cheek, radius,
                lambda dx, dy, d: (np.full_like(d, sign_r * shift), np.zeros_like(d)))
    return out


def enlarge_eyes(bgr, amount=0):
    """大眼：以眼中心为圆心做径向放大"""
    if amount <= 0:
        return bgr.copy()
    pts = get_landmarks(bgr)
    if pts is None or len(pts) < 363:
        return bgr.copy()

    left_center = (pts[33] + pts[133]) / 2.0
    right_center = (pts[362] + pts[263]) / 2.0
    left_w = float(np.linalg.norm(pts[33] - pts[133]))
    right_w = float(np.linalg.norm(pts[362] - pts[263]))
    if left_w <= 1 or right_w <= 1:
        return bgr.copy()

    k = amount / 100.0 * 0.07                        # 径向放大系数

    out = _warp(bgr, left_center, 1.7 * left_w,
                lambda dx, dy, d: (dx * k, dy * k))
    out = _warp(out, right_center, 1.7 * right_w,
                lambda dx, dy, d: (dx * k, dy * k))
    return out


def smile(bgr, amount=0):
    """一键微笑：两侧嘴角向上、向外轻推"""
    if amount <= 0:
        return bgr.copy()
    pts = get_landmarks(bgr)
    if pts is None or len(pts) < 292:
        return bgr.copy()

    left_corner, right_corner = pts[61], pts[291]
    mouth_w = float(np.linalg.norm(left_corner - right_corner))
    if mouth_w <= 1:
        return bgr.copy()

    radius = 0.9 * mouth_w
    mag = amount / 100.0 * 0.085 * mouth_w
    inv = mag / np.sqrt(2.0)

    out = _warp(bgr, left_corner, radius,
                lambda dx, dy, d: (np.full_like(d, -inv), np.full_like(d, -inv)))
    out = _warp(out, right_corner, radius,
                lambda dx, dy, d: (np.full_like(d, inv), np.full_like(d, -inv)))
    return out


_INNER_LIP = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
              308, 324, 318, 402, 317, 14, 87, 178, 88, 95]


def whiten_teeth(bgr, amount=0):
    """牙齿美白：嘴内区域中「亮且低饱和」的像素提亮降饱和"""
    if amount <= 0:
        return bgr.copy()
    pts = get_landmarks(bgr)
    if pts is None or len(pts) <= max(_INNER_LIP):
        return bgr.copy()

    h, w = bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [pts[_INNER_LIP].astype(np.int32)], 255)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, (0, 0, 105), (180, 90, 255))
    teeth = cv2.bitwise_and(mask, bright)
    teeth = cv2.morphologyEx(teeth, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    teeth = cv2.GaussianBlur(teeth, (9, 9), 0)

    h_ch, s_ch, v_ch = cv2.split(hsv)
    v_new = cv2.add(v_ch, int(amount * 0.45))
    s_new = cv2.subtract(s_ch, int(amount * 0.45))
    adjusted = cv2.cvtColor(cv2.merge([h_ch, s_new, v_new]), cv2.COLOR_HSV2BGR)

    alpha = (teeth.astype(np.float32) / 255.0)[:, :, None]
    result = adjusted.astype(np.float32) * alpha + bgr.astype(np.float32) * (1 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


# 人脸外轮廓关键点（用于排除面部区域）
_FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397,
              365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58,
              132, 93, 234, 127, 162, 21, 54, 103, 67, 109]


def recolor_hair(bgr, target_rgb=(0, 0, 0), amount=0):
    """换发色：人像掩码 ∩ 脸部椭圆之外的区域近似头发，再调色"""
    if amount <= 0:
        return bgr.copy()
    pts = get_landmarks(bgr)
    if pts is None:
        return bgr.copy()

    # 1) rembg 人像 alpha（内部有单例缓存，同一张图第二次调用很快）
    from core.matting import remove_background
    rgba = remove_background(bgr)
    person = cv2.resize(rgba[:, :, 3], (bgr.shape[1], bgr.shape[0]),
                        interpolation=cv2.INTER_LINEAR)

    h, w = bgr.shape[:2]

    # 2) 脸部椭圆内部区域（大幅膨胀以越过发际线，宁可少染前沿头发也不染到皮肤）
    face_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(face_mask, [pts[_FACE_OVAL].astype(np.int32)], 255)
    face_mask = cv2.dilate(face_mask, np.ones((55, 55), np.uint8))

    # 3) 头部上方区域（脸部 bbox 向上扩展）
    x_min, y_min = pts.min(axis=0)
    x_max, y_max = pts.max(axis=0)
    fw, fh = x_max - x_min, y_max - y_min
    head_region = np.zeros((h, w), np.uint8)
    y0 = int(max(0, y_min - fh * 1.0))
    y1 = int(min(h, y_min + fh * 0.12))
    x0 = int(max(0, x_min - fw * 0.25))
    x1 = int(min(w, x_max + fw * 0.25))
    cv2.rectangle(head_region, (x0, y0), (x1, y1), 255, -1)

    # 4) 头发 = 人像 ∩ 头部区域 ∩ 非脸部 ∩ 非肤色
    hair = cv2.bitwise_and(person, head_region)
    hair = cv2.bitwise_and(hair, cv2.bitwise_not(face_mask))
    # 肤色检测兜底：无论几何边界怎么切，皮肤像素绝不染色
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCR_CB)
    skin = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))
    skin = cv2.dilate(skin, np.ones((15, 15), np.uint8))
    hair = cv2.bitwise_and(hair, cv2.bitwise_not(skin))
    hair = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    hair = cv2.medianBlur(hair, 9)
    hair = cv2.GaussianBlur(hair, (11, 11), 0)
    if hair.max() == 0:
        return bgr.copy()

    # 5) HSV 色调替换，保留明暗结构
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    target_bgr = np.uint8([[target_rgb[::-1]]])
    target_h = int(cv2.cvtColor(target_bgr, cv2.COLOR_BGR2HSV)[0][0][0])

    h_ch, s_ch, v_ch = cv2.split(hsv)
    new_h = np.full_like(h_ch, target_h)
    new_s = np.clip(s_ch.astype(np.float32) * 1.15 + 40, 0, 255).astype(np.uint8)
    new_v = cv2.add(v_ch, 30)  # 轻微提亮，让深色头发也能显色
    tinted = cv2.cvtColor(cv2.merge([new_h, new_s, new_v]), cv2.COLOR_HSV2BGR)

    alpha = (hair.astype(np.float32) / 255.0 * (amount / 100.0))[:, :, None]
    result = tinted.astype(np.float32) * alpha + bgr.astype(np.float32) * (1 - alpha)
    return np.clip(result, 0, 255).astype(np.uint8)


def retouch_pipeline(bgr, slim=0, eyes=0, smile_amt=0, teeth=0):
    """执行瘦脸 → 大眼 → 微笑 → 牙齿美白"""
    result = bgr.copy()
    if slim > 0:
        result = slim_face(result, slim)
    if eyes > 0:
        result = enlarge_eyes(result, eyes)
    if smile_amt > 0:
        result = smile(result, smile_amt)
    if teeth > 0:
        result = whiten_teeth(result, teeth)
    return result
