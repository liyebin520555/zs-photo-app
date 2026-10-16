# -*- coding: utf-8 -*-
import cv2
import numpy as np
from collections import OrderedDict

# 全局缓存的级联分类器
_face_cascade = None

def _get_face_cascade():
    """获取或创建人脸检测级联分类器（单例缓存）"""
    global _face_cascade
    if _face_cascade is None:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        _face_cascade = cv2.CascadeClassifier(cascade_path)
    return _face_cascade

# 证件照规格：中文名 -> (宽mm, 高mm)，数据见 core/spec_data.py
from core.spec_data import SPECS

def mm_to_px(mm, dpi=300):
    """将毫米转换为像素（取整）"""
    return int(round(mm * dpi / 25.4))

def detect_face(bgr):
    """
    检测图像中最大的人脸。
    :param bgr: BGR格式的numpy图像
    :return: (x, y, w, h) 元组，若未检测到则返回 None
    """
    if bgr is None or len(bgr.shape) < 2:
        return None
    
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    cascade = _get_face_cascade()
    
    # 检测人脸，scaleFactor=1.1, minNeighbors=5
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    if len(faces) == 0:
        return None
    
    # 找到面积最大的人脸
    max_area = 0
    best_face = None
    for (x, y, w, h) in faces:
        area = w * h
        if area > max_area:
            max_area = area
            best_face = (x, y, w, h)
            
    return best_face

def compose_id_photo(rgba_person, spec_name, face_box, bg_bgr):
    """
    按内置规格合成证件照。
    """
    if spec_name not in SPECS:
        raise ValueError(f"未知规格: {spec_name}")

    width_mm, height_mm = SPECS[spec_name]
    return compose_id_photo_custom(rgba_person, width_mm, height_mm, face_box, bg_bgr)


def compose_id_photo_custom(rgba_person, w_mm, h_mm, face_box, bg_bgr, dpi=300):
    """
    按自定义毫米尺寸合成证件照。
    :param w_mm, h_mm: 照片宽高（毫米）
    :param dpi: 输出分辨率
    :return: BGR 格式的证件照图像
    """
    target_w = mm_to_px(w_mm, dpi)
    target_h = mm_to_px(h_mm, dpi)
    
    # 创建纯色背景
    bg = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    bg[:] = bg_bgr
    
    if face_box is None:
        return bg
    
    fx, fy, fw, fh = face_box
    person_h, person_w = rgba_person.shape[:2]
    
    # 计算缩放比例：使人脸高度映射到照片高度的 0.42 倍
    target_face_h = target_h * 0.42
    scale = target_face_h / fh
    
    # 计算缩放后的人像尺寸
    new_w = int(person_w * scale)
    new_h = int(person_h * scale)
    
    if new_w <= 0 or new_h <= 0:
        return bg
    
    # 缩放人像
    resized_person = cv2.resize(rgba_person, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # 计算人脸中心在缩放后图像中的位置
    face_center_x_in_scaled = fx * scale + fw * scale / 2
    face_top_y_in_scaled = fy * scale
    
    # 目标位置：
    # 1. 人脸中心 x 对齐照片水平中心
    # 2. face_box 顶部位于照片高度约 0.25 处
    target_face_center_x = target_w / 2
    target_face_top_y = target_h * 0.25
    
    # 计算偏移量 (dx, dy)
    # 缩放后人脸中心 x 需要移动到 target_face_center_x
    dx = int(target_face_center_x - face_center_x_in_scaled)
    # 缩放后人脸顶部 y 需要移动到 target_face_top_y
    dy = int(target_face_top_y - face_top_y_in_scaled)
    
    # 计算人像在画布上的起始坐标（人像需要平移 (dx, dy) 才能对准目标位置）
    start_x = dx
    start_y = dy
    
    # 定义混合区域
    # 人像在画布上的范围: [start_x, start_x + new_w], [start_y, start_y + new_h]
    # 画布范围: [0, target_w], [0, target_h]
    
    # 计算交集
    src_x1 = max(0, -start_x)
    src_y1 = max(0, -start_y)
    src_x2 = min(new_w, target_w - start_x)
    src_y2 = min(new_h, target_h - start_y)
    
    dst_x1 = max(0, start_x)
    dst_y1 = max(0, start_y)
    dst_x2 = min(target_w, start_x + new_w)
    dst_y2 = min(target_h, start_y + new_h)
    
    # 检查是否有有效区域
    if src_x1 < src_x2 and src_y1 < src_y2:
        # 提取人像对应区域
        roi_person = resized_person[src_y1:src_y2, src_x1:src_x2]
        
        # 分离通道
        bgr_part = roi_person[:, :, :3].astype(np.float32)
        alpha_part = roi_person[:, :, 3].astype(np.float32) / 255.0
        
        # 提取背景对应区域
        roi_bg = bg[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
        
        # Alpha 混合
        # result = alpha * person + (1 - alpha) * bg
        blended = np.zeros_like(roi_bg)
        for c in range(3):
            blended[:, :, c] = alpha_part * bgr_part[:, :, c] + (1 - alpha_part) * roi_bg[:, :, c]
        
        # 更新背景
        bg[dst_y1:dst_y2, dst_x1:dst_x2] = blended.astype(np.uint8)
    
    return bg

def make_layout_sheet(id_photo_bgr, spec_name):
    """
    生成六寸相纸排版图。
    :param id_photo_bgr: 单张证件照 BGR 图像
    :param spec_name: 规格名称
    :return: 排版后的 BGR 图像
    """
    if spec_name not in SPECS:
        raise ValueError(f"未知规格: {spec_name}")
    
    width_mm, height_mm = SPECS[spec_name]
    photo_w = mm_to_px(width_mm)
    photo_h = mm_to_px(height_mm)
    
    # 六寸相纸尺寸 (300dpi)
    sheet_w = 1200
    sheet_h = 1800
    
    # 边距和间距
    margin = 15
    gap = mm_to_px(2)  # 2mm 间距
    
    # 计算每行每列能放几张
    # 可用宽度 = sheet_w - 2 * margin
    # 每张占用宽度 = photo_w + gap (最后一张不需要gap，但为了简化计算，通常假设均匀分布或按最大数量计算)
    # 更精确的计算：
    # n_cols * photo_w + (n_cols - 1) * gap <= sheet_w - 2 * margin
    # n_cols <= (sheet_w - 2 * margin + gap) / (photo_w + gap)
    
    avail_w = sheet_w - 2 * margin
    avail_h = sheet_h - 2 * margin
    
    if photo_w > avail_w or photo_h > avail_h:
        # 如果单张都放不下，返回空白或报错，这里简单处理
        return np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)
        
    n_cols = int((avail_w + gap) / (photo_w + gap))
    n_rows = int((avail_h + gap) / (photo_h + gap))
    
    # 确保至少能放一张
    n_cols = max(1, n_cols)
    n_rows = max(1, n_rows)
    
    # 计算实际使用的总宽度和高度，以便居中
    total_w = n_cols * photo_w + (n_cols - 1) * gap
    total_h = n_rows * photo_h + (n_rows - 1) * gap
    
    start_x = (sheet_w - total_w) // 2
    start_y = (sheet_h - total_h) // 2
    
    # 创建白色背景画布
    sheet = np.full((sheet_h, sheet_w, 3), 255, dtype=np.uint8)
    
    # 放置照片并画线
    for row in range(n_rows):
        for col in range(n_cols):
            x = start_x + col * (photo_w + gap)
            y = start_y + row * (photo_h + gap)
            
            # 粘贴照片
            sheet[y:y+photo_h, x:x+photo_w] = id_photo_bgr
            
            # 画裁切参考线 (1px 淡灰色)
            # 上边
            if y > 0:
                sheet[y-1, x:x+photo_w] = [200, 200, 200]
            # 下边
            if y + photo_h < sheet_h:
                sheet[y+photo_h, x:x+photo_w] = [200, 200, 200]
            # 左边
            if x > 0:
                sheet[y:y+photo_h, x-1] = [200, 200, 200]
            # 右边
            if x + photo_w < sheet_w:
                sheet[y:y+photo_h, x+photo_w] = [200, 200, 200]
                
    return sheet
