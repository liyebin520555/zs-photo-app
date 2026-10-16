# -*- coding: utf-8 -*-
"""core/garment.py — 程序化服装模板库（无需任何生成模型，纯 OpenCV 绘制）

按「男士正装 / 女士正装 / 中老年服饰 / 青少年休闲」分类。
模板在人像 RGBA 上按肩部关键点定位贴合。
"""

import numpy as np
import cv2

# 模板: 名称 -> (分类, 款式, 主色BGR, 次色BGR(衬衫/内搭), 领带BGR 或 None)
# 款式: suit_tie=西装配领带, suit=西装, shirt=衬衫, polo=POLO, knit=开衫/毛衣, tunic=中山装
GARMENTS = {
    # ---- 男士正装 ----
    "黑西装·红领带": ("男士正装", "suit_tie", (20, 18, 22), (245, 240, 240), (52, 34, 178)),
    "黑西装·深蓝领带": ("男士正装", "suit_tie", (20, 18, 22), (245, 240, 240), (120, 60, 20)),
    "黑西装·酒红领带": ("男士正装", "suit_tie", (20, 18, 22), (245, 240, 240), (64, 0, 128)),
    "藏蓝西装·红领带": ("男士正装", "suit_tie", (95, 52, 28), (245, 240, 240), (52, 34, 178)),
    "藏蓝西装·深蓝领带": ("男士正装", "suit_tie", (95, 52, 28), (245, 240, 240), (120, 60, 20)),
    "深灰西装·红领带": ("男士正装", "suit_tie", (75, 70, 70), (245, 240, 240), (52, 34, 178)),
    "深灰西装·深蓝领带": ("男士正装", "suit_tie", (75, 70, 70), (245, 240, 240), (120, 60, 20)),
    "黑西装": ("男士正装", "suit", (20, 18, 22), (245, 240, 240), None),
    "藏蓝西装": ("男士正装", "suit", (95, 52, 28), (245, 240, 240), None),
    "深灰西装": ("男士正装", "suit", (75, 70, 70), (245, 240, 240), None),
    "白衬衫(男)": ("男士正装", "shirt", (245, 240, 240), (235, 235, 235), None),
    "浅蓝衬衫(男)": ("男士正装", "shirt", (240, 225, 210), (235, 225, 210), None),
    # ---- 女士正装 ----
    "黑西装外套·白衬衫": ("女士正装", "suit", (25, 22, 28), (248, 244, 244), None),
    "藏蓝西装外套·白衬衫": ("女士正装", "suit", (100, 58, 32), (248, 244, 244), None),
    "米白西装外套·白衬衫": ("女士正装", "suit", (215, 222, 232), (250, 248, 246), None),
    "浅灰西装外套·白衬衫": ("女士正装", "suit", (160, 158, 158), (248, 244, 244), None),
    "黑西装外套·浅蓝衬衫": ("女士正装", "suit", (25, 22, 28), (240, 225, 210), None),
    "藏蓝西装外套·浅蓝衬衫": ("女士正装", "suit", (100, 58, 32), (240, 225, 210), None),
    "白衬衫(女)": ("女士正装", "shirt", (248, 244, 244), (238, 238, 238), None),
    "浅蓝衬衫(女)": ("女士正装", "shirt", (240, 225, 210), (235, 225, 210), None),
    # ---- 中老年服饰 ----
    "中山装·深灰": ("中老年服饰", "tunic", (70, 66, 66), (60, 56, 56), None),
    "中山装·藏蓝": ("中老年服饰", "tunic", (90, 50, 26), (80, 44, 24), None),
    "深灰开衫": ("中老年服饰", "knit", (85, 80, 80), (110, 104, 104), None),
    "藏蓝开衫": ("中老年服饰", "knit", (105, 58, 30), (130, 76, 40), None),
    "驼色开衫": ("中老年服饰", "knit", (95, 140, 185), (115, 160, 200), None),
    "酒红毛衣": ("中老年服饰", "knit", (60, 30, 110), (80, 45, 125), None),
    "深棕毛衣": ("中老年服饰", "knit", (40, 70, 100), (55, 88, 118), None),
    "墨绿毛衣": ("中老年服饰", "knit", (55, 90, 55), (70, 108, 70), None),
    # ---- 青少年休闲 ----
    "白POLO": ("青少年休闲", "polo", (245, 240, 240), (225, 220, 220), None),
    "天蓝POLO": ("青少年休闲", "polo", (235, 205, 135), (222, 190, 120), None),
    "藏青POLO": ("青少年休闲", "polo", (110, 60, 35), (95, 52, 32), None),
    "白T恤": ("青少年休闲", "shirt", (246, 242, 242), (236, 232, 232), None),
    "浅灰T恤": ("青少年休闲", "shirt", (200, 198, 198), (188, 186, 186), None),
    "浅粉T恤": ("青少年休闲", "shirt", (200, 205, 240), (188, 193, 228), None),
}


def get_categories():
    """返回分类列表（按固定顺序）"""
    order = ["男士正装", "女士正装", "中老年服饰", "青少年休闲"]
    return [c for c in order if any(g[0] == c for g in GARMENTS.values())]


def get_names(category):
    """返回某分类下的模板名列表"""
    return [name for name, g in GARMENTS.items() if g[0] == category]


def _shade(color, factor):
    """按系数调暗颜色"""
    return tuple(int(c * factor) for c in color)


def render_garment(name, shoulder_w_px, gender="male", notch_depth=None):
    """绘制服装模板，返回 RGBA 图像。

    :param shoulder_w_px: 肩宽像素（决定整体尺寸）
    :param gender: male/female，影响肩形与领口
    :param notch_depth: 领口下凹深度（像素）；None 则按肩宽比例。
        耸肩照片肩膀接近下巴时需加大，避免服装盖住脖子。
    :return: (H, W, 4) uint8
    """
    if name not in GARMENTS:
        raise ValueError(f"未知服装模板: {name}")
    _, style, c1, c2, c3 = GARMENTS[name]

    sw = float(shoulder_w_px)
    w = int(sw * 1.7)
    h = int(sw * 1.6)
    cx = w / 2.0
    shoulder_y = h * 0.12          # 肩线位置（画布内）
    sh_h = h - shoulder_y          # 躯干高度

    # 领口：宽度与深度联动，深 V 时加宽
    notch_d = float(notch_depth) if notch_depth is not None else sw * 0.06
    notch_d = float(np.clip(notch_d, sw * 0.04, sh_h * 0.30))
    notch_w = max(sw * 0.09, notch_d * 0.75)

    canvas = np.zeros((h, w, 4), np.uint8)
    solid = np.zeros((h, w, 3), np.uint8)
    solid[:] = c1

    # ---- 底层躯干（西装款式用衬衫色打底，其余用主色）----
    base_color = c2 if style in ("suit_tie", "suit") else c1
    solid[:] = base_color
    taper = 0.62 if gender == "female" else 0.56   # 下摆收窄比例
    pts = np.array([
        [cx - sw * 0.50, shoulder_y + sh_h * 0.10],
        [cx - sw * 0.46, shoulder_y],
        [cx - sw * 0.28, shoulder_y - sh_h * 0.02],
        [cx - notch_w, shoulder_y],
        [cx - notch_w * 0.7, shoulder_y + notch_d],    # 领口左
        [cx + notch_w * 0.7, shoulder_y + notch_d],    # 领口右
        [cx + notch_w, shoulder_y],
        [cx + sw * 0.28, shoulder_y - sh_h * 0.02],
        [cx + sw * 0.46, shoulder_y],
        [cx + sw * 0.50, shoulder_y + sh_h * 0.10],
        [cx + sw * taper + sw * 0.06, h],
        [cx - sw * taper - sw * 0.06, h],
    ], np.float32)
    body_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(body_mask, [pts.astype(np.int32)], 255)
    body_mask = cv2.GaussianBlur(body_mask, (13, 13), 0)

    # ---- 侧边阴影（立体感）----
    shade = solid.astype(np.float32)
    grad_x = np.abs(np.linspace(-1, 1, w, dtype=np.float32))
    grad = np.tile((grad_x * 0.28)[None, :, None], (h, 1, 3))
    shade = np.clip(shade * (1.0 - grad), 0, 255).astype(np.uint8)
    solid = shade

    # ---- 款式细节 ----
    overlay = np.zeros((h, w, 3), np.uint8)
    omask = np.zeros((h, w), np.uint8)

    if style in ("suit_tie", "suit"):
        # 西装左右两片（中间留出衬衫 V 区）
        tie_v_bottom = shoulder_y + sh_h * 0.62
        for sgn in (-1, 1):
            panel = np.array([
                [cx + sgn * sw * 0.485, shoulder_y + sh_h * 0.06],
                [cx + sgn * sw * 0.40, shoulder_y - sh_h * 0.015],
                [cx + sgn * sw * 0.115, shoulder_y + sh_h * 0.01],   # 内上（领口角）
                [cx + sgn * sw * 0.05, h * 0.92],                    # 内下（V 底）
                [cx + sgn * sw * (taper + 0.05), h],
                [cx + sgn * sw * (taper - 0.16), h],
            ], np.float32)
            cv2.fillPoly(overlay, [panel.astype(np.int32)], c1)
            cv2.fillPoly(omask, [panel.astype(np.int32)], 255)
            # 翻领：沿内缘的深色条
            lapel = np.array([
                [cx + sgn * sw * 0.115, shoulder_y + sh_h * 0.01],
                [cx + sgn * sw * 0.175, shoulder_y + sh_h * 0.05],
                [cx + sgn * sw * 0.105, shoulder_y + sh_h * 0.42],
                [cx + sgn * sw * 0.05, h * 0.92],
                [cx + sgn * sw * 0.085, h * 0.92],
            ], np.float32)
            cv2.fillPoly(overlay, [lapel.astype(np.int32)], _shade(c1, 0.55))
            cv2.fillPoly(omask, [lapel.astype(np.int32)], 255)
        if style == "suit_tie" and c3 is not None:
            # 领带：结 + 带身，从领口到 V 底
            knot_top = shoulder_y + notch_d * 0.4
            knot = np.array([
                [cx - sw * 0.045, knot_top],
                [cx + sw * 0.045, knot_top],
                [cx + sw * 0.038, knot_top + sh_h * 0.10],
                [cx - sw * 0.038, knot_top + sh_h * 0.10],
            ], np.float32)
            band = np.array([
                [cx - sw * 0.038, knot_top + sh_h * 0.10],
                [cx + sw * 0.038, knot_top + sh_h * 0.10],
                [cx + sw * 0.052, tie_v_bottom],
                [cx - sw * 0.052, tie_v_bottom],
            ], np.float32)
            cv2.fillPoly(overlay, [knot.astype(np.int32)], c3)
            cv2.fillPoly(overlay, [band.astype(np.int32)], _shade(c3, 0.85))
            cv2.fillPoly(omask, [knot.astype(np.int32)], 255)
            cv2.fillPoly(omask, [band.astype(np.int32)], 255)

    elif style == "polo":
        collar = np.array([
            [cx - sw * 0.13, shoulder_y],
            [cx - sw * 0.02, shoulder_y + sh_h * 0.14],
            [cx + sw * 0.02, shoulder_y + sh_h * 0.14],
            [cx + sw * 0.13, shoulder_y],
            [cx + sw * 0.09, shoulder_y - sh_h * 0.03],
            [cx - sw * 0.09, shoulder_y - sh_h * 0.03],
        ], np.float32)
        cv2.fillPoly(overlay, [collar.astype(np.int32)], c2)
        cv2.fillPoly(omask, [collar.astype(np.int32)], 255)

    elif style == "tunic":
        collar = np.array([
            [cx - sw * 0.16, shoulder_y - sh_h * 0.01],
            [cx + sw * 0.16, shoulder_y - sh_h * 0.01],
            [cx + sw * 0.13, shoulder_y + sh_h * 0.10],
            [cx - sw * 0.13, shoulder_y + sh_h * 0.10],
        ], np.float32)
        cv2.fillPoly(overlay, [collar.astype(np.int32)], _shade(c1, 0.7))
        cv2.fillPoly(omask, [collar.astype(np.int32)], 255)
        line = np.array([
            [cx - sw * 0.012, shoulder_y + sh_h * 0.05],
            [cx + sw * 0.012, shoulder_y + sh_h * 0.05],
            [cx + sw * 0.012, h],
            [cx - sw * 0.012, h],
        ], np.float32)
        cv2.fillPoly(overlay, [line.astype(np.int32)], _shade(c1, 0.6))
        cv2.fillPoly(omask, [line.astype(np.int32)], 255)

    elif style == "knit":
        placket = np.array([
            [cx - sw * 0.03, shoulder_y],
            [cx + sw * 0.03, shoulder_y],
            [cx + sw * 0.03, h],
            [cx - sw * 0.03, h],
        ], np.float32)
        cv2.fillPoly(overlay, [placket.astype(np.int32)], c2)
        cv2.fillPoly(omask, [placket.astype(np.int32)], 255)

    # ---- 合成：body 为底，细节叠加 ----
    out = np.zeros((h, w, 4), np.uint8)
    body3 = cv2.bitwise_and(solid, solid, mask=body_mask)
    alpha = body_mask.copy()
    blend = cv2.bitwise_and(overlay, overlay, mask=omask)
    body3 = body3.astype(np.float32)
    blend_f = blend.astype(np.float32)
    m = (omask.astype(np.float32) / 255.0)[:, :, None]
    body3 = blend_f * m + body3 * (1 - m)
    body3 = np.clip(body3, 0, 255).astype(np.uint8)
    out[:, :, :3] = body3
    out[:, :, 3] = alpha
    return out


def _find_shoulders(person_a, chin_y, face_w):
    """从人像 alpha 轮廓检测肩线：从头肩交界开始找第一处明显变宽的行。

    耸肩/紧身裁剪时肩膀可能高于下巴，所以扫描范围覆盖下巴上方。
    :return: (肩中心x, 肩线y, 该行可见宽度) 或 None
    """
    h, w = person_a.shape
    threshold_w = face_w * 1.30
    y_start = int(max(2, chin_y - face_w * 0.8))
    y_end = int(min(h - 2, chin_y + face_w * 1.2))
    step = max(2, int(face_w * 0.02))
    for y in range(y_start, y_end, step):
        row = person_a[y] > 128
        xs = np.where(row)[0]
        if len(xs) < 10:
            continue
        span = float(xs[-1] - xs[0])
        if span >= threshold_w:
            # 验证接下来几行仍然宽（排除手指等窄物体干扰）
            ok = 0
            for y2 in range(y, min(h, y + int(face_w * 0.3)), step):
                row2 = person_a[y2] > 128
                xs2 = np.where(row2)[0]
                if len(xs2) and (xs2[-1] - xs2[0]) >= threshold_w * 0.9:
                    ok += 1
            if ok >= 3:
                return float((xs[0] + xs[-1]) / 2.0), float(y), span
    return None


def apply_garment(rgba_person, garment_name, landmarks, gender="male"):
    """在人像 RGBA 上贴合服装模板，返回新的 RGBA。

    :param rgba_person: (H, W, 4) 抠图后人像
    :param garment_name: 模板名
    :param landmarks: mediapipe 关键点 (N, 2)
    :param gender: male/female（肩形）
    :return: (H, W, 4)
    """
    if landmarks is None or len(landmarks) < 455:
        return rgba_person

    h, w = rgba_person.shape[:2]
    left, right, chin = landmarks[234], landmarks[454], landmarks[152]
    face_w = float(np.linalg.norm(left - right))
    if face_w < 4:
        return rgba_person

    chin_y = float(chin[1])
    face_cx = float((left[0] + right[0]) / 2.0)

    # 从人像轮廓检测真实肩线，失败则退回比例估计
    found = _find_shoulders(rgba_person[:, :, 3], chin_y, face_w)
    notch_depth = None
    if found is not None:
        sh_cx, shoulder_y, visible_span = found
        # 肩膀在检测行以下继续变宽，量取下方最大宽度才是真实肩宽
        h_img, w_img = rgba_person.shape[:2]
        max_span = visible_span
        for y2 in range(int(shoulder_y), min(h_img, int(shoulder_y + face_w * 1.5)), 4):
            row2 = rgba_person[y2, :, 3] > 128
            xs2 = np.where(row2)[0]
            if len(xs2):
                max_span = max(max_span, float(xs2[-1] - xs2[0]))
        shoulder_w = float(np.clip(max_span * 0.98, face_w * 1.2, face_w * 2.2))
        # 领口 V 底延伸到下巴下方，防止服装盖住脖子（耸肩/紧身照常见）
        notch_depth = max(shoulder_w * 0.05, chin_y - shoulder_y + face_w * 0.08)
    else:
        sh_cx = face_cx
        shoulder_y = chin_y + face_w * 0.45
        shoulder_w = face_w * 1.55

    g = render_garment(garment_name, shoulder_w, gender, notch_depth)
    gh, gw = g.shape[:2]
    gys = int(gh * 0.12)                       # 模板内肩线位置（与 render 一致）

    # 贴图位置：模板肩线略高于检测线（盖住肩顶边缘），水平居中
    x0 = int(round(sh_cx - gw / 2.0))
    y0 = int(round(shoulder_y - gys - face_w * 0.08))

    # 允许覆盖区域 = 人像 alpha 膨胀（防止服装悬空在背景上）
    person_a = rgba_person[:, :, 3]
    allow = cv2.dilate(person_a, np.ones((15, 15), np.uint8))

    out = rgba_person.copy()
    # 裁剪到画布范围内
    sx0, sy0 = max(0, -x0), max(0, -y0)
    sx1 = min(gw, w - x0)
    sy1 = min(gh, h - y0)
    if sx1 <= sx0 or sy1 <= sy0:
        return rgba_person
    dx0, dy0 = max(0, x0), max(0, y0)
    dx1, dy1 = dx0 + (sx1 - sx0), dy0 + (sy1 - sy0)

    g_roi = g[sy0:sy1, sx0:sx1].astype(np.float32)
    allow_roi = allow[dy0:dy1, dx0:dx1].astype(np.float32)
    person_roi = out[dy0:dy1, dx0:dx1].astype(np.float32)

    # 服装 alpha × 允许区域，再轻度羽化
    g_alpha = (g_roi[:, :, 3] / 255.0) * (allow_roi / 255.0)
    g_alpha = cv2.GaussianBlur(g_alpha, (5, 5), 0)[:, :, None]

    roi = out[dy0:dy1, dx0:dx1]
    mixed = g_roi[:, :, :3] * g_alpha + person_roi[:, :, :3] * (1 - g_alpha)
    out[dy0:dy1, dx0:dx1, :3] = np.clip(mixed, 0, 255).astype(np.uint8)
    return out
