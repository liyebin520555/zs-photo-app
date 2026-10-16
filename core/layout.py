# -*- coding: utf-8 -*-
"""证件照打印排版模块。"""

import numpy as np
from typing import Tuple, Dict, Optional
import cv2


PAPER_SIZES: Dict[str, Tuple[int, int]] = {
    "5寸": (89, 127),
    "6寸": (102, 152),
    "7寸": (127, 178),
    "A4": (210, 297),
    "6寸横版": (152, 102),
    "A4横版": (297, 210),
}


def layout_preview_info(
    id_photo_bgr: np.ndarray,
    spec_name: str,
    paper_name: str = "6寸",
    dpi: int = 300,
) -> str:
    """返回排版预览信息字符串。

    Args:
        id_photo_bgr: 证件照 BGR 图像。
        spec_name: 证件照规格名称（如"小二寸"）。
        paper_name: 纸型名称。
        dpi: 输出 DPI。

    Returns:
        排版信息字符串，例如 "6寸相纸可排 8 张（2列 × 4行）"。
    """
    from core.spec_data import SPECS

    photo_w_mm, photo_h_mm = SPECS[spec_name]
    paper_w_mm, paper_h_mm = PAPER_SIZES[paper_name]

    photo_w_px = int(round(photo_w_mm * dpi / 25.4))
    photo_h_px = int(round(photo_h_mm * dpi / 25.4))
    paper_w_px = int(round(paper_w_mm * dpi / 25.4))
    paper_h_px = int(round(paper_h_mm * dpi / 25.4))

    margin_px = int(round(3.0 * dpi / 25.4))
    gap_px = int(round(2.0 * dpi / 25.4))

    # 与 make_sheet 一致：末尾不需要间隙，所以用 (可用+gap)//(照片+gap)
    cols = max(1, (paper_w_px - 2 * margin_px + gap_px) // (photo_w_px + gap_px))
    rows = max(1, (paper_h_px - 2 * margin_px + gap_px) // (photo_h_px + gap_px))

    total = cols * rows
    return f"{paper_name}相纸可排 {total} 张（{cols}列 × {rows}行）"


def make_sheet(
    id_photo_bgr: np.ndarray,
    spec_name: str,
    paper_name: str = "6寸",
    dpi: int = 300,
    margin_mm: float = 3.0,
    gap_mm: float = 2.0,
    draw_guide: bool = True,
    border: bool = False,
) -> np.ndarray:
    """生成排版后的相纸图像。

    Args:
        id_photo_bgr: 证件照 BGR 图像（H×W×3）。
        spec_name: 证件照规格名称。
        paper_name: 纸型名称。
        dpi: 输出 DPI。
        margin_mm: 纸张边缘留白（mm）。
        gap_mm: 照片间距（mm）。
        draw_guide: 是否绘制裁切参考线。
        border: 是否绘制照片边框。

    Returns:
        BGR 相纸图像。
    """
    from core.spec_data import SPECS

    photo_w_mm, photo_h_mm = SPECS[spec_name]
    paper_w_mm, paper_h_mm = PAPER_SIZES[paper_name]

    photo_w_px = int(round(photo_w_mm * dpi / 25.4))
    photo_h_px = int(round(photo_h_mm * dpi / 25.4))
    paper_w_px = int(round(paper_w_mm * dpi / 25.4))
    paper_h_px = int(round(paper_h_mm * dpi / 25.4))

    margin_px = int(round(margin_mm * dpi / 25.4))
    gap_px = int(round(gap_mm * dpi / 25.4))

    # 照片尺寸与规格不一致时，缩放到规格尺寸，避免贴图越界
    if id_photo_bgr.shape[0] != photo_h_px or id_photo_bgr.shape[1] != photo_w_px:
        id_photo_bgr = cv2.resize(id_photo_bgr, (photo_w_px, photo_h_px),
                                  interpolation=cv2.INTER_AREA)

    cols = max(1, (paper_w_px - 2 * margin_px + gap_px) // (photo_w_px + gap_px))
    rows = max(1, (paper_h_px - 2 * margin_px + gap_px) // (photo_h_px + gap_px))

    total = cols * rows

    if photo_w_px > paper_w_px - 2 * margin_px or photo_h_px > paper_h_px - 2 * margin_px:
        canvas = np.full((paper_h_px, paper_w_px, 3), 255, dtype=np.uint8)
        scale_x = paper_w_px / photo_w_px
        scale_y = paper_h_px / photo_h_px
        scale = min(scale_x, scale_y)
        new_w = int(round(photo_w_px * scale))
        new_h = int(round(photo_h_px * scale))
        x0 = (paper_w_px - new_w) // 2
        y0 = (paper_h_px - new_h) // 2
        canvas[y0:y0 + new_h, x0:x0 + new_w] = cv2.resize(
            id_photo_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA
        )
        return canvas

    canvas = np.full((paper_h_px, paper_w_px, 3), 255, dtype=np.uint8)

    x_start = margin_px
    y_start = margin_px

    for r in range(rows):
        for c in range(cols):
            x = x_start + c * (photo_w_px + gap_px)
            y = y_start + r * (photo_h_px + gap_px)

            canvas[y:y + photo_h_px, x:x + photo_w_px] = id_photo_bgr

            # 裁切参考线画在照片外侧 1px，避免被照片盖住
            if draw_guide:
                if y > 0:
                    canvas[y - 1, x:x + photo_w_px] = (200, 200, 200)
                if y + photo_h_px < paper_h_px:
                    canvas[y + photo_h_px, x:x + photo_w_px] = (200, 200, 200)
                if x > 0:
                    canvas[y:y + photo_h_px, x - 1] = (200, 200, 200)
                if x + photo_w_px < paper_w_px:
                    canvas[y:y + photo_h_px, x + photo_w_px] = (200, 200, 200)

            if border:
                cv2.rectangle(canvas, (x, y), (x + photo_w_px, y + photo_h_px),
                              (0, 0, 0), 1)

    return canvas
