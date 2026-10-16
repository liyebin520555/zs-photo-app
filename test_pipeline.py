# -*- coding: utf-8 -*-
"""端到端 headless 测试：证件照管线 + 文档扫描管线"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
os.makedirs(OUT, exist_ok=True)

# ---------- 证件照管线 ----------
from core.beautify import beautify_pipeline
from core.photo_specs import detect_face, compose_id_photo, make_layout_sheet, SPECS, mm_to_px
from core.matting import remove_background, change_background

img = cv2.imdecode(np.fromfile('test_portrait.jpg', dtype=np.uint8), cv2.IMREAD_COLOR)
print('原图尺寸:', img.shape)

beautified = beautify_pipeline(img, 40, 20, 30)
print('美颜 OK')

face = detect_face(beautified)
print('人脸框:', face)
assert face is not None, '人脸检测失败'

rgba = remove_background(beautified)
print('抠图 OK:', rgba.shape, 'alpha非零占比: %.2f' % ((rgba[:, :, 3] > 128).mean()))

photo = compose_id_photo(rgba, '一寸', face, (219, 142, 67))
assert photo.shape[:2] == (mm_to_px(35), mm_to_px(25)), photo.shape
print('一寸证件照 OK:', photo.shape)

sheet = make_layout_sheet(photo, '一寸')
print('六寸排版 OK:', sheet.shape)

ok, enc = cv2.imencode('.jpg', photo)
enc.tofile(os.path.join(OUT, 'id_photo_1inch.jpg'))
ok, enc = cv2.imencode('.jpg', sheet)
enc.tofile(os.path.join(OUT, 'layout_6inch.jpg'))
ok, enc = cv2.imencode('.jpg', beautified)
enc.tofile(os.path.join(OUT, 'beautified.jpg'))

# ---------- 文档扫描管线 ----------
from core.scanner import find_document_contour, four_point_transform, enhance_document

# 合成一张带透视的"文档"测试图
doc = np.full((600, 800, 3), 160, dtype=np.uint8)
page = np.full((480, 360, 3), 245, dtype=np.uint8)
for i in range(20):
    y = 30 + i * 22
    cv2.putText(page, 'The quick brown fox jumps over the lazy dog',
                (15, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 40), 1, cv2.LINE_AA)
src_pts = np.float32([[60, 50], [330, 30], [350, 500], [45, 520]])
M = cv2.getPerspectiveTransform(np.float32([[0, 0], [360, 0], [360, 480], [0, 480]]), src_pts)
warped = cv2.warpPerspective(page, M, (800, 600))
mask = np.zeros((600, 800), np.uint8)
cv2.fillPoly(mask, [src_pts.astype(np.int32)], 255)
test_doc = np.where(mask[:, :, None] > 0, warped, doc)

quad = find_document_contour(test_doc)
print('检测到四边形:', None if quad is None else quad.astype(int).tolist())

fixed = four_point_transform(test_doc, quad if quad is not None else np.float32([[45, 30], [350, 30], [350, 520], [45, 520]]))
print('透视矫正 OK:', fixed.shape)

for mode in ['original', 'enhanced', 'gray', 'bw']:
    r = enhance_document(fixed, mode)
    print('滤镜', mode, 'OK:', r.shape)

ok, enc = cv2.imencode('.jpg', test_doc)
enc.tofile(os.path.join(OUT, 'test_doc_input.jpg'))
ok, enc = cv2.imencode('.jpg', enhance_document(fixed, 'bw'))
enc.tofile(os.path.join(OUT, 'test_doc_bw.jpg'))

print('ALL_CORE_TESTS_PASSED')
