# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QComboBox,
    QSlider, QScrollArea, QDialog, QFormLayout, QLineEdit, QColorDialog,
    QMessageBox, QCheckBox, QGroupBox, QFileDialog, QApplication
)
from PySide6.QtCore import Qt, QThread, Signal, QTimer
from PySide6.QtGui import QImage, QPixmap, QColor

from core.beautify import beautify_pipeline
from core.retouch import retouch_pipeline, recolor_hair, level_head, get_landmarks
from core.matting import remove_background, change_background
from core.photo_specs import SPECS, detect_face, compose_id_photo, compose_id_photo_custom
from core.layout import make_sheet, layout_preview_info, PAPER_SIZES
from core.comfy_client import (
    is_available, edit_image, edit_image_stream, PRESET_PROMPTS, PRESET_PROMPTS_REF,
    resolve_models, find_server, interrupt as comfy_interrupt,
)


class LocalWorker(QThread):
    finished_ok = Signal(np.ndarray)
    failed = Signal(str)

    def __init__(self, orig_bgr, spec_name, custom_size=None, bg_bgr=(255, 255, 255),
                 smooth=40, white=20, sharp=30, slim=0, eyes=0, smile=0, teeth=0,
                 hair_target_rgb=None, hair_amount=60, auto_level=False, parent=None):
        super().__init__(parent)
        self.orig_bgr = orig_bgr
        self.spec_name = spec_name
        self.custom_size = custom_size
        self.bg_bgr = bg_bgr
        self.smooth = smooth
        self.white = white
        self.sharp = sharp
        self.slim = slim
        self.eyes = eyes
        self.smile = smile
        self.teeth = teeth
        self.hair_target_rgb = hair_target_rgb   # None 表示不改发色
        self.hair_amount = hair_amount
        self.auto_level = auto_level

    def run(self):
        try:
            img = self.orig_bgr

            # 0. 人像摆正（按双眼连线水平矫正）
            if self.auto_level:
                img, _angle = level_head(img)

            # 1. 美颜
            img = beautify_pipeline(img, self.smooth, self.white, self.sharp)

            # 2. 精修
            img = retouch_pipeline(img, self.slim, self.eyes, self.smile, self.teeth)

            # 3. 发色
            if self.hair_target_rgb is not None:
                img = recolor_hair(img, self.hair_target_rgb, self.hair_amount)

            # 4. 人脸检测
            face_box = detect_face(img)
            if face_box is None:
                self.failed.emit("未检测到人脸")
                return

            # 5. 抠图
            rgba = remove_background(img)

            # 6. 合成证件照
            if self.custom_size:
                w_mm, h_mm, dpi = self.custom_size
                result = compose_id_photo_custom(rgba, w_mm, h_mm, face_box, self.bg_bgr, dpi=dpi)
            else:
                result = compose_id_photo(rgba, self.spec_name, face_box, self.bg_bgr)

            self.finished_ok.emit(result)
        except Exception as e:
            self.failed.emit(str(e))


class PreviewWorker(QThread):
    """实时预览：在缩略图上跑摆正/美颜/精修/发色/抠图（慢速部分），
    服装贴合与换背景合成由 UI 线程用缓存即时完成。"""
    matting_ready = Signal(np.ndarray, np.ndarray, np.ndarray)  # rgba, face_box, landmarks
    failed = Signal(str)

    def __init__(self, small_bgr, smooth=40, white=20, sharp=30,
                 slim=0, eyes=0, smile=0, teeth=0,
                 hair_target_rgb=None, hair_amount=60, auto_level=False, parent=None):
        super().__init__(parent)
        self.small_bgr = small_bgr
        self.smooth = smooth
        self.white = white
        self.sharp = sharp
        self.slim = slim
        self.eyes = eyes
        self.smile = smile
        self.teeth = teeth
        self.hair_target_rgb = hair_target_rgb
        self.hair_amount = hair_amount
        self.auto_level = auto_level

    def run(self):
        try:
            img = self.small_bgr
            if self.auto_level:
                img, _ = level_head(img)
            img = beautify_pipeline(img, self.smooth, self.white, self.sharp)
            img = retouch_pipeline(img, self.slim, self.eyes, self.smile, self.teeth)
            if self.hair_target_rgb is not None:
                img = recolor_hair(img, self.hair_target_rgb, self.hair_amount)
            face_box = detect_face(img)
            if face_box is None:
                self.failed.emit("未检测到人脸")
                return
            rgba = remove_background(img)
            lm = get_landmarks(img)
            lm_arr = np.zeros((1, 2), np.float32) if lm is None else lm
            self.matting_ready.emit(rgba, np.asarray(face_box, np.float64), lm_arr)
        except Exception as e:
            self.failed.emit(str(e))


class AIWorker(QThread):
    """后台执行 ComfyUI 生成式编辑（websocket 流式，实时预览）"""
    finished_ok = Signal(np.ndarray)
    failed = Signal(str)
    progress = Signal(str)
    preview_frame = Signal(np.ndarray)   # 采样过程预览帧

    def __init__(self, orig_bgr, prompt, ref_bgr=None, unet=None, lora=None,
                 steps=None, cfg=None, timeout=1800, parent=None):
        super().__init__(parent)
        self.orig_bgr = orig_bgr
        self.prompt = prompt
        self.ref_bgr = ref_bgr
        self.unet = unet
        self.lora = lora
        self.steps = steps
        self.cfg = cfg
        self.timeout = timeout
        self._stopped = False

    def stop(self):
        self._stopped = True
        try:
            comfy_interrupt()
        except Exception:
            pass

    def run(self):
        try:
            result = None
            for kind, payload in edit_image_stream(
                    self.orig_bgr, self.prompt,
                    ref_image_bgr=self.ref_bgr,
                    unet=self.unet, lora=self.lora,
                    steps=self.steps, cfg=self.cfg,
                    timeout=self.timeout):
                if self._stopped:
                    return
                if kind == "progress":
                    self.progress.emit(payload)
                elif kind == "preview":
                    self.preview_frame.emit(payload)
                elif kind == "done":
                    result = payload
            if self._stopped:
                return
            if result is None:
                self.failed.emit("AI 编辑失败（无结果返回）")
            else:
                self.finished_ok.emit(result)
        except Exception as e:
            if not self._stopped:
                self.failed.emit(str(e))


def _bgr_to_pixmap(bgr, w, h):
    """BGR ndarray -> 等比缩放到 w×h 的 QPixmap"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ih, iw, _ = rgb.shape
    qimg = QImage(rgb.data, iw, ih, 3 * iw, QImage.Format_RGB888)
    return QPixmap.fromImage(qimg).scaled(w, h, Qt.KeepAspectRatio, Qt.SmoothTransformation)


class GenerateResultDialog(QDialog):
    """生成结果预览：成品大图 + 保存单张 / 保存排版图"""

    def __init__(self, image_bgr, parent=None, info=""):
        super().__init__(parent)
        self.setWindowTitle("生成结果预览")
        self.resize(560, 700)
        self.image_bgr = image_bgr

        lay = QVBoxLayout(self)
        h, w = image_bgr.shape[:2]
        self.view = QLabel()
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setMinimumSize(520, 560)
        self.view.setStyleSheet("border:1px solid #ccc; background:#fafafa;")
        self.view.setPixmap(_bgr_to_pixmap(image_bgr, 520, 560))
        lay.addWidget(self.view)

        self.info_label = QLabel(f"尺寸：{w}×{h} px    {info}")
        self.info_label.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.info_label)

        btns = QHBoxLayout()
        self.save_single_btn = QPushButton("保存单张")
        self.save_sheet_btn = QPushButton("保存排版图")
        close_btn = QPushButton("关闭")
        btns.addWidget(self.save_single_btn)
        btns.addWidget(self.save_sheet_btn)
        btns.addStretch()
        btns.addWidget(close_btn)
        lay.addLayout(btns)
        close_btn.clicked.connect(self.accept)


class AIPreviewDialog(QDialog):
    """AI 生成结果对比预览：左原右新，应用 / 放弃"""

    def __init__(self, before_bgr, after_bgr, parent=None, note=""):
        super().__init__(parent)
        self.setWindowTitle("AI 编辑结果预览")
        self.resize(760, 620)
        self.after_bgr = after_bgr
        self.applied = False

        lay = QVBoxLayout(self)
        row = QHBoxLayout()
        for title, img in (("原图", before_bgr), ("AI 结果", after_bgr)):
            col = QVBoxLayout()
            cap = QLabel(title)
            cap.setAlignment(Qt.AlignCenter)
            view = QLabel()
            view.setAlignment(Qt.AlignCenter)
            view.setFixedSize(350, 440)
            view.setStyleSheet("border:1px solid #ccc; background:#fafafa;")
            view.setPixmap(_bgr_to_pixmap(img, 350, 440))
            col.addWidget(cap)
            col.addWidget(view)
            row.addLayout(col)
        lay.addLayout(row)

        self.note_label = QLabel(note)
        self.note_label.setAlignment(Qt.AlignCenter)
        self.note_label.setWordWrap(True)
        lay.addWidget(self.note_label)

        btns = QHBoxLayout()
        apply_btn = QPushButton("应用为新底图")
        discard_btn = QPushButton("放弃")
        btns.addStretch()
        btns.addWidget(apply_btn)
        btns.addWidget(discard_btn)
        lay.addLayout(btns)
        apply_btn.clicked.connect(self._apply)
        discard_btn.clicked.connect(self.reject)

    def _apply(self):
        self.applied = True
        self.accept()


class PhotoTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_bgr = None
        self.result_bgr = None
        self.custom_size = None  # (w_mm, h_mm, dpi)
        self.last_spec_name = "一寸"
        
        self._setup_ui()
        self._connect_signals()
        self._check_comfy_status()

        # 实时预览：滑杆变化后 400ms 防抖触发
        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(400)
        self._preview_timer.timeout.connect(self._run_preview)
        self.preview_worker = None
        self._matting_cache = {}   # 抠图缓存: 参数key -> (rgba, face_box, landmarks)

    def _setup_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ==================== 左列：原图预览 ====================
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        self.original_label = QLabel("原图")
        self.original_label.setFixedSize(360, 460)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setStyleSheet("border: 2px solid #ccc; border-radius: 5px; background-color: #f5f5f5;")
        left_layout.addWidget(self.original_label)
        
        self.open_btn = QPushButton("打开图片")
        left_layout.addWidget(self.open_btn)
        
        left_layout.addStretch()
        main_layout.addWidget(left_widget)

        # ==================== 中列：控制面板 ====================
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFixedWidth(320)
        
        panel_widget = QWidget()
        panel_layout = QVBoxLayout(panel_widget)
        panel_layout.setSpacing(15)

        # 1. 规格选择
        spec_group = QGroupBox("规格")
        spec_layout = QHBoxLayout(spec_group)
        self.spec_combo = QComboBox()
        self.spec_combo.setEditable(True)
        self.spec_combo.addItems(list(SPECS.keys()))
        spec_layout.addWidget(self.spec_combo)
        
        self.custom_size_btn = QPushButton("自定义尺寸")
        spec_layout.addWidget(self.custom_size_btn)
        panel_layout.addWidget(spec_group)

        # 2. 背景色选择
        bg_group = QGroupBox("背景色")
        bg_layout = QHBoxLayout(bg_group)
        self.bg_combo = QComboBox()
        self.bg_combo.addItems(["白色", "蓝色", "红色", "深蓝", "浅灰", "自定义"])
        bg_layout.addWidget(self.bg_combo)
        
        self.custom_bg_btn = QPushButton("自定义")
        bg_layout.addWidget(self.custom_bg_btn)
        panel_layout.addWidget(bg_group)

        # 3. 美颜滑杆
        beauty_group = QGroupBox("美颜")
        beauty_layout = QVBoxLayout(beauty_group)
        
        self.smooth_slider = QSlider(Qt.Horizontal)
        self.smooth_slider.setRange(0, 100)
        self.smooth_slider.setValue(40)
        self.smooth_label = QLabel("40")
        self._add_slider_row(beauty_layout, "磨皮", self.smooth_slider, self.smooth_label)
        
        self.white_slider = QSlider(Qt.Horizontal)
        self.white_slider.setRange(0, 100)
        self.white_slider.setValue(20)
        self.white_label = QLabel("20")
        self._add_slider_row(beauty_layout, "美白", self.white_slider, self.white_label)
        
        self.sharp_slider = QSlider(Qt.Horizontal)
        self.sharp_slider.setRange(0, 100)
        self.sharp_slider.setValue(30)
        self.sharp_label = QLabel("30")
        self._add_slider_row(beauty_layout, "锐化", self.sharp_slider, self.sharp_label)
        
        panel_layout.addWidget(beauty_group)

        # 4. 精修滑杆
        retouch_group = QGroupBox("精修")
        retouch_layout = QVBoxLayout(retouch_group)

        self.level_check = QCheckBox("自动摆正头部（按双眼水平矫正）")
        retouch_layout.addWidget(self.level_check)
        
        self.slim_slider = QSlider(Qt.Horizontal)
        self.slim_slider.setRange(0, 100)
        self.slim_slider.setValue(0)
        self.slim_label = QLabel("0")
        self._add_slider_row(retouch_layout, "瘦脸", self.slim_slider, self.slim_label)
        
        self.eyes_slider = QSlider(Qt.Horizontal)
        self.eyes_slider.setRange(0, 100)
        self.eyes_slider.setValue(0)
        self.eyes_label = QLabel("0")
        self._add_slider_row(retouch_layout, "大眼", self.eyes_slider, self.eyes_label)
        
        self.smile_slider = QSlider(Qt.Horizontal)
        self.smile_slider.setRange(0, 100)
        self.smile_slider.setValue(0)
        self.smile_label = QLabel("0")
        self._add_slider_row(retouch_layout, "微笑", self.smile_slider, self.smile_label)
        
        self.teeth_slider = QSlider(Qt.Horizontal)
        self.teeth_slider.setRange(0, 100)
        self.teeth_slider.setValue(0)
        self.teeth_label = QLabel("0")
        self._add_slider_row(retouch_layout, "牙齿美白", self.teeth_slider, self.teeth_label)
        
        panel_layout.addWidget(retouch_group)

        # 5. 发色选择
        hair_group = QGroupBox("发色")
        hair_layout = QVBoxLayout(hair_group)
        
        self.hair_combo = QComboBox()
        self.hair_combo.addItems(["原色", "自然黑", "深棕", "浅棕", "亚麻金", "红棕", "自定义"])
        hair_layout.addWidget(self.hair_combo)
        
        self.custom_hair_btn = QPushButton("自定义发色")
        hair_layout.addWidget(self.custom_hair_btn)
        
        self.hair_amount_slider = QSlider(Qt.Horizontal)
        self.hair_amount_slider.setRange(0, 100)
        self.hair_amount_slider.setValue(60)
        self.hair_amount_label = QLabel("60")
        self._add_slider_row(hair_layout, "发色强度", self.hair_amount_slider, self.hair_amount_label)
        
        panel_layout.addWidget(hair_group)

        # 分隔线
        line = QLabel()
        line.setFixedHeight(2)
        line.setStyleSheet("background-color: #ccc;")
        panel_layout.addWidget(line)

        # 6. AI 编辑
        ai_group = QGroupBox("AI 生成式编辑（ComfyUI）")
        ai_layout = QVBoxLayout(ai_group)
        tip = QLabel("这里的 AI 编辑是真实生成，效果自然（约 30~60 秒/次），生成过程可实时预览。")
        tip.setWordWrap(True)
        ai_layout.addWidget(tip)

        self.comfy_status_label = QLabel("检测中...")
        self.comfy_status_label.setWordWrap(True)
        ai_layout.addWidget(self.comfy_status_label)

        # --- 模型自定义选择 ---
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("模型"))
        self.ai_model_combo = QComboBox()
        self.ai_model_combo.setMinimumWidth(200)
        model_row.addWidget(self.ai_model_combo, 1)
        ai_layout.addLayout(model_row)

        lora_row = QHBoxLayout()
        lora_row.addWidget(QLabel("加速"))
        self.ai_lora_combo = QComboBox()
        lora_row.addWidget(self.ai_lora_combo, 1)
        lora_row.addWidget(QLabel("步数"))
        from PySide6.QtWidgets import QSpinBox
        self.ai_steps_spin = QSpinBox()
        self.ai_steps_spin.setRange(1, 50)
        self.ai_steps_spin.setValue(4)
        lora_row.addWidget(self.ai_steps_spin)
        ai_layout.addLayout(lora_row)

        # --- 生成过程实时预览小窗 ---
        self.ai_preview_label = QLabel("生成时这里显示实时画面")
        self.ai_preview_label.setFixedSize(280, 200)
        self.ai_preview_label.setAlignment(Qt.AlignCenter)
        self.ai_preview_label.setStyleSheet("border: 1px solid #bbb; background: #222; color: #aaa;")
        self.ai_preview_label.hide()
        ai_layout.addWidget(self.ai_preview_label)

        self.ai_preset_combo = QComboBox()
        self.ai_preset_combo.addItem("— 选择预设 —")
        self.ai_preset_combo.addItems(list(PRESET_PROMPTS.keys()))
        self.ai_preset_combo.addItems(list(PRESET_PROMPTS_REF.keys()))
        ai_layout.addWidget(self.ai_preset_combo)

        self.ai_prompt_edit = QLineEdit()
        self.ai_prompt_edit.setPlaceholderText("或输入自定义提示词，如：换蓝色polo衫")
        ai_layout.addWidget(self.ai_prompt_edit)

        ref_row = QHBoxLayout()
        self.ai_ref_btn = QPushButton("参考图(可选)")
        self.ai_ref_btn.setCheckable(True)
        self.ai_ref_label = QLabel("未选择")
        self.ai_ref_label.setWordWrap(True)
        ref_row.addWidget(self.ai_ref_btn)
        ref_row.addWidget(self.ai_ref_label, 1)
        ai_layout.addLayout(ref_row)

        self.ai_run_btn = QPushButton("开始 AI 编辑")
        self.ai_stop_btn = QPushButton("取消")
        self.ai_stop_btn.setEnabled(False)
        ai_layout.addWidget(self.ai_run_btn)
        ai_layout.addWidget(self.ai_stop_btn)

        self.ai_progress_label = QLabel("")
        self.ai_progress_label.setWordWrap(True)
        ai_layout.addWidget(self.ai_progress_label)

        panel_layout.addWidget(ai_group)

        # 7. 排版设置
        layout_group = QGroupBox("排版")
        layout_widget_layout = QVBoxLayout(layout_group)
        
        self.paper_combo = QComboBox()
        self.paper_combo.addItems(list(PAPER_SIZES.keys()))
        layout_widget_layout.addWidget(self.paper_combo)
        
        self.guide_check = QCheckBox("裁切线")
        self.guide_check.setChecked(True)
        layout_widget_layout.addWidget(self.guide_check)
        
        self.border_check = QCheckBox("描边")
        self.border_check.setChecked(False)
        layout_widget_layout.addWidget(self.border_check)
        
        self.layout_info_label = QLabel("")
        self.layout_info_label.setWordWrap(True)
        layout_widget_layout.addWidget(self.layout_info_label)
        
        panel_layout.addWidget(layout_group)

        panel_layout.addStretch()

        scroll_area.setWidget(panel_widget)
        main_layout.addWidget(scroll_area)

        # ==================== 右列：实时预览 ====================
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        cap = QLabel("实时预览（跟随左侧参数自动更新）")
        cap.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(cap)

        self.result_label = QLabel("打开图片后这里显示实时预览")
        self.result_label.setFixedSize(420, 540)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setStyleSheet("border: 2px solid #ccc; border-radius: 5px; background-color: #f5f5f5;")
        right_layout.addWidget(self.result_label)

        self.preview_status_label = QLabel("")
        self.preview_status_label.setAlignment(Qt.AlignCenter)
        self.preview_status_label.setStyleSheet("color: #666;")
        right_layout.addWidget(self.preview_status_label)

        # 生成 / 保存
        self.generate_btn = QPushButton("生成证件照（高清成品）")
        self.save_single_btn = QPushButton("保存单张")
        self.save_sheet_btn = QPushButton("保存排版图")
        right_layout.addWidget(self.generate_btn)
        row2 = QHBoxLayout()
        row2.addWidget(self.save_single_btn)
        row2.addWidget(self.save_sheet_btn)
        right_layout.addLayout(row2)

        right_layout.addStretch()
        main_layout.addWidget(right_widget)

    def _add_slider_row(self, layout, title, slider, value_label):
        row = QHBoxLayout()
        row.addWidget(QLabel(title))
        row.addWidget(slider)
        row.addWidget(value_label)
        layout.addLayout(row)
        
        # 连接滑杆信号更新标签
        slider.valueChanged.connect(lambda val: value_label.setText(str(val)))

    def _connect_signals(self):
        # 打开图片
        self.open_btn.clicked.connect(self._on_open_image)
        
        # 自定义尺寸
        self.custom_size_btn.clicked.connect(self._on_custom_size)
        
        # 自定义背景色
        self.custom_bg_btn.clicked.connect(self._on_custom_bg_color)
        
        # 自定义发色
        self.custom_hair_btn.clicked.connect(self._on_custom_hair_color)
        
        # 生成证件照
        self.generate_btn.clicked.connect(self._on_generate)
        
        # 保存单张
        self.save_single_btn.clicked.connect(self._on_save_single)
        
        # 保存排版图
        self.save_sheet_btn.clicked.connect(self._on_save_sheet)
        
        # AI 按钮
        self.ai_run_btn.clicked.connect(self._on_ai_edit)
        self.ai_stop_btn.clicked.connect(self._on_ai_stop)
        self.ai_ref_btn.clicked.connect(self._on_ai_pick_ref)
        self.ai_preset_combo.currentIndexChanged.connect(self._on_ai_preset_changed)
        self.ai_lora_combo.currentIndexChanged.connect(self._on_ai_lora_changed)

        # 纸型切换时刷新排版提示
        self.paper_combo.currentIndexChanged.connect(lambda _: self._update_layout_info())

        # 实时预览：任何参数变化都重新调度
        for slider in [self.smooth_slider, self.white_slider, self.sharp_slider,
                       self.slim_slider, self.eyes_slider, self.smile_slider,
                       self.teeth_slider, self.hair_amount_slider]:
            slider.valueChanged.connect(lambda _v: self._schedule_preview())
        self.hair_combo.currentIndexChanged.connect(lambda _i: self._schedule_preview())
        self.level_check.stateChanged.connect(lambda _s: self._schedule_preview())
        # 换背景/规格/服装变化：命中缓存则即时重合成
        self.bg_combo.currentIndexChanged.connect(lambda _i: self._schedule_preview())
        self.spec_combo.currentIndexChanged.connect(lambda _i: self._schedule_preview())

    def _check_comfy_status(self):
        """检查 ComfyUI 是否可用，并填充模型选择下拉"""
        self.ai_ref_path = None
        try:
            ok = is_available()
            if ok:
                info = resolve_models(force=True)
                if info.get("ok"):
                    # 模型下拉（保持当前选择）
                    self.ai_model_combo.blockSignals(True)
                    self.ai_model_combo.clear()
                    self.ai_model_combo.addItems(info.get("unet_options") or [info["unet"]])
                    cur = info["unet"]
                    idx = self.ai_model_combo.findText(cur)
                    self.ai_model_combo.setCurrentIndex(idx if idx >= 0 else 0)
                    self.ai_model_combo.blockSignals(False)

                    # LoRA 下拉
                    self.ai_lora_combo.blockSignals(True)
                    self.ai_lora_combo.clear()
                    self.ai_lora_combo.addItem("无（不用加速LoRA，较慢）", None)
                    for l in info.get("lora_options") or [info.get("lora")]:
                        if l:
                            self.ai_lora_combo.addItem(l, l)
                    if info.get("lora"):
                        idx = self.ai_lora_combo.findData(info["lora"])
                        self.ai_lora_combo.setCurrentIndex(idx if idx >= 0 else 0)
                    self.ai_lora_combo.blockSignals(False)
                    self.ai_steps_spin.setValue(int(info.get("steps", 4)))

                    self.comfy_status_label.setText(
                        f"已连接 · 默认 {info['steps']} 步 / cfg {info['cfg']}")
                else:
                    self.comfy_status_label.setText(f"已连接但模型缺失：{info.get('note','')}")
                self.comfy_status_label.setStyleSheet("color: green;")
                self.ai_run_btn.setEnabled(True)
            else:
                self.comfy_status_label.setText(
                    "未检测到 ComfyUI。\n请先启动 ComfyUI（默认端口 8166/8188）。")
                self.comfy_status_label.setStyleSheet("color: red;")
                self.ai_run_btn.setEnabled(False)
        except Exception:
            self.comfy_status_label.setText("ComfyUI 未启动")
            self.comfy_status_label.setStyleSheet("color: red;")
            self.ai_run_btn.setEnabled(False)

    def _on_ai_preset_changed(self, _idx):
        """选择预设时填充提示词框"""
        name = self.ai_preset_combo.currentText()
        if name.startswith("—"):
            return
        if name in PRESET_PROMPTS:
            self.ai_prompt_edit.setText(PRESET_PROMPTS[name])
        elif name in PRESET_PROMPTS_REF:
            self.ai_prompt_edit.setText(PRESET_PROMPTS_REF[name])
            if not getattr(self, "ai_ref_path", None):
                QMessageBox.information(self, "提示",
                    "该预设需要一张参考图（如服装/发型照片），请点「参考图(可选)」选择。")

    def _on_ai_pick_ref(self):
        """选择参考图（图2，如服装/发型参考）"""
        if self.ai_ref_btn.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self, "选择参考图", "", "Image Files (*.jpg *.jpeg *.png *.bmp)")
            if not path:
                self.ai_ref_btn.setChecked(False)
                self.ai_ref_path = None
                self.ai_ref_label.setText("未选择")
                return
            try:
                data = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(data, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("无法读取")
                self.ai_ref_path = path
                self.ai_ref_img = img
                self.ai_ref_label.setText(os.path.basename(path))
            except Exception as e:
                self.ai_ref_btn.setChecked(False)
                self.ai_ref_path = None
                self.ai_ref_label.setText("未选择")
                QMessageBox.warning(self, "错误", f"读取参考图失败: {e}")
        else:
            self.ai_ref_path = None
            self.ai_ref_img = None
            self.ai_ref_label.setText("未选择")

    def _on_ai_lora_changed(self, _idx):
        """切换加速 LoRA 时自动匹配推荐步数"""
        name = self.ai_lora_combo.currentText()
        if "4steps" in name or "4步" in name:
            self.ai_steps_spin.setValue(4)
        elif "8steps" in name or "8步" in name:
            self.ai_steps_spin.setValue(8)
        elif name.startswith("无"):
            self.ai_steps_spin.setValue(20)

    def _on_ai_stop(self):
        """取消 AI 编辑"""
        if getattr(self, "ai_worker", None) and self.ai_worker.isRunning():
            self.ai_worker.stop()
            self.ai_progress_label.setText("正在取消…")

    def _on_open_image(self):
        """打开图片文件"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if not file_path:
            return

        # 使用 np.fromfile + cv2.imdecode 支持中文路径
        try:
            data = np.fromfile(file_path, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                QMessageBox.warning(self, "错误", "无法读取图片文件")
                return

            self.original_bgr = img
            self.result_bgr = None
            # 换了新图，旧抠图缓存必须作废，否则预览会一直显示上一张
            self._matting_cache.clear()
            if self.preview_worker is not None and self.preview_worker.isRunning():
                self.preview_worker.disconnect(self)
            self._update_original_preview()
            self._schedule_preview()   # 打开后立即给出基准预览
        except Exception as e:
            QMessageBox.warning(self, "错误", f"读取图片失败: {str(e)}")

    def _update_original_preview(self):
        """更新原图预览"""
        if self.original_bgr is None:
            return
        
        # BGR -> RGB
        rgb = cv2.cvtColor(self.original_bgr, cv2.COLOR_BGR2RGB)
        
        # 转换为 QImage
        h, w, _ = rgb.shape
        bytes_per_line = 3 * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        
        # 等比缩放到 360x460
        scaled_pixmap = pixmap.scaled(360, 460, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.original_label.setPixmap(scaled_pixmap)

    def _on_custom_size(self):
        """自定义尺寸对话框"""
        dialog = QDialog(self)
        dialog.setWindowTitle("自定义尺寸")
        form = QFormLayout()
        
        w_input = QLineEdit("25")
        h_input = QLineEdit("35")
        dpi_input = QLineEdit("300")
        
        form.addRow("宽 (mm):", w_input)
        form.addRow("高 (mm):", h_input)
        form.addRow("DPI:", dpi_input)
        
        btn_box = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        btn_box.addWidget(ok_btn)
        btn_box.addWidget(cancel_btn)
        form.addRow(btn_box)
        
        dialog.setLayout(form)
        
        def on_ok():
            try:
                w_mm = float(w_input.text())
                h_mm = float(h_input.text())
                dpi = int(dpi_input.text())
                if w_mm <= 0 or h_mm <= 0 or dpi <= 0:
                    raise ValueError("数值必须大于0")
                
                self.custom_size = (w_mm, h_mm, dpi)
                self.spec_combo.setCurrentText(f"自定义 {int(w_mm)}x{int(h_mm)}mm")
                dialog.accept()
            except Exception as e:
                QMessageBox.warning(dialog, "错误", f"输入无效: {str(e)}")
        
        ok_btn.clicked.connect(on_ok)
        cancel_btn.clicked.connect(dialog.reject)
        
        if dialog.exec() == QDialog.Accepted:
            self._update_layout_info()

    def _on_custom_bg_color(self):
        """自定义背景色"""
        color = QColorDialog.getColor(QColor("#FFFFFF"), self, "选择背景色")
        if color.isValid():
            # QColor 是 RGB，转为 BGR 元组存储
            self.custom_bg_bgr = (color.blue(), color.green(), color.red())
            self.bg_combo.setCurrentText("自定义")

    def _on_custom_hair_color(self):
        """自定义发色"""
        color = QColorDialog.getColor(QColor("#000000"), self, "选择发色")
        if color.isValid():
            # 存 RGB 元组
            self.custom_hair_rgb = (color.red(), color.green(), color.blue())
            self.hair_combo.setCurrentText("自定义")

    def _get_bg_color(self, bg_name):
        """获取背景色 BGR 元组"""
        if bg_name == "自定义":
            return getattr(self, 'custom_bg_bgr', (255, 255, 255))
        colors = {
            "白色": (255, 255, 255),
            "蓝色": (219, 142, 67),     # 标准证件照蓝 RGB(67,142,219)
            "红色": (0, 0, 255),        # RGB(255,0,0)
            "深蓝": (139, 0, 0),        # 深蓝 RGB(0,0,139)
            "浅灰": (200, 200, 200),
        }
        return colors.get(bg_name, (255, 255, 255))

    def _get_hair_target_rgb(self, hair_name):
        """获取发色目标 RGB 值"""
        if hair_name == "自定义":
            return getattr(self, 'custom_hair_rgb', (30, 30, 30))
        colors = {
            "自然黑": (30, 30, 30),
            "深棕": (60, 40, 20),
            "浅棕": (120, 80, 50),
            "亚麻金": (180, 150, 100),
            "红棕": (150, 60, 30),
        }
        return colors.get(hair_name)

    def _schedule_preview(self):
        """参数变化后调度实时预览（防抖）"""
        if self.original_bgr is None:
            return
        # 生成证件照过程中暂停实时预览
        if getattr(self, 'local_worker', None) is not None and self.local_worker.isRunning():
            return
        self._preview_timer.start()

    def _matting_key(self):
        """影响抠图结果的参数组合（缓存键）"""
        hair = self._resolve_hair_target_rgb()
        return (self.level_check.isChecked(),
                self.smooth_slider.value(), self.white_slider.value(),
                self.sharp_slider.value(), self.slim_slider.value(),
                self.eyes_slider.value(), self.smile_slider.value(),
                self.teeth_slider.value(),
                tuple(hair) if hair else None, self.hair_amount_slider.value())

    def _resolve_spec(self):
        """返回 (spec_name, custom_size)"""
        spec_name = self.spec_combo.currentText()
        custom_size = self.custom_size if "自定义" in spec_name else None
        return spec_name, custom_size

    def _run_preview(self):
        """缩略图上执行慢速部分（摆正/美颜/精修/发色/抠图），命中缓存则直接合成"""
        if self.original_bgr is None:
            return
        key = self._matting_key()
        if key in self._matting_cache:
            self._compose_preview()
            return
        # 上一轮预览还在跑就跳过，等下次触发
        if self.preview_worker is not None and self.preview_worker.isRunning():
            self._preview_timer.start()
            return

        small = self._downscale(self.original_bgr, 500)
        self.preview_worker = PreviewWorker(
            small_bgr=small,
            smooth=self.smooth_slider.value(),
            white=self.white_slider.value(),
            sharp=self.sharp_slider.value(),
            slim=self.slim_slider.value(),
            eyes=self.eyes_slider.value(),
            smile=self.smile_slider.value(),
            teeth=self.teeth_slider.value(),
            hair_target_rgb=self._resolve_hair_target_rgb(),
            hair_amount=self.hair_amount_slider.value(),
            auto_level=self.level_check.isChecked(),
        )
        self.preview_worker.matting_ready.connect(self._on_matting_ready)
        self.preview_worker.failed.connect(lambda _m: None)
        self.preview_worker.start()

    def _on_matting_ready(self, rgba, face_box, lm):
        """抠图完成：缓存后即时合成"""
        self._matting_cache.clear()
        self._matting_cache[self._matting_key()] = (rgba, face_box, lm)
        self._compose_preview()

    def _compose_preview(self):
        """用缓存的抠图结果做换背景合成（UI 线程，毫秒级）"""
        key = self._matting_key()
        if key not in self._matting_cache:
            return
        rgba, face_box, _lm = self._matting_cache[key]

        # 换背景 + 规格合成
        spec_name, custom_size = self._resolve_spec()
        bg_bgr = self._get_bg_color(self.bg_combo.currentText())
        try:
            if custom_size:
                w_mm, h_mm, dpi = custom_size
                result = compose_id_photo_custom(rgba, w_mm, h_mm, face_box, bg_bgr, dpi=dpi)
            else:
                result = compose_id_photo(rgba, spec_name, face_box, bg_bgr)
        except Exception:
            result = change_background(rgba, bg_bgr)  # 规格异常时退回仅换底

        self.preview_status_label.setText(f"预览 {result.shape[1]}×{result.shape[0]} · 规格：{spec_name}")
        self._update_preview_label(self.result_label, result)

    def _downscale(self, bgr, max_side):
        h, w = bgr.shape[:2]
        scale = min(1.0, max_side / max(h, w))
        if scale >= 1.0:
            return bgr.copy()
        return cv2.resize(bgr, (int(w * scale), int(h * scale)),
                          interpolation=cv2.INTER_AREA)

    def _update_preview_label(self, label, bgr_img):
        rgb = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        label.setPixmap(pixmap.scaled(label.width(), label.height(),
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def _resolve_hair_target_rgb(self):
        """解析发色下拉框为 RGB 元组，原色返回 None"""
        name = self.hair_combo.currentText()
        if name == "原色":
            return None
        if name == "自定义":
            return getattr(self, 'custom_hair_rgb', (30, 30, 30))
        colors = {
            "自然黑": (30, 30, 30),
            "深棕": (60, 40, 20),
            "浅棕": (120, 80, 50),
            "亚麻金": (180, 150, 100),
            "红棕": (150, 60, 30),
        }
        return colors.get(name, (30, 30, 30))

    def _on_generate(self):
        """生成证件照"""
        if self.original_bgr is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return
        
        # 禁用按钮
        self._disable_buttons()

        # 显示处理中（不动实时预览画面）
        self.preview_status_label.setText("正在生成高清成品…")
        self.preview_status_label.setStyleSheet("color: #c00;")

        # 获取当前参数
        spec_name = self.spec_combo.currentText()
        custom_size = self.custom_size if "自定义" in spec_name else None
        
        # 创建并启动 LocalWorker（背景色在 UI 线程解析，含对话框取色）
        spec_name, custom_size = self._resolve_spec()
        self.local_worker = LocalWorker(
            orig_bgr=self.original_bgr,
            spec_name=spec_name,
            custom_size=custom_size,
            bg_bgr=self._get_bg_color(self.bg_combo.currentText()),
            smooth=self.smooth_slider.value(),
            white=self.white_slider.value(),
            sharp=self.sharp_slider.value(),
            slim=self.slim_slider.value(),
            eyes=self.eyes_slider.value(),
            smile=self.smile_slider.value(),
            teeth=self.teeth_slider.value(),
            hair_target_rgb=self._resolve_hair_target_rgb(),
            hair_amount=self.hair_amount_slider.value(),
            auto_level=self.level_check.isChecked(),
        )
        
        self.local_worker.finished_ok.connect(self._on_generate_success)
        self.local_worker.failed.connect(self._on_generate_failed)
        self.local_worker.start()

    def _on_generate_success(self, result_bgr):
        """生成成功：弹窗预览成品，实时预览保持不变"""
        self.result_bgr = result_bgr
        # 记录本次成功使用的规格（供自定义尺寸排版回退用）
        spec_name = self.spec_combo.currentText()
        if "自定义" not in spec_name and spec_name in SPECS:
            self.last_spec_name = spec_name
        self.preview_status_label.setText("高清成品已生成 ✔")
        self.preview_status_label.setStyleSheet("color: #080;")
        self._enable_buttons()
        self._update_layout_info()

        info = f"规格：{spec_name} · 背景：{self.bg_combo.currentText()}"
        dlg = GenerateResultDialog(result_bgr, self, info=info)
        dlg.save_single_btn.clicked.connect(self._on_save_single)
        dlg.save_sheet_btn.clicked.connect(self._on_save_sheet)
        dlg.exec()

    def _on_generate_failed(self, error_msg):
        """生成失败"""
        self.preview_status_label.setText("")
        self.preview_status_label.setStyleSheet("color: #666;")
        QMessageBox.warning(self, "错误", f"生成失败: {error_msg}")
        self._enable_buttons()

    def _on_ai_edit(self):
        """AI 生成式编辑（ComfyUI，流式实时预览）"""
        if self.original_bgr is None:
            QMessageBox.warning(self, "提示", "请先打开图片")
            return

        prompt = self.ai_prompt_edit.text().strip()
        if not prompt:
            QMessageBox.warning(self, "提示", "请先选择预设或输入提示词")
            return
        if not is_available():
            QMessageBox.warning(self, "提示",
                "未检测到 ComfyUI，请先启动（默认端口 8166/8188）")
            return

        ref = getattr(self, "ai_ref_img", None)
        unet = self.ai_model_combo.currentText() or None
        lora = self.ai_lora_combo.currentData()
        steps = self.ai_steps_spin.value()

        self._disable_buttons()
        self.ai_progress_label.setText("准备中…")
        self.ai_run_btn.setEnabled(False)
        self.ai_stop_btn.setEnabled(True)
        self.ai_preview_label.show()
        self.ai_preview_label.setText("等待首个预览帧…")

        self.ai_worker = AIWorker(orig_bgr=self.original_bgr, prompt=prompt,
                                  ref_bgr=ref, unet=unet, lora=lora, steps=steps)
        self.ai_worker.progress.connect(lambda m: self.ai_progress_label.setText(m))
        self.ai_worker.preview_frame.connect(self._on_ai_preview_frame)
        self.ai_worker.finished_ok.connect(self._on_ai_edit_success)
        self.ai_worker.failed.connect(self._on_ai_edit_failed)
        self.ai_worker.start()

    def _on_ai_preview_frame(self, frame_bgr):
        """显示生成过程的实时预览帧"""
        self._update_preview_label(self.ai_preview_label, frame_bgr)

    def _on_ai_edit_success(self, result_bgr):
        """AI 编辑成功：先弹对比预览，用户确认后才替换底图"""
        self.ai_preview_label.hide()
        self.ai_progress_label.setText("生成完成，请确认效果")
        self._enable_buttons()
        self.ai_run_btn.setEnabled(True)
        self.ai_stop_btn.setEnabled(False)

        dlg = AIPreviewDialog(self.original_bgr, result_bgr, self,
                              note="满意就「应用为新底图」（可继续叠加美颜/排版），不满意就「放弃」重新生成。")
        if dlg.exec() == QDialog.Accepted and dlg.applied:
            self.original_bgr = dlg.after_bgr
            self.result_bgr = None
            # 底图变了，旧抠图缓存必须作废
            self._matting_cache.clear()
            if self.preview_worker is not None and self.preview_worker.isRunning():
                self.preview_worker.disconnect(self)
            self._update_original_preview()
            self._schedule_preview()
            self.ai_progress_label.setText("已应用为新底图 ✔")
        else:
            self.ai_progress_label.setText("已放弃本次结果")

    def _on_ai_edit_failed(self, error_msg):
        """AI 编辑失败"""
        self.ai_preview_label.hide()
        self.ai_progress_label.setText("")
        QMessageBox.warning(self, "错误", f"AI 编辑失败:\n{error_msg}")
        self._enable_buttons()
        self.ai_run_btn.setEnabled(is_available())
        self.ai_stop_btn.setEnabled(False)

    def _on_save_single(self):
        """保存单张证件照"""
        if self.result_bgr is None:
            QMessageBox.warning(self, "提示", "请先生成证件照")
            return
        
        spec_name = self.spec_combo.currentText().replace(" ", "")
        default_name = f"证件照_{spec_name}.jpg"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存证件照", default_name, "JPEG Files (*.jpg *.jpeg);;PNG Files (*.png)"
        )
        if not file_path:
            return
        
        # 使用 cv2.imencode + tofile 支持中文路径
        try:
            ext = os.path.splitext(file_path)[1]
            if ext.lower() in ['.jpg', '.jpeg']:
                success, encoded = cv2.imencode('.jpg', self.result_bgr)
            else:
                success, encoded = cv2.imencode('.png', self.result_bgr)
            
            if success:
                encoded.tofile(file_path)
            else:
                raise Exception("编码失败")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存失败: {str(e)}")

    def _on_save_sheet(self):
        """保存排版图"""
        if self.result_bgr is None:
            QMessageBox.warning(self, "提示", "请先生成证件照")
            return
        
        paper_name = self.paper_combo.currentText()
        
        # 确定规格名：如果是自定义，使用最近一次生成的规格名或"一寸"
        spec_name = self.spec_combo.currentText()
        if "自定义" in spec_name:
            sheet_spec_name = self.last_spec_name if hasattr(self, 'last_spec_name') else "一寸"
        else:
            sheet_spec_name = spec_name
        
        # 生成排版图
        try:
            sheet_bgr = make_sheet(
                id_photo_bgr=self.result_bgr,
                spec_name=sheet_spec_name,
                paper_name=paper_name,
                dpi=300,
                margin_mm=3,
                gap_mm=2,
                draw_guide=self.guide_check.isChecked(),
                border=self.border_check.isChecked()
            )
            
            default_name = f"排版_{paper_name}_{sheet_spec_name}.jpg"
            file_path, _ = QFileDialog.getSaveFileName(
                self, "保存排版图", default_name, "JPEG Files (*.jpg *.jpeg);;PNG Files (*.png)"
            )
            if not file_path:
                return
            
            ext = os.path.splitext(file_path)[1]
            if ext.lower() in ['.jpg', '.jpeg']:
                success, encoded = cv2.imencode('.jpg', sheet_bgr)
            else:
                success, encoded = cv2.imencode('.png', sheet_bgr)
            
            if success:
                encoded.tofile(file_path)
            else:
                raise Exception("编码失败")
        except Exception as e:
            QMessageBox.warning(self, "错误", f"保存排版图失败: {str(e)}")

    def _update_layout_info(self):
        """更新排版预览信息"""
        if self.result_bgr is None:
            return
        
        paper_name = self.paper_combo.currentText()
        spec_name = self.spec_combo.currentText()
        
        try:
            info = layout_preview_info(None, spec_name, paper_name)
            self.layout_info_label.setText(info)
        except Exception:
            self.layout_info_label.setText("无法获取排版信息")

    def _disable_buttons(self):
        """禁用所有按钮"""
        for btn in [self.open_btn, self.custom_size_btn, self.custom_bg_btn,
                    self.custom_hair_btn, self.generate_btn, self.save_single_btn,
                    self.save_sheet_btn, self.ai_ref_btn, self.ai_preset_combo,
                    self.ai_prompt_edit, self.ai_model_combo, self.ai_lora_combo,
                    self.ai_steps_spin]:
            btn.setEnabled(False)

    def _enable_buttons(self):
        """启用所有按钮"""
        for btn in [self.open_btn, self.custom_size_btn, self.custom_bg_btn,
                    self.custom_hair_btn, self.generate_btn, self.save_single_btn,
                    self.save_sheet_btn, self.ai_ref_btn, self.ai_preset_combo,
                    self.ai_prompt_edit, self.ai_model_combo, self.ai_lora_combo,
                    self.ai_steps_spin]:
            btn.setEnabled(True)

        # AI 运行按钮根据 ComfyUI 状态启用
        try:
            self.ai_run_btn.setEnabled(is_available())
        except Exception:
            self.ai_run_btn.setEnabled(False)
