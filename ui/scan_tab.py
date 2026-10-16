# -*- coding: utf-8 -*-
import sys
import os
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QComboBox, QFileDialog
)
from PySide6.QtGui import QPixmap, QImage
from PySide6.QtCore import Qt
from core.scanner import find_document_contour, four_point_transform, enhance_document


class ScanTab(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.original_image = None  # BGR numpy array
        self.vertices = None  # 4x2 numpy array of vertices
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QHBoxLayout()
        self.setLayout(main_layout)

        # Left column: original image preview and buttons
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)

        self.original_label = QLabel("原图预览")
        self.original_label.setFixedWidth(420)
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumHeight(300)
        left_layout.addWidget(self.original_label)

        open_btn = QPushButton("打开图片")
        open_btn.clicked.connect(self._on_open_image)
        left_layout.addWidget(open_btn)

        detect_btn = QPushButton("自动检测边缘")
        detect_btn.clicked.connect(self._on_detect_edges)
        left_layout.addWidget(detect_btn)

        main_layout.addWidget(left_widget)

        # Middle column: edge detection result preview
        middle_widget = QWidget()
        middle_layout = QVBoxLayout()
        middle_widget.setLayout(middle_layout)

        self.edge_label = QLabel("边缘检测结果")
        self.edge_label.setFixedWidth(300)
        self.edge_label.setAlignment(Qt.AlignCenter)
        self.edge_label.setMinimumHeight(300)
        middle_layout.addWidget(self.edge_label)

        self.edge_info_label = QLabel("说明文字")
        self.edge_info_label.setWordWrap(True)
        middle_layout.addWidget(self.edge_info_label)

        main_layout.addWidget(middle_widget)

        # Right column: result preview, filter combo, save button
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)

        self.result_label = QLabel("结果预览")
        self.result_label.setFixedWidth(380)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumHeight(300)
        right_layout.addWidget(self.result_label)

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["原图", "增强", "灰度", "黑白"])
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        right_layout.addWidget(self.filter_combo)

        save_btn = QPushButton("保存结果")
        save_btn.clicked.connect(self._on_save_result)
        right_layout.addWidget(save_btn)

        main_layout.addWidget(right_widget)

    def _numpy_to_qpixmap(self, img_bgr):
        """Convert BGR numpy array to QPixmap"""
        if img_bgr is None:
            return QPixmap()
        if img_bgr.ndim == 2:
            img_bgr = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2BGR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(qimg)

    def _on_open_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "打开图片", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return
        # Read image using cv2.imdecode to handle Chinese paths
        data = np.fromfile(file_path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            return
        self.original_image = img
        self.vertices = None
        # Update original preview
        pixmap = self._numpy_to_qpixmap(img)
        self.original_label.setPixmap(pixmap.scaled(
            420, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))
        # Reset edge and result previews
        self.edge_label.setText("边缘检测结果")
        self.edge_info_label.setText("")
        self.result_label.setText("结果预览")

    def _on_detect_edges(self):
        if self.original_image is None:
            return
        img = self.original_image
        h, w = img.shape[:2]
        # Try to find document contour
        vertices = find_document_contour(img)
        if vertices is not None and len(vertices) == 4:
            self.vertices = vertices.astype(np.int32)
            # Draw the quadrilateral on a copy of the original image
            vis = img.copy()
            cv2.polylines(vis, [self.vertices.reshape(4, 1, 2)], True, (0, 255, 0), 3)
            for pt in self.vertices:
                cv2.circle(vis, tuple(pt), 6, (0, 255, 0), -1)
            pixmap = self._numpy_to_qpixmap(vis)
            self.edge_label.setPixmap(pixmap.scaled(
                300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.edge_info_label.setText("已检测到文档边缘")
        else:
            # Fallback to image corners
            self.vertices = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.int32)
            vis = img.copy()
            cv2.polylines(vis, [self.vertices.reshape(4, 1, 2)], True, (0, 255, 0), 3)
            for pt in self.vertices:
                cv2.circle(vis, tuple(pt), 6, (0, 255, 0), -1)
            pixmap = self._numpy_to_qpixmap(vis)
            self.edge_label.setPixmap(pixmap.scaled(
                300, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
            ))
            self.edge_info_label.setText("未检测到文档边缘，将按原图四角矫正")

    def _get_filter_mode(self):
        index = self.filter_combo.currentIndex()
        modes = ["original", "enhanced", "gray", "bw"]
        return modes[index] if index < len(modes) else "original"

    def _on_filter_changed(self, index=None):
        if self.original_image is None or self.vertices is None:
            return
        mode = self._get_filter_mode()
        transformed = four_point_transform(self.original_image, self.vertices)
        result = enhance_document(transformed, mode)
        pixmap = self._numpy_to_qpixmap(result)
        self.result_label.setPixmap(pixmap.scaled(
            380, 300, Qt.KeepAspectRatio, Qt.SmoothTransformation
        ))

    def _on_save_result(self):
        if self.original_image is None or self.vertices is None:
            return
        mode = self._get_filter_mode()
        transformed = four_point_transform(self.original_image, self.vertices)
        result = enhance_document(transformed, mode)
        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "PNG (*.png);;JPEG (*.jpg *.jpeg)"
        )
        if not file_path:
            return
        # Determine extension
        ext = os.path.splitext(file_path)[1].lower()
        if ext in [".jpg", ".jpeg"]:
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        else:
            encode_param = []
        success, encoded = cv2.imencode(ext if ext.startswith('.') else '.png', result, encode_param)
        if success:
            # Use np.tofile to handle Chinese paths
            encoded.tofile(file_path)
