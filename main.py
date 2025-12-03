import sys
import cv2
import numpy as np
import os
import time
import base64
import tempfile  # [新增] 用于创建临时目录
import shutil    # [新增] 用于清理资源
import ctypes  # [新增] 用于设置 Windows 任务栏图标 ID

# 移除对 resources_rc 和 qdarkstyle 的依赖，实现完全独立的单文件运行
# try: import resources_rc ... (已移除)
# try: import qdarkstyle ... (已移除)

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QPushButton, QSlider, QFileDialog, QTabWidget, 
                             QScrollArea, QFrame, QLineEdit, QComboBox, QGroupBox, 
                             QGridLayout, QColorDialog, QMessageBox, QSpinBox, 
                             QDoubleSpinBox, QFormLayout, QCheckBox, QListWidget, 
                             QListWidgetItem, QAbstractItemView, QSplitter, QSizePolicy,
                             QTableWidget, QTableWidgetItem, QHeaderView)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPointF, QRectF, QSize, QPoint
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QWheelEvent, QMouseEvent, QAction, QIcon, QPen, QBrush, QKeySequence, QShortcut

# ==========================================
# 1. 嵌入式资源管理器 (核心修改)
# ==========================================
class ResourceManager:
    """
    在运行时自动生成所需的 SVG 图标文件，解决 Base64 显示问题，
    同时实现单文件运行，无需外部 .rc 或 .py 资源文件。
    """
    def __init__(self):
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp(prefix="gemini_style_")
        self.icons = {}
        self._create_svg_files()

    def _create_svg_files(self):
        # 定义 SVG 内容 (纯白色 #ffffff，适应暗色背景)
        # 这里的 path 数据绘制了标准的三角形箭头
        svg_up = '''<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">
          <path fill="#ffffff" d="M6 3 L10 9 L2 9 Z"/>
        </svg>'''

        svg_down = '''<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">
          <path fill="#ffffff" d="M6 9 L2 3 L10 3 Z"/>
        </svg>'''

        # 写入文件
        up_path = os.path.join(self.temp_dir, "up.svg")
        down_path = os.path.join(self.temp_dir, "down.svg")

        with open(up_path, "w") as f:
            f.write(svg_up)
        with open(down_path, "w") as f:
            f.write(svg_down)

        # 存储为 Qt 样式表可用的路径格式 (Windows 下需替换反斜杠)
        self.icons['up'] = up_path.replace("\\", "/")
        self.icons['down'] = down_path.replace("\\", "/")

    def get_icon_url(self, name):
        """返回 CSS url(...) 格式的字符串"""
        path = self.icons.get(name, "")
        return f'url("{path}")'

    def cleanup(self):
        """程序退出时清理临时文件"""
        try:
            shutil.rmtree(self.temp_dir)
        except:
            pass

# 初始化全局资源管理器
RES_MANAGER = ResourceManager()

# ==========================================
# 资源定义
# ==========================================

# 复选框仍使用 Base64 (Qt默认样式对PNG Base64支持较好)
ICON_CHECK_WHITE = 'url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAABmJLR0QA/wD/AP+gvaeTAAAAbklEQVQ4je2S0Q3AIAxD78M6Q3dI958u0g3Sqf+0Ug0i+sA5X6wEcOzYJAn8M2RInz1CDsCR09puvZaF/CAlhHnN77y2CwBExN3+rNfS6iBvB3k7yN9B/g7+d/C/g/8d/O/g/8d/O/g/8d/O/g7yDvIC7wB/k0315YVvOAAAAAASUVORK5CYII=")'

# 获取箭头图标的物理文件路径 (最稳健方案)
ICON_UP_PATH = RES_MANAGER.get_icon_url("up")
ICON_DOWN_PATH = RES_MANAGER.get_icon_url("down")


# ==========================================
# 辅助函数与核心逻辑 (保持不变)
# ==========================================
def print_progress(current, total, message=""):
    if sys.stdout is None: return
    bar_length = 30
    if total > 0:
        percent = float(current) / total
    else:
        percent = 0
    arrow = '▓' * int(round(percent * bar_length))
    spaces = '░' * (bar_length - len(arrow))
    sys.stdout.write(f"\r[{arrow}{spaces}] {int(percent * 100)}% ({current}/{total}) | {message}")
    sys.stdout.flush()

def calculate_history_limit(img):
    if img is None: return 20
    h, w = img.shape[:2]
    pixels = h * w
    base_limit = 20
    ref_pixels = 1920 * 1080 
    if pixels <= ref_pixels:
        return base_limit
    ratio = pixels / ref_pixels
    limit = int(base_limit / ratio)
    return max(3, limit)

def resource_path(relative_path):
    """ 获取资源的绝对路径，适配 PyInstaller 打包后的路径 """
    try:
        # PyInstaller 会创建一个临时文件夹，并将路径存储在 _MEIPASS 中
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

class ImageProcessingThread(QThread):
    result_ready = pyqtSignal(int, object, object) 

    def __init__(self, cv_img, index, params):
        super().__init__()
        self.cv_img = cv_img.copy() if cv_img is not None else None
        self.index = index
        self.params = params

    def run(self):
        if self.cv_img is None:
             self.result_ready.emit(self.index, None, None)
             return
        try:
            method = self.params.get('method', 'bilateral')
            s_space = float(self.params.get('sigma_space', 25))
            s_color = float(self.params.get('sigma_color', 15))
            
            if method == 'html_hard':
                u_img = cv2.UMat(self.cv_img)
                pre_smooth = cv2.bilateralFilter(u_img, 5, s_color, s_space)
                if self.isInterruptionRequested(): self.result_ready.emit(self.index, None, None); return
                
                gray_u = cv2.cvtColor(pre_smooth, cv2.COLOR_BGR2GRAY)
                grad_x = cv2.Sobel(gray_u, cv2.CV_32F, 1, 0, ksize=3)
                grad_y = cv2.Sobel(gray_u, cv2.CV_32F, 0, 1, ksize=3)
                abs_grad_x = cv2.convertScaleAbs(grad_x)
                abs_grad_y = cv2.convertScaleAbs(grad_y)
                magnitude = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                
                core_thresh = self.params.get('core_edge_thresh', 50)
                _, edge_mask = cv2.threshold(magnitude, core_thresh, 255, cv2.THRESH_BINARY)
                
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
                edge_mask_np = edge_mask.get()
                edge_mask_np = cv2.morphologyEx(edge_mask_np, cv2.MORPH_CLOSE, kernel)
                
                img_np = pre_smooth.get()
                flat_mask = cv2.bitwise_not(edge_mask_np)
                
                num_labels, labels = cv2.connectedComponents(flat_mask, connectivity=4)
                if self.isInterruptionRequested(): self.result_ready.emit(self.index, None, None); return

                flat_labels = labels.flatten()
                flat_img = img_np.reshape(-1, 3)
                
                counts = np.bincount(flat_labels, minlength=num_labels)
                counts[counts == 0] = 1 
                
                sum_b = np.bincount(flat_labels, weights=flat_img[:, 0], minlength=num_labels)
                sum_g = np.bincount(flat_labels, weights=flat_img[:, 1], minlength=num_labels)
                sum_r = np.bincount(flat_labels, weights=flat_img[:, 2], minlength=num_labels)
                
                mean_b = (sum_b / counts).astype(np.uint8)
                mean_g = (sum_g / counts).astype(np.uint8)
                mean_r = (sum_r / counts).astype(np.uint8)
                
                mean_colors = np.stack((mean_b, mean_g, mean_r), axis=1)
                result_np = mean_colors[labels]
                
                edge_indices = (flat_mask == 0)
                result_np[edge_indices] = img_np[edge_indices]
                result_u = cv2.UMat(result_np)

            else:
                u_img = cv2.UMat(self.cv_img)
                result_u = u_img
                if method == 'bilateral':
                    for _ in range(2):
                        result_u = cv2.bilateralFilter(result_u, 9, s_color, s_space)
                elif method == 'meanshift':
                    temp_img_np = u_img.get()
                    result_np = cv2.pyrMeanShiftFiltering(temp_img_np, sp=float(s_space), sr=float(s_color), maxLevel=1)
                    result_u = cv2.UMat(result_np)

            if self.isInterruptionRequested(): self.result_ready.emit(self.index, None, None); return

            result_np = result_u.get()
            if self.params.get('enable_kmeans', False):
                k = self.params.get('k_value', 8)
                z = result_np.reshape((-1, 3)).astype(np.float32)
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
                ret, label, center = cv2.kmeans(z, int(k), None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
                center = np.uint8(center)
                res = center[label.flatten()]
                result_np = res.reshape((result_np.shape))
                result_u = cv2.UMat(result_np)

            edge_mode = self.params.get('edge_mode', 'none')
            final_edges_mask = None 
            if edge_mode != 'none':
                gray_final = cv2.cvtColor(result_u, cv2.COLOR_BGR2GRAY)
                edges_out = None
                t1 = self.params.get('canny_t1', 50)
                t2 = self.params.get('canny_t2', 150)
                
                if edge_mode == 'canny':
                    edges_out = cv2.Canny(gray_final, t1, t2)
                elif edge_mode == 'sobel': 
                    grad_x = cv2.Sobel(gray_final, cv2.CV_32F, 1, 0, ksize=3)
                    grad_y = cv2.Sobel(gray_final, cv2.CV_32F, 0, 1, ksize=3)
                    abs_grad_x = cv2.convertScaleAbs(grad_x)
                    abs_grad_y = cv2.convertScaleAbs(grad_y)
                    magnitude = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
                    _, edges_thresh = cv2.threshold(magnitude, t1, 255, cv2.THRESH_BINARY)
                    edges_out = cv2.UMat(edges_thresh)
                elif edge_mode == 'adaptive':
                    gray_np_final = gray_final.get()
                    b_size = t1 if t1 % 2 == 1 else t1 + 1
                    if b_size < 3: b_size = 3
                    edges_np = cv2.adaptiveThreshold(gray_np_final, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, b_size, t2)
                    edges_out = cv2.UMat(edges_np)

                if edges_out is not None:
                    final_edges_mask = edges_out.get() 

            final_image_np = result_u.get()
            self.result_ready.emit(self.index, final_image_np, final_edges_mask)

        except Exception as e:
            print(f"Processing Error in thread {self.index}: {e}")
            self.result_ready.emit(self.index, None, None)

class DraggableColorPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setFixedSize(162, 100)
        self.is_dragging = False; self.drag_start_pos = QPoint()
        self.row_in = self.create_color_row("IN")
        self.row_out = self.create_color_row("OUT")
        layout = QVBoxLayout(self); layout.setContentsMargins(0, 0, 0, 0)
        self.container = QFrame(); self.container.setObjectName("ColorPanelFrame")
        self.container.setStyleSheet("""
            #ColorPanelFrame { background-color: rgba(30, 30, 30, 200); border: 1px solid #555; border-radius: 8px; }
            QLabel { background-color: transparent; }
        """)
        inner_layout = QVBoxLayout(self.container)
        lbl_title = QLabel("Color Inspector"); lbl_title.setStyleSheet("color: #aaa; font-size: 10px; font-weight: bold;"); lbl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        inner_layout.addWidget(lbl_title); inner_layout.addLayout(self.row_in['layout']); inner_layout.addLayout(self.row_out['layout'])
        layout.addWidget(self.container)

    def create_color_row(self, label_text):
        row_layout = QHBoxLayout()
        lbl = QLabel(label_text); lbl.setFixedWidth(30); lbl.setStyleSheet("color: #ccc; font-weight: bold;")
        swatch = QLabel(); swatch.setFixedSize(24, 16); swatch.setStyleSheet("background-color: transparent; border: 1px solid #555;")
        hex_val = QLabel("-------"); hex_val.setStyleSheet("color: #fff; font-family: Consolas;")
        row_layout.addWidget(lbl); row_layout.addWidget(swatch); row_layout.addWidget(hex_val); row_layout.addStretch()
        return {'layout': row_layout, 'swatch': swatch, 'hex': hex_val}

    def update_colors(self, c1_bgr, c2_bgr):
        def set_row(row_dict, bgr):
            if bgr is not None:
                qcol = QColor(int(bgr[2]), int(bgr[1]), int(bgr[0]))
                hex_str = qcol.name().upper()
                row_dict['swatch'].setStyleSheet(f"background-color: {hex_str}; border: 1px solid #888; border-radius: 2px;")
                row_dict['hex'].setText(hex_str)
            else:
                row_dict['swatch'].setStyleSheet("background-color: transparent; border: 1px solid #555;"); row_dict['hex'].setText("-------")
        set_row(self.row_in, c1_bgr); set_row(self.row_out, c2_bgr)

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self.is_dragging = True; self.drag_start_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft(); event.accept()
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_dragging:
            self.move(event.globalPosition().toPoint() - self.drag_start_pos); event.accept()
    def mouseReleaseEvent(self, event: QMouseEvent): self.is_dragging = False

class OverlayControls(QWidget):
    toggle_signal = pyqtSignal(bool); reset_signal = pyqtSignal(); zoom_100_signal = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QHBoxLayout(self); layout.setContentsMargins(5, 5, 5, 5); layout.setSpacing(5)
        btn_style = """
            QPushButton { background-color: rgba(40, 40, 40, 200); color: white; border: 1px solid #555; border-radius: 4px; padding: 4px 10px; font-size: 12px; }
            QPushButton:hover { background-color: rgba(60, 60, 60, 230); border-color: #0078d4; }
            QPushButton:checked { background-color: #0078d4; border-color: #0078d4; }
            QPushButton:pressed { background-color: #0078d4; border-color: #0078d4; color: white; }
        """
        self.btn_reset = QPushButton("复位"); self.btn_reset.setStyleSheet(btn_style); self.btn_reset.clicked.connect(self.reset_signal.emit)
        self.btn_100 = QPushButton("100%"); self.btn_100.setStyleSheet(btn_style); self.btn_100.clicked.connect(self.zoom_100_signal.emit)
        self.btn_toggle = QPushButton("当前: 原图"); self.btn_toggle.setCheckable(True); self.btn_toggle.setStyleSheet(btn_style); self.btn_toggle.clicked.connect(self.on_toggle_clicked)
        layout.addWidget(self.btn_reset); layout.addWidget(self.btn_100); layout.addWidget(self.btn_toggle)

    def on_toggle_clicked(self):
        is_processed = self.btn_toggle.isChecked()
        self.btn_toggle.setText("当前: 结果" if is_processed else "当前: 原图")
        self.toggle_signal.emit(is_processed)
    def set_toggle_state(self, checked):
        self.btn_toggle.setChecked(checked); self.btn_toggle.setText("当前: 结果" if checked else "当前: 原图")

class UndoControls(QWidget):
    undo_signal = pyqtSignal(); redo_signal = pyqtSignal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        layout = QHBoxLayout(self); layout.setContentsMargins(5, 5, 5, 5); layout.setSpacing(5)
        btn_style = """
            QPushButton { background-color: rgba(40, 40, 40, 200); color: white; border: 1px solid #555; border-radius: 4px; padding: 4px 10px; font-size: 12px; }
            QPushButton:hover { background-color: rgba(60, 60, 60, 230); border-color: #0078d4; }
            QPushButton:pressed { background-color: #0078d4; border-color: #0078d4; color: white; }
            QPushButton:disabled { background-color: rgba(30, 30, 30, 150); color: #888; border-color: #444; }
        """
        self.btn_undo = QPushButton("↩ 撤销"); self.btn_undo.setStyleSheet(btn_style); self.btn_undo.clicked.connect(self.undo_signal.emit); self.btn_undo.setToolTip("返回上一次处理结果 (Ctrl+Z)"); self.btn_undo.setEnabled(False)
        self.btn_redo = QPushButton("↪ 恢复"); self.btn_redo.setStyleSheet(btn_style); self.btn_redo.clicked.connect(self.redo_signal.emit); self.btn_redo.setToolTip("重做已撤销的操作 (Ctrl+Shift+Z / Ctrl+Y)"); self.btn_redo.setEnabled(False)
        layout.addWidget(self.btn_undo); layout.addWidget(self.btn_redo)
    def update_states(self, can_undo, can_redo):
        self.btn_undo.setEnabled(can_undo); self.btn_redo.setEnabled(can_redo)

class SyncImageViewer(QWidget):
    mouse_hover = pyqtSignal(int, int) 
    def __init__(self, parent=None):
        super().__init__(parent)
        self.img_pixmap = None; self.scale_factor = 1.0; self.offset = QPointF(0, 0)
        self.is_dragging = False; self.last_mouse_pos = QPointF(0, 0); self.setMouseTracking(True); self.bg_color = QColor("#1e1e1e")
        self.controls = OverlayControls(self); self.controls.show()
        self.undo_controls = UndoControls(self); self.undo_controls.show()

    def set_image(self, cv_img):
        if cv_img is None: self.img_pixmap = None; self.update(); return
        h, w, ch = cv_img.shape; bytes_per_line = ch * w
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB).copy() 
        qt_img = QImage(rgb_img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.img_pixmap = QPixmap.fromImage(qt_img); self.update()

    def resizeEvent(self, event):
        cw = self.controls.width(); self.controls.move(self.width() - cw - 10, 10); self.undo_controls.move(10, 10); super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing, False); painter.fillRect(self.rect(), self.bg_color)
        if self.img_pixmap and not self.img_pixmap.isNull():
            cx, cy = self.width() / 2, self.height() / 2
            painter.translate(cx + self.offset.x(), cy + self.offset.y())
            painter.scale(self.scale_factor, self.scale_factor)
            x = -self.img_pixmap.width() / 2; y = -self.img_pixmap.height() / 2
            painter.drawPixmap(QPointF(x, y), self.img_pixmap)
            pen = QPen(QColor(255, 255, 255, 50)); pen.setWidthF(1.0 / self.scale_factor); painter.setPen(pen); painter.drawRect(QRectF(x, y, self.img_pixmap.width(), self.img_pixmap.height()))

    def wheelEvent(self, event: QWheelEvent):
        zoom_in = event.angleDelta().y() > 0; multiplier = 1.1 if zoom_in else 0.9
        mouse_pos = event.position(); center = QPointF(self.width() / 2, self.height() / 2)
        delta = mouse_pos - center; old_scale = self.scale_factor; new_scale = old_scale * multiplier
        new_scale = max(0.05, min(50.0, new_scale)); factor = new_scale / old_scale
        self.offset = delta - (delta - self.offset) * factor; self.scale_factor = new_scale; self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton: self.is_dragging = True; self.last_mouse_pos = event.position(); self.setCursor(Qt.CursorShape.ClosedHandCursor)
    def mouseMoveEvent(self, event: QMouseEvent):
        if self.is_dragging:
            delta = event.position() - self.last_mouse_pos; self.offset += delta; self.last_mouse_pos = event.position(); self.update()
        if self.img_pixmap:
            cx, cy = self.width() / 2, self.height() / 2
            local_x = (event.position().x() - (cx + self.offset.x())) / self.scale_factor
            local_y = (event.position().y() - (cy + self.offset.y())) / self.scale_factor
            img_x = int(local_x + self.img_pixmap.width() / 2); img_y = int(local_y + self.img_pixmap.height() / 2)
            self.mouse_hover.emit(img_x, img_y)
        else: self.mouse_hover.emit(-1, -1)
    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton: self.is_dragging = False; self.setCursor(Qt.CursorShape.ArrowCursor)
    def reset_view_center(self): self.offset = QPointF(0, 0); self.update()
    def set_100_percent_zoom(self): self.scale_factor = 1.0; self.offset = QPointF(0, 0); self.update()

# ==========================================
# 主窗口
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("MinimalistStylizer")
        self.resize(1600, 950)
        # [新增] 设置运行时的窗口图标
        # 这里的 "app_icon.ico" 必须与 main.py 在同一目录下，或者打包时已包含
        icon_path = resource_path("app_icon.ico")
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        self.images_data = []; self.current_tab_index = -1; self.processing_threads = {}
        self.batch_queue = []; self.batch_total = 0; self.batch_finished = 0
        self.stop_requested = False; self.param_block_signal = False; self.batch_edge_mode_state = 0
        
        self.init_icons()
        self.init_ui()
        self.apply_theme() 
        self.init_shortcuts() 
        
        self.color_panel = DraggableColorPanel(self) 
        self.color_panel.show()
        geom = self.geometry()
        self.color_panel.move(geom.x() + 360, geom.y() + 100)

    def init_shortcuts(self):
        QShortcut(QKeySequence("Ctrl+Z"), self, self.undo_current_image)
        QShortcut(QKeySequence("Ctrl+Y"), self, self.redo_current_image)
        QShortcut(QKeySequence("Ctrl+Shift+Z"), self, self.redo_current_image)

    def init_icons(self):
        self.icon_overlay = QIcon(QPixmap(QImage.fromData(b'<?xml version="1.0" encoding="UTF-8"?><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="4" width="16" height="16" rx="2" fill="#0078d4" stroke="white" stroke-width="2"/></svg>')))
        self.icon_clean = QIcon(QPixmap(QImage.fromData(b'<?xml version="1.0" encoding="UTF-8"?><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="4" width="16" height="16" rx="2" fill="#0078d4"/></svg>')))
        self.icon_edge = QIcon(QPixmap(QImage.fromData(b'<?xml version="1.0" encoding="UTF-8"?><svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><rect x="4" y="4" width="16" height="16" rx="2" stroke="white" stroke-width="2"/></svg>')))

    def init_ui(self):
        main_widget = QWidget(); self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget); main_layout.setContentsMargins(0, 0, 0, 0); main_layout.setSpacing(0)

        # ================= LEFT SIDEBAR =================
        sidebar_left = QFrame(); sidebar_left.setFixedWidth(340); sidebar_left.setObjectName("sidebar")
        left_layout = QVBoxLayout(sidebar_left); left_layout.setContentsMargins(0, 0, 0, 0); left_layout.setSpacing(0)

        scroll_controls = QScrollArea(); scroll_controls.setWidgetResizable(True); scroll_controls.setFrameShape(QFrame.Shape.NoFrame)
        scroll_content = QWidget(); scroll_layout = QVBoxLayout(scroll_content); scroll_layout.setContentsMargins(10, 10, 10, 10)

        algo_box = QGroupBox("核心算法 (Core Algorithm)"); algo_layout = QVBoxLayout()
        self.combo_algo = QComboBox(); self.combo_algo.addItems(["硬边缘 (Sobel+连通域 - 推荐)", "双边滤波 (柔和)", "均值漂移 (旧版)"])
        self.combo_algo.setToolTip("<p><b>1. 硬边缘:</b> 模拟HTML版插画算法，物理防溢色，边缘锐利。</p><p><b>2. 双边滤波:</b> 传统的保边去噪算法。</p><p><b>3. 均值漂移:</b> 油画涂抹感强。</p>")
        self.combo_algo.currentIndexChanged.connect(self.update_param_labels) 
        algo_layout.addWidget(self.combo_algo)
        
        self.core_edge_container = QWidget(); core_layout = QVBoxLayout(self.core_edge_container); core_layout.setContentsMargins(0, 5, 0, 5)
        lbl_core = QLabel("防溢色阈值 (Wall Thresh):"); lbl_core.setToolTip("仅对【硬边缘】算法有效。值越大，墙壁越少，越容易溢色；值越小，墙壁越密，噪点越多。"); self.core_edge_container.setToolTip(lbl_core.toolTip())
        core_input_layout = QHBoxLayout()
        self.slider_core_edge = QSlider(Qt.Orientation.Horizontal); self.slider_core_edge.setRange(1, 255); self.slider_core_edge.setValue(50)
        self.inp_core_edge = QSpinBox(); self.inp_core_edge.setRange(1, 255); self.inp_core_edge.setValue(50); self.inp_core_edge.setFixedWidth(60); self.inp_core_edge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_core_edge.valueChanged.connect(self.inp_core_edge.setValue); self.inp_core_edge.valueChanged.connect(self.slider_core_edge.setValue)
        core_input_layout.addWidget(self.slider_core_edge); core_input_layout.addWidget(self.inp_core_edge)
        core_layout.addWidget(lbl_core); core_layout.addLayout(core_input_layout)
        algo_layout.addWidget(self.core_edge_container)

        btn_reset_params = QPushButton("重置默认参数"); btn_reset_params.setStyleSheet("margin-top: 5px; font-size: 11px;"); btn_reset_params.clicked.connect(self.reset_parameters)
        algo_layout.addWidget(btn_reset_params); algo_box.setLayout(algo_layout); scroll_layout.addWidget(algo_box)
        
        def create_slider_input(label_text, min_val, max_val, default_val, tooltip=""):
            container = QWidget(); layout = QVBoxLayout(container); layout.setContentsMargins(0, 5, 0, 5); layout.setSpacing(2)
            lbl_layout = QHBoxLayout(); lbl = QLabel(label_text)
            if tooltip: lbl.setToolTip(tooltip); container.setToolTip(tooltip)
            lbl_layout.addWidget(lbl)
            inp = QSpinBox(); inp.setRange(min_val, max_val); inp.setValue(default_val); inp.setFixedWidth(60); inp.setAlignment(Qt.AlignmentFlag.AlignCenter)
            input_layout = QHBoxLayout(); slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(min_val, max_val); slider.setValue(default_val)
            if tooltip: slider.setToolTip(tooltip)
            slider.valueChanged.connect(inp.setValue); inp.valueChanged.connect(slider.setValue)
            input_layout.addWidget(slider); input_layout.addWidget(inp); layout.addLayout(lbl_layout); layout.addLayout(input_layout)
            return container, slider, inp

        group_smooth = QGroupBox("1. 颜色归一化"); gs_layout = QVBoxLayout()
        self.c_s1, self.slider_s1, self.inp_s1 = create_slider_input("空间半径 (Space):", 1, 100, 25, "")
        self.c_s2, self.slider_s2, self.inp_s2 = create_slider_input("颜色半径 (Color):", 1, 100, 15, "")
        gs_layout.addWidget(self.c_s1); gs_layout.addWidget(self.c_s2); group_smooth.setLayout(gs_layout); scroll_layout.addWidget(group_smooth)

        group_kmeans = QGroupBox("2. 极简风去噪 (K-Means)"); gk_layout = QVBoxLayout()
        self.chk_kmeans = QPushButton("启用颜色量化"); self.chk_kmeans.setCheckable(True); self.chk_kmeans.setChecked(True); self.chk_kmeans.setToolTip("<p>强制将全图颜色限制在固定的数量内。</p>")
        k_layout_container = QWidget(); k_vlayout = QVBoxLayout(k_layout_container); k_vlayout.setContentsMargins(0,0,0,0)
        k_head_layout = QHBoxLayout(); k_head_layout.addWidget(QLabel("色块数量 (K):"))
        btn_rec_k = QPushButton("推荐"); btn_rec_k.setFixedWidth(40); btn_rec_k.setFixedHeight(18); btn_rec_k.setStyleSheet("font-size:10px; padding:0px;") 
        btn_rec_k.clicked.connect(self.auto_recommend_k); btn_rec_k.setToolTip("<p style='font-size:13px; font-weight:normal;'>分析图像直方图，自动推荐最能保留 95% 色彩信息的色块数量。</p>") 
        k_head_layout.addWidget(btn_rec_k); k_vlayout.addLayout(k_head_layout)
        self.inp_k = QSpinBox(); self.inp_k.setRange(2, 64); self.inp_k.setValue(8); self.inp_k.setFixedWidth(60)
        self.slider_k = QSlider(Qt.Orientation.Horizontal); self.slider_k.setRange(2, 64); self.slider_k.setValue(8)
        self.inp_k.valueChanged.connect(self.slider_k.setValue); self.slider_k.valueChanged.connect(self.inp_k.setValue)
        k_input_layout = QHBoxLayout(); k_input_layout.addWidget(self.slider_k); k_input_layout.addWidget(self.inp_k)
        k_vlayout.addLayout(k_input_layout); gk_layout.addWidget(self.chk_kmeans); gk_layout.addWidget(k_layout_container); group_kmeans.setLayout(gk_layout); scroll_layout.addWidget(group_kmeans)

        group_edge = QGroupBox("3. 边缘增强 & 防溢色"); group_edge.setToolTip("<p><b>边缘检测作用:</b> 仅用于生成覆盖在色块上方的黑色描边层，<b>不再影响核心色块形状</b>。</p>")
        ge_layout = QVBoxLayout()
        self.combo_edge = QComboBox(); self.combo_edge.addItems(["Sobel (防溢色/厚重)", "Canny (细线)", "Lineart (自适应)", "无"]); self.combo_edge.currentIndexChanged.connect(self.toggle_edge_params)
        self.edge_params_container = QWidget(); ep_layout = QVBoxLayout(self.edge_params_container); ep_layout.setContentsMargins(0,0,0,0)
        self.c_e1, self.slider_e1, self.inp_e1 = create_slider_input("边缘阈值 (Threshold):", 1, 255, 50, "")
        self.c_e2, self.slider_e2, self.inp_e2 = create_slider_input("辅助阈值 (High):", 1, 255, 150, "")
        ep_layout.addWidget(self.c_e1); ep_layout.addWidget(self.c_e2)
        ge_layout.addWidget(self.combo_edge); ge_layout.addWidget(self.edge_params_container); group_edge.setLayout(ge_layout); scroll_layout.addWidget(group_edge)

        scroll_layout.addStretch(); scroll_controls.setWidget(scroll_content); left_layout.addWidget(scroll_controls)

        group_io = QGroupBox("文件操作"); io_layout = QVBoxLayout(); io_layout.setContentsMargins(10, 10, 10, 10)
        btn_import = QPushButton(" 导入图片 (批量)"); btn_import.setIcon(QIcon.fromTheme("document-open")); btn_import.setFixedHeight(35); btn_import.clicked.connect(self.import_images)
        btn_batch_save = QPushButton("导出已选中"); btn_batch_save.setToolTip("根据当前勾选的图片进行导出。如果图片处于【仅描边】模式，将导出透明背景的PNG。"); btn_batch_save.clicked.connect(self.batch_export_images)
        io_layout.addWidget(btn_import); io_layout.addWidget(btn_batch_save); group_io.setLayout(io_layout); left_layout.addWidget(group_io)
        main_layout.addWidget(sidebar_left)

        # ================= CENTER VIEW =================
        center_frame = QFrame(); center_layout = QVBoxLayout(center_frame); center_layout.setContentsMargins(0, 0, 0, 0)
        self.image_viewer = SyncImageViewer(self); self.image_viewer.mouse_hover.connect(self.handle_pixel_hover)
        self.image_viewer.controls.reset_signal.connect(self.image_viewer.reset_view_center)
        self.image_viewer.controls.zoom_100_signal.connect(self.image_viewer.set_100_percent_zoom)
        self.image_viewer.controls.toggle_signal.connect(self.on_view_toggle_changed)
        self.image_viewer.undo_controls.undo_signal.connect(self.undo_current_image)
        self.image_viewer.undo_controls.redo_signal.connect(self.redo_current_image)
        center_layout.addWidget(self.image_viewer); main_layout.addWidget(center_frame)

        # ================= RIGHT SIDEBAR =================
        sidebar_right = QFrame(); sidebar_right.setFixedWidth(300); sidebar_right.setObjectName("sidebar_right")
        right_layout = QVBoxLayout(sidebar_right); right_layout.setContentsMargins(10, 10, 10, 10)

        layer_group = QGroupBox("图层与批处理"); layer_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding); self.layer_group_layout = QVBoxLayout()
        self.layer_table = QTableWidget(); self.layer_table.setColumnCount(3); self.layer_table.horizontalHeader().hide(); self.layer_table.verticalHeader().hide()
        self.layer_table.setShowGrid(False); self.layer_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows); self.layer_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.layer_table.setFocusPolicy(Qt.FocusPolicy.NoFocus); self.layer_table.setColumnWidth(0, 40); self.layer_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch); self.layer_table.setColumnWidth(2, 40) 
        self.layer_table.cellClicked.connect(self.on_table_cell_clicked)
        self.layer_group_layout.addWidget(self.layer_table)
        
        sel_btn_layout = QHBoxLayout()
        btn_sel_all = QPushButton("全选"); btn_sel_all.clicked.connect(lambda: self.set_all_layers_checked(True))
        btn_sel_none = QPushButton("反选"); btn_sel_none.clicked.connect(self.invert_layer_selection)
        self.btn_batch_toggle_mode = QPushButton(); self.btn_batch_toggle_mode.setFixedSize(30, 26); self.btn_batch_toggle_mode.setIcon(self.icon_overlay)
        self.btn_batch_toggle_mode.setToolTip("一键切换选中图层的显示模式 (叠加 -> 纯色 -> 描边)"); self.btn_batch_toggle_mode.clicked.connect(self.batch_toggle_edge_mode)
        btn_del = QPushButton("删除"); btn_del.clicked.connect(self.delete_selected_layers)
        sel_btn_layout.addWidget(btn_sel_all); sel_btn_layout.addWidget(btn_sel_none); sel_btn_layout.addWidget(self.btn_batch_toggle_mode); sel_btn_layout.addWidget(btn_del)
        self.layer_group_layout.addLayout(sel_btn_layout)
        layer_group.setLayout(self.layer_group_layout); right_layout.addWidget(layer_group)
        
        self.btn_process = QPushButton("执行处理 (选中的图片)"); self.btn_process.setFixedHeight(60)
        self.btn_process.setStyleSheet("QPushButton { background-color: #0078d4; color: white; font-size: 14px; font-weight: bold; border-radius: 5px; } QPushButton:hover { background-color: #1084e0; } QPushButton:pressed { background-color: #005a9e; } QPushButton:disabled { background-color: #555; color: #aaa; }")
        self.btn_process.clicked.connect(self.toggle_processing); right_layout.addWidget(self.btn_process); main_layout.addWidget(sidebar_right)
        
        self.update_param_labels(0); self.toggle_edge_params(0)

    def apply_theme(self):
        # 定义统一颜色
        BLUE_ACCENT = "#0078d4"
        BLUE_HOVER = "#1084e0"
        BLUE_PRESSED = "#005a9e"
        DARK_BG = "#202020"
        PANEL_BG = "#333333"
        BORDER_COL = "#555555"
        TEXT_COL = "#e0e0e0"

        # 使用 f-string 注入变量，确保图标变量被正确替换
        qss = f"""
        QMainWindow, QWidget {{ 
            background-color: {DARK_BG}; 
            color: {TEXT_COL}; 
            font-family: "Segoe UI", "Microsoft YaHei"; 
            selection-background-color: {BLUE_ACCENT};
            selection-color: white;
        }}
        
        #sidebar, #sidebar_right {{ 
            background-color: #2b2b2b; 
            border-right: 1px solid #1f1f1f; 
            border-left: 1px solid #1f1f1f; 
        }}
        
        QGroupBox {{ 
            border: 1px solid #444; 
            margin-top: 8px; 
            padding-top: 10px; 
            font-weight: bold; 
            border-radius: 4px; 
        }}
        QGroupBox::title {{ 
            subcontrol-origin: margin; 
            subcontrol-position: top left; 
            padding: 0 5px; 
            background-color: #2b2b2b; 
        }}
        
        QTableWidget {{ background-color: {PANEL_BG}; border: 1px solid {BORDER_COL}; color: #eee; outline: none; }}
        QTableWidget::item {{ padding: 2px; border-bottom: 1px solid #3a3a3a; }}
        QTableWidget::item:selected {{ background-color: #444; border: 1px solid {BLUE_ACCENT}; }}
        
        QCheckBox {{ spacing: 0px; margin-left: 10px; }}
        QCheckBox::indicator {{ width: 18px; height: 18px; border: 1px solid #666; background: {PANEL_BG}; border-radius: 3px; }}
        QCheckBox::indicator:hover {{ border-color: #888; background: #444; }}
        QCheckBox::indicator:checked {{ background-color: {BLUE_ACCENT}; border-color: {BLUE_ACCENT}; image: {ICON_CHECK_WHITE}; }}
        
        /* 修复 QSpinBox 的样式与箭头 */
        QSpinBox, QDoubleSpinBox {{ 
            background-color: {PANEL_BG}; 
            border: 1px solid {BORDER_COL}; 
            color: #eee; 
            padding: 4px; 
            border-radius: 4px; 
        }}
        
        QSpinBox:focus, QDoubleSpinBox:focus {{
            border: 1px solid {BLUE_ACCENT};
        }}

        QSpinBox::up-button, QDoubleSpinBox::up-button {{ 
            subcontrol-origin: border; 
            subcontrol-position: top right; 
            width: 16px; 
            border-left: 1px solid {BORDER_COL}; 
            border-bottom: 1px solid {BORDER_COL}; 
            background-color: #3a3a3a; 
            border-top-right-radius: 4px; 
            margin-top: 1px; 
            margin-right: 1px;
        }}
        
        QSpinBox::down-button, QDoubleSpinBox::down-button {{ 
            subcontrol-origin: border; 
            subcontrol-position: bottom right; 
            width: 16px; 
            border-left: 1px solid {BORDER_COL}; 
            border-top: 0px solid {BORDER_COL}; 
            background-color: #3a3a3a; 
            border-bottom-right-radius: 4px; 
            margin-bottom: 1px; 
            margin-right: 1px;
        }}
        
        /* [核心修复] 使用运行时生成的 SVG 文件路径 */
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ 
            image: {ICON_UP_PATH}; 
            width: 12px; 
            height: 12px; 
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ 
            image: {ICON_DOWN_PATH}; 
            width: 12px; 
            height: 12px; 
        }}

        /* 统一的按钮 Hover 状态 (蓝色边框/高亮) */
        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover, 
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{ 
            background-color: #444;
            border-left: 1px solid {BLUE_HOVER};
        }}

        /* 统一的按钮 Pressed 状态 (蓝色背景) */
        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed, 
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {{ 
            background-color: {BLUE_ACCENT}; 
        }}
        
        QLineEdit, QComboBox {{ background-color: {PANEL_BG}; border: 1px solid {BORDER_COL}; color: #eee; padding: 4px; border-radius: 4px; }}
        QLineEdit:focus, QComboBox:focus {{ border: 1px solid {BLUE_ACCENT}; }}

        QToolTip {{ font-size: 13px; font-family: "Microsoft YaHei"; color: #ffffff; background-color: {PANEL_BG}; border: 1px solid {BLUE_ACCENT}; padding: 5px; border-radius: 3px; }}
        
        QPushButton {{ background-color: #3a3a3a; border: 1px solid {BORDER_COL}; padding: 6px 12px; border-radius: 4px; }}
        QPushButton:hover {{ background-color: #454545; border-color: {BLUE_HOVER}; }}
        QPushButton:checked {{ background-color: {BLUE_ACCENT}; border-color: {BLUE_PRESSED}; color: white; }}
        QPushButton:pressed {{ background-color: {BLUE_PRESSED}; border-color: {BLUE_PRESSED}; color: white; }}
        """
        self.setStyleSheet(qss)
    
    def closeEvent(self, event):
        # 退出时清理临时文件
        RES_MANAGER.cleanup()
        super().closeEvent(event)

    def _update_param_ui(self, container, label_text, tooltip, slider, spinbox):
        label = container.findChild(QLabel)
        if label:
            label.setText(label_text)
            label.setToolTip(tooltip)
        container.setToolTip(tooltip)
        slider.setToolTip(tooltip)
        spinbox.setToolTip(tooltip)

    def toggle_edge_params(self, index):
        self.edge_params_container.setVisible(index < 3) # 3 is None
        
        # 默认显示两个参数
        self.c_e1.setVisible(True)
        self.c_e2.setVisible(True)
        
        if index == 0: # Sobel
            self._update_param_ui(self.c_e1, "Sobel 阈值:", 
                                  "<p><b>Sobel 阈值：</b>控制黑色描边线条的灵敏度。</p>"
                                  "<ul>"
                                  "<li><b>值越小：</b>线条越密集、厚重，保留更多细节纹理。</li>"
                                  "<li><b>值越大：</b>线条越稀疏、干净，仅保留主要轮廓。</li>"
                                  "</ul>"
                                  "<p><i>注：此参数仅影响后期黑线叠加，不影响色块形状。</i></p>", 
                                  self.slider_e1, self.inp_e1)            # 隐藏无效参数
            self.c_e2.setVisible(False)
            self.inp_e1.setValue(50)
            
        elif index == 1: # Canny
            self._update_param_ui(self.c_e1, "低阈值 (Low):", 
                                  "Canny 边缘检测的低阈值。\n低于此值的像素点会被丢弃 (弱边缘)。", 
                                  self.slider_e1, self.inp_e1)
            self._update_param_ui(self.c_e2, "高阈值 (High):", 
                                  "Canny 边缘检测的高阈值。\n高于此值的像素点被认为是强边缘 (一定会保留)。", 
                                  self.slider_e2, self.inp_e2)
            self.inp_e1.setValue(50)
            self.inp_e2.setValue(150)
            
        elif index == 2: # Lineart
            self._update_param_ui(self.c_e1, "块大小 (Block):", 
                                  "Lineart 自适应阈值的邻域块大小 (必须为奇数)。\n决定了局部对比度的计算范围。", 
                                  self.slider_e1, self.inp_e1)
            self._update_param_ui(self.c_e2, "常数 C:", 
                                  "从平均值中减去的常数。\n用于微调黑白分界线，去除背景噪点。", 
                                  self.slider_e2, self.inp_e2)
            self.inp_e1.setValue(7)
            self.inp_e2.setValue(2)

    def update_param_labels(self, index):
        # 当核心算法不是硬边缘时，隐藏防溢色滑块
        self.core_edge_container.setVisible(index == 0)
        
        if index == 0: # HTML Hard
            self._update_param_ui(self.c_s1, "预处理平滑 (Space):", 
                                  "<p><b>Space Sigma (Blur):</b> 预处理时的模糊程度。值越大，细节越少，生成的色块越大。</p>", 
                                  self.slider_s1, self.inp_s1)
            self._update_param_ui(self.c_s2, "颜色相似度 (Color):", 
                                  "<p><b>Color Sigma (Sim):</b> 颜色归类的容差。值越小，保留更多颜色细节；值越大，颜色越单一。</p>", 
                                  self.slider_s2, self.inp_s2)
        elif index == 1: # Bilateral
            self._update_param_ui(self.c_s1, "双边直径 (Diameter):", 
                                  "<p><b>Diameter/Space:</b> 滤波时的像素邻域直径。直接决定磨皮/平滑的范围。</p>", 
                                  self.slider_s1, self.inp_s1)
            self._update_param_ui(self.c_s2, "色彩权重 (Sigma Color):", 
                                  "<p><b>Sigma Color:</b> 颜色混合的阈值。值越大，差异大的颜色也会被混合。</p>", 
                                  self.slider_s2, self.inp_s2)
        elif index == 2: # MeanShift
            self._update_param_ui(self.c_s1, "物理半径 (Spatial):", 
                                  "<p><b>Spatial Radius (sp):</b> 均值漂移的物理空间窗口大小。</p>", 
                                  self.slider_s1, self.inp_s1)
            self._update_param_ui(self.c_s2, "色差半径 (Color):", 
                                  "<p><b>Color Radius (sr):</b> 均值漂移的颜色空间窗口大小。决定涂抹感强弱。</p>", 
                                  self.slider_s2, self.inp_s2)

    def reset_parameters(self):
        self.param_block_signal = True
        self.combo_algo.setCurrentIndex(0) 
        self.inp_s1.setValue(25)
        self.inp_s2.setValue(15)
        self.inp_core_edge.setValue(50) 
        self.chk_kmeans.setChecked(True)
        self.inp_k.setValue(8)
        self.combo_edge.setCurrentIndex(0) 
        self.inp_e1.setValue(50)
        self.inp_e2.setValue(150)
        self.param_block_signal = False
        print("所有参数已重置为默认值。")

    def auto_recommend_k(self):
        if self.current_tab_index < 0:
            QMessageBox.warning(self, "提示", "请先选择一张图片。")
            return
        img = self.images_data[self.current_tab_index]['original']
        if img is None: return
        try:
            small = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
            data = small.reshape(-1, 3).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            _, labels, _ = cv2.kmeans(data, 16, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            counts = np.bincount(labels.flatten())
            counts[::-1].sort()
            total_pixels = 100 * 100
            cumulative = 0
            rec_k = 0
            for c in counts:
                cumulative += c
                rec_k += 1
                if cumulative >= total_pixels * 0.95:
                    break
            self.inp_k.setValue(rec_k)
            QMessageBox.information(self, "推荐完成", f"根据当前图像内容，推荐 K 值设为: {rec_k}")
        except Exception as e:
            print(f"Recommendation failed: {e}")

    # ================= 历史记录核心逻辑 =================
    
    def get_current_params(self):
        """获取当前UI参数快照"""
        return {
            'method_idx': self.combo_algo.currentIndex(),
            'sigma_space': self.inp_s1.value(),
            'sigma_color': self.inp_s2.value(),
            'core_edge_thresh': self.inp_core_edge.value(), 
            'kmeans': self.chk_kmeans.isChecked(),
            'k_value': self.inp_k.value(),
            'edge_idx': self.combo_edge.currentIndex(),
            'edge_t1': self.inp_e1.value(),
            'edge_t2': self.inp_e2.value()
        }

    def apply_params(self, params):
        """将参数快照应用到UI，不触发重算"""
        self.param_block_signal = True
        try:
            self.combo_algo.setCurrentIndex(params.get('method_idx', 0))
            self.inp_s1.setValue(params.get('sigma_space', 25))
            self.inp_s2.setValue(params.get('sigma_color', 15))
            self.inp_core_edge.setValue(params.get('core_edge_thresh', 50)) 
            self.chk_kmeans.setChecked(params.get('kmeans', True))
            self.inp_k.setValue(params.get('k_value', 8))
            self.combo_edge.setCurrentIndex(params.get('edge_idx', 0))
            self.inp_e1.setValue(params.get('edge_t1', 50))
            self.inp_e2.setValue(params.get('edge_t2', 150))
        finally:
            self.param_block_signal = False

    def push_history(self, idx, processed_img, edges, params_snapshot):
        if idx < 0 or idx >= len(self.images_data): return
        data = self.images_data[idx]
        
        current_hist_idx = data['history_idx']
        if current_hist_idx < len(data['history']) - 1:
            data['history'] = data['history'][:current_hist_idx+1]
        
        snapshot = {
            'processed': processed_img,
            'edges': edges,
            'params': params_snapshot 
        }
        
        data['history'].append(snapshot)
        
        limit = calculate_history_limit(processed_img)
        if len(data['history']) > limit:
            data['history'].pop(0) 
        else:
            data['history_idx'] += 1 
            data['history_idx'] = len(data['history']) - 1

        self.update_undo_redo_buttons()

    def undo_current_image(self):
        idx = self.current_tab_index
        if idx < 0: return
        data = self.images_data[idx]
        
        if data['history_idx'] > 0:
            data['history_idx'] -= 1
            self.restore_snapshot(idx)

    def redo_current_image(self):
        idx = self.current_tab_index
        if idx < 0: return
        data = self.images_data[idx]
        
        if data['history_idx'] < len(data['history']) - 1:
            data['history_idx'] += 1
            self.restore_snapshot(idx)

    def restore_snapshot(self, idx):
        data = self.images_data[idx]
        hist_idx = data['history_idx']
        snapshot = data['history'][hist_idx]
        
        data['processed'] = snapshot['processed']
        data['edges'] = snapshot['edges']
        
        self.apply_params(snapshot['params'])
        
        self.refresh_current_view()
        self.update_undo_redo_buttons()

    def update_undo_redo_buttons(self):
        if self.current_tab_index < 0:
            self.image_viewer.undo_controls.update_states(False, False)
            return
            
        data = self.images_data[self.current_tab_index]
        can_undo = data['history_idx'] > 0
        can_redo = data['history_idx'] < len(data['history']) - 1
        
        self.image_viewer.undo_controls.update_states(can_undo, can_redo)

    # ================= 导入导出逻辑 =================
    def import_images(self):
        files, _ = QFileDialog.getOpenFileNames(self, "导入图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.webp)")
        if files:
            new_count = 0
            for f in files:
                try:
                    img_data = np.fromfile(f, dtype=np.uint8)
                    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
                    if img is None: continue
                    short_name = f.replace("\\", "/").split('/')[-1]
                    
                    self.images_data.append({
                        'path': f, 'name': short_name, 'original': img,
                        'processed': None, 'edges': None, 'is_processed': False,
                        'edge_mode': 0,
                        'history': [],
                        'history_idx': -1
                    })
                    
                    thumb_size = 50
                    container = QPixmap(thumb_size, thumb_size)
                    container.fill(Qt.GlobalColor.transparent)
                    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, _ = rgb_img.shape
                    qim = QImage(rgb_img.data, w, h, 3 * w, QImage.Format.Format_RGB888)
                    pixmap = QPixmap.fromImage(qim)
                    scaled_pix = pixmap.scaled(thumb_size, thumb_size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    painter = QPainter(container)
                    x = (thumb_size - scaled_pix.width()) // 2
                    y = (thumb_size - scaled_pix.height()) // 2
                    painter.drawPixmap(x, y, scaled_pix)
                    painter.end()
                    
                    row = self.layer_table.rowCount()
                    self.layer_table.insertRow(row)
                    self.layer_table.setRowHeight(row, 58)
                    
                    chk_widget = QWidget()
                    chk_layout = QHBoxLayout(chk_widget)
                    chk_layout.setContentsMargins(0,0,0,0) 
                    chk_layout.setAlignment(Qt.AlignmentFlag.AlignCenter) 
                    checkbox = QCheckBox()
                    checkbox.setChecked(True) 
                    chk_layout.addWidget(checkbox)
                    self.layer_table.setCellWidget(row, 0, chk_widget)
                    
                    item = QTableWidgetItem(short_name)
                    item.setIcon(QIcon(container))
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    self.layer_table.setItem(row, 1, item)
                    
                    edge_widget = QWidget()
                    edge_layout = QHBoxLayout(edge_widget)
                    edge_layout.setContentsMargins(0,0,0,0)
                    edge_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    btn_edge = QPushButton()
                    btn_edge.setFixedSize(28, 28)
                    btn_edge.setIcon(self.icon_overlay) 
                    btn_edge.setIconSize(QSize(20, 20))
                    btn_edge.setToolTip("点击切换描边模式:\n1. 叠加 (Overlay)\n2. 仅结果 (Clean)\n3. 仅描边 (Edge Only)")
                    btn_edge.clicked.connect(self.on_edge_btn_clicked)
                    edge_layout.addWidget(btn_edge)
                    self.layer_table.setCellWidget(row, 2, edge_widget)
                    
                    new_count += 1
                except Exception as e:
                    print(f"Error loading {f}: {e}")

            if new_count > 0:
                last_idx = len(self.images_data) - 1
                self.layer_table.selectRow(last_idx)
                self.current_tab_index = last_idx
                self.refresh_current_view()

    def on_edge_btn_clicked(self):
        btn = self.sender()
        pos = btn.mapTo(self.layer_table.viewport(), QPoint(0,0))
        index = self.layer_table.indexAt(pos)
        if not index.isValid(): return
        
        row = index.row()
        data = self.images_data[row]
        current_mode = data['edge_mode']
        new_mode = (current_mode + 1) % 3
        data['edge_mode'] = new_mode
        
        if new_mode == 0: btn.setIcon(self.icon_overlay)
        elif new_mode == 1: btn.setIcon(self.icon_clean)
        elif new_mode == 2: btn.setIcon(self.icon_edge)
        
        if row == self.current_tab_index:
            if not self.image_viewer.controls.btn_toggle.isChecked():
                self.image_viewer.controls.btn_toggle.setChecked(True)
                self.image_viewer.controls.on_toggle_clicked() 
            else:
                self.refresh_current_view()

    # [Added] 批量切换显示模式的槽函数
    def batch_toggle_edge_mode(self):
        # 更新全局状态：0->1->2->0
        self.batch_edge_mode_state = (self.batch_edge_mode_state + 1) % 3
        
        # 更新主按钮图标
        if self.batch_edge_mode_state == 0: self.btn_batch_toggle_mode.setIcon(self.icon_overlay)
        elif self.batch_edge_mode_state == 1: self.btn_batch_toggle_mode.setIcon(self.icon_clean)
        elif self.batch_edge_mode_state == 2: self.btn_batch_toggle_mode.setIcon(self.icon_edge)
        
        # 遍历并更新选中行
        needs_refresh = False
        for i in range(self.layer_table.rowCount()):
            chk = self.get_row_checkbox(i, 0)
            if chk and chk.isChecked():
                # 更新数据
                self.images_data[i]['edge_mode'] = self.batch_edge_mode_state
                
                # 更新行内小按钮图标
                widget = self.layer_table.cellWidget(i, 2)
                if widget:
                    btn = widget.findChild(QPushButton)
                    if btn:
                        if self.batch_edge_mode_state == 0: btn.setIcon(self.icon_overlay)
                        elif self.batch_edge_mode_state == 1: btn.setIcon(self.icon_clean)
                        elif self.batch_edge_mode_state == 2: btn.setIcon(self.icon_edge)
                
                # 如果当前显示的图片在批量更新范围内，标记刷新
                if i == self.current_tab_index:
                    needs_refresh = True

        if needs_refresh:
            self.refresh_current_view()

    def on_table_cell_clicked(self, row, col):
        if col == 1:
            if row >= 0 and row < len(self.images_data):
                self.current_tab_index = row
                self.refresh_current_view()

    def get_row_checkbox(self, row, col=0):
        widget = self.layer_table.cellWidget(row, col)
        if widget:
            return widget.findChild(QCheckBox)
        return None
        
    def update_row_edge_icon(self, row):
        if row < 0 or row >= len(self.images_data): return
        widget = self.layer_table.cellWidget(row, 2)
        if widget:
            btn = widget.findChild(QPushButton)
            if btn:
                has_edges = self.images_data[row].get('edges') is not None
                btn.setEnabled(has_edges)

    def delete_selected_layers(self):
        indices_to_remove = []
        for i in range(self.layer_table.rowCount()):
            chk = self.get_row_checkbox(i, 0)
            if chk and chk.isChecked():
                indices_to_remove.append(i)
        
        if not indices_to_remove:
            return

        for i in reversed(indices_to_remove):
            self.layer_table.removeRow(i)
            del self.images_data[i]
        
        if not self.images_data:
            self.image_viewer.set_image(None)
            self.current_tab_index = -1
        else:
            if self.current_tab_index >= len(self.images_data):
                self.current_tab_index = len(self.images_data) - 1
            self.layer_table.selectRow(self.current_tab_index)
            self.refresh_current_view()

    def set_all_layers_checked(self, checked):
        for i in range(self.layer_table.rowCount()):
            chk = self.get_row_checkbox(i, 0)
            if chk: chk.setChecked(checked)

    def invert_layer_selection(self):
        for i in range(self.layer_table.rowCount()):
            chk = self.get_row_checkbox(i, 0)
            if chk: chk.setChecked(not chk.isChecked())

    def on_view_toggle_changed(self, is_processed):
        self.refresh_current_view()

    def get_current_display_image(self):
        if self.current_tab_index < 0 or self.current_tab_index >= len(self.images_data):
            return None
        
        data = self.images_data[self.current_tab_index]
        show_processed = self.image_viewer.controls.btn_toggle.isChecked()
        
        final_img = None
        
        if not show_processed:
            final_img = data['original']
        else:
            if not data['is_processed'] or data['processed'] is None:
                return data['original']
            
            mode = data['edge_mode']
            
            if mode == 1: # Clean
                final_img = data['processed'].copy()
            
            elif mode == 2: # Edge Only
                if data.get('edges') is not None:
                    mask_inv = cv2.bitwise_not(data['edges'])
                    final_img = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
                else:
                    h, w = data['processed'].shape[:2]
                    final_img = np.ones((h, w, 3), dtype=np.uint8) * 255
            
            else: # 0 = Overlay
                final_img = data['processed'].copy()
                if data.get('edges') is not None:
                    mask_inv = cv2.bitwise_not(data['edges'])
                    mask_bgr = cv2.cvtColor(mask_inv, cv2.COLOR_GRAY2BGR)
                    final_img = cv2.bitwise_and(final_img, mask_bgr)
        
        return final_img

    def refresh_current_view(self):
        img_to_show = self.get_current_display_image()
        if img_to_show is None:
            self.image_viewer.set_image(None)
            self.color_panel.update_colors(None, None)
            return
        
        self.image_viewer.set_image(img_to_show)
        self.update_undo_redo_buttons() 

    def handle_pixel_hover(self, x, y):
        if self.current_tab_index < 0: 
            self.color_panel.update_colors(None, None)
            return
        
        data = self.images_data[self.current_tab_index]
        c1 = None
        c2 = None
        
        def get_col(img):
            if img is None: return None
            if 0 <= y < img.shape[0] and 0 <= x < img.shape[1]:
                return img[y, x] # BGR
            return None
            
        if x >= 0 and y >= 0:
            c1 = get_col(data['original'])
            c2 = get_col(data['processed']) if data.get('processed') is not None else None
            
        self.color_panel.update_colors(c1, c2)

    def toggle_processing(self):
        if self.stop_requested: return 
        if self.processing_threads:
            self.stop_requested = True
            self.btn_process.setText("正在停止...")
            self.btn_process.setStyleSheet("background-color: #e81123; color: white; font-size: 14px; font-weight: bold; border-radius: 5px;")
            self.batch_queue.clear() 
        else:
            self.run_batch_processing()

    def run_batch_processing(self):
        indices = [i for i in range(self.layer_table.rowCount()) if self.get_row_checkbox(i).isChecked()]
        if not indices:
            QMessageBox.warning(self, "提示", "请先在右侧图层列表中勾选至少一张图片。")
            return
        
        idx = self.combo_edge.currentIndex()
        e_mode = 'none'
        if idx == 0: e_mode = 'sobel'
        elif idx == 1: e_mode = 'canny'
        elif idx == 2: e_mode = 'adaptive'

        # [Fix 1] 在点击执行时获取参数快照，并锁定在 processing loop 中
        params = {
            'method': 'html_hard' if self.combo_algo.currentIndex() == 0 else 'bilateral' if self.combo_algo.currentIndex() == 1 else 'meanshift',
            'sigma_color': self.inp_s2.value(), 
            'sigma_space': self.inp_s1.value(),
            'sp': self.inp_s1.value(), 'sr': self.inp_s2.value(),
            'enable_kmeans': self.chk_kmeans.isChecked(),
            'k_value': self.inp_k.value(),
            'edge_mode': e_mode,
            'canny_t1': self.inp_e1.value(), 'canny_t2': self.inp_e2.value(),
            'core_edge_thresh': self.inp_core_edge.value(),
            'overlay_edges': True 
        }

        ui_snapshot = self.get_current_params()
        
        self.stop_requested = False
        self.btn_process.setText("停止执行")
        self.btn_process.setStyleSheet("background-color: #e81123; color: white; font-size: 14px; font-weight: bold; border-radius: 5px;")
        self.btn_process.clicked.disconnect()
        self.btn_process.clicked.connect(self.toggle_processing) 
        
        if not self.image_viewer.controls.btn_toggle.isChecked():
             self.image_viewer.controls.btn_toggle.setChecked(True)
             self.image_viewer.controls.on_toggle_clicked()

        print(f"\n=== 开始批量处理: 共 {len(indices)} 张图片 ===")
        self.batch_queue = indices
        self.batch_total = len(indices)
        self.batch_finished = 0
        
        self.process_next_in_queue(params, ui_snapshot)

    def process_next_in_queue(self, params, ui_snapshot):
        if self.stop_requested or not self.batch_queue:
            self.finish_batch_processing()
            return

        idx = self.batch_queue.pop(0)
        original_img = self.images_data[idx]['original']
        
        thread = ImageProcessingThread(original_img, idx, params)
        thread.result_ready.connect(lambda i, res, edges: self.on_batch_item_finished(i, res, edges, params, ui_snapshot))
        thread.finished.connect(lambda: self.processing_threads.pop(idx, None))
        self.processing_threads[idx] = thread
        thread.start()

    def on_batch_item_finished(self, idx, processed_img, edges, params, ui_snapshot):
        if processed_img is not None:
            self.images_data[idx]['processed'] = processed_img
            self.images_data[idx]['edges'] = edges
            self.images_data[idx]['is_processed'] = True
            
            # [Fix 1] 使用处理开始时捕获的 ui_snapshot 存入历史
            self.push_history(idx, processed_img, edges, ui_snapshot)
            
            self.update_row_edge_icon(idx)
        
        self.batch_finished += 1
        name = self.images_data[idx]['name']
        print_progress(self.batch_finished, self.batch_total, f"Completed: {name}")
        if idx == self.current_tab_index: self.refresh_current_view()
        self.process_next_in_queue(params, ui_snapshot)

    def finish_batch_processing(self):
        self.btn_process.setText("执行处理 (选中的图片)")
        self.btn_process.setStyleSheet("QPushButton { background-color: #0078d4; color: white; font-size: 14px; font-weight: bold; border-radius: 5px; } QPushButton:hover { background-color: #1084e0; } QPushButton:pressed { background-color: #005a9e; }")
        self.btn_process.clicked.disconnect()
        self.btn_process.clicked.connect(self.run_batch_processing)
        
        if self.stop_requested:
            print("\n>>> 任务已中断。")
        else:
            print("\n>>> 所有任务处理完成。")

    def batch_export_images(self):
        indices = [i for i in range(self.layer_table.rowCount()) if self.get_row_checkbox(i).isChecked()]
        if not indices: return
        dir_path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if not dir_path: return
        
        count = 0
        for idx in indices:
            data = self.images_data[idx]
            if not data.get('is_processed'): continue
            
            mode = data['edge_mode']
            final = None
            
            # [Revised Logic] Handle Transparency for Mode 2 (Edge Only)
            if mode == 2: # Edge Only - Transparent BG
                 if data.get('edges') is not None:
                     # edges is 255 for edge, 0 for bg
                     edges = data['edges']
                     h, w = edges.shape
                     # Create BGRA (4 channel)
                     # RGB = Black (0,0,0), Alpha = Edges
                     b = np.zeros((h,w), dtype=np.uint8)
                     g = np.zeros((h,w), dtype=np.uint8)
                     r = np.zeros((h,w), dtype=np.uint8)
                     a = edges # 255 = visible edge
                     final = cv2.merge([b, g, r, a])
            
            elif mode == 1: # Clean
                final = data['processed']
                
            else: # Mode 0: Overlay (default)
                base = data['processed'].copy()
                if data.get('edges') is not None:
                    mask = cv2.cvtColor(cv2.bitwise_not(data['edges']), cv2.COLOR_GRAY2BGR)
                    final = cv2.bitwise_and(base, mask)
                else:
                    final = base
            
            if final is not None:
                # [Revised Logic] Filename handling with _output
                base_name = os.path.splitext(data['name'])[0] + "_output"
                ext = ".png"
                save_path = os.path.join(dir_path, base_name + ext)
                
                # Duplicate check
                counter = 1
                while os.path.exists(save_path):
                    save_path = os.path.join(dir_path, f"{base_name}_{counter}{ext}")
                    counter += 1
                
                # Use imencode + tofile to support unicode paths (Chinese)
                is_success, im_buf_arr = cv2.imencode(ext, final)
                if is_success:
                    im_buf_arr.tofile(save_path)
                    count += 1
                
        QMessageBox.information(self, "完成", f"导出 {count} 张图片")

if __name__ == '__main__':
    try:
        # [新增] 设置 AppUserModelID
        # 这行代码告诉 Windows：“我是一个独立的应用程序”，从而让任务栏显示正确的图标
        myappid = 'mycompany.minimaliststylizer.v1.0' # 任意唯一的字符串
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except:
        pass

    try:
        QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except: pass
    app = QApplication(sys.argv)
    font = app.font()
    font.setFamily("Microsoft YaHei")
    font.setPointSize(9)
    app.setFont(font)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())