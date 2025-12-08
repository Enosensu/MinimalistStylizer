import sys
import os
import tempfile
import shutil
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QSlider, QSpinBox, QFrame)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon

# ==========================================
# 1. 资源管理器 (用于生成 SpinBox 的箭头图标)
# ==========================================
class ResourceManager:
    """
    在运行时自动生成所需的 SVG 图标文件，解决 Base64 显示问题，
    实现单文件运行，无需外部资源文件。
    """
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp(prefix="gemini_style_")
        self.icons = {}
        self._create_svg_files()

    def _create_svg_files(self):
        # 纯白色箭头 SVG #ffffff
        svg_up = '''<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">
          <path fill="#ffffff" d="M6 3 L10 9 L2 9 Z"/>
        </svg>'''

        svg_down = '''<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 12 12">
          <path fill="#ffffff" d="M6 9 L2 3 L10 3 Z"/>
        </svg>'''

        up_path = os.path.join(self.temp_dir, "up.svg")
        down_path = os.path.join(self.temp_dir, "down.svg")

        with open(up_path, "w") as f: f.write(svg_up)
        with open(down_path, "w") as f: f.write(svg_down)

        self.icons['up'] = up_path.replace("\\", "/")
        self.icons['down'] = down_path.replace("\\", "/")

    def get_icon_url(self, name):
        path = self.icons.get(name, "")
        return f'url("{path}")'

    def cleanup(self):
        try: shutil.rmtree(self.temp_dir)
        except: pass

# 初始化资源 (全局)
RES_MANAGER = ResourceManager()
ICON_UP_PATH = RES_MANAGER.get_icon_url("up")
ICON_DOWN_PATH = RES_MANAGER.get_icon_url("down")

# ==========================================
# 2. 可复用的滑块组件类
# ==========================================
class SliderBlock(QWidget):
    """
    包含标题、滑块(Slider)和数值框(SpinBox)的组合控件。
    实现了双向绑定。
    """
    def __init__(self, label_text, min_val, max_val, default_val, tooltip="", parent=None):
        super().__init__(parent)
        self.setup_ui(label_text, min_val, max_val, default_val, tooltip)

    def setup_ui(self, label_text, min_val, max_val, default_val, tooltip):
        # 主垂直布局
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(5)

        # 1. 标题行
        lbl_layout = QHBoxLayout()
        self.label = QLabel(label_text)
        if tooltip:
            self.label.setToolTip(tooltip)
            self.setToolTip(tooltip)
        lbl_layout.addWidget(self.label)
        layout.addLayout(lbl_layout)

        # 2. 控件行 (滑块 + 输入框)
        input_layout = QHBoxLayout()
        
        # 滑块
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(min_val, max_val)
        self.slider.setValue(default_val)
        if tooltip: self.slider.setToolTip(tooltip)

        # 输入框
        self.spinbox = QSpinBox()
        self.spinbox.setRange(min_val, max_val)
        self.spinbox.setValue(default_val)
        self.spinbox.setFixedWidth(60)
        self.spinbox.setAlignment(Qt.AlignmentFlag.AlignCenter)
        if tooltip: self.spinbox.setToolTip(tooltip)

        # 双向绑定信号
        self.slider.valueChanged.connect(self.spinbox.setValue)
        self.spinbox.valueChanged.connect(self.slider.setValue)

        input_layout.addWidget(self.slider)
        input_layout.addWidget(self.spinbox)
        layout.addLayout(input_layout)

    def get_value(self):
        return self.slider.value()

    def set_value(self, val):
        self.slider.setValue(val)

# ==========================================
# 3. 主窗口 (演示容器)
# ==========================================
class DemoWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slider UI Reference")
        self.resize(400, 300)
        
        # 样式定义
        self.apply_theme()

        # 主容器
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # --- 演示控件 1 ---
        self.slider1 = SliderBlock(
            label_text="参数 A (Sobel Threshold):",
            min_val=0, max_val=255, default_val=50,
            tooltip="控制边缘检测的灵敏度"
        )
        
        # --- 演示控件 2 ---
        self.slider2 = SliderBlock(
            label_text="参数 B (Smoothing):",
            min_val=1, max_val=100, default_val=25,
            tooltip="控制平滑程度"
        )

        # 添加到容器框中模拟侧边栏效果
        container = QFrame()
        container.setObjectName("DemoContainer") # 用于CSS定位
        v_layout = QVBoxLayout(container)
        v_layout.addWidget(self.slider1)
        v_layout.addWidget(self.slider2)
        v_layout.addStretch()
        
        main_layout.addWidget(container)

    def apply_theme(self):
        # 提取自原项目的配色方案
        BLUE_ACCENT = "#0078d4"
        BLUE_HOVER = "#1084e0"
        DARK_BG = "#202020"
        PANEL_BG = "#333333"
        BORDER_COL = "#555555"
        TEXT_COL = "#e0e0e0"

        qss = f"""
        QMainWindow, QWidget {{ 
            background-color: {DARK_BG}; 
            color: {TEXT_COL}; 
            font-family: "Segoe UI", "Microsoft YaHei"; 
        }}
        
        #DemoContainer {{
            background-color: #2b2b2b; 
            border: 1px solid #444;
            border-radius: 8px;
            padding: 10px;
        }}

        /* === SpinBox 样式核心 === */
        QSpinBox, QDoubleSpinBox {{ 
            background-color: {PANEL_BG}; 
            border: 1px solid {BORDER_COL}; 
            color: #eee; 
            padding: 4px; 
            border-radius: 4px; 
        }}
        
        QSpinBox:focus {{ border: 1px solid {BLUE_ACCENT}; }}

        /* 上按钮 */
        QSpinBox::up-button {{ 
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
        
        /* 下按钮 */
        QSpinBox::down-button {{ 
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
        
        /* 引用生成的 SVG 图标 */
        QSpinBox::up-arrow {{ image: {ICON_UP_PATH}; width: 10px; height: 10px; }}
        QSpinBox::down-arrow {{ image: {ICON_DOWN_PATH}; width: 10px; height: 10px; }}

        /* 按钮交互状态 */
        QSpinBox::up-button:hover, QSpinBox::down-button:hover {{ 
            background-color: #444; border-left: 1px solid {BLUE_HOVER};
        }}
        QSpinBox::up-button:pressed, QSpinBox::down-button:pressed {{ 
            background-color: {BLUE_ACCENT}; 
        }}
        
        /* === ToolTip === */
        QToolTip {{ 
            font-size: 12px; color: #ffffff; 
            background-color: {PANEL_BG}; 
            border: 1px solid {BLUE_ACCENT}; 
            padding: 4px; 
        }}
        """
        self.setStyleSheet(qss)
    
    def closeEvent(self, event):
        RES_MANAGER.cleanup()
        super().closeEvent(event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 字体设置
    font = app.font()
    font.setFamily("Microsoft YaHei")
    font.setPointSize(9)
    app.setFont(font)
    
    window = DemoWindow()
    window.show()
    sys.exit(app.exec())