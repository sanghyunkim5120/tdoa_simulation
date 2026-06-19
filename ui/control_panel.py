from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QFrame, QScrollArea
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont


BUTTON_STYLE = """
QPushButton {{
    background-color: {bg};
    color: {fg};
    border: 1px solid {border};
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 12px;
    font-family: 'Malgun Gothic';
    font-weight: bold;
}}
QPushButton:hover {{
    background-color: {hover};
    border-color: {fg};
}}
QPushButton:pressed {{
    background-color: {fg};
    color: #0A0A1A;
}}
QPushButton:disabled {{
    background-color: #1A1A2A;
    color: #444466;
    border-color: #222233;
}}
"""


def _make_style(bg, fg, border, hover):
    return BUTTON_STYLE.format(bg=bg, fg=fg, border=border, hover=hover)


class ControlPanel(QWidget):
    sig_add_satellite = pyqtSignal()
    sig_remove_satellite = pyqtSignal()
    sig_start_calculation = pyqtSignal()
    sig_stop_calculation = pyqtSignal()
    sig_show_actual = pyqtSignal()
    sig_reset = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(200)
        self.setStyleSheet('background-color: #0D0D20;')
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(12, 16, 12, 16)

        # 타이틀
        title = QLabel('TDOA 시뮬레이션')
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            'color: #7B8CDE; font-size: 14px; font-weight: bold;'
            'font-family: "Malgun Gothic"; padding: 4px;'
        )
        layout.addWidget(title)

        layout.addWidget(_divider())

        # 위성 제어
        sec1 = QLabel('▌ 위성 제어')
        sec1.setStyleSheet('color: #555588; font-size: 11px;'
                           'font-family: "Malgun Gothic";')
        layout.addWidget(sec1)

        self.btn_add = QPushButton('＋ 위성 추가')
        self.btn_add.setStyleSheet(
            _make_style('#0D1F0D', '#00FF88', '#00AA55', '#0D2D1A'))
        self.btn_add.clicked.connect(self.sig_add_satellite)
        layout.addWidget(self.btn_add)

        self.btn_remove = QPushButton('－ 위성 제거')
        self.btn_remove.setStyleSheet(
            _make_style('#1F0D0D', '#FF6B6B', '#AA3333', '#2D1A1A'))
        self.btn_remove.clicked.connect(self.sig_remove_satellite)
        layout.addWidget(self.btn_remove)

        layout.addWidget(_divider())

        # 계산 제어
        sec2 = QLabel('▌ 위치 계산')
        sec2.setStyleSheet('color: #555588; font-size: 11px;'
                           'font-family: "Malgun Gothic";')
        layout.addWidget(sec2)

        self.btn_start = QPushButton('▶ 위치 계산 시작')
        self.btn_start.setStyleSheet(
            _make_style('#0D0D1F', '#7B8CDE', '#3344AA', '#151535'))
        self.btn_start.clicked.connect(self.sig_start_calculation)
        layout.addWidget(self.btn_start)

        self.btn_stop = QPushButton('■ 위치 계산 중지')
        self.btn_stop.setStyleSheet(
            _make_style('#1F1A0D', '#FFD700', '#AA8800', '#2D2510'))
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.sig_stop_calculation)
        layout.addWidget(self.btn_stop)

        layout.addWidget(_divider())

        # 결과 보기
        sec3 = QLabel('▌ 결과')
        sec3.setStyleSheet('color: #555588; font-size: 11px;'
                           'font-family: "Malgun Gothic";')
        layout.addWidget(sec3)

        self.btn_actual = QPushButton('★ 실제 위치 보기')
        self.btn_actual.setStyleSheet(
            _make_style('#1A0D0D', '#FF69B4', '#AA2255', '#280D18'))
        self.btn_actual.clicked.connect(self.sig_show_actual)
        layout.addWidget(self.btn_actual)

        layout.addWidget(_divider())

        self.btn_reset = QPushButton('↺ 초기화')
        self.btn_reset.setStyleSheet(
            _make_style('#0D1A1F', '#4ECDC4', '#1A7A77', '#0D2226'))
        self.btn_reset.clicked.connect(self.sig_reset)
        layout.addWidget(self.btn_reset)

        layout.addStretch()

        # 상태 레이블
        self.lbl_status = QLabel('대기 중')
        self.lbl_status.setAlignment(Qt.AlignCenter)
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet(
            'color: #888899; font-size: 10px;'
            'font-family: "Malgun Gothic";'
            'background-color: #0A0A18; border-radius: 4px; padding: 6px;'
        )
        layout.addWidget(self.lbl_status)

        # 위성 정보 레이블
        self.lbl_info = QLabel('위성: 0개\n쌍곡선: 0/0개')
        self.lbl_info.setAlignment(Qt.AlignCenter)
        self.lbl_info.setStyleSheet(
            'color: #6677AA; font-size: 10px;'
            'font-family: "Malgun Gothic";'
            'background-color: #0A0A18; border-radius: 4px; padding: 6px;'
        )
        layout.addWidget(self.lbl_info)

    def set_status(self, text: str, color: str = '#888899'):
        self.lbl_status.setText(text)
        self.lbl_status.setStyleSheet(
            f'color: {color}; font-size: 10px;'
            'font-family: "Malgun Gothic";'
            'background-color: #0A0A18; border-radius: 4px; padding: 6px;'
        )

    def set_info(self, n_sat: int, active_pairs: int, total_pairs: int):
        self.lbl_info.setText(
            f'위성: {n_sat}개\n쌍곡선: {active_pairs}/{total_pairs}개'
        )

    def set_calculating(self, calculating: bool):
        self.btn_start.setEnabled(not calculating)
        self.btn_stop.setEnabled(calculating)
        self.btn_add.setEnabled(not calculating)
        self.btn_remove.setEnabled(not calculating)


def _divider():
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setStyleSheet('color: #1A1A3A;')
    return line
