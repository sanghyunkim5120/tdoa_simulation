import numpy as np
from PyQt5.QtWidgets import QMainWindow, QWidget, QHBoxLayout, QVBoxLayout
from PyQt5.QtCore import QTimer

from .earth_widget import EarthWidget
from .tdoa_widget import TDOAWidget
from .control_panel import ControlPanel
from core.satellite import Satellite, UserPosition
from core.tdoa_calculator import compute_tdoa_pairs, add_measurement_noise


MAX_SATELLITES = 4
ORBIT_INTERVAL_MS = 80
LIVE_CALC_INTERVAL_MS = 200   # 실시간 TDOA 재계산 주기


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('TDOA 기반 위치 측정 시뮬레이션')
        self.resize(1300, 750)
        self.setStyleSheet('background-color: #0A0A1A;')

        self.satellites = []
        self.user_position = UserPosition()
        self._next_sat_id = 1
        self._active_pairs = 0
        self._show_actual = False
        self._calculating = False

        self._build_ui()
        self._connect_signals()

        for _ in range(3):
            self._add_satellite()

        self._orbit_timer = QTimer()
        self._orbit_timer.timeout.connect(self._orbit_tick)
        self._orbit_timer.start(ORBIT_INTERVAL_MS)

        self._hyp_timer = QTimer()
        self._hyp_timer.timeout.connect(self._hyperbola_tick)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setSpacing(0)
        root.setContentsMargins(0, 0, 0, 0)

        left_wrap = QWidget()
        left_wrap.setStyleSheet('background-color: #0A0A1A;')
        ll = QVBoxLayout(left_wrap)
        ll.setContentsMargins(8, 8, 4, 8)
        self.earth_widget = EarthWidget()
        ll.addWidget(self.earth_widget)

        right_wrap = QWidget()
        right_wrap.setStyleSheet('background-color: #0A0A1A;')
        rl = QVBoxLayout(right_wrap)
        rl.setContentsMargins(4, 8, 8, 8)
        self.tdoa_widget = TDOAWidget()
        rl.addWidget(self.tdoa_widget)

        self.ctrl = ControlPanel()
        root.addWidget(left_wrap, stretch=5)
        root.addWidget(right_wrap, stretch=5)
        root.addWidget(self.ctrl, stretch=0)

    def _connect_signals(self):
        c = self.ctrl
        c.sig_add_satellite.connect(self._add_satellite)
        c.sig_remove_satellite.connect(self._remove_satellite)
        c.sig_start_calculation.connect(self._start_calculation)
        c.sig_stop_calculation.connect(self._stop_calculation)
        c.sig_show_actual.connect(self._toggle_actual)
        c.sig_reset.connect(self._reset)

        self.earth_widget.sig_interaction_start.connect(
            lambda: self._orbit_timer.stop())
        self.earth_widget.sig_interaction_end.connect(
            lambda: self._orbit_timer.start(ORBIT_INTERVAL_MS))

    # ── 위성 제어 ──────────────────────────────────────────
    def _add_satellite(self):
        if len(self.satellites) >= MAX_SATELLITES:
            self.ctrl.set_status(f'최대 {MAX_SATELLITES}개까지 추가 가능', '#FF6B6B')
            return
        sat = Satellite(self._next_sat_id)
        self._next_sat_id += 1
        self.satellites.append(sat)
        self._sync_tdoa()
        # 위성 수 변경 시 오른쪽(TDOA)도 즉시 반영
        self.tdoa_widget.redraw()
        self._update_info()
        self.ctrl.set_status(f'S{sat.sat_id} 추가됨', '#00FF88')

    def _remove_satellite(self):
        if not self.satellites:
            self.ctrl.set_status('위성이 없습니다', '#FF6B6B')
            return
        removed = self.satellites.pop()
        self._sync_tdoa()
        self.tdoa_widget.redraw()
        self._update_info()
        self.ctrl.set_status(f'S{removed.sat_id} 제거됨', '#FF6B6B')

    # ── 계산 제어 ──────────────────────────────────────────
    def _start_calculation(self):
        if len(self.satellites) < 2:
            self.ctrl.set_status('위성 2개 이상 필요', '#FFD700')
            return
        self._calculating = True
        self.ctrl.set_calculating(True)
        self.ctrl.set_status('실시간 계산 중...', '#7B8CDE')
        self._hyperbola_tick()                       # 즉시 첫 계산
        self._hyp_timer.start(LIVE_CALC_INTERVAL_MS)

    def _stop_calculation(self):
        self._hyp_timer.stop()
        self._calculating = False
        self.ctrl.set_calculating(False)
        self.ctrl.set_status('계산 중지됨', '#FFD700')

    def _hyperbola_tick(self):
        """위성 현재 위치로 TDOA 전체 쌍 실시간 재계산."""
        rng = np.random.default_rng()
        true_pairs = compute_tdoa_pairs(self.satellites, self.user_position)
        noisy_pairs = add_measurement_noise(true_pairs, rng)
        n = len(self.satellites)
        total = n * (n - 1) // 2
        self._active_pairs = total
        self.tdoa_widget.noisy_pairs = noisy_pairs
        self.tdoa_widget.set_active_pairs(total)
        self._update_info()

    # ── 실제 위치 토글 ─────────────────────────────────────
    def _toggle_actual(self):
        self._show_actual = not self._show_actual
        # 2D 위젯
        self.tdoa_widget.show_actual = self._show_actual
        self.tdoa_widget.redraw()
        # 3D 위젯: show 상태 전달 (user_position 변경 없이)
        self.earth_widget.set_show_user(self._show_actual)
        if self._show_actual:
            self.ctrl.set_status('실제 위치 표시 중', '#FF69B4')
        else:
            self.ctrl.set_status('실제 위치 숨김', '#888899')

    # ── 초기화 ─────────────────────────────────────────────
    def _reset(self):
        self._hyp_timer.stop()
        self._calculating = False
        self._show_actual = False
        self.ctrl.set_calculating(False)

        self.satellites.clear()
        self._next_sat_id = 1
        self._active_pairs = 0
        # 새 사용자 위치 생성
        self.user_position = UserPosition()

        # 3D: 별 숨기고 새 위치 적용
        self.earth_widget.set_show_user(False, self.user_position)

        for _ in range(3):
            self._add_satellite()

        self._sync_tdoa()
        self.tdoa_widget.show_actual = False
        self.tdoa_widget.reset()
        self._update_info()
        self.ctrl.set_status('초기화 완료', '#4ECDC4')

    # ── 내부 헬퍼 ──────────────────────────────────────────
    def _sync_tdoa(self):
        self.tdoa_widget.satellites = self.satellites
        self.tdoa_widget.user_position = self.user_position
        self.tdoa_widget.n_satellites = len(self.satellites)

    def _orbit_tick(self):
        dt = ORBIT_INTERVAL_MS / 1000.0
        for sat in self.satellites:
            sat.update(dt)
        self.earth_widget.update_scene(self.satellites, self.user_position)

    def _update_info(self):
        n = len(self.satellites)
        total = n * (n - 1) // 2
        self.ctrl.set_info(n, self._active_pairs, total)
