import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import pyqtSignal


class EarthWidget(FigureCanvas):
    sig_interaction_start = pyqtSignal()
    sig_interaction_end = pyqtSignal()

    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 5), facecolor='#0A0A1A')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.ax = self.fig.add_subplot(111, projection='3d')
        self._setup_axes()
        self._draw_earth()

        self.satellites = []
        self.user_position = None
        self._show_user = False          # ← 별 표시 여부 상태

        self._sat_scatters = []
        self._sat_labels = []
        self._orbit_lines = []
        self._user_scatter = None
        self._user_spike_line = None
        self._prev_sat_count = -1

        self._mouse_down = False
        self.mpl_connect('button_press_event', self._on_mouse_press)
        self.mpl_connect('button_release_event', self._on_mouse_release)
        self.mpl_connect('scroll_event', self._on_scroll)

    def _on_mouse_press(self, event):
        self._mouse_down = True
        self.sig_interaction_start.emit()

    def _on_mouse_release(self, event):
        self._mouse_down = False
        self.sig_interaction_end.emit()

    def _on_scroll(self, event):
        self.draw_idle()

    def _setup_axes(self):
        ax = self.ax
        ax.set_facecolor('#0A0A1A')
        self.fig.patch.set_facecolor('#0A0A1A')
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor('#1A1A2A')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        lim = 12000
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        ax.set_box_aspect([1, 1, 1])

    def _draw_earth(self):
        from core.satellite import EARTH_RADIUS
        r = EARTH_RADIUS
        u = np.linspace(0, 2 * np.pi, 48)
        v = np.linspace(0, np.pi, 24)
        x = r * np.outer(np.cos(u), np.sin(v))
        y = r * np.outer(np.sin(u), np.sin(v))
        z = r * np.outer(np.ones(len(u)), np.cos(v))
        self.ax.plot_surface(x, y, z, color='#1B4F8A', alpha=0.80,
                             linewidth=0, antialiased=False, zorder=1)
        for lat_deg in range(-60, 61, 30):
            lat = np.radians(lat_deg)
            lon = np.linspace(0, 2 * np.pi, 72)
            self.ax.plot(r * np.cos(lat) * np.cos(lon),
                         r * np.cos(lat) * np.sin(lon),
                         r * np.sin(lat) * np.ones_like(lon),
                         color='#2A6DB5', lw=0.4, alpha=0.4, zorder=2)
        for lon_deg in range(0, 360, 30):
            lon = np.radians(lon_deg)
            lat = np.linspace(-np.pi / 2, np.pi / 2, 72)
            self.ax.plot(r * np.cos(lat) * np.cos(lon),
                         r * np.cos(lat) * np.sin(lon),
                         r * np.sin(lat),
                         color='#2A6DB5', lw=0.4, alpha=0.4, zorder=2)
        self.ax.set_title('위성 궤도 (3D)', color='#E0E0FF', fontsize=11,
                          pad=8, fontfamily='Malgun Gothic')

    # ── 위성 아티스트 재생성 ──────────────────────────────
    def _rebuild_satellite_artists(self):
        for sc in self._sat_scatters:
            sc.remove()
        for ln in self._orbit_lines:
            ln.remove()
        for txt in self._sat_labels:
            txt.remove()
        self._sat_scatters.clear()
        self._orbit_lines.clear()
        self._sat_labels.clear()

        for sat in self.satellites:
            xs, ys, zs = sat.get_orbit_path_3d(80)
            ln, = self.ax.plot(xs, ys, zs, color=sat.color,
                               lw=0.8, alpha=0.35, zorder=3)
            self._orbit_lines.append(ln)

            p = sat.get_position_3d()
            sc = self.ax.scatter([p[0]], [p[1]], [p[2]],
                                 color=sat.color, s=120,
                                 depthshade=False, zorder=10,
                                 edgecolors='white', linewidths=0.8)
            self._sat_scatters.append(sc)

            txt = self.ax.text(p[0], p[1], p[2], f'  S{sat.sat_id}',
                               color=sat.color, fontsize=8,
                               fontfamily='Malgun Gothic', zorder=11)
            self._sat_labels.append(txt)

        # 위성 재생성 후 별 표시 상태 유지
        self._apply_user_marker()

    # ── 사용자 마커 표시/숨김 ─────────────────────────────
    def _clear_user_marker(self):
        if self._user_scatter:
            try:
                self._user_scatter.remove()
            except Exception:
                pass
            self._user_scatter = None
        if self._user_spike_line:
            try:
                self._user_spike_line[0].remove()
            except Exception:
                pass
            self._user_spike_line = None

    def _apply_user_marker(self):
        """_show_user 상태에 따라 마커를 그리거나 지움"""
        self._clear_user_marker()
        if self._show_user and self.user_position:
            from core.satellite import EARTH_RADIUS
            pu = self.user_position.get_position_3d()
            direction = pu / np.linalg.norm(pu)
            surface = direction * EARTH_RADIUS
            outer = direction * (EARTH_RADIUS + 2500)

            spike = self.ax.plot(
                [surface[0], outer[0]],
                [surface[1], outer[1]],
                [surface[2], outer[2]],
                color='#FF3366', lw=1.5, alpha=0.8,
                linestyle='--', zorder=20)
            self._user_spike_line = spike

            self._user_scatter = self.ax.scatter(
                [outer[0]], [outer[1]], [outer[2]],
                color='#FF3366', s=300, marker='*',
                depthshade=False, zorder=20,
                edgecolors='#FFAAAA', linewidths=1.2)

    def set_show_user(self, show: bool, user_position=None):
        """실제 위치 보기 ON/OFF. user_position 은 초기화 시에만 갱신."""
        if user_position is not None:
            self.user_position = user_position
        self._show_user = show
        self._apply_user_marker()
        self.draw_idle()

    # ── 매 프레임 업데이트 ────────────────────────────────
    def update_scene(self, satellites, user_position):
        if self._mouse_down:
            return

        self.satellites = satellites
        self.user_position = user_position

        if len(satellites) != self._prev_sat_count:
            self._prev_sat_count = len(satellites)
            self._rebuild_satellite_artists()
            self.draw_idle()
            return

        for i, sat in enumerate(self.satellites):
            p = sat.get_position_3d()
            self._sat_scatters[i]._offsets3d = ([p[0]], [p[1]], [p[2]])
            self._sat_labels[i].set_position((p[0], p[1]))
            self._sat_labels[i].set_3d_properties(p[2], 'z')

        self.draw_idle()
