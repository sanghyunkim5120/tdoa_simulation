import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MultipleLocator
from PyQt5.QtWidgets import QSizePolicy
from scipy import ndimage

from core.tdoa_calculator import (
    generate_hyperbola_2d, build_heatmap, estimate_position
)

GRID_RANGE = 15000.0
RESOLUTION = 600

HYPER_COLORS = ['#00FFAA', '#FF6B35', '#4ECDC4', '#FFE66D',
                '#A8DADC', '#FF8FA3', '#C77DFF', '#80FFDB',
                '#F4A261', '#E76F51', '#2A9D8F', '#E9C46A',
                '#264653', '#F72585', '#7209B7', '#3A0CA3',
                '#4361EE', '#4CC9F0', '#06D6A0', '#FFB703',
                '#FB8500', '#8338EC', '#3A86FF', '#FF006E',
                '#FFBE0B', '#FB5607', '#FF006E', '#8338EC']


def _term(axis: str, val: float) -> str:
    """좌표 항 표현: 'axis-val' 또는 'axis+|val|' (부호 자동 처리)"""
    if val >= 0:
        return f'{axis}-{val:.0f}'
    else:
        return f'{axis}+{abs(val):.0f}'


class TDOAWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(5, 6.5), facecolor='#0A0A1A')
        super().__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        gs = GridSpec(2, 1, figure=self.fig,
                      height_ratios=[4, 1.6],
                      hspace=0.48,
                      top=0.95, bottom=0.03,
                      left=0.09, right=0.97)
        self.ax = self.fig.add_subplot(gs[0])
        self.ax_eq = self.fig.add_subplot(gs[1])
        self._setup_axes()
        self._setup_eq_axes()

        self.satellites = []
        self.user_position = None
        self.active_pairs = 0
        self.show_actual = False
        self._estimated_pos = None
        self.n_satellites = 0

        self.noisy_pairs: list = []

    def _setup_axes(self):
        ax = self.ax
        self.fig.patch.set_facecolor('#0A0A1A')
        ax.set_facecolor('#0D0D20')
        ax.tick_params(colors='#7777AA', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#1A1A3A')
        ax.set_xlabel('X (km)', color='#7777AA', fontsize=8,
                      fontfamily='Malgun Gothic')
        ax.set_ylabel('Y (km)', color='#7777AA', fontsize=8,
                      fontfamily='Malgun Gothic')
        ax.set_title('TDOA 위치 측정 (2D 투영)', color='#E0E0FF',
                     fontsize=11, fontfamily='Malgun Gothic')
        ax.set_xlim(-GRID_RANGE, GRID_RANGE)
        ax.set_ylim(-GRID_RANGE, GRID_RANGE)
        ax.set_aspect('equal')
        ax.grid(True, color='#1A1A3A', lw=0.5, alpha=0.5)
        ax.xaxis.set_major_locator(MultipleLocator(5000))
        ax.yaxis.set_major_locator(MultipleLocator(5000))
        # 원점 십자선
        ax.axhline(0, color='#2A2A4A', lw=0.9, alpha=0.9, zorder=1)
        ax.axvline(0, color='#2A2A4A', lw=0.9, alpha=0.9, zorder=1)

    def _setup_eq_axes(self):
        ax = self.ax_eq
        ax.set_facecolor('#08081A')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

    def reset(self):
        self.active_pairs = 0
        self.show_actual = False
        self._estimated_pos = None
        self.noisy_pairs = []
        self.redraw()

    def set_active_pairs(self, n: int):
        self.active_pairs = n
        self.redraw()

    def show_actual_position(self, show: bool):
        self.show_actual = show
        self.redraw()

    def _draw_equation_panel(self):
        ax = self.ax_eq
        ax.cla()
        ax.set_facecolor('#08081A')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')

        # 패널 구분선 (ylim=0~1 이므로 y=1.0 이 상단)
        ax.axhline(1.0, color='#1A1A3A', lw=1.0, clip_on=False)

        active = self.noisy_pairs[:self.active_pairs] if self.noisy_pairs else []

        if not active:
            ax.text(0.5, 0.5, '위치 계산 시작 후 쌍곡선 방정식이 표시됩니다',
                    color='#444466', fontsize=8, ha='center', va='center',
                    fontfamily='Malgun Gothic', transform=ax.transAxes)
            return

        # 헤더
        ax.text(0.5, 0.97, '━━  쌍곡선 방정식  ━━',
                color='#6677AA', fontsize=8, ha='center', va='top',
                fontfamily='Malgun Gothic', fontweight='bold',
                transform=ax.transAxes)

        n = len(active)
        y_start = 0.82
        y_step = (y_start - 0.04) / max(n, 1)

        for idx, (pi, pj, delta_d, id_i, id_j) in enumerate(active):
            color = HYPER_COLORS[idx % len(HYPER_COLORS)]
            y = y_start - idx * y_step

            eq = (f'H{idx+1}  S{id_i}-S{id_j}:  '
                  f'√(({_term("x", pi[0])})²+({_term("y", pi[1])})²)'
                  f'  -  √(({_term("x", pj[0])})²+({_term("y", pj[1])})²)'
                  f'  =  {delta_d:+.0f} km')

            ax.text(0.01, y, eq,
                    color=color, fontsize=6.5,
                    fontfamily='Malgun Gothic',
                    va='top', transform=ax.transAxes,
                    clip_on=True)

    def redraw(self):
        self.ax.cla()
        self._setup_axes()

        if not self.satellites or not self.user_position:
            self._draw_equation_panel()
            self.draw_idle()
            return

        # ── 히트맵 (곱셈 확률, 노이즈 포함) ──────────────────
        if self.active_pairs > 0 and self.noisy_pairs:
            X, Y, score = build_heatmap(
                self.noisy_pairs, self.active_pairs,
                GRID_RANGE, RESOLUTION)

            if score.max() > 0:
                self.ax.pcolormesh(X, Y, score,
                                   cmap='plasma', alpha=0.6,
                                   shading='auto',
                                   vmin=0, vmax=1.0)

                threshold_50 = np.exp(-0.5)
                if score.max() >= threshold_50:
                    self.ax.contour(X, Y, score,
                                    levels=[threshold_50],
                                    colors=['#FFFFFF'],
                                    linewidths=1.0, alpha=0.4,
                                    linestyles='--')

            # ── 쌍곡선 ────────────────────────────────────────
            for idx, (pi, pj, noisy_delta_d, id_i, id_j) in \
                    enumerate(self.noisy_pairs[:self.active_pairs]):
                Xh, Yh, Zh = generate_hyperbola_2d(
                    pi, pj, noisy_delta_d, GRID_RANGE, 320)
                color = HYPER_COLORS[idx % len(HYPER_COLORS)]

                self.ax.contour(Xh, Yh, Zh,
                                levels=[noisy_delta_d],
                                colors=[color],
                                linewidths=1.8, alpha=0.95)
                if noisy_delta_d != 0:
                    self.ax.contour(Xh, Yh, Zh,
                                    levels=[-noisy_delta_d],
                                    colors=[color],
                                    linewidths=0.9, alpha=0.35,
                                    linestyles='--')

                # 메인 플롯 좌상단 번호 배지
                self.ax.annotate(
                    f'H{idx+1}',
                    xy=(0.01, 0.97 - idx * 0.045),
                    xycoords='axes fraction',
                    color=color, fontsize=7, fontweight='bold',
                    fontfamily='Malgun Gothic',
                    bbox=dict(boxstyle='round,pad=0.25',
                              facecolor='#0A0A1A', alpha=0.75,
                              edgecolor=color, linewidth=0.6))

            # ── 추정 위치: 위성 수 및 완료 여부에 따라 피크 표시 ──
            total_pairs = len(self.noisy_pairs)
            all_shown = (self.active_pairs >= total_pairs > 0)

            if all_shown and self.n_satellites == 3:
                peaks = _find_peaks(score, X, Y, GRID_RANGE, RESOLUTION,
                                    min_relative_height=0.8, max_peaks=2)
            elif all_shown and self.n_satellites >= 4:
                peaks = _find_peaks(score, X, Y, GRID_RANGE, RESOLUTION,
                                    min_relative_height=0.8, max_peaks=1)
            else:
                peaks = []

            self._estimated_pos = peaks[0] if peaks else None

            for i, (px, py, peak_score) in enumerate(peaks):
                label = f'추정 위치 {i+1}' if len(peaks) > 1 else '추정 위치'
                color = '#FF2244' if i == 0 else '#FF9900'
                self.ax.scatter(
                    [px], [py],
                    color=color, s=160, marker='o',
                    zorder=12, label=label,
                    edgecolors='white', lw=1.5)
                self.ax.annotate(
                    f'{label}\n({px:.0f}, {py:.0f}) km',
                    (px, py),
                    color=color, fontsize=8, fontweight='bold',
                    xytext=(8, 6), textcoords='offset points',
                    fontfamily='Malgun Gothic',
                    bbox=dict(boxstyle='round,pad=0.3',
                              facecolor='#0A0A1A', alpha=0.8))

        # ── 위성 위치 (삼각형) ────────────────────────────────
        # noisy_pairs도 매 tick 현재 위치로 재계산되므로 항상 일치
        for sat in self.satellites:
            p = sat.get_position_2d()
            self.ax.scatter([p[0]], [p[1]], color=sat.color,
                            s=80, zorder=8, marker='^',
                            edgecolors='white', lw=0.5)
            self.ax.annotate(
                f'S{sat.sat_id}', (p[0], p[1]),
                color=sat.color, fontsize=9, fontweight='bold',
                xytext=(8, 4), textcoords='offset points',
                fontfamily='Malgun Gothic')

        # ── 실제 위치 ─────────────────────────────────────────
        if self.show_actual and self.user_position:
            pu = self.user_position.get_position_2d()
            self.ax.scatter([pu[0]], [pu[1]],
                            color='#00FF88', s=200, marker='*',
                            zorder=12, label='실제 위치',
                            edgecolors='white', lw=0.5)
            self.ax.annotate(
                f'실제 위치\n({pu[0]:.0f}, {pu[1]:.0f}) km',
                (pu[0], pu[1]),
                color='#00FF88', fontsize=8, fontweight='bold',
                xytext=(8, 6), textcoords='offset points',
                fontfamily='Malgun Gothic',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#0A0A1A', alpha=0.8))

        # ── 후보 영역 크기 정보 ───────────────────────────────
        if self.active_pairs > 0 and self.noisy_pairs:
            try:
                region_km2 = _estimate_region_area(score, GRID_RANGE, RESOLUTION)
            except NameError:
                X, Y, score = build_heatmap(
                    self.noisy_pairs, self.active_pairs, GRID_RANGE, RESOLUTION)
                region_km2 = _estimate_region_area(score, GRID_RANGE, RESOLUTION)
            area_txt = f'{region_km2:,.0f} km²' if region_km2 < 1e7 \
                else f'{region_km2/1e6:.1f}×10⁶ km²'
            self.ax.annotate(
                f'후보 영역: {area_txt}',
                xy=(0.5, 0.97), xycoords='axes fraction',
                color='#FFD700', fontsize=9, ha='center', va='top',
                fontfamily='Malgun Gothic',
                bbox=dict(boxstyle='round,pad=0.3',
                          facecolor='#0A0A1A', alpha=0.8))

        # ── 위성/쌍곡선 수 ────────────────────────────────────
        n_sat = len(self.satellites)
        n_pairs = n_sat * (n_sat - 1) // 2
        self.ax.annotate(
            f'위성 {n_sat}개 | 쌍곡선 {self.active_pairs}/{n_pairs}개',
            xy=(0.02, 0.03), xycoords='axes fraction',
            color='#AAAACC', fontsize=8, va='bottom',
            fontfamily='Malgun Gothic',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='#0A0A1A', alpha=0.7))

        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            self.ax.legend(handles, labels,
                           facecolor='#1A1A3A', edgecolor='#333355',
                           labelcolor='#E0E0FF', fontsize=8,
                           loc='upper right',
                           prop={'family': 'Malgun Gothic', 'size': 8})

        # ── 방정식 패널 갱신 ──────────────────────────────────
        self._draw_equation_panel()

        self.draw_idle()


def _estimate_region_area(score: np.ndarray, grid_range: float,
                           resolution: int) -> float:
    """score >= e^(-0.5) 픽셀 수 × 픽셀 면적 = 후보 영역 km²"""
    threshold = np.exp(-0.5)
    cell_size = (2 * grid_range / resolution) ** 2
    return float(np.sum(score >= threshold) * cell_size)


def _find_peaks(score: np.ndarray, X: np.ndarray, Y: np.ndarray,
                grid_range: float, resolution: int,
                min_relative_height: float = 0.35,
                min_separation_km: float = 1500.0,
                max_peaks: int = 4) -> list:
    """
    로컬 최댓값(피크)을 탐지하고 (x, y, score) 리스트 반환.
    - min_relative_height: 전체 최댓값 대비 최소 비율
    - min_separation_km: 피크 간 최소 거리 (중복 제거)
    """
    if score.max() == 0:
        return []

    cell_km = 2 * grid_range / resolution
    sep_px = max(1, int(min_separation_km / cell_km))

    fp = np.ones((sep_px, sep_px), dtype=bool)
    local_max = (ndimage.maximum_filter(score, footprint=fp) == score)
    threshold = score.max() * min_relative_height
    candidates = np.argwhere(local_max & (score >= threshold))

    candidates = sorted(candidates,
                        key=lambda rc: score[rc[0], rc[1]],
                        reverse=True)

    peaks = []
    used = []
    for rc in candidates:
        r, c = rc
        px, py = float(X[r, c]), float(Y[r, c])
        too_close = any(
            np.hypot(px - ex, py - ey) < min_separation_km
            for ex, ey, _ in used)
        if not too_close:
            used.append((px, py, float(score[r, c])))
            peaks.append((px, py, float(score[r, c])))
        if len(peaks) >= max_peaks:
            break

    return peaks


def _peak_radius(score: np.ndarray, X: np.ndarray, Y: np.ndarray,
                 cx: float, cy: float,
                 grid_range: float, resolution: int) -> float:
    """
    해당 피크 주변의 1-sigma 신뢰 영역 반지름 추정.
    피크 중심 기준으로 score >= peak * e^(-0.5) 인 픽셀들의
    평균 거리를 반지름으로 사용.
    """
    cell_km = 2 * grid_range / resolution
    ci = int((cx + grid_range) / (2 * grid_range) * resolution)
    ri = int((cy + grid_range) / (2 * grid_range) * resolution)
    ci = np.clip(ci, 0, resolution - 1)
    ri = np.clip(ri, 0, resolution - 1)

    peak_val = score[ri, ci]
    if peak_val == 0:
        return grid_range * 0.1

    mask = score >= peak_val * np.exp(-0.5)
    if mask.sum() == 0:
        return cell_km * 2

    dist = np.sqrt((X[mask] - cx) ** 2 + (Y[mask] - cy) ** 2)
    return float(np.percentile(dist, 90))
