import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import ndimage

from core.satellite import Satellite, UserPosition, EARTH_RADIUS
from core.tdoa_calculator import (
    compute_tdoa_pairs, add_measurement_noise,
    generate_hyperbola_2d, build_heatmap,
)

# ── 상수 ──────────────────────────────────────────────────────
MAX_SATELLITES = 4
GRID_RANGE = 15_000.0
HEATMAP_RES = 150
HYPER_RES = 120
ANIM_DT = 0.2  # 초 / 프레임

HYPER_COLORS = [
    '#00FFAA', '#FF6B35', '#4ECDC4', '#FFE66D',
    '#A8DADC', '#FF8FA3', '#C77DFF', '#80FFDB',
]

# ── 직렬화 헬퍼 ───────────────────────────────────────────────
def sat_to_dict(sat):
    return {
        'sat_id': sat.sat_id,
        'orbit_radius': sat.orbit_radius,
        'inclination': float(sat.inclination),
        'raan': float(sat.raan),
        'angle': float(sat.angle),
        'angular_speed': float(sat.angular_speed),
        'color': sat.color,
    }

def dict_to_sat(d):
    sat = Satellite(
        sat_id=d['sat_id'],
        orbit_radius=d['orbit_radius'],
        inclination=d['inclination'],
        raan=d['raan'],
        initial_angle=d['angle'],
        angular_speed=d['angular_speed'],
    )
    sat.color = d['color']
    return sat

def pairs_to_list(pairs):
    return [[p[0].tolist(), p[1].tolist(), float(p[2]), int(p[3]), int(p[4])]
            for p in pairs]

def list_to_pairs(lst):
    return [(np.array(x[0]), np.array(x[1]), float(x[2]), int(x[3]), int(x[4]))
            for x in lst]

def _term(axis, val):
    return f'{axis}-{val:.0f}' if val >= 0 else f'{axis}+{abs(val):.0f}'

# ── 상태 초기화 ───────────────────────────────────────────────
def init_state():
    if 'initialized' in st.session_state:
        return
    sats = [Satellite(i) for i in range(1, 4)]
    user = UserPosition()
    st.session_state.satellites = [sat_to_dict(s) for s in sats]
    st.session_state.user_lat = float(user.lat)
    st.session_state.user_lon = float(user.lon)
    st.session_state.calculating = False
    st.session_state.show_actual = False
    st.session_state.noisy_pairs = []
    st.session_state.next_sat_id = 4
    st.session_state.status = '대기 중'
    st.session_state.active_pairs = 0
    st.session_state.initialized = True

# ── 피크 탐지 ─────────────────────────────────────────────────
def _find_peaks(score, X, Y, min_rel=0.8, min_sep=1500.0, max_peaks=4):
    if score.max() == 0:
        return []
    res = score.shape[0]
    sep_px = max(1, int(min_sep / (2 * GRID_RANGE / res)))
    fp = np.ones((sep_px, sep_px), dtype=bool)
    local_max = ndimage.maximum_filter(score, footprint=fp) == score
    cands = sorted(
        np.argwhere(local_max & (score >= score.max() * min_rel)).tolist(),
        key=lambda rc: score[rc[0], rc[1]], reverse=True,
    )
    peaks, used = [], []
    for rc in cands:
        r, c = rc
        px, py = float(X[r, c]), float(Y[r, c])
        if not any(np.hypot(px - ex, py - ey) < min_sep for ex, ey, _ in used):
            used.append((px, py, float(score[r, c])))
            peaks.append((px, py, float(score[r, c])))
        if len(peaks) >= max_peaks:
            break
    return peaks

# ── 3D 지구 시각화 ────────────────────────────────────────────
def build_earth_fig(satellites, user_position, show_actual):
    fig = go.Figure()
    r = EARTH_RADIUS
    u = np.linspace(0, 2 * np.pi, 36)
    v = np.linspace(0, np.pi, 18)
    fig.add_trace(go.Surface(
        x=r * np.outer(np.cos(u), np.sin(v)),
        y=r * np.outer(np.sin(u), np.sin(v)),
        z=r * np.outer(np.ones(36), np.cos(v)),
        colorscale=[[0, '#1B4F8A'], [1, '#2A6DB5']],
        opacity=0.8, showscale=False, hoverinfo='skip',
    ))

    for sat in satellites:
        p = sat.get_position_3d()
        xs, ys, zs = sat.get_orbit_path_3d(80)
        fig.add_trace(go.Scatter3d(
            x=xs, y=ys, z=zs, mode='lines',
            line=dict(color=sat.color, width=1),
            opacity=0.4, showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter3d(
            x=[p[0]], y=[p[1]], z=[p[2]],
            mode='markers+text',
            marker=dict(color=sat.color, size=6,
                        line=dict(color='white', width=1)),
            text=[f'S{sat.sat_id}'],
            textposition='top center',
            textfont=dict(color=sat.color, size=11),
            showlegend=False,
        ))

    if show_actual and user_position:
        pu = user_position.get_position_3d()
        d = pu / np.linalg.norm(pu)
        sp, op = d * EARTH_RADIUS, d * (EARTH_RADIUS + 2500)
        fig.add_trace(go.Scatter3d(
            x=[sp[0], op[0]], y=[sp[1], op[1]], z=[sp[2], op[2]],
            mode='lines',
            line=dict(color='#FF3366', width=2, dash='dash'),
            showlegend=False, hoverinfo='skip',
        ))
        fig.add_trace(go.Scatter3d(
            x=[op[0]], y=[op[1]], z=[op[2]],
            mode='markers',
            marker=dict(color='#FF3366', size=10, symbol='diamond',
                        line=dict(color='#FFAAAA', width=1)),
            showlegend=False, hoverinfo='skip',
        ))

    lim = 12000
    fig.update_layout(
        paper_bgcolor='#0A0A1A',
        scene=dict(
            bgcolor='#0A0A1A',
            xaxis=dict(range=[-lim, lim], visible=False),
            yaxis=dict(range=[-lim, lim], visible=False),
            zaxis=dict(range=[-lim, lim], visible=False),
            aspectmode='cube',
        ),
        title=dict(text='위성 궤도 (3D)',
                   font=dict(color='#E0E0FF', size=13), x=0.5),
        margin=dict(l=0, r=0, t=40, b=0),
        showlegend=False,
        uirevision='earth',
    )
    return fig

# ── 2D TDOA 시각화 ────────────────────────────────────────────
def build_tdoa_fig(satellites, user_position, noisy_pairs, active_pairs,
                   show_actual, n_sat):
    fig = go.Figure()
    x1d = np.linspace(-GRID_RANGE, GRID_RANGE, HEATMAP_RES)
    x1d_h = np.linspace(-GRID_RANGE, GRID_RANGE, HYPER_RES)

    if satellites and active_pairs > 0 and noisy_pairs:
        X, Y, score = build_heatmap(noisy_pairs, active_pairs, GRID_RANGE, HEATMAP_RES)

        if score.max() > 0:
            fig.add_trace(go.Heatmap(
                x=x1d, y=x1d, z=score,
                colorscale='Plasma', opacity=0.6,
                showscale=False, hoverinfo='skip',
            ))

        for idx, (pi, pj, delta_d, id_i, id_j) in enumerate(noisy_pairs[:active_pairs]):
            _, _, Zh = generate_hyperbola_2d(pi, pj, delta_d, GRID_RANGE, HYPER_RES)
            color = HYPER_COLORS[idx % len(HYPER_COLORS)]
            eps = max(abs(delta_d) * 0.001, 0.5)

            fig.add_trace(go.Contour(
                x=x1d_h, y=x1d_h, z=Zh,
                contours=dict(start=delta_d, end=delta_d + eps, size=2 * eps,
                              coloring='none', showlabels=False),
                line=dict(color=color, width=2),
                showscale=False, hoverinfo='skip', showlegend=False,
            ))
            if delta_d != 0:
                nd = -delta_d
                fig.add_trace(go.Contour(
                    x=x1d_h, y=x1d_h, z=Zh,
                    contours=dict(start=nd, end=nd + eps, size=2 * eps,
                                  coloring='none', showlabels=False),
                    line=dict(color=color, width=1, dash='dash'),
                    opacity=0.35,
                    showscale=False, hoverinfo='skip', showlegend=False,
                ))

        all_shown = active_pairs >= len(noisy_pairs) > 0
        if all_shown and n_sat == 3:
            peaks = _find_peaks(score, X, Y, max_peaks=2)
        elif all_shown and n_sat >= 4:
            peaks = _find_peaks(score, X, Y, max_peaks=1)
        else:
            peaks = []

        for i, (px, py, _) in enumerate(peaks):
            label = f'추정 위치 {i + 1}' if len(peaks) > 1 else '추정 위치'
            c = '#FF2244' if i == 0 else '#FF9900'
            fig.add_trace(go.Scatter(
                x=[px], y=[py], mode='markers+text',
                marker=dict(color=c, size=14, symbol='circle',
                            line=dict(color='white', width=2)),
                text=[f'{label}<br>({px:.0f}, {py:.0f}) km'],
                textposition='top right',
                textfont=dict(color=c, size=10),
                showlegend=False,
            ))

    for sat in satellites:
        p = sat.get_position_2d()
        fig.add_trace(go.Scatter(
            x=[p[0]], y=[p[1]], mode='markers+text',
            marker=dict(color=sat.color, size=10, symbol='triangle-up',
                        line=dict(color='white', width=1)),
            text=[f'S{sat.sat_id}'],
            textposition='top right',
            textfont=dict(color=sat.color, size=11),
            showlegend=False,
        ))

    if show_actual and user_position:
        pu = user_position.get_position_2d()
        fig.add_trace(go.Scatter(
            x=[pu[0]], y=[pu[1]], mode='markers+text',
            marker=dict(color='#00FF88', size=16, symbol='star',
                        line=dict(color='white', width=1)),
            text=[f'실제 위치<br>({pu[0]:.0f}, {pu[1]:.0f}) km'],
            textposition='top right',
            textfont=dict(color='#00FF88', size=10),
            showlegend=False,
        ))

    fig.update_layout(
        paper_bgcolor='#0A0A1A',
        plot_bgcolor='#0D0D20',
        xaxis=dict(range=[-GRID_RANGE, GRID_RANGE], color='#7777AA',
                   title='X (km)', gridcolor='#1A1A3A',
                   zeroline=True, zerolinecolor='#2A2A4A', dtick=5000),
        yaxis=dict(range=[-GRID_RANGE, GRID_RANGE], color='#7777AA',
                   title='Y (km)', gridcolor='#1A1A3A',
                   zeroline=True, zerolinecolor='#2A2A4A', dtick=5000,
                   scaleanchor='x', scaleratio=1),
        title=dict(text='TDOA 위치 측정 (2D 투영)',
                   font=dict(color='#E0E0FF', size=13), x=0.5),
        margin=dict(l=50, r=10, t=40, b=50),
        showlegend=False,
        uirevision='tdoa',
    )
    return fig

# ── 앱 시작 ───────────────────────────────────────────────────
st.set_page_config(
    page_title='TDOA 시뮬레이션',
    page_icon='🛰️',
    layout='wide',
)

st.markdown("""
<style>
.stApp { background-color: #0A0A1A !important; }
section[data-testid="stSidebar"] { background-color: #0D0D20 !important; }
.stButton > button {
    width: 100%;
    border-radius: 6px;
    font-weight: bold;
    background-color: #1A1A2E;
    color: #A0A0CC;
    border: 1px solid #333355;
}
.stButton > button:hover { border-color: #7B8CDE; color: #E0E0FF; }
code { font-size: 11px !important; }
</style>
""", unsafe_allow_html=True)

init_state()

n_sat_pre = len(st.session_state.satellites)

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🛰️ TDOA 시뮬레이션')
    st.markdown('---')

    st.markdown('**▌ 위성 제어**')
    ca, cb = st.columns(2)
    with ca:
        if st.button('＋ 추가',
                     disabled=st.session_state.calculating,
                     use_container_width=True):
            if n_sat_pre >= MAX_SATELLITES:
                st.session_state.status = f'최대 {MAX_SATELLITES}개'
            else:
                new_sat = Satellite(st.session_state.next_sat_id)
                st.session_state.next_sat_id += 1
                st.session_state.satellites.append(sat_to_dict(new_sat))
                st.session_state.status = f'S{new_sat.sat_id} 추가됨'
    with cb:
        if st.button('－ 제거',
                     disabled=st.session_state.calculating,
                     use_container_width=True):
            if not st.session_state.satellites:
                st.session_state.status = '위성 없음'
            else:
                rid = st.session_state.satellites[-1]['sat_id']
                st.session_state.satellites.pop()
                st.session_state.status = f'S{rid} 제거됨'

    st.markdown('---')

    st.markdown('**▌ 위치 계산**')
    if st.session_state.calculating:
        if st.button('■ 계산 중지', use_container_width=True):
            st.session_state.calculating = False
            st.session_state.active_pairs = 0
            st.session_state.status = '계산 중지됨'
    else:
        if st.button('▶ 계산 시작', use_container_width=True):
            if n_sat_pre < 2:
                st.session_state.status = '위성 2개 이상 필요'
            else:
                st.session_state.calculating = True
                st.session_state.status = '실시간 계산 중...'

    st.markdown('---')

    st.markdown('**▌ 결과**')
    if st.button('★ 실제 위치 보기/숨기기', use_container_width=True):
        st.session_state.show_actual = not st.session_state.show_actual
        st.session_state.status = ('실제 위치 표시' if st.session_state.show_actual
                                   else '실제 위치 숨김')

    st.markdown('---')
    if st.button('↺ 초기화', use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown('---')

    n_display = len(st.session_state.satellites)
    total_pairs = n_display * (n_display - 1) // 2
    st.markdown(f'**상태:** {st.session_state.status}')
    st.markdown(f'**위성:** {n_display}개 | **쌍곡선:** {st.session_state.active_pairs}/{total_pairs}개')
    if st.session_state.calculating:
        st.info('🔄 계산 & 애니메이션 진행 중')

# ── 차트 + 애니메이션 (fragment: 사이드바 깜빡임 없이 차트만 갱신) ──
@st.fragment(run_every=ANIM_DT if st.session_state.calculating else None)
def render_charts():
    # 최신 상태 읽기
    satellites = [dict_to_sat(d) for d in st.session_state.satellites]
    user_position = UserPosition(st.session_state.user_lat, st.session_state.user_lon)
    n_sat = len(satellites)
    noisy_pairs = (list_to_pairs(st.session_state.noisy_pairs)
                   if st.session_state.noisy_pairs else [])
    active_pairs = st.session_state.active_pairs

    # 계산 중일 때만 궤도 갱신 + TDOA 재계산
    if st.session_state.calculating and n_sat >= 2:
        for sat in satellites:
            sat.update(ANIM_DT)
        st.session_state.satellites = [sat_to_dict(s) for s in satellites]

        rng = np.random.default_rng()
        true_pairs = compute_tdoa_pairs(satellites, user_position)
        noisy = add_measurement_noise(true_pairs, rng)
        noisy_pairs = noisy
        active_pairs = n_sat * (n_sat - 1) // 2
        st.session_state.noisy_pairs = pairs_to_list(noisy)
        st.session_state.active_pairs = active_pairs

    # 차트 렌더링 (key를 고정하면 Plotly.react()로 데이터만 교체 → 깜빡임 없음)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(
            build_earth_fig(satellites, user_position, st.session_state.show_actual),
            use_container_width=True,
            key='earth_chart',
        )
    with col2:
        st.plotly_chart(
            build_tdoa_fig(satellites, user_position, noisy_pairs,
                           active_pairs, st.session_state.show_actual, n_sat),
            use_container_width=True,
            key='tdoa_chart',
        )

        if noisy_pairs and active_pairs > 0:
            with st.expander('쌍곡선 방정식', expanded=True):
                for idx, (pi, pj, delta_d, id_i, id_j) in enumerate(
                        noisy_pairs[:active_pairs]):
                    color = HYPER_COLORS[idx % len(HYPER_COLORS)]
                    eq = (f'H{idx + 1} S{id_i}-S{id_j}: '
                          f'√(({_term("x", pi[0])})²+({_term("y", pi[1])})²) '
                          f'− √(({_term("x", pj[0])})²+({_term("y", pj[1])})²) '
                          f'= {delta_d:+.0f} km')
                    st.markdown(
                        f'<code style="color:{color}">{eq}</code>',
                        unsafe_allow_html=True,
                    )

render_charts()
