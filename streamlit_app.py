import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy import ndimage

from core.satellite import Satellite, UserPosition, EARTH_RADIUS
from core.tdoa_calculator import compute_tdoa_pairs, add_measurement_noise, build_heatmap

# ── 상수 ──────────────────────────────────────────────────────
MAX_SATELLITES = 4
GRID_RANGE     = 15_000.0
HEATMAP_RES    = 100

HYPER_COLORS = [
    '#00FFAA', '#FF6B35', '#4ECDC4', '#FFE66D',
    '#A8DADC', '#FF8FA3', '#C77DFF', '#80FFDB',
]

# ── 직렬화 ────────────────────────────────────────────────────
def sat_to_dict(sat):
    return {'sat_id': sat.sat_id, 'orbit_radius': sat.orbit_radius,
            'inclination': float(sat.inclination), 'raan': float(sat.raan),
            'angle': float(sat.angle), 'angular_speed': float(sat.angular_speed),
            'color': sat.color}

def dict_to_sat(d):
    sat = Satellite(sat_id=d['sat_id'], orbit_radius=d['orbit_radius'],
                    inclination=d['inclination'], raan=d['raan'],
                    initial_angle=d['angle'], angular_speed=d['angular_speed'])
    sat.color = d['color']
    return sat

def pairs_to_list(pairs):
    return [[p[0].tolist(), p[1].tolist(), float(p[2]), int(p[3]), int(p[4])] for p in pairs]

def list_to_pairs(lst):
    return [(np.array(x[0]), np.array(x[1]), float(x[2]), int(x[3]), int(x[4])) for x in lst]

def _term(axis, val):
    return f'{axis}-{val:.0f}' if val >= 0 else f'{axis}+{abs(val):.0f}'

# ── 쌍곡선 파라메트릭 계산 ─────────────────────────────────────
def _hyperbola_branch(p1, p2, delta_d, grid_range=GRID_RANGE, num_pts=400):
    """di - dj = delta_d 인 쌍곡선 한 가지(branch) 좌표 반환."""
    center = (p1 + p2) / 2.0
    dx, dy = p2[0] - p1[0], p2[1] - p1[1]
    c = np.hypot(dx, dy) / 2.0          # 초점간 거리의 절반
    a = abs(delta_d) / 2.0              # 쌍곡선 a 파라미터

    if a < 1e-3:
        # 퇴화 케이스: 수직이등분선
        perp = np.array([-dy, dx], dtype=float)
        norm = np.linalg.norm(perp)
        if norm < 1e-9:
            return [], []
        perp = perp / norm * grid_range
        return ([center[0]-perp[0], center[0]+perp[0]],
                [center[1]-perp[1], center[1]+perp[1]])

    if a >= c - 1e-6:
        return [], []   # 물리적으로 불가능한 경우

    b = np.sqrt(c**2 - a**2)
    angle = np.arctan2(dy, dx)
    ca, sa = np.cos(angle), np.sin(angle)

    # delta_d>0 → p1이 더 먼 쪽 branch, delta_d<0 → p2가 더 먼 쪽
    sign = 1 if delta_d > 0 else -1

    T = min(np.arccosh(max(grid_range / a, 1.0 + 1e-9)), 5.5)
    t = np.linspace(-T, T, num_pts)
    x_loc = sign * a * np.cosh(t)
    y_loc = b * np.sinh(t)

    xw = center[0] + ca * x_loc - sa * y_loc
    yw = center[1] + sa * x_loc + ca * y_loc

    mask = (np.abs(xw) <= grid_range * 1.05) & (np.abs(yw) <= grid_range * 1.05)
    if not np.any(mask):
        return [], []

    # 그리드 경계에서 None으로 선 끊기
    xs, ys = [], []
    prev_in = False
    for x, y, m in zip(xw, yw, mask):
        if m:
            xs.append(float(x)); ys.append(float(y))
            prev_in = True
        elif prev_in:
            xs.append(None); ys.append(None)
            prev_in = False
    return xs, ys

# ── 상태 초기화 ───────────────────────────────────────────────
def init_state():
    if 'initialized' in st.session_state:
        return
    sats = [Satellite(i) for i in range(1, 4)]
    user = UserPosition()
    st.session_state.satellites   = [sat_to_dict(s) for s in sats]
    st.session_state.user_lat     = float(user.lat)
    st.session_state.user_lon     = float(user.lon)
    st.session_state.calculating  = False
    st.session_state.show_actual  = False
    st.session_state.noisy_pairs  = []
    st.session_state.next_sat_id  = 4
    st.session_state.status       = '대기 중'
    st.session_state.active_pairs = 0
    st.session_state.initialized  = True

# ── 피크 탐지 ─────────────────────────────────────────────────
def _find_peaks(score, X, Y, min_rel=0.8, min_sep=1500.0, max_peaks=4):
    if score.max() == 0:
        return []
    sep_px = max(1, int(min_sep / (2 * GRID_RANGE / score.shape[0])))
    fp = np.ones((sep_px, sep_px), dtype=bool)
    local_max = ndimage.maximum_filter(score, footprint=fp) == score
    cands = sorted(np.argwhere(local_max & (score >= score.max() * min_rel)).tolist(),
                   key=lambda rc: score[rc[0], rc[1]], reverse=True)
    peaks, used = [], []
    for rc in cands:
        r, c = rc
        px, py = float(X[r, c]), float(Y[r, c])
        if not any(np.hypot(px-ex, py-ey) < min_sep for ex, ey, _ in used):
            used.append((px, py, float(score[r, c])))
            peaks.append((px, py, float(score[r, c])))
        if len(peaks) >= max_peaks:
            break
    return peaks

# ── 3D 지구 ───────────────────────────────────────────────────
@st.cache_data
def _earth_surface_data():
    u, v = np.linspace(0, 2*np.pi, 20), np.linspace(0, np.pi, 10)
    r = EARTH_RADIUS
    return (r*np.outer(np.cos(u), np.sin(v)),
            r*np.outer(np.sin(u), np.sin(v)),
            r*np.outer(np.ones(20), np.cos(v)))

def build_earth_fig(satellites, user_position, show_actual):
    fig = go.Figure()
    xs, ys, zs = _earth_surface_data()
    fig.add_trace(go.Surface(x=xs, y=ys, z=zs,
                             colorscale=[[0,'#1B4F8A'],[1,'#2A6DB5']],
                             opacity=0.8, showscale=False, hoverinfo='skip'))
    for sat in satellites:
        p = sat.get_position_3d()
        ox, oy, oz = sat.get_orbit_path_3d(40)
        fig.add_trace(go.Scatter3d(x=ox, y=oy, z=oz, mode='lines',
                                   line=dict(color=sat.color, width=1),
                                   opacity=0.4, showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]],
                                   mode='markers+text',
                                   marker=dict(color=sat.color, size=6,
                                               line=dict(color='white', width=1)),
                                   text=[f'S{sat.sat_id}'], textposition='top center',
                                   textfont=dict(color=sat.color, size=11),
                                   showlegend=False))
    if show_actual and user_position:
        pu = user_position.get_position_3d()
        d  = pu / np.linalg.norm(pu)
        sp, op = d * EARTH_RADIUS, d * (EARTH_RADIUS + 2500)
        fig.add_trace(go.Scatter3d(x=[sp[0],op[0]], y=[sp[1],op[1]], z=[sp[2],op[2]],
                                   mode='lines',
                                   line=dict(color='#FF3366', width=2, dash='dash'),
                                   showlegend=False, hoverinfo='skip'))
        fig.add_trace(go.Scatter3d(x=[op[0]], y=[op[1]], z=[op[2]],
                                   mode='text', text=['★'],
                                   textfont=dict(color='#FF3366', size=22),
                                   showlegend=False, hoverinfo='skip'))
    lim = 12000
    fig.update_layout(
        paper_bgcolor='#0A0A1A',
        scene=dict(bgcolor='#0A0A1A',
                   xaxis=dict(range=[-lim,lim], visible=False),
                   yaxis=dict(range=[-lim,lim], visible=False),
                   zaxis=dict(range=[-lim,lim], visible=False),
                   aspectmode='cube'),
        title=dict(text='위성 궤도 (3D)', font=dict(color='#E0E0FF', size=13), x=0.5),
        margin=dict(l=0,r=0,t=40,b=0), showlegend=False, uirevision='earth')
    return fig

# ── 2D TDOA ───────────────────────────────────────────────────
def build_tdoa_fig(satellites, user_position, noisy_pairs, active_pairs, show_actual, n_sat):
    fig = go.Figure()
    x1d = np.linspace(-GRID_RANGE, GRID_RANGE, HEATMAP_RES)

    if satellites and active_pairs > 0 and noisy_pairs:
        # 히트맵
        X, Y, score = build_heatmap(noisy_pairs, active_pairs, GRID_RANGE, HEATMAP_RES)
        if score.max() > 0:
            fig.add_trace(go.Heatmap(x=x1d, y=x1d, z=score,
                                     colorscale='Plasma', opacity=0.55,
                                     showscale=False, hoverinfo='skip'))

        # 쌍곡선 — 파라메트릭 곡선으로 직접 계산
        for idx, (pi, pj, delta_d, id_i, id_j) in enumerate(noisy_pairs[:active_pairs]):
            color = HYPER_COLORS[idx % len(HYPER_COLORS)]

            # 실제 branch (사용자가 있는 쪽) — 실선
            bx1, by1 = _hyperbola_branch(pi, pj, delta_d)
            if bx1:
                fig.add_trace(go.Scatter(x=bx1, y=by1, mode='lines',
                                         line=dict(color=color, width=2),
                                         showlegend=False, hoverinfo='skip'))
            # 반대 branch — 점선
            bx2, by2 = _hyperbola_branch(pi, pj, -delta_d)
            if bx2:
                fig.add_trace(go.Scatter(x=bx2, y=by2, mode='lines',
                                         line=dict(color=color, width=1, dash='dash'),
                                         opacity=0.35, showlegend=False, hoverinfo='skip'))

        # 추정 위치
        all_shown = active_pairs >= len(noisy_pairs) > 0
        if all_shown and n_sat == 3:
            peaks = _find_peaks(score, X, Y, max_peaks=2)
        elif all_shown and n_sat >= 4:
            peaks = _find_peaks(score, X, Y, max_peaks=1)
        else:
            peaks = []

        for i, (px, py, _) in enumerate(peaks):
            label = f'추정 위치 {i+1}' if len(peaks) > 1 else '추정 위치'
            c = '#FF2244' if i == 0 else '#FF9900'
            fig.add_trace(go.Scatter(x=[px], y=[py], mode='markers+text',
                                     marker=dict(color=c, size=14, symbol='circle',
                                                 line=dict(color='white', width=2)),
                                     text=[f'{label}<br>({px:.0f},{py:.0f})km'],
                                     textposition='top right',
                                     textfont=dict(color=c, size=10), showlegend=False))

    # 위성 위치
    for sat in satellites:
        p = sat.get_position_2d()
        fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode='markers+text',
                                 marker=dict(color=sat.color, size=10, symbol='triangle-up',
                                             line=dict(color='white', width=1)),
                                 text=[f'S{sat.sat_id}'], textposition='top right',
                                 textfont=dict(color=sat.color, size=11), showlegend=False))

    if show_actual and user_position:
        pu = user_position.get_position_2d()
        fig.add_trace(go.Scatter(x=[pu[0]], y=[pu[1]], mode='markers+text',
                                 marker=dict(color='#00FF88', size=16, symbol='star',
                                             line=dict(color='white', width=1)),
                                 text=[f'실제 위치<br>({pu[0]:.0f},{pu[1]:.0f})km'],
                                 textposition='top right',
                                 textfont=dict(color='#00FF88', size=10), showlegend=False))

    fig.update_layout(
        paper_bgcolor='#0A0A1A', plot_bgcolor='#0D0D20',
        xaxis=dict(range=[-GRID_RANGE,GRID_RANGE], color='#7777AA', title='X (km)',
                   gridcolor='#1A1A3A', zeroline=True, zerolinecolor='#2A2A4A', dtick=5000),
        yaxis=dict(range=[-GRID_RANGE,GRID_RANGE], color='#7777AA', title='Y (km)',
                   gridcolor='#1A1A3A', zeroline=True, zerolinecolor='#2A2A4A', dtick=5000,
                   scaleanchor='x', scaleratio=1),
        title=dict(text='TDOA 위치 측정 (2D 투영)', font=dict(color='#E0E0FF', size=13), x=0.5),
        margin=dict(l=50,r=10,t=40,b=50), showlegend=False, uirevision='tdoa')
    return fig

# ── 앱 ────────────────────────────────────────────────────────
st.set_page_config(page_title='TDOA 시뮬레이션', page_icon='🛰️', layout='wide')
st.markdown("""
<style>
.stApp{background-color:#0A0A1A!important}
section[data-testid="stSidebar"]{background-color:#0D0D20!important}
.stButton>button{width:100%;border-radius:6px;font-weight:bold;
  background-color:#1A1A2E;color:#A0A0CC;border:1px solid #333355}
.stButton>button:hover{border-color:#7B8CDE;color:#E0E0FF}
.stButton>button:disabled{opacity:0.35;cursor:not-allowed}
code{font-size:11px!important}
</style>""", unsafe_allow_html=True)

init_state()

# ── 버튼 콜백 (on_click으로 상태를 재실행 전에 먼저 업데이트) ──
def _add_satellite():
    n = len(st.session_state.satellites)
    if n >= MAX_SATELLITES:
        st.session_state.status = f'최대 {MAX_SATELLITES}개'
    else:
        s = Satellite(st.session_state.next_sat_id)
        st.session_state.next_sat_id += 1
        st.session_state.satellites.append(sat_to_dict(s))
        st.session_state.calculating  = False
        st.session_state.noisy_pairs  = []
        st.session_state.active_pairs = 0
        st.session_state.status = f'S{s.sat_id} 추가됨'

def _remove_satellite():
    if not st.session_state.satellites:
        st.session_state.status = '위성 없음'
    else:
        rid = st.session_state.satellites[-1]['sat_id']
        st.session_state.satellites.pop()
        st.session_state.calculating  = False
        st.session_state.noisy_pairs  = []
        st.session_state.active_pairs = 0
        st.session_state.status = f'S{rid} 제거됨'

def _start_calc():
    n = len(st.session_state.satellites)
    if n < 2:
        st.session_state.status = '위성 2개 이상 필요'
        return
    sats = [dict_to_sat(d) for d in st.session_state.satellites]
    user = UserPosition(st.session_state.user_lat, st.session_state.user_lon)
    rng  = np.random.default_rng()
    noisy = add_measurement_noise(compute_tdoa_pairs(sats, user), rng)
    st.session_state.noisy_pairs  = pairs_to_list(noisy)
    st.session_state.active_pairs = n * (n - 1) // 2
    st.session_state.calculating  = True
    st.session_state.status       = '계산 완료'

def _stop_calc():
    st.session_state.calculating  = False
    st.session_state.noisy_pairs  = []
    st.session_state.active_pairs = 0
    st.session_state.status       = '대기 중'

def _toggle_actual():
    st.session_state.show_actual = not st.session_state.show_actual
    st.session_state.status = ('실제 위치 표시' if st.session_state.show_actual
                               else '실제 위치 숨김')

def _reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]

# ── 사이드바 ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 🛰️ TDOA 시뮬레이션')
    st.markdown('---')

    st.markdown('**▌ 위성 제어**')
    ca, cb = st.columns(2)
    with ca:
        st.button('＋ 추가', on_click=_add_satellite, use_container_width=True)
    with cb:
        st.button('－ 제거', on_click=_remove_satellite, use_container_width=True)

    st.markdown('---')
    st.markdown('**▌ 위치 계산**')
    st.button('▶ 계산 시작', on_click=_start_calc,
              disabled=st.session_state.calculating, use_container_width=True)
    st.button('■ 계산 중지', on_click=_stop_calc,
              disabled=not st.session_state.calculating, use_container_width=True)

    st.markdown('---')
    st.markdown('**▌ 결과**')
    st.button('★ 실제 위치 보기/숨기기', on_click=_toggle_actual, use_container_width=True)

    st.markdown('---')
    st.button('↺ 초기화', on_click=_reset, use_container_width=True)

    st.markdown('---')
    n_d = len(st.session_state.satellites)
    st.markdown(f'**상태:** {st.session_state.status}')
    st.markdown(f'**위성:** {n_d}개 | **쌍곡선:** {st.session_state.active_pairs}/{n_d*(n_d-1)//2}개')
    if st.session_state.calculating:
        st.success('📌 TDOA 계산 결과 표시 중')

# ── 렌더링 ────────────────────────────────────────────────────
sats     = [dict_to_sat(d) for d in st.session_state.satellites]
user_pos = UserPosition(st.session_state.user_lat, st.session_state.user_lon)
n_sat    = len(sats)
noisy_pairs  = list_to_pairs(st.session_state.noisy_pairs) if st.session_state.noisy_pairs else []
active_pairs = st.session_state.active_pairs

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(build_earth_fig(sats, user_pos, st.session_state.show_actual),
                    use_container_width=True, key='earth_chart')
with col2:
    st.plotly_chart(build_tdoa_fig(sats, user_pos, noisy_pairs, active_pairs,
                                   st.session_state.show_actual, n_sat),
                    use_container_width=True, key='tdoa_chart')

    if noisy_pairs and active_pairs > 0:
        with st.expander('쌍곡선 방정식', expanded=True):
            for idx, (pi, pj, delta_d, id_i, id_j) in enumerate(noisy_pairs[:active_pairs]):
                color = HYPER_COLORS[idx % len(HYPER_COLORS)]
                eq = (f'H{idx+1} S{id_i}-S{id_j}: '
                      f'√(({_term("x",pi[0])})²+({_term("y",pi[1])})²) '
                      f'− √(({_term("x",pj[0])})²+({_term("y",pj[1])})²) '
                      f'= {delta_d:+.0f} km')
                st.markdown(f'<code style="color:{color}">{eq}</code>',
                            unsafe_allow_html=True)
