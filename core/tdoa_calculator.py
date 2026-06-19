import numpy as np
from itertools import combinations

# TDOA 측정 오차 (거리 환산) - 클수록 초기 후보 영역이 넓음
MEASUREMENT_NOISE_STD = 0   # km


def compute_tdoa_pairs(satellites: list, user_pos) -> list:
    """실제 TDOA 쌍 계산 — 측정 시점 위치를 스냅샷으로 저장"""
    pairs = []
    for si, sj in combinations(satellites, 2):
        pi = si.get_position_2d().copy()
        pj = sj.get_position_2d().copy()
        pu = user_pos.get_position_2d()
        di = np.linalg.norm(pi - pu)
        dj = np.linalg.norm(pj - pu)
        pairs.append((pi, pj, di - dj, si.sat_id, sj.sat_id))
    return pairs


def add_measurement_noise(pairs: list, rng: np.random.Generator) -> list:
    """각 TDOA 쌍에 측정 오차(Gaussian) 추가."""
    noisy = []
    for pi, pj, delta_d, id_i, id_j in pairs:
        noise = rng.normal(0, MEASUREMENT_NOISE_STD)
        noisy.append((pi, pj, delta_d + noise, id_i, id_j))
    return noisy


def generate_hyperbola_2d(si_pos, sj_pos, delta_d: float,
                           grid_range: float, resolution: int = 300):
    """signed d_i - d_j 필드 반환 → contour at delta_d = 쌍곡선"""
    x = np.linspace(-grid_range, grid_range, resolution)
    y = np.linspace(-grid_range, grid_range, resolution)
    X, Y = np.meshgrid(x, y)
    di = np.sqrt((X - si_pos[0])**2 + (Y - si_pos[1])**2)
    dj = np.sqrt((X - sj_pos[0])**2 + (Y - sj_pos[1])**2)
    return X, Y, di - dj


def build_heatmap(noisy_pairs: list, active_pairs: int,
                  grid_range: float, resolution: int = 220):
    """
    곱셈 확률 히트맵.
    각 쌍의 log-Gaussian을 합산 → exp → 쌍이 많을수록 후보 영역 급감.
    """
    x = np.linspace(-grid_range, grid_range, resolution)
    y = np.linspace(-grid_range, grid_range, resolution)
    X, Y = np.meshgrid(x, y)

    if not noisy_pairs or active_pairs == 0:
        return X, Y, np.zeros_like(X)

    sigma = MEASUREMENT_NOISE_STD if MEASUREMENT_NOISE_STD > 0 else 30
    log_score = np.zeros_like(X, dtype=np.float64)

    for pi, pj, noisy_delta_d, *_ in noisy_pairs[:active_pairs]:
        di = np.sqrt((X - pi[0])**2 + (Y - pi[1])**2)
        dj = np.sqrt((X - pj[0])**2 + (Y - pj[1])**2)
        diff = (di - dj) - noisy_delta_d
        log_score += -(diff ** 2) / (2 * sigma ** 2)

    # 수치 안정 정규화 후 exp
    log_score -= log_score.max()
    score = np.exp(log_score)
    return X, Y, score


def estimate_position(noisy_pairs: list, active_pairs: int,
                      grid_range: float, resolution: int = 220):
    """히트맵 최댓값 위치 = 추정 위치"""
    X, Y, score = build_heatmap(noisy_pairs, active_pairs, grid_range, resolution)
    if score.max() == 0:
        return None
    idx = np.unravel_index(np.argmax(score), score.shape)
    return np.array([X[idx], Y[idx]])
