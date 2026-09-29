import math
from pathlib import Path
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# =========================================================
# User configuration
# =========================================================

TARGET_VOLTAGE = 3.3
H_REF_FOR_ACC = 8

CSV_PATH = "right.csv"

# Ground truth is used ONLY for evaluation.
# It is not used for endpoint localization or trajectory reconstruction.
#
# Format: (x, y), ordered from early to late exposure.
GT_FULL_POINTS = [
    (2, 3),
    (3, 3),
    (4, 3),
    (4, 4),
    (4, 5),
    (4, 6),  # hidden point if PRED_STEPS = 1
]

PRED_STEPS = 1


# =========================================================
# Dynamic-programming parameters
# =========================================================
#
# The cost/reward terms retain the same physical meaning as
# the SI formulation:
#
#   λ : response reward
#   μ : spatial-continuity penalty
#   ρ : weak-signal / path-length penalty
#   Jmax : maximum local vertical displacement
#
# For the measured 8x8 L-shaped trajectories, the path is
# represented as an ordered 2D sequence p_k=(x_k,y_k).
#
# This is a local 2D extension of the column-wise DP idea:
# x is not allowed to reverse, but Δx=0 is allowed, so the
# same x-column may contain several consecutive y positions.
# =========================================================

LAMBDA_RESPONSE = 1.0      # λ
MU_STEP = 0.18             # μ

RHO_BASE = 0.10
RHO_WEAK = 0.20
WEAK_SIGNAL_THRESHOLD = 0.12

ACTIVE_RESPONSE_THRESHOLD = 0.05

J_MAX = 1
MAX_X_STEP = 1

# Residual response should increase toward the endpoint.
MIN_RESPONSE_RISE = 1e-9

MIN_RECON_POINTS = 2


# =========================================================
# Reconstruction smoothing
# =========================================================

SMOOTH_WIN = 3
KEEP_SMOOTH_ENDPOINTS = True

# Raw DP path is the primary reconstruction output used for
# quantitative reconstruction metrics.
USE_SMOOTH_FOR_PRED = True

# Plot-only smoothing
PLOT_SMOOTH_CURVE = True
CURVE_SMOOTH_METHOD = "catmull_rom"
CURVE_POINTS_PER_SEGMENT = 30
CHAIKIN_ITERATIONS = 3


# =========================================================
# Kalman prediction
# =========================================================
#
# To stay close to the scalar three-state formulation in the SI,
# the same [position, velocity, acceleration] Kalman model is
# applied independently to x and y.
# This is mathematically equivalent to a block-diagonal 2D
# constant-acceleration Kalman model.
# =========================================================

KF_FIT_WIN = 5
KF_DT = 1.0
KF_PROCESS_VAR = 0.05
KF_MEAS_VAR = 0.8


# =========================================================
# Optional speed estimation from relaxation inversion
# =========================================================

ESTIMATE_SPEED_FROM_RELAXATION = True
READABLE_WINDOW_S = 40.0
RELAXATION_GRID_N = 20000

A1 = 3.87821e-7
TAU1 = 0.34921
A2 = 3.50008e-7
TAU2 = 2.97588
A3 = 3.00247e-7
TAU3 = 30.47546
Y0 = 4.46074e-8

I_MAX_T0 = A1 + A2 + A3 + Y0
TRANSIMPEDANCE_GAIN = TARGET_VOLTAGE / I_MAX_T0


# =========================================================
# Output
# =========================================================

OUTPUT_DIR = "measured_8x8_outputs"


# =========================================================
# IO
# =========================================================

def load_voltage_matrix(csv_path: str) -> np.ndarray:
    V = np.loadtxt(csv_path, delimiter=",")
    if V.ndim != 2:
        raise ValueError("Voltage matrix must be 2D.")
    if np.any(~np.isfinite(V)):
        raise ValueError("Voltage matrix contains NaN or Inf.")
    return V.astype(float)


# =========================================================
# Relaxation model and inversion
# =========================================================

def relaxation_model_current(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    return (
        A1 * np.exp(-t / TAU1)
        + A2 * np.exp(-t / TAU2)
        + A3 * np.exp(-t / TAU3)
        + Y0
    )


def current_to_voltage(current: np.ndarray) -> np.ndarray:
    return np.asarray(current, dtype=float) * TRANSIMPEDANCE_GAIN


def invert_relaxation_time_from_voltage(
    voltage_values: np.ndarray,
    t_max: float = READABLE_WINDOW_S,
    n_grid: int = RELAXATION_GRID_N,
) -> np.ndarray:
    voltage_values = np.asarray(voltage_values, dtype=float)

    t_grid = np.linspace(0.0, t_max, n_grid)
    i_grid = relaxation_model_current(t_grid)
    v_grid = current_to_voltage(i_grid)

    v_min = float(np.min(v_grid))
    v_max = float(np.max(v_grid))
    voltage_values = np.clip(voltage_values, v_min, v_max)

    return np.interp(
        voltage_values,
        v_grid[::-1],
        t_grid[::-1],
    )


# =========================================================
# DP helpers
# =========================================================

def normalize_voltage(V: np.ndarray) -> np.ndarray:
    V = np.asarray(V, dtype=float)
    vmin = float(np.min(V))
    vmax = float(np.max(V))
    denom = max(vmax - vmin, 1e-12)
    return (V - vmin) / denom


def find_global_endpoint(V: np.ndarray) -> Tuple[int, int]:
    """
    Endpoint localization corresponding to the global-response
    maximum assumption in the SI.
    """
    y_end, x_end = np.unravel_index(
        np.argmax(V),
        V.shape,
    )
    return int(x_end), int(y_end)


def rho_penalty(v_norm: float) -> float:
    weak = 0.0

    if WEAK_SIGNAL_THRESHOLD > 0:
        deficit = max(
            0.0,
            WEAK_SIGNAL_THRESHOLD - float(v_norm),
        )
        weak = (
            RHO_WEAK
            * deficit
            / WEAK_SIGNAL_THRESHOLD
        )

    return RHO_BASE + weak


def node_reward(v_norm: float) -> float:
    return (
        LAMBDA_RESPONSE * float(v_norm)
        - rho_penalty(float(v_norm))
    )


def allowed_transition(
    p_xy: Tuple[int, int],
    q_xy: Tuple[int, int],
    Vn: np.ndarray,
) -> bool:
    """
    Forward temporal transition.

    Conditions:
      - no x-direction reversal;
      - Δx=0 is allowed for vertical/L-shaped segments;
      - Δx <= MAX_X_STEP;
      - |Δy| <= J_MAX;
      - residual response increases toward the endpoint.
    """
    px, py = p_xy
    qx, qy = q_xy

    dx = qx - px
    dy = qy - py

    if dx < 0:
        return False

    if dx > MAX_X_STEP:
        return False

    if dx == 0 and dy == 0:
        return False

    if abs(dy) > J_MAX:
        return False

    vp = float(Vn[py, px])
    vq = float(Vn[qy, qx])

    if vq < vp + MIN_RESPONSE_RISE:
        return False

    return True


def reconstruct_path_dp_2d(
    V: np.ndarray,
) -> Tuple[np.ndarray, Tuple[int, int], np.ndarray, Dict]:
    """
    GT-independent 2D DP reconstruction.

    The global response maximum is used as the endpoint.
    Search is restricted to x <= x_end.
    """
    V = np.asarray(V, dtype=float)
    H, W = V.shape

    Vn = normalize_voltage(V)
    endpoint = find_global_endpoint(V)
    x_end, y_end = endpoint

    candidate_nodes: List[Tuple[int, int]] = []

    for x in range(0, x_end + 1):
        for y in range(H):
            if (
                Vn[y, x] >= ACTIVE_RESPONSE_THRESHOLD
                or (x == x_end and y == y_end)
            ):
                candidate_nodes.append((x, y))

    # Response ordering makes the graph acyclic under the
    # monotonic-response transition rule.
    candidate_nodes.sort(
        key=lambda p: (
            float(Vn[p[1], p[0]]),
            p[0],
            p[1],
        )
    )

    if endpoint not in candidate_nodes:
        candidate_nodes.append(endpoint)

    best_score: Dict[Tuple[int, int], float] = {}
    predecessor: Dict[
        Tuple[int, int],
        Optional[Tuple[int, int]]
    ] = {}
    path_length: Dict[Tuple[int, int], int] = {}

    for q in candidate_nodes:
        qx, qy = q
        local_reward = node_reward(Vn[qy, qx])

        best_score[q] = local_reward
        predecessor[q] = None
        path_length[q] = 1

        for p in candidate_nodes:
            if p == q:
                continue

            if (
                float(Vn[p[1], p[0]])
                >= float(Vn[qy, qx])
            ):
                continue

            if p not in best_score:
                continue

            if not allowed_transition(
                p,
                q,
                Vn,
            ):
                continue

            px, py = p

            step_distance = float(
                np.hypot(
                    qx - px,
                    qy - py,
                )
            )

            transition_penalty = (
                MU_STEP * step_distance
            )

            score = (
                best_score[p]
                + local_reward
                - transition_penalty
            )

            length = path_length[p] + 1

            if (
                score > best_score[q] + 1e-12
                or (
                    abs(
                        score - best_score[q]
                    ) <= 1e-12
                    and length > path_length[q]
                )
            ):
                best_score[q] = score
                predecessor[q] = p
                path_length[q] = length

    if endpoint not in best_score:
        raise RuntimeError(
            "Global endpoint is absent from the DP state set."
        )

    path_rev = []
    current = endpoint

    while current is not None:
        path_rev.append(current)
        current = predecessor[current]

    path = path_rev[::-1]

    # Diagnostic fallback only if the optimum degenerates
    # into the endpoint alone.
    if len(path) < MIN_RECON_POINTS:
        valid_prev = []

        for p in candidate_nodes:
            if p == endpoint:
                continue

            if allowed_transition(
                p,
                endpoint,
                Vn,
            ):
                valid_prev.append(p)

        if len(valid_prev) > 0:
            best_prev = max(
                valid_prev,
                key=lambda p: (
                    node_reward(
                        Vn[p[1], p[0]]
                    )
                    - MU_STEP
                    * np.hypot(
                        endpoint[0] - p[0],
                        endpoint[1] - p[1],
                    )
                ),
            )
            path = [
                best_prev,
                endpoint,
            ]

    path_xy = np.asarray(
        path,
        dtype=float,
    )

    diagnostics = {
        "candidate_count":
            int(len(candidate_nodes)),
        "path_length":
            int(len(path_xy)),
        "best_score_endpoint":
            float(best_score[endpoint]),
    }

    return (
        path_xy,
        endpoint,
        Vn,
        diagnostics,
    )


# =========================================================
# Smoothing
# =========================================================

def smooth_xy_trajectory(
    xy: np.ndarray,
    win: int = 3,
    keep_endpoints: bool = True,
) -> np.ndarray:
    xy = np.asarray(
        xy,
        dtype=float,
    )

    n = len(xy)

    if n <= 2 or win <= 1:
        return xy.copy()

    win = min(
        win,
        n,
    )

    if win % 2 == 0:
        win = max(
            1,
            win - 1,
        )

    if win <= 1:
        return xy.copy()

    pad = win // 2
    out = np.zeros_like(
        xy,
        dtype=float,
    )

    for dim in range(2):
        values = xy[:, dim]
        padded = np.pad(
            values,
            (pad, pad),
            mode="edge",
        )

        kernel = (
            np.ones(win)
            / win
        )

        out[:, dim] = np.convolve(
            padded,
            kernel,
            mode="valid",
        )

    if keep_endpoints:
        out[0] = xy[0]
        out[-1] = xy[-1]

    return out


# =========================================================
# Plot-only smoothing
# =========================================================

def catmull_rom_spline(
    points: np.ndarray,
    points_per_segment: int = 30,
) -> np.ndarray:
    P = np.asarray(
        points,
        dtype=float,
    )

    if len(P) < 3:
        return P.copy()

    P_ext = np.vstack([
        P[0],
        P,
        P[-1],
    ])

    curve = []

    for i in range(
        1,
        len(P_ext) - 2,
    ):
        p0 = P_ext[i - 1]
        p1 = P_ext[i]
        p2 = P_ext[i + 1]
        p3 = P_ext[i + 2]

        t_values = np.linspace(
            0,
            1,
            points_per_segment,
            endpoint=False,
        )

        for t in t_values:
            t2 = t * t
            t3 = t2 * t

            point = 0.5 * (
                (2 * p1)
                + (-p0 + p2) * t
                + (
                    2 * p0
                    - 5 * p1
                    + 4 * p2
                    - p3
                ) * t2
                + (
                    -p0
                    + 3 * p1
                    - 3 * p2
                    + p3
                ) * t3
            )

            curve.append(point)

    curve.append(P[-1])

    return np.asarray(
        curve,
        dtype=float,
    )


def chaikin_smooth_polyline(
    points: np.ndarray,
    iterations: int = 3,
) -> np.ndarray:
    P = np.asarray(
        points,
        dtype=float,
    )

    if len(P) < 3:
        return P.copy()

    curve = P.copy()

    for _ in range(iterations):
        new_points = [curve[0]]

        for i in range(
            len(curve) - 1
        ):
            p = curve[i]
            q = curve[i + 1]

            Q = (
                0.75 * p
                + 0.25 * q
            )

            R = (
                0.25 * p
                + 0.75 * q
            )

            new_points.extend([
                Q,
                R,
            ])

        new_points.append(
            curve[-1]
        )

        curve = np.asarray(
            new_points,
            dtype=float,
        )

    return curve


def make_display_curve(
    points: np.ndarray,
) -> np.ndarray:
    if (
        not PLOT_SMOOTH_CURVE
        or len(points) < 3
    ):
        return np.asarray(
            points,
            dtype=float,
        )

    method = (
        CURVE_SMOOTH_METHOD
        .lower()
        .strip()
    )

    if method == "catmull_rom":
        return catmull_rom_spline(
            points,
            points_per_segment=
                CURVE_POINTS_PER_SEGMENT,
        )

    if method == "chaikin":
        return chaikin_smooth_polyline(
            points,
            iterations=
                CHAIKIN_ITERATIONS,
        )

    return np.asarray(
        points,
        dtype=float,
    )


# =========================================================
# Scalar three-state Kalman filter
# =========================================================

def kalman_predict_scalar_ca(
    observations: np.ndarray,
    steps: int,
    fit_win: int = KF_FIT_WIN,
    dt: float = KF_DT,
    process_var: float = KF_PROCESS_VAR,
    meas_var: float = KF_MEAS_VAR,
) -> Optional[np.ndarray]:
    """
    Scalar constant-acceleration Kalman model:

        state = [position, velocity, acceleration]^T

    The same function is applied independently to x and y.
    """
    obs = np.asarray(
        observations[-fit_win:],
        dtype=float,
    )

    if (
        len(obs) < 3
        or steps <= 0
    ):
        return None

    v0 = (
        obs[1]
        - obs[0]
    ) / dt

    a0 = (
        obs[2]
        - 2.0 * obs[1]
        + obs[0]
    ) / (dt ** 2)

    state = np.array(
        [
            obs[0],
            v0,
            a0,
        ],
        dtype=float,
    )

    P = np.eye(
        3,
        dtype=float,
    )

    F = np.array(
        [
            [
                1.0,
                dt,
                0.5 * dt * dt,
            ],
            [
                0.0,
                1.0,
                dt,
            ],
            [
                0.0,
                0.0,
                1.0,
            ],
        ],
        dtype=float,
    )

    Hm = np.array(
        [[1.0, 0.0, 0.0]],
        dtype=float,
    )

    Q = process_var * np.array(
        [
            [
                dt ** 4 / 4.0,
                dt ** 3 / 2.0,
                dt ** 2 / 2.0,
            ],
            [
                dt ** 3 / 2.0,
                dt ** 2,
                dt,
            ],
            [
                dt ** 2 / 2.0,
                dt,
                1.0,
            ],
        ],
        dtype=float,
    )

    R = np.array(
        [[meas_var]],
        dtype=float,
    )

    for z in obs[1:]:
        state = F @ state
        P = (
            F @ P @ F.T
            + Q
        )

        residual = np.array(
            [[z - (Hm @ state)[0]]],
            dtype=float,
        )

        S = (
            Hm @ P @ Hm.T
            + R
        )

        K = (
            P
            @ Hm.T
            @ np.linalg.inv(S)
        )

        state = (
            state
            + (K @ residual).flatten()
        )

        P = (
            np.eye(3)
            - K @ Hm
        ) @ P

        state[1] = np.clip(
            state[1],
            -3.0,
            3.0,
        )

        state[2] = np.clip(
            state[2],
            -2.0,
            2.0,
        )

    state[0] = obs[-1]

    pred = []

    for _ in range(steps):
        state = F @ state
        P = (
            F @ P @ F.T
            + Q
        )

        state[1] = np.clip(
            state[1],
            -3.0,
            3.0,
        )

        state[2] = np.clip(
            state[2],
            -2.0,
            2.0,
        )

        pred.append(
            float(state[0])
        )

    return np.asarray(
        pred,
        dtype=float,
    )


def predict_trajectory_2d(
    xy_hist: np.ndarray,
    steps: int,
    W_limit: int,
    H_limit: int,
) -> Optional[np.ndarray]:
    xy_hist = np.asarray(
        xy_hist,
        dtype=float,
    )

    if (
        len(xy_hist) < 3
        or steps <= 0
    ):
        return None

    x_pred = kalman_predict_scalar_ca(
        xy_hist[:, 0],
        steps=steps,
    )

    y_pred = kalman_predict_scalar_ca(
        xy_hist[:, 1],
        steps=steps,
    )

    if (
        x_pred is None
        or y_pred is None
    ):
        return None

    pred = np.column_stack([
        x_pred,
        y_pred,
    ])

    pred[:, 0] = np.clip(
        pred[:, 0],
        0,
        W_limit - 1,
    )

    pred[:, 1] = np.clip(
        pred[:, 1],
        0,
        H_limit - 1,
    )

    return pred


# =========================================================
# Metrics
# =========================================================

def resample_polyline_by_arclength(
    xy: np.ndarray,
    n_samples: int,
) -> np.ndarray:
    xy = np.asarray(
        xy,
        dtype=float,
    )

    if n_samples <= 0:
        return np.empty(
            (0, 2),
            dtype=float,
        )

    if len(xy) == 0:
        return np.empty(
            (0, 2),
            dtype=float,
        )

    if len(xy) == 1:
        return np.repeat(
            xy,
            n_samples,
            axis=0,
        )

    ds = np.sqrt(
        np.sum(
            np.diff(
                xy,
                axis=0,
            ) ** 2,
            axis=1,
        )
    )

    s = np.concatenate([
        [0.0],
        np.cumsum(ds),
    ])

    total = float(s[-1])

    if total <= 1e-12:
        return np.repeat(
            xy[:1],
            n_samples,
            axis=0,
        )

    s_new = np.linspace(
        0.0,
        total,
        n_samples,
    )

    x_new = np.interp(
        s_new,
        s,
        xy[:, 0],
    )

    y_new = np.interp(
        s_new,
        s,
        xy[:, 1],
    )

    return np.column_stack([
        x_new,
        y_new,
    ])


def measured_2d_metrics(
    gt_xy: np.ndarray,
    rec_xy: np.ndarray,
    ref_span_px: float = H_REF_FOR_ACC,
) -> dict:
    gt_xy = np.asarray(
        gt_xy,
        dtype=float,
    )

    rec_xy = np.asarray(
        rec_xy,
        dtype=float,
    )

    if (
        len(gt_xy) == 0
        or len(rec_xy) == 0
    ):
        return {
            "mae_pixel": math.nan,
            "rmse_pixel": math.nan,
            "accuracy_percent": math.nan,
        }

    if len(rec_xy) != len(gt_xy):
        rec_eval = (
            resample_polyline_by_arclength(
                rec_xy,
                len(gt_xy),
            )
        )
    else:
        rec_eval = rec_xy.copy()

    err = (
        rec_eval
        - gt_xy
    )

    dist = np.sqrt(
        np.sum(
            err ** 2,
            axis=1,
        )
    )

    mae = float(
        np.mean(dist)
    )

    rmse = float(
        np.sqrt(
            np.mean(
                dist ** 2
            )
        )
    )

    acc = float(
        max(
            0.0,
            100.0
            * (
                1.0
                - rmse
                / ref_span_px
            ),
        )
    )

    return {
        "mae_pixel": mae,
        "rmse_pixel": rmse,
        "accuracy_percent": acc,
    }


def estimate_mean_speed_pixels_per_step(
    xy_path: np.ndarray,
) -> float:
    xy_path = np.asarray(
        xy_path,
        dtype=float,
    )

    if len(xy_path) < 2:
        return math.nan

    step_dist = np.sqrt(
        np.sum(
            np.diff(
                xy_path,
                axis=0,
            ) ** 2,
            axis=1,
        )
    )

    return float(
        np.mean(step_dist)
    )


def estimate_mean_speed_from_relaxation(
    xy_path: np.ndarray,
    V_full: np.ndarray,
    collection_time_s: float = READABLE_WINDOW_S,
) -> float:
    xy_path = np.asarray(
        xy_path,
        dtype=float,
    )

    if len(xy_path) < 2:
        return math.nan

    H, W = V_full.shape

    x_idx = np.clip(
        np.rint(
            xy_path[:, 0]
        ).astype(int),
        0,
        W - 1,
    )

    y_idx = np.clip(
        np.rint(
            xy_path[:, 1]
        ).astype(int),
        0,
        H - 1,
    )

    voltages = np.array(
        [
            V_full[y, x]
            for x, y
            in zip(
                x_idx,
                y_idx,
            )
        ],
        dtype=float,
    )

    t_since = (
        invert_relaxation_time_from_voltage(
            voltages
        )
    )

    t_since = np.clip(
        t_since,
        0.0,
        collection_time_s,
    )

    t_event = (
        collection_time_s
        - t_since
    )

    dxy = np.diff(
        xy_path,
        axis=0,
    )

    dist = np.sqrt(
        np.sum(
            dxy ** 2,
            axis=1,
        )
    )

    dt = np.diff(
        t_event
    )

    valid = (
        np.isfinite(dt)
        & (np.abs(dt) > 1e-6)
    )

    if np.sum(valid) == 0:
        return math.nan

    speed = (
        dist[valid]
        / np.abs(
            dt[valid]
        )
    )

    speed = speed[
        np.isfinite(speed)
    ]

    if len(speed) == 0:
        return math.nan

    return float(
        np.mean(speed)
    )


def estimate_prediction_direction_2d(
    pred_xy: Optional[np.ndarray],
    start_xy: np.ndarray,
) -> float:
    if (
        pred_xy is None
        or len(pred_xy) < 1
    ):
        return math.nan

    start_xy = np.asarray(
        start_xy,
        dtype=float,
    )

    dx = float(
        pred_xy[-1, 0]
        - start_xy[0]
    )

    dy = float(
        pred_xy[-1, 1]
        - start_xy[1]
    )

    if (
        abs(dx) < 1e-12
        and abs(dy) < 1e-12
    ):
        return math.nan

    return float(
        np.degrees(
            np.arctan2(
                dy,
                dx,
            )
        )
    )


# =========================================================
# Plotting
# =========================================================

def plot_all(
    V_full: np.ndarray,
    gt_visible: np.ndarray,
    gt_hidden: np.ndarray,
    rec_raw: np.ndarray,
    rec_smooth: np.ndarray,
    pred_xy: Optional[np.ndarray],
    endpoint: Tuple[int, int],
    out_png: Path,
):
    cmap = LinearSegmentedColormap.from_list(
        "w2p",
        [
            (1, 1, 1),
            (0.72, 0.52, 0.95),
        ],
    )

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    hm = ax.imshow(
        V_full,
        cmap=cmap,
        vmin=0,
        vmax=TARGET_VOLTAGE,
        origin="upper",
        aspect="equal",
    )

    plt.colorbar(
        hm,
        ax=ax,
        label="Voltage (V)",
    )

    if len(gt_visible) > 0:
        ax.plot(
            gt_visible[:, 0],
            gt_visible[:, 1],
            "-o",
            color="gray",
            alpha=0.55,
            lw=1.6,
            ms=5,
            label="GT available",
        )

    if len(gt_hidden) > 0:
        ax.plot(
            gt_hidden[:, 0],
            gt_hidden[:, 1],
            "o",
            color="m",
            ms=7,
            label="Hidden GT",
        )

    ax.plot(
        rec_raw[:, 0],
        rec_raw[:, 1],
        "-o",
        color="red",
        lw=1.4,
        ms=5,
        label="Reconstructed(raw DP)",
    )

    rec_curve = make_display_curve(
        rec_smooth
    )

    ax.plot(
        rec_curve[:, 0],
        rec_curve[:, 1],
        "-",
        color="green",
        lw=2.3,
        label="Reconstructed(smoothed)",
    )

    ax.plot(
        rec_smooth[:, 0],
        rec_smooth[:, 1],
        "o",
        color="green",
        ms=5,
    )

    x_end, y_end = endpoint

    ax.scatter(
        [x_end],
        [y_end],
        marker="*",
        s=180,
        color="orange",
        edgecolor="black",
        linewidth=0.8,
        label="Global-response endpoint",
        zorder=10,
    )

    if (
        pred_xy is not None
        and len(pred_xy) > 0
    ):
        plot_pred = np.vstack([
            rec_smooth[-1],
            pred_xy,
        ])

        pred_curve = (
            make_display_curve(plot_pred)
            if len(plot_pred) >= 3
            else plot_pred
        )

        ax.plot(
            pred_curve[:, 0],
            pred_curve[:, 1],
            "--",
            color="blue",
            lw=2.0,
            label="Predicted",
        )

        ax.plot(
            pred_xy[:, 0],
            pred_xy[:, 1],
            "o",
            color="blue",
            ms=5,
        )

    H, W = V_full.shape

    ax.set_xlim(
        -0.5,
        W - 0.5,
    )

    ax.set_ylim(
        H - 0.5,
        -0.5,
    )

    ax.set_xticks(
        np.arange(W)
    )

    ax.set_yticks(
        np.arange(H)
    )

    ax.grid(
        alpha=0.22
    )

    ax.set_xlabel(
        "Column"
    )

    ax.set_ylabel(
        "Row"
    )

    ax.set_title(
        "Device-measured 8x8 FTF trajectory reconstruction and prediction"
    )

    #ax.legend(
     #   loc="best",
     #   fontsize=9,
   # )

    plt.tight_layout()

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def save_path_csv(
    rec_raw: np.ndarray,
    rec_smooth: np.ndarray,
    pred_xy: Optional[np.ndarray],
    out_csv: Path,
):
    rows = []

    for i in range(
        len(rec_raw)
    ):
        rows.append(
            {
                "type": "reconstruction",
                "index": i,
                "x_raw":
                    rec_raw[i, 0],
                "y_raw":
                    rec_raw[i, 1],
                "x_smoothed":
                    rec_smooth[i, 0],
                "y_smoothed":
                    rec_smooth[i, 1],
            }
        )

    if pred_xy is not None:
        for i, p in enumerate(
            pred_xy
        ):
            rows.append(
                {
                    "type": "prediction",
                    "index": i,
                    "x_raw": np.nan,
                    "y_raw": np.nan,
                    "x_smoothed": p[0],
                    "y_smoothed": p[1],
                }
            )

    pd.DataFrame(
        rows
    ).to_csv(
        out_csv,
        index=False,
    )


# =========================================================
# Main
# =========================================================

def main():
    print(
        "RUNNING DEVICE-MEASURED 8x8 TRAJECTORY ANALYSIS "
        "- DP + SCALAR-CA KALMAN"
    )
    print("=" * 80)

    out_dir = Path(
        OUTPUT_DIR
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    V_full = load_voltage_matrix(
        CSV_PATH
    )

    H, W = V_full.shape

    print(
        f"Input matrix shape: {H} x {W}"
    )

    gt_full = np.asarray(
        GT_FULL_POINTS,
        dtype=float,
    )

    if PRED_STEPS > 0:
        gt_visible = (
            gt_full[:-PRED_STEPS]
        )
        gt_hidden = (
            gt_full[-PRED_STEPS:]
        )
    else:
        gt_visible = gt_full
        gt_hidden = np.empty(
            (0, 2),
            dtype=float,
        )

    # Reconstruction is completely independent of GT.
    (
        rec_raw,
        endpoint,
        Vn,
        dp_diagnostics,
    ) = reconstruct_path_dp_2d(
        V_full
    )

    rec_smooth = (
        smooth_xy_trajectory(
            rec_raw,
            win=SMOOTH_WIN,
            keep_endpoints=
                KEEP_SMOOTH_ENDPOINTS,
        )
    )

    # Primary reconstruction metrics: raw DP result
    rec_raw_metrics = (
        measured_2d_metrics(
            gt_visible,
            rec_raw,
            ref_span_px=
                H_REF_FOR_ACC,
        )
    )

    # Smoothed metrics are auxiliary only.
    rec_smooth_metrics = (
        measured_2d_metrics(
            gt_visible,
            rec_smooth,
            ref_span_px=
                H_REF_FOR_ACC,
        )
    )

    path_for_pred = (
        rec_smooth
        if USE_SMOOTH_FOR_PRED
        else rec_raw
    )

    pred_xy = (
        predict_trajectory_2d(
            path_for_pred,
            steps=PRED_STEPS,
            W_limit=W,
            H_limit=H,
        )
    )

    if (
        pred_xy is not None
        and len(gt_hidden) > 0
    ):
        pred_metrics = (
            measured_2d_metrics(
                gt_hidden,
                pred_xy,
                ref_span_px=
                    H_REF_FOR_ACC,
            )
        )
    else:
        pred_metrics = {
            "mae_pixel": math.nan,
            "rmse_pixel": math.nan,
            "accuracy_percent": math.nan,
        }

    speed_pixel_per_step = (
        estimate_mean_speed_pixels_per_step(
            path_for_pred
        )
    )

    if ESTIMATE_SPEED_FROM_RELAXATION:
        mean_speed_pixel_per_s = (
            estimate_mean_speed_from_relaxation(
                path_for_pred,
                V_full,
                collection_time_s=
                    READABLE_WINDOW_S,
            )
        )
    else:
        mean_speed_pixel_per_s = (
            math.nan
        )

    direction = (
        estimate_prediction_direction_2d(
            pred_xy,
            start_xy=
                path_for_pred[-1],
        )
    )

    stem = Path(
        CSV_PATH
    ).stem

    out_png = (
        out_dir
        / (
            f"{stem}_device_measured_8x8_"
            f"dp_reconstruction_prediction.png"
        )
    )

    out_path_csv = (
        out_dir
        / f"{stem}_reconstructed_path.csv"
    )

    plot_all(
        V_full,
        gt_visible,
        gt_hidden,
        rec_raw,
        rec_smooth,
        pred_xy,
        endpoint,
        out_png,
    )

    save_path_csv(
        rec_raw,
        rec_smooth,
        pred_xy,
        out_path_csv,
    )

    x_end, y_end = endpoint

    summary = {
        "csv_path":
            CSV_PATH,

        "endpoint_x":
            x_end,

        "endpoint_y":
            y_end,

        "dp_candidate_count":
            dp_diagnostics[
                "candidate_count"
            ],

        "dp_path_length":
            dp_diagnostics[
                "path_length"
            ],

        "lambda_response":
            LAMBDA_RESPONSE,

        "mu_step":
            MU_STEP,

        "rho_base":
            RHO_BASE,

        "rho_weak":
            RHO_WEAK,

        "weak_signal_threshold":
            WEAK_SIGNAL_THRESHOLD,

        "active_response_threshold":
            ACTIVE_RESPONSE_THRESHOLD,

        "J_max":
            J_MAX,

        "max_x_step":
            MAX_X_STEP,

        "raw_mae_pixel":
            rec_raw_metrics[
                "mae_pixel"
            ],

        "raw_rmse_pixel":
            rec_raw_metrics[
                "rmse_pixel"
            ],

        "raw_accuracy_percent":
            rec_raw_metrics[
                "accuracy_percent"
            ],

        "smoothed_mae_pixel":
            rec_smooth_metrics[
                "mae_pixel"
            ],

        "smoothed_rmse_pixel":
            rec_smooth_metrics[
                "rmse_pixel"
            ],

        "smoothed_accuracy_percent":
            rec_smooth_metrics[
                "accuracy_percent"
            ],

        "prediction_mae_pixel":
            pred_metrics[
                "mae_pixel"
            ],

        "prediction_rmse_pixel":
            pred_metrics[
                "rmse_pixel"
            ],

        "prediction_accuracy_percent":
            pred_metrics[
                "accuracy_percent"
            ],

        "kalman_fit_window":
            KF_FIT_WIN,

        "kalman_dt":
            KF_DT,

        "kalman_process_variance":
            KF_PROCESS_VAR,

        "kalman_measurement_variance":
            KF_MEAS_VAR,

        "mean_speed_pixel_per_s_from_relaxation":
            mean_speed_pixel_per_s,

        "aux_mean_displacement_pixel_per_step":
            speed_pixel_per_step,

        "prediction_direction_deg":
            direction,
    }

    out_summary_csv = (
        out_dir
        / (
            f"{stem}_summary_"
            f"dp_kalman.csv"
        )
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        out_summary_csv,
        index=False,
    )

    print(
        f"Detected endpoint: "
        f"(x={x_end}, y={y_end})"
    )

    print(
        f"DP path length: "
        f"{len(rec_raw)}"
    )

    print(
        "\nRaw reconstructed path:"
    )

    for p in rec_raw:
        print(
            f"  ({p[0]:.0f}, {p[1]:.0f})"
        )

    print(
        "\nSmoothed reconstructed path:"
    )

    for p in rec_smooth:
        print(
            f"  ({p[0]:.3f}, {p[1]:.3f})"
        )

    print(
        "\nPrimary reconstruction metrics "
        "(raw DP path):"
    )

    print(
        f"  MAE  = "
        f"{rec_raw_metrics['mae_pixel']:.4f} pixel"
    )

    print(
        f"  RMSE = "
        f"{rec_raw_metrics['rmse_pixel']:.4f} pixel"
    )

    print(
        f"  Score = "
        f"{rec_raw_metrics['accuracy_percent']:.2f}%"
    )

    print(
        "\nAuxiliary smoothed-path metrics:"
    )

    print(
        f"  MAE  = "
        f"{rec_smooth_metrics['mae_pixel']:.4f} pixel"
    )

    print(
        f"  RMSE = "
        f"{rec_smooth_metrics['rmse_pixel']:.4f} pixel"
    )

    print(
        f"  Score = "
        f"{rec_smooth_metrics['accuracy_percent']:.2f}%"
    )

    if pred_xy is not None:
        print(
            "\nPredicted points:"
        )

        for p in pred_xy:
            print(
                f"  ({p[0]:.3f}, {p[1]:.3f})"
            )

    print(
        "\nPrediction metrics:"
    )

    print(
        f"  MAE  = "
        f"{pred_metrics['mae_pixel']:.4f} pixel"
    )

    print(
        f"  RMSE = "
        f"{pred_metrics['rmse_pixel']:.4f} pixel"
    )

    print(
        f"  Score = "
        f"{pred_metrics['accuracy_percent']:.2f}%"
    )

    if math.isnan(
        mean_speed_pixel_per_s
    ):
        print(
            "\nMean speed from relaxation: N/A"
        )
    else:
        print(
            f"\nMean speed from relaxation: "
            f"{mean_speed_pixel_per_s:.4f} pixel/s"
        )

    print(
        f"Prediction direction: "
        f"{direction:.2f} deg"
        if not math.isnan(direction)
        else "Prediction direction: N/A"
    )

    print(
        "\nSaved:"
    )

    print(
        f"  Figure: {out_png}"
    )

    print(
        f"  Path CSV: {out_path_csv}"
    )

    print(
        f"  Summary CSV: {out_summary_csv}"
    )

    print("=" * 80)


if __name__ == "__main__":
    main()
