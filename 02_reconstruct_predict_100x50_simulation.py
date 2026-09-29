import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import warnings
warnings.filterwarnings("ignore")


# =========================================================
# Configuration
# =========================================================

TARGET_VOLTAGE = 3.3

SAMPLE_INTERVAL_S = 0.1
READABLE_WINDOW_S = 40.0

PRED_STEPS = 3

SMOOTH_WIN = 5
USE_SMOOTH_FOR_PRED = False

KF_FIT_WIN = 6
KF_DT = 1.0
KF_PROCESS_VAR = 0.05
KF_MEAS_VAR = 0.8

H_REF_FOR_ACC = 50


# =========================================================
# DP parameters
# =========================================================
#
# These correspond directly to the SI notation:
#
#   λ  -> VOLTAGE_REWARD
#   μ  -> SMOOTH_PENALTY
#   ρx -> MISSING_COL_PENALTY for weak/invalid columns
#   Jmax -> MAX_JUMP
#
# The simulated sinusoidal trajectory is monotonic in x,
# therefore the column-wise y(x) DP formulation in the SI
# can be used directly without modification.
# =========================================================

CANDIDATE_TOP_K = 8
CANDIDATE_ALPHA = 0.6
MIN_COL_MAX_RATIO = 0.05

VOLTAGE_REWARD = 2.0
SMOOTH_PENALTY = 1.8
MISSING_COL_PENALTY = 0.8
MAX_JUMP = 4


# =========================================================
# Triple-exponential relaxation parameters
# =========================================================

A1 = 3.87821e-7
TAU1 = 0.34921

A2 = 3.50008e-7
TAU2 = 2.97588

A3 = 3.00247e-7
TAU3 = 30.47546

Y0 = 4.46074e-8

I_MAX_T0 = (
    A1 + A2 + A3 + Y0
)

TRANSIMPEDANCE_GAIN = (
    TARGET_VOLTAGE
    / I_MAX_T0
)


# =========================================================
# Relaxation model and inversion
# =========================================================

def relaxation_model_current(
    t: np.ndarray,
) -> np.ndarray:
    t = np.asarray(
        t,
        dtype=float,
    )

    return (
        A1 * np.exp(-t / TAU1)
        + A2 * np.exp(-t / TAU2)
        + A3 * np.exp(-t / TAU3)
        + Y0
    )


def current_to_voltage(
    current: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(
            current,
            dtype=float,
        )
        * TRANSIMPEDANCE_GAIN
    )


def voltage_to_current(
    voltage: np.ndarray,
) -> np.ndarray:
    return (
        np.asarray(
            voltage,
            dtype=float,
        )
        / TRANSIMPEDANCE_GAIN
    )


def invert_relaxation_time_from_voltage(
    voltage_values,
    t_max=READABLE_WINDOW_S,
    n_grid=20000,
):
    voltage_values = np.asarray(
        voltage_values,
        dtype=float,
    )

    t_grid = np.linspace(
        0.0,
        t_max,
        n_grid,
    )

    i_grid = (
        relaxation_model_current(
            t_grid
        )
    )

    v_grid = (
        current_to_voltage(
            i_grid
        )
    )

    return np.interp(
        voltage_values,
        v_grid[::-1],
        t_grid[::-1],
    )


# =========================================================
# Simulated trajectory generation
# =========================================================

def generate_sine_trajectory_100x50(
    noise_ratio=0.05,
    seed=42,
):
    rng = np.random.default_rng(
        seed
    )

    width_pixels = 100
    height_pixels = 50

    pixel_size = 0.5
    total_distance = 40.0
    velocity = 1.0

    total_time = (
        total_distance
        / velocity
    )

    num_points = 2000

    x_physical = np.linspace(
        0,
        total_distance,
        num_points,
    )

    amplitude = 10.0
    wavelength = 8.0
    frequency = 1.0 / wavelength

    y_physical = (
        12.5
        + amplitude
        * np.sin(
            2
            * np.pi
            * frequency
            * x_physical
        )
    )

    x_pixels = np.clip(
        x_physical / pixel_size,
        0,
        width_pixels - 1,
    )

    y_pixels = np.clip(
        y_physical / pixel_size,
        0,
        height_pixels - 1,
    )

    times = (
        x_physical
        / velocity
    )

    collection_time = total_time

    time_since_exposure = (
        collection_time
        - times
    )

    currents = (
        relaxation_model_current(
            time_since_exposure
        )
    )

    voltages = (
        current_to_voltage(
            currents
        )
    )

    voltage_matrix = np.zeros(
        (
            height_pixels,
            width_pixels,
        ),
        dtype=float,
    )

    gt_columns = {}

    for i in range(
        len(x_pixels)
    ):
        x_int = int(
            round(
                x_pixels[i]
            )
        )

        y_int = int(
            round(
                y_pixels[i]
            )
        )

        if (
            0 <= x_int < width_pixels
            and 0 <= y_int < height_pixels
        ):
            voltage_matrix[
                y_int,
                x_int,
            ] = max(
                voltage_matrix[
                    y_int,
                    x_int,
                ],
                voltages[i],
            )

            gt_columns.setdefault(
                x_int,
                [],
            ).append(
                (
                    y_pixels[i],
                    times[i],
                    voltages[i],
                )
            )

    noise_level = (
        np.max(voltages)
        * noise_ratio
    )

    noise = rng.normal(
        0,
        noise_level,
        (
            height_pixels,
            width_pixels,
        ),
    )

    voltage_matrix = np.maximum(
        0,
        voltage_matrix + noise,
    )

    gt_x = np.array(
        sorted(
            gt_columns.keys()
        ),
        dtype=int,
    )

    gt_y = np.array(
        [
            np.mean(
                [
                    p[0]
                    for p
                    in gt_columns[x]
                ]
            )
            for x
            in gt_x
        ],
        dtype=float,
    )

    gt_t = np.array(
        [
            np.mean(
                [
                    p[1]
                    for p
                    in gt_columns[x]
                ]
            )
            for x
            in gt_x
        ],
        dtype=float,
    )

    gt_v = np.array(
        [
            np.max(
                [
                    p[2]
                    for p
                    in gt_columns[x]
                ]
            )
            for x
            in gt_x
        ],
        dtype=float,
    )

    gt_df = pd.DataFrame(
        {
            "x_pixel":
                gt_x,
            "y_pixel":
                gt_y,
            "time_s":
                gt_t,
            "voltage_V":
                gt_v,
        }
    )

    meta = {
        "pixel_size_m":
            pixel_size,

        "total_distance_m":
            total_distance,

        "total_time_s":
            total_time,

        "velocity_mps":
            velocity,

        "width_pixels":
            width_pixels,

        "height_pixels":
            height_pixels,

        "collection_time_s":
            collection_time,
    }

    return (
        voltage_matrix,
        gt_df,
        meta,
    )


# =========================================================
# Endpoint localization
# =========================================================

def local_box_smooth_3x3(
    V: np.ndarray,
) -> np.ndarray:
    """
    Small local averaging used only for robust endpoint localization.
    It suppresses isolated pixel noise without changing the DP input.
    """
    V = np.asarray(
        V,
        dtype=float,
    )

    H, W = V.shape

    padded = np.pad(
        V,
        (
            (1, 1),
            (1, 1),
        ),
        mode="edge",
    )

    out = np.zeros_like(
        V,
        dtype=float,
    )

    for dy in range(3):
        for dx in range(3):
            out += padded[
                dy:dy + H,
                dx:dx + W,
            ]

    return out / 9.0


def find_global_endpoint(
    V: np.ndarray,
    cluster_alpha: float = 0.60,
):
    """
    Robust data-driven endpoint localization.

    1. A 3x3 locally averaged response map is used to identify the
       strongest endpoint region and its x-column.
    2. Within that column, the contiguous high-response cluster
       around the detected peak is found from the ORIGINAL V matrix.
    3. The center row of that cluster is used as y_end.

    Ground truth is never used for endpoint localization.

    This keeps the physical "strongest residual response = endpoint"
    assumption while avoiding an isolated noisy pixel from defining
    the endpoint coordinate in the 100x50 simulated image.
    """
    V = np.asarray(
        V,
        dtype=float,
    )

    Vs = local_box_smooth_3x3(
        V
    )

    y_peak, x_end = np.unravel_index(
        np.argmax(Vs),
        Vs.shape,
    )

    x_end = int(x_end)
    y_peak = int(y_peak)

    col = V[:, x_end]

    col_max = float(
        np.max(col)
    )

    if col_max <= 0:
        return (
            x_end,
            y_peak,
        )

    active_rows = np.where(
        col
        >= cluster_alpha
        * col_max
    )[0]

    if len(active_rows) == 0:
        return (
            x_end,
            int(np.argmax(col)),
        )

    # Split active rows into contiguous groups.
    groups = []
    start = int(
        active_rows[0]
    )
    prev = int(
        active_rows[0]
    )

    for y in active_rows[1:]:
        y = int(y)

        if y == prev + 1:
            prev = y
        else:
            groups.append(
                (
                    start,
                    prev,
                )
            )
            start = y
            prev = y

    groups.append(
        (
            start,
            prev,
        )
    )

    # Select the group containing the smoothed-map peak.
    # If none contains it, select the nearest group.
    chosen = None

    for g0, g1 in groups:
        if g0 <= y_peak <= g1:
            chosen = (
                g0,
                g1,
            )
            break

    if chosen is None:
        chosen = min(
            groups,
            key=lambda g: min(
                abs(y_peak - g[0]),
                abs(y_peak - g[1]),
            ),
        )

    y_center = (
        chosen[0]
        + chosen[1]
    ) / 2.0

    # Conventional half-up rounding rather than Python banker's rounding.
    y_end = int(
        np.floor(
            y_center + 0.5
        )
    )

    return (
        x_end,
        y_end,
    )


# =========================================================
# Candidate construction
# =========================================================

def build_candidates_per_column_adaptive(
    V,
    top_k=CANDIDATE_TOP_K,
    alpha=CANDIDATE_ALPHA,
    min_col_max_ratio=
        MIN_COL_MAX_RATIO,
):
    H, W = V.shape

    Vmax = (
        float(V.max())
        + 1e-12
    )

    candidates = []
    valid_col = np.ones(
        W,
        dtype=bool,
    )

    for x in range(W):
        col = V[:, x]
        col_max = float(
            col.max()
        )

        if (
            col_max
            < min_col_max_ratio
            * Vmax
        ):
            valid_col[x] = False

            candidates.append(
                [
                    int(
                        np.argmax(col)
                    )
                ]
            )

            continue

        thr = (
            alpha
            * col_max
        )

        idx = np.where(
            col >= thr
        )[0]

        if len(idx) == 0:
            idx = np.argsort(
                col
            )[-top_k:]
        else:
            idx = idx[
                np.argsort(
                    col[idx]
                )[-top_k:]
            ]

        idx = idx[
            np.argsort(
                col[idx]
            )[::-1]
        ]

        candidates.append(
            [
                int(i)
                for i
                in idx
            ]
        )

    return (
        candidates,
        valid_col,
    )


# =========================================================
# Column-wise DP reconstruction
# =========================================================

def dp_reconstruct_with_fixed_end(
    V,
    candidates,
    valid_col,
    y_end,
    smooth_penalty=
        SMOOTH_PENALTY,
    voltage_reward=
        VOLTAGE_REWARD,
    missing_col_penalty=
        MISSING_COL_PENALTY,
    max_jump=
        MAX_JUMP,
):
    """
    Column-wise dynamic programming.

    This retains the SI y(x) formulation for the simulated
    monotonic-x trajectory.
    """
    H, W = V.shape

    dp = []
    prev = []

    y0_list = candidates[0]

    dp0 = np.zeros(
        len(y0_list)
    )

    prev0 = -np.ones(
        len(y0_list),
        dtype=int,
    )

    for i, y in enumerate(
        y0_list
    ):
        v = float(
            V[y, 0]
        )

        base = (
            -voltage_reward * v
            if valid_col[0]
            else missing_col_penalty
        )

        dp0[i] = base

    dp.append(dp0)
    prev.append(prev0)

    for x in range(
        1,
        W,
    ):
        y_list = candidates[x]
        y_prev_list = (
            candidates[x - 1]
        )

        dp_prev = dp[x - 1]

        dp_x = np.full(
            len(y_list),
            np.inf,
        )

        prev_x = np.full(
            len(y_list),
            -1,
            dtype=int,
        )

        for i, y in enumerate(
            y_list
        ):
            v = float(
                V[y, x]
            )

            base = (
                -voltage_reward * v
                if valid_col[x]
                else missing_col_penalty
            )

            best_cost = np.inf
            best_j = -1

            for j, y_prev in enumerate(
                y_prev_list
            ):
                jump = abs(
                    y - y_prev
                )

                if jump > max_jump:
                    continue

                cost = (
                    dp_prev[j]
                    + smooth_penalty
                    * jump
                    + base
                )

                if cost < best_cost:
                    best_cost = cost
                    best_j = j

            # Robust fallback for isolated weak/noisy columns.
            if best_j < 0:
                for j, y_prev in enumerate(
                    y_prev_list
                ):
                    jump = abs(
                        y - y_prev
                    )

                    cost = (
                        dp_prev[j]
                        + smooth_penalty
                        * jump
                        + base
                        + 5.0
                    )

                    if cost < best_cost:
                        best_cost = cost
                        best_j = j

            dp_x[i] = best_cost
            prev_x[i] = best_j

        dp.append(dp_x)
        prev.append(prev_x)

    last_candidates = (
        candidates[-1]
    )

    if y_end in last_candidates:
        end_i = (
            last_candidates
            .index(y_end)
        )
    else:
        end_i = int(
            np.argmin(
                [
                    abs(
                        y - y_end
                    )
                    for y
                    in last_candidates
                ]
            )
        )

    path_y = np.zeros(
        W,
        dtype=int,
    )

    path_y[-1] = (
        last_candidates[end_i]
    )

    cur_i = end_i

    for x in range(
        W - 1,
        0,
        -1,
    ):
        cur_i = int(
            prev[x][cur_i]
        )

        if cur_i < 0:
            cur_i = 0

        path_y[x - 1] = (
            candidates[
                x - 1
            ][cur_i]
        )

    return (
        np.arange(W),
        path_y,
    )


# =========================================================
# Smoothing
# =========================================================

def smooth_trajectory(
    y,
    win=SMOOTH_WIN,
):
    y = np.asarray(
        y,
        dtype=float,
    )

    if win <= 1:
        return y.copy()

    if win % 2 == 0:
        win += 1

    pad = win // 2

    y_pad = np.pad(
        y,
        (pad, pad),
        mode="edge",
    )

    kernel = (
        np.ones(win)
        / win
    )

    return np.convolve(
        y_pad,
        kernel,
        mode="valid",
    )


# =========================================================
# Scalar three-state Kalman prediction
# =========================================================

def predict_from_last_point_kalman(
    x_hist,
    y_hist,
    steps,
    H_limit,
    fit_win=KF_FIT_WIN,
    dt=KF_DT,
    process_var=
        KF_PROCESS_VAR,
    meas_var=
        KF_MEAS_VAR,
):
    """
    Three-state constant-acceleration Kalman filter:

        state = [y, v_y, a_y]^T

    This is identical in form to the current SI description.
    """
    x_hist = np.asarray(
        x_hist[-fit_win:],
        dtype=float,
    )

    y_hist = np.asarray(
        y_hist[-fit_win:],
        dtype=float,
    )

    if (
        len(x_hist) < 3
        or steps <= 0
    ):
        return (
            None,
            None,
        )

    order = np.argsort(
        x_hist
    )

    x_hist = (
        x_hist[order]
    )

    y_hist = (
        y_hist[order]
    )

    v0 = (
        y_hist[1]
        - y_hist[0]
    ) / dt

    a0 = (
        y_hist[2]
        - 2.0 * y_hist[1]
        + y_hist[0]
    ) / (dt ** 2)

    state = np.array(
        [
            y_hist[0],
            v0,
            a0,
        ],
        dtype=float,
    )

    P = np.diag(
        [
            1.0,
            1.0,
            1.0,
        ]
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
        [
            [
                1.0,
                0.0,
                0.0,
            ]
        ],
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

    for z in y_hist[1:]:
        state = F @ state

        P = (
            F @ P @ F.T
            + Q
        )

        residual = np.array(
            [
                [
                    z
                    - (
                        Hm
                        @ state
                    )[0]
                ]
            ],
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
            + (
                K
                @ residual
            ).flatten()
        )

        P = (
            np.eye(3)
            - K @ Hm
        ) @ P

        state[1] = np.clip(
            state[1],
            -5.0,
            5.0,
        )

        state[2] = np.clip(
            state[2],
            -3.0,
            3.0,
        )

    state[0] = (
        y_hist[-1]
    )

    y_pred = []

    for _ in range(steps):
        state = F @ state

        P = (
            F @ P @ F.T
            + Q
        )

        state[1] = np.clip(
            state[1],
            -5.0,
            5.0,
        )

        state[2] = np.clip(
            state[2],
            -3.0,
            3.0,
        )

        y_pred.append(
            state[0]
        )

    y_pred = np.clip(
        np.asarray(
            y_pred,
            dtype=float,
        ),
        0,
        H_limit - 1,
    )

    x_last = int(
        round(
            x_hist[-1]
        )
    )

    x_pred = np.arange(
        x_last + 1,
        x_last + 1 + steps,
        dtype=int,
    )

    return (
        x_pred,
        y_pred,
    )


# =========================================================
# Metrics
# =========================================================

def regression_metrics(
    y_true,
    y_pred,
    ref_span_px=None,
):
    y_true = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float,
    )

    err = (
        y_pred
        - y_true
    )

    mae = np.mean(
        np.abs(err)
    )

    rmse = np.sqrt(
        np.mean(
            err ** 2
        )
    )

    out = {
        "mae":
            float(mae),
        "rmse":
            float(rmse),
    }

    if (
        ref_span_px is not None
        and ref_span_px > 0
    ):
        out[
            "acc_rmse_pct"
        ] = float(
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

    return out


def estimate_prediction_direction(
    x_pred,
    y_pred,
):
    if (
        x_pred is None
        or y_pred is None
        or len(x_pred) < 2
    ):
        return np.nan

    dx = float(
        x_pred[-1]
        - x_pred[0]
    )

    dy = float(
        y_pred[-1]
        - y_pred[0]
    )

    return float(
        np.degrees(
            np.arctan2(
                dy,
                dx,
            )
        )
    )


def estimate_mean_speed_only(
    x_path,
    y_path,
    V_full,
    pixel_size_m,
    collection_time_s,
):
    x_path = np.asarray(
        x_path,
        dtype=int,
    )

    y_path = np.asarray(
        np.round(y_path),
        dtype=int,
    )

    H, W = V_full.shape

    y_path = np.clip(
        y_path,
        0,
        H - 1,
    )

    x_path = np.clip(
        x_path,
        0,
        W - 1,
    )

    voltages = np.array(
        [
            V_full[y, x]
            for x, y
            in zip(
                x_path,
                y_path,
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

    dx_pix = np.diff(
        x_path.astype(float)
    )

    dy_pix = np.diff(
        y_path.astype(float)
    )

    dt = np.diff(
        t_event.astype(float)
    )

    valid = (
        dt > 1e-6
    )

    if np.sum(valid) == 0:
        return np.nan

    dx_m = (
        dx_pix[valid]
        * pixel_size_m
    )

    dy_m = (
        dy_pix[valid]
        * pixel_size_m
    )

    dt_valid = (
        dt[valid]
    )

    vx = (
        dx_m
        / dt_valid
    )

    vy = (
        dy_m
        / dt_valid
    )

    speed = np.sqrt(
        vx ** 2
        + vy ** 2
    )

    return float(
        np.mean(speed)
    )


# =========================================================
# Plot
# =========================================================

def plot_main_figure(
    V_full,
    gt_visible_df,
    gt_hidden_df,
    x_rec,
    y_raw,
    y_smooth,
    x_pred,
    y_pred,
    endpoint,
    out_png="fig_main.png",
):
    cmap = LinearSegmentedColormap.from_list(
        "w2p",
        [
            (1, 1, 1),
            (0.72, 0.52, 0.95),
        ],
    )

    fig, ax = plt.subplots(
        figsize=(12, 6)
    )

    ax.imshow(
        V_full,
        cmap=cmap,
        vmin=0,
        vmax=TARGET_VOLTAGE,
        origin="upper",
        aspect="auto",
    )

    ax.plot(
        gt_visible_df[
            "x_pixel"
        ],
        gt_visible_df[
            "y_pixel"
        ],
        color="gray",
        lw=1.2,
        alpha=0.7,
        label="GT available",
    )

    ax.plot(
        gt_hidden_df[
            "x_pixel"
        ],
        gt_hidden_df[
            "y_pixel"
        ],
        "mo",
        ms=7,
        label="Hidden GT",
    )

    ax.plot(
        x_rec,
        y_raw,
        color="red",
        lw=1.4,
        label="Reconstructed(raw)",
    )

    ax.plot(
        x_rec,
        y_smooth,
        color="green",
        lw=2.2,
        label="Reconstructed(smoothed)",
    )

    x_end, y_end = endpoint

    ax.plot(
        [x_end],
        [y_end],
        marker="*",
        color="orange",
        markeredgecolor="black",
        markersize=14,
        linestyle="None",
        label="Global-response endpoint",
    )

    if (
        x_pred is not None
        and y_pred is not None
    ):
        ax.plot(
            x_pred,
            y_pred,
            "--o",
            color="blue",
            lw=2.0,
            ms=6,
            label="Predicted(KF)",
        )

    ax.set_xlim(
        0,
        V_full.shape[1] - 1,
    )

    ax.set_ylim(
        V_full.shape[0] - 1,
        0,
    )

    ax.set_xlabel(
        "Column"
    )

    ax.set_ylabel(
        "Row"
    )

    ax.legend(
    loc="upper right",
    fontsize=9,
    )

    plt.tight_layout()

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


def plot_raw_curve_only(
    x_rec,
    y_raw,
    H,
    W,
    out_png=
        "fig_reconstructed_raw.png",
):
    fig, ax = plt.subplots(
        figsize=(12, 6),
        facecolor="white",
    )

    ax.set_facecolor(
        "white"
    )

    ax.plot(
        x_rec,
        y_raw,
        color="red",
        lw=2.0,
    )

    ax.set_xlim(
        0,
        W - 1,
    )

    ax.set_ylim(
        H - 1,
        0,
    )

    ax.set_xlabel(
        "Column"
    )

    ax.set_ylabel(
        "Row"
    )

    plt.tight_layout()

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


def plot_smoothed_curve_only(
    x_rec,
    y_smooth,
    H,
    W,
    out_png=
        "fig_reconstructed_smoothed.png",
):
    fig, ax = plt.subplots(
        figsize=(12, 6),
        facecolor="white",
    )

    ax.set_facecolor(
        "white"
    )

    ax.plot(
        x_rec,
        y_smooth,
        color="green",
        lw=2.2,
    )

    ax.set_xlim(
        0,
        W - 1,
    )

    ax.set_ylim(
        H - 1,
        0,
    )

    ax.set_xlabel(
        "Column"
    )

    ax.set_ylabel(
        "Row"
    )

    plt.tight_layout()

    fig.savefig(
        out_png,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


# =========================================================
# Main
# =========================================================

def main():
    print(
        "RUNNING 100x50 SIMULATION "
        "- COLUMN-WISE DP + KALMAN"
    )

    voltage_matrix, gt_df, meta = (
        generate_sine_trajectory_100x50(
            noise_ratio=0.05,
            seed=42,
        )
    )

    H, W = (
        voltage_matrix.shape
    )

    # -----------------------------------------------------
    # Prediction benchmark split
    # -----------------------------------------------------
    #
    # GT is used here only to define the held-out future
    # prediction horizon in the synthetic benchmark.
    # Endpoint localization inside the visible fused matrix
    # is performed independently from GT.
    # -----------------------------------------------------

    gt_hidden_df = (
        gt_df
        .tail(PRED_STEPS)
        .reset_index(drop=True)
    )

    gt_visible_df = (
        gt_df
        .iloc[:-PRED_STEPS]
        .reset_index(drop=True)
    )

    visible_limit_x = int(
        gt_visible_df[
            "x_pixel"
        ].iloc[-1]
    )

    V_visible_split = (
        voltage_matrix[
            :,
            :visible_limit_x + 1,
        ]
    )

    # -----------------------------------------------------
    # GT-independent endpoint localization
    # -----------------------------------------------------

    x_end, y_end = (
        find_global_endpoint(
            V_visible_split
        )
    )

    V_obs = (
        V_visible_split[
            :,
            :x_end + 1,
        ]
    )

    print(
        f"Detected endpoint: "
        f"(x={x_end}, y={y_end})"
    )

    # -----------------------------------------------------
    # DP reconstruction
    # -----------------------------------------------------

    candidates, valid_col = (
        build_candidates_per_column_adaptive(
            V_obs
        )
    )

    x_rec, y_raw = (
        dp_reconstruct_with_fixed_end(
            V_obs,
            candidates,
            valid_col,
            y_end=y_end,
        )
    )

    y_smooth = (
        smooth_trajectory(
            y_raw,
            win=SMOOTH_WIN,
        )
    )

    # -----------------------------------------------------
    # Reconstruction metrics
    # -----------------------------------------------------

    y_true_rec = np.interp(
        x_rec,
        gt_visible_df[
            "x_pixel"
        ],
        gt_visible_df[
            "y_pixel"
        ],
    )

    rec_raw_metrics = (
        regression_metrics(
            y_true_rec,
            y_raw,
            ref_span_px=
                H_REF_FOR_ACC,
        )
    )

    rec_smooth_metrics = (
        regression_metrics(
            y_true_rec,
            y_smooth,
            ref_span_px=
                H_REF_FOR_ACC,
        )
    )

    # -----------------------------------------------------
    # Prediction
    # -----------------------------------------------------

    path_for_pred = (
        y_smooth
        if USE_SMOOTH_FOR_PRED
        else y_raw
    )

    fit_n = min(
        KF_FIT_WIN,
        len(x_rec),
    )

    x_hist = (
        x_rec[-fit_n:]
    )

    y_hist = (
        path_for_pred[
            -fit_n:
        ]
    )

    x_pred, y_pred = (
        predict_from_last_point_kalman(
            x_hist,
            y_hist,
            PRED_STEPS,
            H_limit=H,
        )
    )

    pred_metrics = {
        "mae": np.nan,
        "rmse": np.nan,
        "acc_rmse_pct": np.nan,
    }

    pred_direction = np.nan

    if (
        x_pred is not None
        and y_pred is not None
        and len(gt_hidden_df) > 0
    ):
        y_true_pred = (
            gt_hidden_df[
                "y_pixel"
            ].values
        )

        n_eval = min(
            len(y_true_pred),
            len(y_pred),
        )

        pred_metrics = (
            regression_metrics(
                y_true_pred[
                    :n_eval
                ],
                y_pred[:n_eval],
                ref_span_px=
                    H_REF_FOR_ACC,
            )
        )

        pred_direction = (
            estimate_prediction_direction(
                x_pred[:n_eval],
                y_pred[:n_eval],
            )
        )

    # -----------------------------------------------------
    # Mean speed
    # -----------------------------------------------------

    mean_speed = (
        estimate_mean_speed_only(
            x_rec,
            path_for_pred,
            voltage_matrix,
            pixel_size_m=
                meta[
                    "pixel_size_m"
                ],
            collection_time_s=
                meta[
                    "collection_time_s"
                ],
        )
    )

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------

    plot_main_figure(
        voltage_matrix,
        gt_visible_df,
        gt_hidden_df,
        x_rec,
        y_raw,
        y_smooth,
        x_pred,
        y_pred,
        endpoint=(
            x_end,
            y_end,
        ),
        out_png=
            "fig_main.png",
    )

    plot_raw_curve_only(
        x_rec,
        y_raw,
        H,
        W,
        out_png=
            "fig_reconstructed_raw.png",
    )

    plot_smoothed_curve_only(
        x_rec,
        y_smooth,
        H,
        W,
        out_png=
            "fig_reconstructed_smoothed.png",
    )

    # -----------------------------------------------------
    # Save summary
    # -----------------------------------------------------

    summary = {
        "endpoint_x":
            x_end,

        "endpoint_y":
            y_end,

        "lambda_voltage_reward":
            VOLTAGE_REWARD,

        "mu_smooth_penalty":
            SMOOTH_PENALTY,

        "rho_missing_col_penalty":
            MISSING_COL_PENALTY,

        "J_max":
            MAX_JUMP,

        "raw_mae_pixel":
            rec_raw_metrics[
                "mae"
            ],

        "raw_rmse_pixel":
            rec_raw_metrics[
                "rmse"
            ],

        "raw_accuracy_percent":
            rec_raw_metrics[
                "acc_rmse_pct"
            ],

        "smoothed_mae_pixel":
            rec_smooth_metrics[
                "mae"
            ],

        "smoothed_rmse_pixel":
            rec_smooth_metrics[
                "rmse"
            ],

        "smoothed_accuracy_percent":
            rec_smooth_metrics[
                "acc_rmse_pct"
            ],

        "prediction_mae_pixel":
            pred_metrics[
                "mae"
            ],

        "prediction_rmse_pixel":
            pred_metrics[
                "rmse"
            ],

        "prediction_accuracy_percent":
            pred_metrics[
                "acc_rmse_pct"
            ],

        "kalman_fit_window":
            KF_FIT_WIN,

        "kalman_dt":
            KF_DT,

        "kalman_process_variance":
            KF_PROCESS_VAR,

        "kalman_measurement_variance":
            KF_MEAS_VAR,

        "mean_speed_mps":
            mean_speed,

        "prediction_direction_deg":
            pred_direction,
    }

    pd.DataFrame(
        [summary]
    ).to_csv(
        "simulation_100x50_summary.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Console output
    # -----------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "Evaluation Summary"
    )

    print(
        "=" * 60
    )

    print(
        "1. Reconstruction"
    )

    print(
        f"   Raw MAE             : "
        f"{rec_raw_metrics['mae']:.4f} pixel"
    )

    print(
        f"   Raw RMSE            : "
        f"{rec_raw_metrics['rmse']:.4f} pixel"
    )

    print(
        f"   Raw Accuracy        : "
        f"{rec_raw_metrics['acc_rmse_pct']:.2f}%"
    )

    print(
        f"   Smoothed MAE        : "
        f"{rec_smooth_metrics['mae']:.4f} pixel"
    )

    print(
        f"   Smoothed RMSE       : "
        f"{rec_smooth_metrics['rmse']:.4f} pixel"
    )

    print(
        f"   Smoothed Accuracy   : "
        f"{rec_smooth_metrics['acc_rmse_pct']:.2f}%"
    )

    print(
        "\n2. Prediction"
    )

    print(
        f"   MAE                 : "
        f"{pred_metrics['mae']:.4f} pixel"
    )

    print(
        f"   RMSE                : "
        f"{pred_metrics['rmse']:.4f} pixel"
    )

    print(
        f"   Accuracy            : "
        f"{pred_metrics['acc_rmse_pct']:.2f}%"
    )

    print(
        "\n3. Speed"
    )

    print(
        f"   Mean Speed          : "
        f"{mean_speed:.4f} m/s"
    )

    print(
        "\n4. Prediction Direction"
    )

    print(
        f"   Direction Angle     : "
        f"{pred_direction:.4f} deg"
    )

    print(
        "=" * 60
    )

    print(
        "Saved figures/files:"
    )

    print(
        " - fig_main.png"
    )

    print(
        " - fig_reconstructed_raw.png"
    )

    print(
        " - fig_reconstructed_smoothed.png"
    )

    print(
        " - simulation_100x50_summary.csv"
    )

    print(
        "=" * 60
    )


if __name__ == "__main__":
    main()
