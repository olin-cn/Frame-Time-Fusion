import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# =========================================================
# Configuration
# =========================================================
TARGET_VOLTAGE = 3.3
SAMPLE_INTERVAL_S = 0.1
READABLE_WINDOW_S = 40.0
PRED_STEPS = 3
SMOOTH_WIN = 5
USE_SMOOTH_FOR_PRED = False
PRED_FIT_WIN = 6
H_REF_FOR_ACC = 50

# Triple-exponential relaxation parameters
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

def voltage_to_current(voltage: np.ndarray) -> np.ndarray:
    return np.asarray(voltage, dtype=float) / TRANSIMPEDANCE_GAIN

def invert_relaxation_time_from_voltage(voltage_values, t_max=READABLE_WINDOW_S, n_grid=20000):
    voltage_values = np.asarray(voltage_values, dtype=float)

    t_grid = np.linspace(0.0, t_max, n_grid)
    i_grid = relaxation_model_current(t_grid)
    v_grid = current_to_voltage(i_grid)

    return np.interp(voltage_values, v_grid[::-1], t_grid[::-1])

# =========================================================
# Simulated trajectory generation
# =========================================================
def generate_sine_trajectory_100x50(noise_ratio=0.05, seed=42):
    rng = np.random.default_rng(seed)

    width_pixels = 100
    height_pixels = 50
    pixel_size = 0.5
    total_distance = 40.0
    velocity = 1.0
    total_time = total_distance / velocity

    num_points = 2000
    x_physical = np.linspace(0, total_distance, num_points)
    amplitude = 10.0
    wavelength = 8.0
    frequency = 1.0 / wavelength
    y_physical = 12.5 + amplitude * np.sin(2 * np.pi * frequency * x_physical)

    x_pixels = np.clip(x_physical / pixel_size, 0, width_pixels - 1)
    y_pixels = np.clip(y_physical / pixel_size, 0, height_pixels - 1)
    times = x_physical / velocity

    collection_time = total_time
    time_since_exposure = collection_time - times
    currents = relaxation_model_current(time_since_exposure)
    voltages = current_to_voltage(currents)

    voltage_matrix = np.zeros((height_pixels, width_pixels), dtype=float)
    gt_columns = {}

    for i in range(len(x_pixels)):
        x_int = int(round(x_pixels[i]))
        y_int = int(round(y_pixels[i]))
        if 0 <= x_int < width_pixels and 0 <= y_int < height_pixels:
            voltage_matrix[y_int, x_int] = max(voltage_matrix[y_int, x_int], voltages[i])
            gt_columns.setdefault(x_int, []).append((y_pixels[i], times[i], voltages[i]))

    noise_level = np.max(voltages) * noise_ratio
    noise = rng.normal(0, noise_level, (height_pixels, width_pixels))
    voltage_matrix = np.maximum(0, voltage_matrix + noise)

    gt_x = np.array(sorted(gt_columns.keys()), dtype=int)
    gt_y = np.array([np.mean([p[0] for p in gt_columns[x]]) for x in gt_x], dtype=float)
    gt_t = np.array([np.mean([p[1] for p in gt_columns[x]]) for x in gt_x], dtype=float)
    gt_v = np.array([np.max([p[2] for p in gt_columns[x]]) for x in gt_x], dtype=float)

    gt_df = pd.DataFrame({
        'x_pixel': gt_x,
        'y_pixel': gt_y,
        'time_s': gt_t,
        'voltage_V': gt_v
    })

    meta = {
        'pixel_size_m': pixel_size,
        'total_distance_m': total_distance,
        'total_time_s': total_time,
        'velocity_mps': velocity,
        'width_pixels': width_pixels,
        'height_pixels': height_pixels,
        'collection_time_s': collection_time,
    }
    return voltage_matrix, gt_df, meta

# =========================================================
# Reconstruction helpers
# =========================================================
def build_candidates_per_column_adaptive(V, top_k=8, alpha=0.6, min_col_max_ratio=0.05):
    H, W = V.shape
    Vmax = float(V.max()) + 1e-12
    candidates = []
    valid_col = np.ones(W, dtype=bool)

    for x in range(W):
        col = V[:, x]
        col_max = float(col.max())

        if col_max < min_col_max_ratio * Vmax:
            valid_col[x] = False
            candidates.append([int(np.argmax(col))])
            continue

        thr = alpha * col_max
        idx = np.where(col >= thr)[0]

        if len(idx) == 0:
            idx = np.argsort(col)[-top_k:]
        else:
            idx = idx[np.argsort(col[idx])[-top_k:]]

        idx = idx[np.argsort(col[idx])[::-1]]
        candidates.append([int(i) for i in idx])

    return candidates, valid_col

def dp_reconstruct_with_fixed_end(
    V, candidates, valid_col, y_end,
    smooth_penalty=1.8,
    voltage_reward=2.0,
    missing_col_penalty=0.8,
    max_jump=4
):
    H, W = V.shape
    dp, prev = [], []

    y0_list = candidates[0]
    dp0 = np.zeros(len(y0_list))
    prev0 = -np.ones(len(y0_list), dtype=int)

    for i, y in enumerate(y0_list):
        v = float(V[y, 0])
        base = (-voltage_reward * v) if valid_col[0] else missing_col_penalty
        dp0[i] = base

    dp.append(dp0)
    prev.append(prev0)

    for x in range(1, W):
        y_list = candidates[x]
        y_prev_list = candidates[x - 1]
        dp_prev = dp[x - 1]
        dp_x = np.full(len(y_list), np.inf)
        prev_x = np.full(len(y_list), -1)

        for i, y in enumerate(y_list):
            v = float(V[y, x])
            base = (-voltage_reward * v) if valid_col[x] else missing_col_penalty

            best_cost, best_j = np.inf, -1
            for j, y_prev in enumerate(y_prev_list):
                jump = abs(y - y_prev)
                if jump > max_jump:
                    continue
                cost = dp_prev[j] + smooth_penalty * jump + base
                if cost < best_cost:
                    best_cost, best_j = cost, j

            if best_j < 0:
                for j, y_prev in enumerate(y_prev_list):
                    jump = abs(y - y_prev)
                    cost = dp_prev[j] + smooth_penalty * jump + base + 5.0
                    if cost < best_cost:
                        best_cost, best_j = cost, j

            dp_x[i] = best_cost
            prev_x[i] = best_j

        dp.append(dp_x)
        prev.append(prev_x)

    last_candidates = candidates[-1]
    if y_end in last_candidates:
        end_i = last_candidates.index(y_end)
    else:
        end_i = int(np.argmin([abs(y - y_end) for y in last_candidates]))

    path_y = np.zeros(W, dtype=int)
    path_y[-1] = last_candidates[end_i]
    cur_i = end_i

    for x in range(W - 1, 0, -1):
        cur_i = int(prev[x][cur_i])
        if cur_i < 0:
            cur_i = 0
        path_y[x - 1] = candidates[x - 1][cur_i]

    return np.arange(W), path_y

def smooth_trajectory(y, win=SMOOTH_WIN):
    y = np.asarray(y, dtype=float)
    if win <= 1:
        return y.copy()
    if win % 2 == 0:
        win += 1
    pad = win // 2
    y_pad = np.pad(y, (pad, pad), mode='edge')
    kernel = np.ones(win) / win
    return np.convolve(y_pad, kernel, mode='valid')

def predict_from_last_point_quadratic(x_hist, y_hist, steps, H_limit, fit_win=PRED_FIT_WIN):
    x_hist = np.asarray(x_hist[-fit_win:], dtype=float)
    y_hist = np.asarray(y_hist[-fit_win:], dtype=float)

    if len(x_hist) < 5 or steps <= 0:
        return None, None

    x_last = x_hist[-1]
    y_last = y_hist[-1]
    h = x_hist - x_last
    dy = y_hist - y_last

    X = np.column_stack([h, h**2])
    weights = np.linspace(0.4, 1.0, len(h))
    coef = np.linalg.lstsq(weights[:, None] * X, weights * dy, rcond=None)[0]
    b, c = coef

    h_pred = np.arange(1, steps + 1, dtype=float)
    y_pred = y_last + b * h_pred + c * (h_pred ** 2)
    y_pred = np.clip(y_pred, 0, H_limit - 1)

    x_pred = np.arange(int(x_last) + 1, int(x_last) + 1 + steps)
    return x_pred, y_pred

# =========================================================
# Metrics
# =========================================================
def regression_metrics(y_true, y_pred, ref_span_px=None):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    err = y_pred - y_true
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err ** 2))

    out = {
        'mae': float(mae),
        'rmse': float(rmse)
    }
    if ref_span_px is not None and ref_span_px > 0:
        out['acc_rmse_pct'] = float(max(0.0, 100.0 * (1.0 - rmse / ref_span_px)))
    return out

def estimate_prediction_direction(x_pred, y_pred):
    if x_pred is None or y_pred is None or len(x_pred) < 2:
        return np.nan

    dx = float(x_pred[-1] - x_pred[0])
    dy = float(y_pred[-1] - y_pred[0])
    return float(np.degrees(np.arctan2(dy, dx)))

def estimate_mean_speed_only(x_path, y_path, V_full, pixel_size_m, collection_time_s):
   
    x_path = np.asarray(x_path, dtype=int)
    y_path = np.asarray(np.round(y_path), dtype=int)

    H, W = V_full.shape
    y_path = np.clip(y_path, 0, H - 1)
    x_path = np.clip(x_path, 0, W - 1)

    voltages = np.array([V_full[y, x] for x, y in zip(x_path, y_path)], dtype=float)

    t_since = invert_relaxation_time_from_voltage(voltages)
    t_since = np.clip(t_since, 0.0, collection_time_s)
    t_event = collection_time_s - t_since

    dx_pix = np.diff(x_path.astype(float))
    dy_pix = np.diff(y_path.astype(float))
    dt = np.diff(t_event.astype(float))

    valid = dt > 1e-6
    if np.sum(valid) == 0:
        return np.nan

    dx_m = dx_pix[valid] * pixel_size_m
    dy_m = dy_pix[valid] * pixel_size_m
    dt_valid = dt[valid]

    vx = dx_m / dt_valid
    vy = dy_m / dt_valid
    speed = np.sqrt(vx ** 2 + vy ** 2)

    return float(np.mean(speed))

# =========================================================
# Plot
# =========================================================
def plot_main_figure(
    V_full, gt_visible_df, gt_hidden_df,
    x_rec, y_raw, y_smooth, x_pred, y_pred,
    out_png='fig_main.png'
):
    cmap = LinearSegmentedColormap.from_list('w2p', [(1, 1, 1), (0.72, 0.52, 0.95)])

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(V_full, cmap=cmap, vmin=0, vmax=TARGET_VOLTAGE, origin='upper', aspect='auto')

    ax.plot(gt_visible_df['x_pixel'], gt_visible_df['y_pixel'],
            color='gray', lw=1.2, alpha=0.7, label='GT available')
    ax.plot(gt_hidden_df['x_pixel'], gt_hidden_df['y_pixel'],
            'mo', ms=7, label='Hidden GT (3-step)')
    ax.plot(x_rec, y_raw, color='red', lw=1.4, label='Reconstructed(raw)')
    ax.plot(x_rec, y_smooth, color='green', lw=2.2, label='Reconstructed(smoothed)')

    if x_pred is not None and y_pred is not None:
        ax.plot(x_pred, y_pred, '--o', color='blue', lw=2.0, ms=6, label='Predicted(3-step)')

    ax.set_xlim(0, V_full.shape[1] - 1)
    ax.set_ylim(V_full.shape[0] - 1, 0)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')
    ax.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close(fig)

def plot_raw_curve_only(x_rec, y_raw, H, W, out_png='fig_reconstructed_raw.png'):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')
    ax.plot(x_rec, y_raw, color='red', lw=2.0)

    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

def plot_smoothed_curve_only(x_rec, y_smooth, H, W, out_png='fig_reconstructed_smoothed.png'):
    fig, ax = plt.subplots(figsize=(12, 6), facecolor='white')
    ax.set_facecolor('white')
    ax.plot(x_rec, y_smooth, color='green', lw=2.2)

    ax.set_xlim(0, W - 1)
    ax.set_ylim(H - 1, 0)
    ax.set_xlabel('Column')
    ax.set_ylabel('Row')

    plt.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# =========================================================
# Main
# =========================================================
def main():
    print("RUNNING NEW VERSION - SPEED FIXED")

    voltage_matrix, gt_df, meta = generate_sine_trajectory_100x50(noise_ratio=0.05, seed=42)
    H, W = voltage_matrix.shape

    gt_hidden_df = gt_df.tail(PRED_STEPS).reset_index(drop=True)
    gt_visible_df = gt_df.iloc[:-PRED_STEPS].reset_index(drop=True)

    observed_end_x = int(gt_visible_df['x_pixel'].iloc[-1])
    observed_end_y = int(round(gt_visible_df['y_pixel'].iloc[-1]))

    V_obs = voltage_matrix[:, :observed_end_x + 1]
    candidates, valid_col = build_candidates_per_column_adaptive(V_obs)
    x_rec, y_raw = dp_reconstruct_with_fixed_end(V_obs, candidates, valid_col, y_end=observed_end_y)
    y_smooth = smooth_trajectory(y_raw, win=SMOOTH_WIN)

    y_true_rec = np.interp(x_rec, gt_visible_df['x_pixel'], gt_visible_df['y_pixel'])
    rec_raw_metrics = regression_metrics(y_true_rec, y_raw, ref_span_px=H_REF_FOR_ACC)
    rec_smooth_metrics = regression_metrics(y_true_rec, y_smooth, ref_span_px=H_REF_FOR_ACC)

  
    path_for_pred = y_smooth if USE_SMOOTH_FOR_PRED else y_raw
    fit_n = min(PRED_FIT_WIN, len(x_rec))
    x_hist = x_rec[-fit_n:]
    y_hist = path_for_pred[-fit_n:]
    x_pred, y_pred = predict_from_last_point_quadratic(x_hist, y_hist, PRED_STEPS, H_limit=H)

    pred_metrics = {'mae': np.nan, 'rmse': np.nan, 'acc_rmse_pct': np.nan}
    pred_direction = np.nan
    if x_pred is not None and y_pred is not None and len(gt_hidden_df) > 0:
        y_true_pred = gt_hidden_df['y_pixel'].values
        pred_metrics = regression_metrics(y_true_pred, y_pred, ref_span_px=H_REF_FOR_ACC)
        pred_direction = estimate_prediction_direction(x_pred, y_pred)

 
    mean_speed = estimate_mean_speed_only(
        x_rec,
        path_for_pred,
        voltage_matrix,
        pixel_size_m=meta['pixel_size_m'],
        collection_time_s=meta['collection_time_s']
    )


    plot_main_figure(
        voltage_matrix, gt_visible_df, gt_hidden_df,
        x_rec, y_raw, y_smooth, x_pred, y_pred,
        out_png='fig_main.png'
    )
    plot_raw_curve_only(x_rec, y_raw, H, W, out_png='fig_reconstructed_raw.png')
    plot_smoothed_curve_only(x_rec, y_smooth, H, W, out_png='fig_reconstructed_smoothed.png')

    print("\n" + "=" * 55)
    print("Evaluation Summary")
    print("=" * 55)

    print("1. Reconstruction Accuracy")
    print(f"   Raw Accuracy        : {rec_raw_metrics['acc_rmse_pct']:.2f}%")
    print(f"   Smoothed Accuracy   : {rec_smooth_metrics['acc_rmse_pct']:.2f}%")

    print("\n2. Prediction Accuracy")
    print(f"   Prediction Accuracy : {pred_metrics['acc_rmse_pct']:.2f}%")

    print("\n3. Speed")
    print(f"   Mean Speed          : {mean_speed:.4f} m/s")

    print("\n4. Prediction Direction")
    print(f"   Direction Angle     : {pred_direction:.4f} deg")

    print("=" * 55)
    print("Saved figures:")
    print(" - fig_main.png")
    print(" - fig_reconstructed_raw.png")
    print(" - fig_reconstructed_smoothed.png")
    print("=" * 55)

if __name__ == '__main__':
    main()