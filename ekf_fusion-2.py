#!/usr/bin/env python3
"""
ekf_fusion.py — Paper-faithful EKF for differential-drive robot
(Huang et al. 2025, Scientific Reports — "Application of multi-sensor fusion
localization algorithm based on recurrent neural networks")

State:   [x, y, theta]                              (paper: X_t = [x_t,y_t,theta_t]^T)
Control: [v, omega] from cmd_vel_filtered.csv        (paper: U_t = [v_t,omega_t]^T,
                                                        "received from the motion controller")
Odometry measurement: [s_t, phi_t]                   (paper: Z_Odometry = [s_t,phi_t]^T —
                                                        incremental displacement + turn angle,
                                                        NOT absolute x,y)
IMU measurement:      [theta_from_orientation]        (paper: Z_IMU = [omega,alpha]^T — rotational
                                                        data only. We use the IMU's fused absolute
                                                        orientation output as a direct theta
                                                        measurement, since that's the only way
                                                        gyro/rotational data can correct a
                                                        position-orientation state without also
                                                        tracking velocity states. Linear
                                                        acceleration channels are DROPPED — with a
                                                        [x,y,theta]-only state and an all-zero H
                                                        row, they were mathematically inert in the
                                                        old code (verified: K's accel columns were
                                                        exactly zero every step).

IMPORTANT — read before trusting the RMSE number this prints:
This script fuses ONLY odometry + IMU, matching the paper's own EKF stage
(see Fig. 2 of the paper: EKF fuses IMU+odometry into a *preliminary* estimate;
absolute position correction only enters later, via LiDAR, in the RNN
complementary-fusion stage). Neither odometry nor cmd_vel is an absolute
position reference — both are dead-reckoning. No EKF weighting scheme can
stop dead-reckoning-only estimates from drifting over a real ~28s drive with
real wheel slip/inertia. The paper's own reported "EKF only" RMSE
(0.065-0.178m) was measured under synthetic Gaussian noise layered onto an
already-accurate simulated trajectory — not real compounding drift — so it
is not a valid target for this stage alone. Real accuracy in this pipeline
should come from lidar_fusion.py (not yet built), which is the only sensor
in the whole system providing anything resembling an absolute position fix.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import time

LOG_DIR = os.path.expanduser('~/ws_mobile/sensor_logs')


# ══════════════════════════════════════════════════════════════════════════════
# EKF — 3D state [x, y, theta], paper-faithful odometry [s,phi] + IMU theta
# ══════════════════════════════════════════════════════════════════════════════

class DiffDriveEKF:
    def __init__(self, init_x=0.0, init_y=0.0, init_theta=0.0):
        self.x = np.array([init_x, init_y, init_theta], dtype=float)
        self.P = np.eye(3) * 1.0

        # Process noise density (per SECOND, not per-tick — predict() scales
        # this by dt itself). This is the fix from the earlier debugging
        # session: the old code added a flat 0.1 every tick regardless of
        # elapsed time, which at 50Hz meant an effective 5.0 m^2/s injection
        # rate — wildly too aggressive. Confirmed via real-data test that
        # this alone wasn't the dominant RMSE driver, but it's still the
        # mathematically correct way to define process noise, so it stays.
        self.Q = np.array([[0.1,  0.0,  0.0],
                           [0.0,  0.1,  0.0],
                           [0.0,  0.0,  0.05]])

        # Odometry measurement noise: [s (displacement, m), phi (turn, rad)]
        # phi noise loosened from 0.01 to 0.35 (sigma ~5.7deg -> ~34deg):
        # real-data diagnostics on this session showed odometry's own
        # dead-reckoned turn-angle integration is off by 10-25+ degrees
        # during turning (correlation with |omega|: 0.302), so it must not
        # be trusted tightly.
        self.R_odom = np.diag([0.02, 0.35])

        # IMU measurement noise: [theta (rad)] — real orientation output,
        # not raw gyro rate. Tightened from 0.02 to 0.005 (sigma ~8.1deg ->
        # ~4deg): real-data diagnostics showed IMU absolute orientation vs
        # ground truth has mean abs error ~0.037deg with no growth over the
        # run, i.e. essentially exact — this is the reliable heading
        # channel and should dominate the odometry-derived turn estimate.
        self.R_imu = np.array([[0.005]])

    # ── Prediction step ───────────────────────────────────────────────────────
    def predict(self, v: float, omega: float, dt: float):
        """Nonlinear differential-drive kinematics, dt-scaled process noise."""
        x, y, th = self.x

        self.x = np.array([
            x  + v * np.cos(th) * dt,
            y  + v * np.sin(th) * dt,
            th + omega * dt
        ])
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))

        F = np.array([
            [1.0, 0.0, -v * np.sin(th) * dt],
            [0.0, 1.0,  v * np.cos(th) * dt],
            [0.0, 0.0,  1.0               ]
        ])

        self.P = F @ self.P @ F.T + self.Q * dt

    # ── Odometry update: paper's Z_Odometry = [s_t, phi_t] ────────────────────
    def update_odometry(self, x_prev, y_prev, theta_prev, s_meas, phi_meas):
        """
        s_meas, phi_meas: distance and turn angle odometry reports it moved
        since the LAST accepted state (x_prev,y_prev,theta_prev) — NOT an
        absolute position fix. h(x) = [dist(x,y ; x_prev,y_prev), theta-theta_prev].
        """
        x, y, th = self.x
        dx, dy = x - x_prev, y - y_prev
        s_pred = np.hypot(dx, dy)
        phi_pred = th - theta_prev

        h_x = np.array([s_pred, phi_pred])
        z   = np.array([s_meas, phi_meas])
        innov = z - h_x
        innov[1] = np.arctan2(np.sin(innov[1]), np.cos(innov[1]))  # wrap turn angle

        # Jacobian of h(x) w.r.t. state, evaluated at current predicted state
        if s_pred > 1e-6:
            H = np.array([
                [dx / s_pred, dy / s_pred, 0.0],
                [0.0,         0.0,         1.0]
            ])
        else:
            # Avoid divide-by-zero when the robot hasn't moved this tick —
            # displacement Jacobian is undefined at s=0, so skip the
            # distance row's contribution (leave turn-angle correction active).
            H = np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0]
            ])

        S = H @ self.P @ H.T + self.R_odom
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        self.x = self.x + K @ innov
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))

        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_odom @ K.T

        return innov, K

    # ── IMU update: absolute heading only ──────────────────────────────────────
    def update_imu_theta(self, theta_meas):
        H = np.array([[0.0, 0.0, 1.0]])
        th = self.x[2]
        innov = np.array([np.arctan2(np.sin(theta_meas - th), np.cos(theta_meas - th))])

        S = H @ self.P @ H.T + self.R_imu
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        self.x = self.x + (K @ innov)
        self.x[2] = np.arctan2(np.sin(self.x[2]), np.cos(self.x[2]))

        I_KH = np.eye(3) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R_imu @ K.T

        return innov, K


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS  (unchanged from previous version)
# ══════════════════════════════════════════════════════════════════════════════

def load_latest(prefix: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(LOG_DIR, f'{prefix}_filtered.csv'))
    if not files:
        files = glob.glob(os.path.join(LOG_DIR, f'{prefix}_*filtered*.csv'))
    if not files:
        raise FileNotFoundError(f"No filtered CSV for '{prefix}' in {LOG_DIR}")
    path = max(files, key=os.path.getctime)
    print(f'  Loading: {os.path.basename(path)}')
    return pd.read_csv(path)


def quat_to_yaw(oz, ow) -> float:
    w, z = float(ow), float(oz)
    return np.arctan2(2.0 * w * z, 1.0 - 2.0 * z * z)


def get_col(df, candidates, default=0.0):
    for col in candidates:
        if col in df.columns:
            return df[col].fillna(default)
    print(f"  Warning: {candidates} not found — using {default}")
    return pd.Series([default] * len(df))


def compute_metrics(pred_x, pred_y, true_x, true_y):
    err = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)
    rmse = np.sqrt(np.mean(err**2))
    mae  = np.mean(err)
    std  = np.std(err)
    return rmse, mae, std, err


# ══════════════════════════════════════════════════════════════════════════════
# MAIN FUSION
# ══════════════════════════════════════════════════════════════════════════════

def run_ekf_fusion():
    print("\n" + "="*70)
    print("  EKF FUSION (paper-faithful) — State=[x,y,θ]")
    print("  Odometry: [s,φ] incremental   IMU: absolute θ only")
    print("="*70)

    imu_df  = load_latest('imu')
    odom_df = load_latest('odom')

    n = min(len(imu_df), len(odom_df))
    imu_df  = imu_df.iloc[:n].reset_index(drop=True)
    odom_df = odom_df.iloc[:n].reset_index(drop=True)
    print(f"  Fusing {n} timesteps")

    # ── Raw odometry position + heading (used to derive s_t, phi_t) ──────────
    odom_x = get_col(odom_df, ['pos_x', 'odom_pos_x', 'x'])
    odom_y = get_col(odom_df, ['pos_y', 'odom_pos_y', 'y'])
    odom_oz = get_col(odom_df, ['orient_z', 'odom_orient_z'])
    odom_ow = get_col(odom_df, ['orient_w', 'odom_orient_w'], default=1.0)
    odom_theta = pd.Series([quat_to_yaw(odom_oz.iloc[i], odom_ow.iloc[i])
                            for i in range(n)])

    # s_t, phi_t are DISTANCE/ANGLE DIFFERENCES — invariant to any fixed
    # rotation/translation of the odometry frame, so no frame-alignment
    # is needed here (unlike the old absolute-position approach).
    d_odom_x = np.diff(odom_x.values, prepend=odom_x.values[0])
    d_odom_y = np.diff(odom_y.values, prepend=odom_y.values[0])
    s_meas   = np.hypot(d_odom_x, d_odom_y)
    s_meas[0] = 0.0
    phi_raw  = np.diff(odom_theta.values, prepend=odom_theta.values[0])
    phi_meas = np.arctan2(np.sin(phi_raw), np.cos(phi_raw))
    phi_meas[0] = 0.0

    # ── IMU absolute orientation (theta measurement) ─────────────────────────
    if 'imu_orient_z' in imu_df.columns and 'imu_orient_w' in imu_df.columns:
        imu_theta = imu_df.apply(
            lambda r: quat_to_yaw(r['imu_orient_z'], r['imu_orient_w']), axis=1)
    elif 'orient_z' in imu_df.columns and 'orient_w' in imu_df.columns:
        imu_theta = imu_df.apply(
            lambda r: quat_to_yaw(r['orient_z'], r['orient_w']), axis=1)
    else:
        print("  Warning: IMU orient cols not found — using odom orientation")
        imu_theta = odom_theta.copy()

    # ── Control input Ut=[v,ω] — commanded velocity (paper-correct) ──────────
    v_lin = get_col(odom_df, ['lin_vel_x', 'linear_x', 'odom_linear_x', 'vel_x'])
    omega = get_col(imu_df,  ['ang_vel_z', 'angular_vel_z', 'imu_angular_vel_z',
                               'angular_z'])
    cmd_path = os.path.join(LOG_DIR, 'cmd_vel_filtered.csv')
    if os.path.exists(cmd_path):
        cmd_df = pd.read_csv(cmd_path)
        cmd_df = cmd_df.iloc[:n].reset_index(drop=True)
        v_lin  = get_col(cmd_df, ['linear_x'])
        omega  = get_col(cmd_df, ['angular_z'])
        print("  Control input: cmd_vel_filtered.csv (commanded velocity)")
    else:
        print("  Warning: no cmd_vel_filtered.csv — falling back to sensor-"
              "derived velocity for prediction.")

    # Timestamps
    if 'timestamp' in odom_df.columns:
        timestamps = odom_df['timestamp'].values
    else:
        timestamps = np.arange(n) / 50.0

    # ── Ground truth ───────────────────────────────────────────────────────
    filtered_gt_path = os.path.join(LOG_DIR, 'ground_truth_filtered.csv')
    has_gt, gt_is_real = False, False

    if os.path.exists(filtered_gt_path):
        gt_df = pd.read_csv(filtered_gt_path).iloc[:n].reset_index(drop=True)
        gt_x  = get_col(gt_df, ['gt_x'])
        gt_y  = get_col(gt_df, ['gt_y'])
        gt_oz = get_col(gt_df, ['gt_orient_z'])
        gt_ow = get_col(gt_df, ['gt_orient_w'])
        gt_theta = pd.Series([quat_to_yaw(gt_oz.iloc[i], gt_ow.iloc[i])
                              for i in range(len(gt_df))])
        has_gt, gt_is_real = True, True
        print("  Ground truth: ground_truth_filtered.csv (real, resampled)")
    else:
        print("\n  ⚠️  No ground truth file found — cannot compute real RMSE.")
        gt_x, gt_y, gt_theta = odom_x.copy(), odom_y.copy(), odom_theta.copy()
        has_gt, gt_is_real = True, False

    # ── Initialise EKF from ground truth's own first pose ────────────────────
    if gt_is_real:
        ekf = DiffDriveEKF(float(gt_x.iloc[0]), float(gt_y.iloc[0]), float(gt_theta.iloc[0]))
        print(f"  Init from ground truth: x={ekf.x[0]:.3f}  y={ekf.x[1]:.3f}  "
              f"θ={ekf.x[2]:.3f} rad")
    else:
        ekf = DiffDriveEKF(0.0, 0.0, float(imu_theta.iloc[0]))
        print(f"  Init from origin (no real GT): x=0  y=0  θ={ekf.x[2]:.3f} rad")

    # ── Run EKF ────────────────────────────────────────────────────────────
    out_x, out_y, out_theta = [], [], []
    innov_s, innov_phi, innov_imu_theta = [], [], []
    cov_trace, runtimes = [], []

    x_prev, y_prev, theta_prev = ekf.x[0], ekf.x[1], ekf.x[2]

    for i in range(n):
        dt = float(timestamps[i] - timestamps[i-1]) if i > 0 else 0.02
        dt = np.clip(dt, 0.001, 0.5)

        t0 = time.perf_counter()

        ekf.predict(float(v_lin.iloc[i]), float(omega.iloc[i]), dt)

        innov_o, _ = ekf.update_odometry(x_prev, y_prev, theta_prev,
                                          float(s_meas[i]), float(phi_meas[i]))
        innov_i, _ = ekf.update_imu_theta(float(imu_theta.iloc[i]))

        t1 = time.perf_counter()
        runtimes.append((t1 - t0) * 1000.0)

        # This step's accepted state becomes next step's "_prev" reference
        x_prev, y_prev, theta_prev = ekf.x[0], ekf.x[1], ekf.x[2]

        out_x.append(ekf.x[0])
        out_y.append(ekf.x[1])
        out_theta.append(ekf.x[2])
        innov_s.append(innov_o[0])
        innov_phi.append(innov_o[1])
        innov_imu_theta.append(innov_i[0])
        cov_trace.append(np.trace(ekf.P))

    # ── Save output CSV ────────────────────────────────────────────────────
    result_df = pd.DataFrame({
        'timestamp':       timestamps,
        'odom_x':          odom_x.values,
        'odom_y':          odom_y.values,
        'ekf_x':           out_x,
        'ekf_y':           out_y,
        'ekf_theta':       out_theta,
        'innov_s':         innov_s,
        'innov_phi':       innov_phi,
        'innov_imu_theta': innov_imu_theta,
        'cov_trace':       cov_trace,
        'runtime_ms':      runtimes
    })
    out_path = os.path.join(LOG_DIR, 'ekf_fused_output.csv')
    result_df.to_csv(out_path, index=False)
    print(f"\n  Saved: ekf_fused_output.csv  ({n} rows)")

    rt = np.array(runtimes)
    print(f"\n  ── EKF Runtime ──")
    print(f"     Mean : {rt.mean():.3f} ms   Std: {rt.std():.3f} ms")

    if has_gt:
        rmse, mae, std, err = compute_metrics(np.array(out_x), np.array(out_y),
                                               gt_x.values[:n], gt_y.values[:n])
        label = "EKF Accuracy vs REAL Ground Truth" if gt_is_real else \
                "EKF 'Accuracy' vs Odometry Proxy — NOT REAL"
        print(f"\n  ── {label} ──")
        print(f"     RMSE : {rmse:.4f} m")
        print(f"     MAE  : {mae:.4f} m")
        print(f"     Std  : {std:.4f} m")
        print(f"\n  NOTE: this EKF stage has no absolute position reference")
        print(f"  (odometry+cmd_vel are both dead-reckoning). If RMSE here is")
        print(f"  close to or worse than the previous absolute-position version,")
        print(f"  that CONFIRMS the paper's own EKF-only design can't close this")
        print(f"  gap without LiDAR — build lidar_fusion.py next, not more EKF tuning.")

        metrics_df = pd.DataFrame({'metric': ['RMSE', 'MAE', 'Std', 'ground_truth_is_real'],
                                    'value_m': [rmse, mae, std, gt_is_real]})
        metrics_df.to_csv(os.path.join(LOG_DIR, 'ekf_metrics.csv'), index=False)
        print(f"     Saved: ekf_metrics.csv")

    # ── Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if has_gt:
        gt_label = 'Ground Truth' if gt_is_real else 'Ground Truth (⚠️ odom proxy)'
        axes[0].plot(gt_x.values[:n], gt_y.values[:n], label=gt_label,
                     color='black', linewidth=2, linestyle='--')
    axes[0].plot(odom_x, odom_y, label='Raw Odometry', alpha=0.5, color='orange')
    axes[0].plot(out_x, out_y, label='EKF (paper-faithful)', color='blue', linewidth=2)
    axes[0].set_xlabel('X (m)'); axes[0].set_ylabel('Y (m)')
    axes[0].set_title('EKF Trajectory'); axes[0].legend(); axes[0].grid(True); axes[0].axis('equal')

    axes[1].plot(innov_s, label='Innovation s (m)', color='red', alpha=0.7)
    axes[1].plot(innov_phi, label='Innovation φ (rad)', color='green', alpha=0.7)
    axes[1].plot(innov_imu_theta, label='Innovation IMU θ (rad)', color='purple', alpha=0.7)
    axes[1].set_xlabel('Timestep'); axes[1].set_ylabel('Innovation')
    axes[1].set_title('Measurement Innovation (Residual)'); axes[1].legend(); axes[1].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(LOG_DIR, 'ekf_trajectory_comparison.png'), dpi=150)
    print(f"\n  Saved: ekf_trajectory_comparison.png")

    print("\n✅ EKF fusion complete!")
    print("   Next: build lidar_fusion.py for absolute position correction,")
    print("   then python3 ~/ws_mobile/scripts/rnn_fusion_v2.py")


if __name__ == '__main__':
    run_ekf_fusion()
