#!/usr/bin/env python3
"""
preprocess_data.py — Complete preprocessing pipeline:
  - NaN handling
  - Remove duplicate timestamps
  - Sort timestamps
  - Outlier removal (IQR)
  - LiDAR ranges expanded to numeric columns (range_0..range_539)
  - Savitzky-Golay filter on ALL numeric cols including orientation/yaw
  - Adaptive window selection
  - Save intermediate stages
  - Print preprocessing statistics
  - ONE global shared timeline across IMU + Odom + LiDAR
  - Output: imu_filtered.csv, odom_filtered.csv, scan_filtered.csv, fusion_ready.csv
"""

import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from pathlib import Path
import glob
import os

# ── Config ────────────────────────────────────────────────────────────────────
LOG_DIR   = Path.home() / "ws_mobile" / "sensor_logs"
SG_WINDOW = 11
SG_POLY   = 3
TARGET_HZ = 50        # common timeline frequency
USE_IQR   = True      # IQR-based outlier clipping — paper only describes
                       # Savitzky-Golay filtering, not statistical outlier
                       # removal. Set False for a stricter paper-replication
                       # run (S-G filtering only, no IQR clipping beforehand).
IQR_SCALE = 3.0
LIDAR_MIN = 0.10
LIDAR_MAX = 2.50
N_BEAMS   = 540       # must match your LiDAR config
# ──────────────────────────────────────────────────────────────────────────────


# ══════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def load_latest(prefix: str) -> pd.DataFrame:
    files = glob.glob(str(LOG_DIR / f"{prefix}_*.csv"))
    files = [f for f in files if "filtered" not in f and "stage" not in f
             and "fusion_ready" not in f]
    if not files:
        raise FileNotFoundError(f"No raw CSV for '{prefix}' in {LOG_DIR}")
    path = max(files, key=os.path.getctime)
    print(f"  Loading: {Path(path).name}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def make_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "time_sec" in df.columns and "time_nsec" in df.columns:
        df["timestamp"] = (df["time_sec"].astype(float)
                           + df["time_nsec"].astype(float) * 1e-9)
    return df


def print_stats(df: pd.DataFrame, label: str, stage: str):
    num = df.select_dtypes(include=[np.number])
    print(f"\n  [{label}] — {stage}")
    print(f"    Rows: {len(df)}   NaN total: {num.isna().sum().sum()}")
    show_cols = [c for c in num.columns if c not in
                 ("timestamp","time_sec","time_nsec") and
                 not c.startswith("range_")][:5]
    for col in show_cols:
        print(f"    {col:22s} mean={num[col].mean():9.4f}  std={num[col].std():9.4f}")


def save_stage(df: pd.DataFrame, prefix: str, stage: str):
    out = LOG_DIR / f"{prefix}_{stage}.csv"
    df.to_csv(out, index=False)
    print(f"  Saved: {out.name}")


# ── Clean steps (run at native frequency) ─────────────────────────────────────

def handle_nan(df, label):
    n = df.isna().sum().sum()
    if n > 0:
        print(f"  [{label}] {n} NaNs found — interpolating")
        # FIX: only interpolate numeric columns. Calling .interpolate() on the
        # whole dataframe crashes (TypeError: Cannot interpolate with str dtype)
        # the moment any text column (frame_id, child_frame_id, covariance
        # strings, etc.) is present — which every sensor df here has.
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].interpolate(method="linear", limit_direction="both")
        df[num_cols] = df[num_cols].ffill().bfill()
    return df

def remove_duplicates(df, label):
    if "timestamp" not in df.columns:
        return df
    before = len(df)
    df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"  [{label}] Removed {removed} duplicate timestamps")
    return df

def sort_by_time(df, label):
    if "timestamp" not in df.columns:
        return df
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def remove_outliers(df, label):
    exclude = {"timestamp", "time_sec", "time_nsec"}
    cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude]
    total = 0
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        mask = (df[col] < q1 - IQR_SCALE*iqr) | (df[col] > q3 + IQR_SCALE*iqr)
        if mask.sum():
            df.loc[mask, col] = np.nan
            total += mask.sum()
    if total:
        print(f"  [{label}] Clipped {total} outliers — re-interpolating")
        # FIX: same str-dtype crash as handle_nan — restrict to the numeric
        # 'cols' list already computed above, not the whole dataframe.
        df[cols] = df[cols].interpolate(method="linear", limit_direction="both")
        df[cols] = df[cols].ffill().bfill()
    return df

def sg_filter(df, label):
    """S-G filter on all numeric cols INCLUDING orientation/yaw. Adaptive window."""
    exclude = {"timestamp", "time_sec", "time_nsec"}
    cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude]
    n = len(df)
    win = SG_WINDOW
    if n < win:
        win = n if n % 2 == 1 else n - 1
    win = max(win, SG_POLY + 2)
    if win % 2 == 0:
        win -= 1
    orient_cols = [c for c in cols if "orient" in c or "yaw" in c or "theta" in c]
    print(f"  [{label}] S-G filter window={win} on {len(cols)} cols "
          f"(orientation cols: {orient_cols})")

    # FIX: quaternion sign-flip continuity, BEFORE filtering.
    # q and -q represent the identical rotation (double cover), but the
    # data source can emit either sign from sample to sample. S-G filtering
    # treats x/y/z/w as independent scalar signals — if consecutive samples
    # flip sign, the filter averages q and -q as if they were far apart
    # rotations, producing a garbage intermediate value. Verified on real
    # session data: this produced an isolated -134 degree heading spike in
    # ground_truth_filtered.csv at one timestep (dot(q_i, q_{i-1}) < 0
    # there), corrupting that single row's x/y/theta after S-G filtering.
    # Fix: walk the raw sequence and flip sign whenever consecutive
    # quaternions have negative dot product, so the whole run stays on one
    # consistent sign branch before filtering ever runs.
    pre_quat_prefixes = set()
    for c in cols:
        if c.endswith("orient_w"):
            prefix = c[:-len("orient_w")]
            if all(f"{prefix}orient_{ax}" in df.columns for ax in ("x", "y", "z", "w")):
                pre_quat_prefixes.add(prefix)
    for prefix in pre_quat_prefixes:
        q = df[[f"{prefix}orient_x", f"{prefix}orient_y",
                f"{prefix}orient_z", f"{prefix}orient_w"]].values.copy()
        flips = 0
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
                flips += 1
        if flips:
            df[f"{prefix}orient_x"] = q[:, 0]
            df[f"{prefix}orient_y"] = q[:, 1]
            df[f"{prefix}orient_z"] = q[:, 2]
            df[f"{prefix}orient_w"] = q[:, 3]
            label_p = prefix if prefix else "(unprefixed)"
            print(f"  [{label}] Fixed {flips} quaternion sign-flip(s) in "
                  f"'{label_p}' group (continuity enforced pre-filter)")

    for col in cols:
        try:
            df[col] = savgol_filter(df[col].values,
                                    window_length=win, polyorder=SG_POLY)
        except Exception as e:
            print(f"  [{label}] Warning: skipped {col}: {e}")

    # Renormalize quaternions after filtering. S-G filters each component
    # (x, y, z, w) independently, which does NOT preserve the unit-norm
    # constraint (x²+y²+z²+w²=1). Left unnormalized, downstream yaw
    # calculations (atan2 on filtered components) are computed from a
    # slightly invalid quaternion. Detects any {prefix}orient_{x,y,z,w}
    # group present (handles both plain 'orient_*' and prefixed 'gt_orient_*').
    quat_prefixes = set()
    for c in cols:
        if c.endswith("orient_w"):
            prefix = c[:-len("orient_w")]
            if all(f"{prefix}orient_{ax}" in df.columns for ax in ("x", "y", "z", "w")):
                quat_prefixes.add(prefix)
    for prefix in quat_prefixes:
        qx = df[f"{prefix}orient_x"].values
        qy = df[f"{prefix}orient_y"].values
        qz = df[f"{prefix}orient_z"].values
        qw = df[f"{prefix}orient_w"].values
        norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
        norm[norm == 0] = 1.0  # avoid div-by-zero in the degenerate all-zero case
        df[f"{prefix}orient_x"] = qx / norm
        df[f"{prefix}orient_y"] = qy / norm
        df[f"{prefix}orient_z"] = qz / norm
        df[f"{prefix}orient_w"] = qw / norm
    if quat_prefixes:
        labels = sorted(p if p else "(unprefixed)" for p in quat_prefixes)
        print(f"  [{label}] Renormalized quaternion(s): {labels} "
              f"(S-G filtering breaks unit-norm)")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# LIDAR: expand ranges string → numeric columns BEFORE filtering
# ══════════════════════════════════════════════════════════════════════════════

def expand_lidar_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert semicolon-separated ranges string into numeric columns range_0..range_N.
    Invalid values (inf, out of [LIDAR_MIN, LIDAR_MAX]) replaced with NaN.
    """
    if "ranges" not in df.columns:
        print("  [LiDAR] No 'ranges' column found — skipping expansion")
        return df

    print(f"  [LiDAR] Expanding ranges string → numeric columns...")
    range_data = df["ranges"].apply(
        lambda s: [float(x) for x in str(s).split(";")]
    )

    # Determine actual beam count from data
    beam_count = range_data.apply(len).mode()[0]
    print(f"  [LiDAR] Detected {beam_count} beams per scan")

    range_array = np.array(range_data.tolist(), dtype=np.float32)

    # Clip to actual beam count
    range_array = range_array[:, :beam_count]

    # Replace inf and out-of-range with NaN
    range_array[~np.isfinite(range_array)] = np.nan
    range_array[range_array < LIDAR_MIN] = np.nan
    range_array[range_array > LIDAR_MAX] = np.nan

    invalid = np.isnan(range_array).sum()
    print(f"  [LiDAR] {invalid} out-of-range/inf values set to NaN")

    # Build range columns
    range_cols = {f"range_{i}": range_array[:, i] for i in range(beam_count)}
    range_df = pd.DataFrame(range_cols)

    # Drop original string column and attach numeric ones
    df = df.drop(columns=["ranges"])
    df = pd.concat([df.reset_index(drop=True), range_df], axis=1)

    return df


# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL TIMELINE: interpolate all sensors onto one shared time grid
# ══════════════════════════════════════════════════════════════════════════════

def build_global_timeline(imu_df, odom_df, lidar_df, gt_df=None, cmd_df=None):
    """
    Build one global time grid from t_start to t_end at TARGET_HZ.
    t_start = latest start among all sensors (all sensors have data here).
    t_end   = earliest end among all sensors.
    """
    labels = ['IMU', 'Odom', 'LiDAR', 'GroundTruth', 'CmdVel']
    dfs    = [imu_df, odom_df, lidar_df, gt_df, cmd_df]

    starts, ends = [], []
    ranges = {}
    for label, df in zip(labels, dfs):
        if df is not None and "timestamp" in df.columns:
            lo, hi = df["timestamp"].min(), df["timestamp"].max()
            starts.append(lo)
            ends.append(hi)
            ranges[label] = (lo, hi)

    # FIX: catch mismatched clock domains (e.g. one sensor stamped with real
    # wall-clock epoch time ~1.7 billion while others use sim-time elapsed
    # seconds in the hundreds) BEFORE silently producing an inverted/empty
    # range. This exact bug happened with cmd_vel when data_logger.py was run
    # without use_sim_time:=true — every downstream file went to 0 rows with
    # no error, just a confusing "0 steps" line.
    all_lo = min(starts)
    all_hi = max(ends)
    if all_hi - all_lo > 0:
        outliers = []
        for label, (lo, hi) in ranges.items():
            # A sensor whose range sits almost entirely outside the union of
            # everyone else's range is on a different clock, not just a
            # different recording duration.
            others_lo = min(r[0] for l, r in ranges.items() if l != label) if len(ranges) > 1 else lo
            others_hi = max(r[1] for l, r in ranges.items() if l != label) if len(ranges) > 1 else hi
            if hi < others_lo - 1.0 or lo > others_hi + 1.0:
                outliers.append((label, lo, hi))
        if outliers:
            print("\n  🛑 CLOCK MISMATCH DETECTED — these sensor(s) are on a "
                  "completely different clock than the rest:")
            for label, lo, hi in outliers:
                print(f"     {label}: {lo:.2f} → {hi:.2f}  "
                      f"(everyone else is roughly {all_lo:.2f} → {all_hi:.2f})")
            print("  🛑 This produces an inverted/empty time range and every "
                  "downstream file silently becomes 0 rows with no crash.")
            print("  🛑 Most likely cause: data_logger.py was run without "
                  "'--ros-args -p use_sim_time:=true', so plain Twist "
                  "messages (cmd_vel) got stamped with real wall-clock time "
                  "instead of simulation time like every other sensor.")
            print("  🛑 Fix: rerun the driving session with use_sim_time:=true, "
                  "then rerun this script. Continuing anyway will produce "
                  "garbage (or empty) output.\n")

    t_start = max(starts)
    t_end   = min(ends)
    dt      = 1.0 / TARGET_HZ
    t_grid  = np.arange(t_start, t_end, dt)

    print(f"\n  Global timeline: {t_start:.3f} → {t_end:.3f}s "
          f"({len(t_grid)} steps @ {TARGET_HZ}Hz)")

    if len(t_grid) == 0:
        raise ValueError(
            f"Global timeline produced 0 steps (t_start={t_start:.3f} >= "
            f"t_end={t_end:.3f}). This means the sensor time ranges don't "
            f"overlap at all — almost always the clock-mismatch issue printed "
            f"above. Stopping here instead of silently writing empty "
            f"'*_filtered.csv' files and a fake '✅ PREPROCESSING COMPLETE'.")

    return t_grid


def resample_to_grid(df: pd.DataFrame, t_grid: np.ndarray,
                     label: str, method: str = "linear") -> pd.DataFrame:
    """
    Interpolate all numeric columns onto t_grid.

    method="linear" (default): correct for continuous physical signals
    (IMU, odometry, ground truth, LiDAR) — the true value genuinely varies
    smoothly between samples.

    method="zoh" (zero-order hold / step function): the value stays
    constant until the next real sample arrives, then jumps. Use this for
    cmd_vel — teleop_twist_keyboard only publishes when you press a NEW
    key (17 raw samples across an entire session is normal), not
    continuously. Linear interpolation between those sparse samples
    fabricates smooth velocity ramps that never actually happened; the
    real commanded velocity was constant until the next keypress. Feeding
    fabricated ramps into the EKF's predict step corrupts most of the
    trajectory, since a 17-sample signal stretched across 1000+ output
    rows is mostly interpolated, not real.
    """
    if "timestamp" not in df.columns:
        print(f"  [{label}] No timestamp — cannot resample")
        return df

    t = df["timestamp"].values.astype(float)
    out = pd.DataFrame({"timestamp": t_grid})

    # FIX: quaternion sign-flip continuity, BEFORE interpolating.
    # q and -q are the identical rotation (double cover), but np.interp
    # treats each x/y/z/w component as an independent scalar signal. If two
    # consecutive source samples happen to have opposite sign (same real
    # rotation, flipped representation), linearly interpolating between
    # them averages q and -q as if they were nearly opposite rotations,
    # producing a garbage intermediate quaternion at every grid point that
    # falls between them. Verified on real session data: ground truth does
    # NOT go through sg_filter() (deliberately unfiltered, see
    # process_ground_truth), so this resampling step was the actual source
    # of an isolated -134 degree heading spike in ground_truth_filtered.csv
    # — not sg_filter, which GT never reaches. Fix applies to every sensor
    # resampled here (IMU/odom already get this in sg_filter, but it's
    # harmless and cheap to also guarantee it here; GT and any future
    # unfiltered sensor need it here since this is the only step they pass
    # through before hitting the shared grid).
    df = df.copy()
    quat_prefixes = set()
    for c in df.columns:
        if c.endswith("orient_w"):
            prefix = c[:-len("orient_w")]
            if all(f"{prefix}orient_{ax}" in df.columns for ax in ("x", "y", "z", "w")):
                quat_prefixes.add(prefix)
    for prefix in quat_prefixes:
        q = df[[f"{prefix}orient_x", f"{prefix}orient_y",
                f"{prefix}orient_z", f"{prefix}orient_w"]].values.copy()
        flips = 0
        for i in range(1, len(q)):
            if np.dot(q[i], q[i - 1]) < 0:
                q[i] = -q[i]
                flips += 1
        if flips:
            df[f"{prefix}orient_x"] = q[:, 0]
            df[f"{prefix}orient_y"] = q[:, 1]
            df[f"{prefix}orient_z"] = q[:, 2]
            df[f"{prefix}orient_w"] = q[:, 3]
            label_p = prefix if prefix else "(unprefixed)"
            print(f"  [{label}] Fixed {flips} quaternion sign-flip(s) in "
                  f"'{label_p}' group before grid interpolation")

    for col in df.columns:
        if col == "timestamp":
            continue
        # FIX: dtype == object misses pandas' newer StringDtype ('string'),
        # which is what read_csv can produce for text columns depending on
        # pandas version/settings. is_numeric_dtype is the correct, version-
        # robust check — this crashed specifically on ground_truth's
        # frame_id/child_frame_id text columns.
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if method == "zoh":
            # Hold each value constant until the next real sample's time.
            idx = np.searchsorted(t, t_grid, side="right") - 1
            idx = np.clip(idx, 0, len(t) - 1)
            out[col] = df[col].values[idx]
        else:
            out[col] = np.interp(t_grid, t, df[col].values.astype(float))

    # Renormalize quaternions after grid interpolation. Even with sign
    # continuity enforced above, component-wise linear interpolation still
    # doesn't preserve unit norm (x^2+y^2+z^2+w^2=1) between grid points —
    # same reason sg_filter renormalizes after its own filtering pass.
    for prefix in quat_prefixes:
        if all(f"{prefix}orient_{ax}" in out.columns for ax in ("x", "y", "z", "w")):
            qx = out[f"{prefix}orient_x"].values
            qy = out[f"{prefix}orient_y"].values
            qz = out[f"{prefix}orient_z"].values
            qw = out[f"{prefix}orient_w"].values
            norm = np.sqrt(qx**2 + qy**2 + qz**2 + qw**2)
            norm[norm == 0] = 1.0
            out[f"{prefix}orient_x"] = qx / norm
            out[f"{prefix}orient_y"] = qy / norm
            out[f"{prefix}orient_z"] = qz / norm
            out[f"{prefix}orient_w"] = qw / norm

    print(f"  [{label}] Resampled {len(df)} → {len(out)} rows onto global grid"
          f"{' (zero-order hold)' if method == 'zoh' else ''}")
    return out


def process_ground_truth():
    print("\n" + "="*55 + "\n  GROUND TRUTH\n" + "="*55)
    df = load_latest("ground_truth")
    df = make_timestamp(df)
    print_stats(df, "GroundTruth", "raw (may include /tf fallback rows)")

    # The logger writes a handful of /tf dead-reckoning rows before the real
    # pose-publisher topic takes over (child_frame_id == 'base_footprint' or
    # 'base_link'). Those are NOT ground truth — drop them so only genuine
    # physics-pose rows (child_frame_id == the robot's model name) remain.
    before = len(df)
    df = df[~df["child_frame_id"].isin(["base_footprint", "base_link"])].copy()
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped {dropped} /tf-fallback row(s) — dead-reckoning, not real GT")
    if len(df) == 0:
        raise ValueError(
            "All ground_truth rows were /tf-fallback — the real pose topic "
            "never fired this session. Check /ground_truth_pose before trusting "
            "any downstream error metric.")

    df = handle_nan(df, "GroundTruth")
    df = remove_duplicates(df, "GroundTruth")
    df = sort_by_time(df, "GroundTruth")
    # Deliberately NO outlier removal and NO Savitzky-Golay filtering here.
    # This is the reference truth signal that everything else gets scored
    # against — smoothing or clipping it would quietly bias every RMSE
    # number downstream. Only the noisy sensor streams get filtered.
    save_stage(df, "ground_truth", "stage1_cleaned")
    print_stats(df, "GroundTruth", "cleaned (pose-only, unfiltered)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# PER-SENSOR PROCESSORS
# ══════════════════════════════════════════════════════════════════════════════

def process_imu():
    print("\n" + "="*55 + "\n  IMU\n" + "="*55)
    df = load_latest("imu")
    df = make_timestamp(df)
    print_stats(df, "IMU", "raw")

    df = handle_nan(df, "IMU")
    df = remove_duplicates(df, "IMU")
    df = sort_by_time(df, "IMU")
    if USE_IQR:
        df = remove_outliers(df, "IMU")
    save_stage(df, "imu", "stage1_cleaned")

    df = sg_filter(df, "IMU")
    save_stage(df, "imu", "stage2_sgfiltered")
    print_stats(df, "IMU", "after S-G filter")
    return df


def process_odom():
    print("\n" + "="*55 + "\n  ODOMETRY\n" + "="*55)
    df = load_latest("odom")
    df = make_timestamp(df)
    print_stats(df, "Odom", "raw")

    df = handle_nan(df, "Odom")
    df = remove_duplicates(df, "Odom")
    df = sort_by_time(df, "Odom")
    if USE_IQR:
        df = remove_outliers(df, "Odom")
    save_stage(df, "odom", "stage1_cleaned")

    df = sg_filter(df, "Odom")
    save_stage(df, "odom", "stage2_sgfiltered")
    print_stats(df, "Odom", "after S-G filter")
    return df


def process_lidar():
    print("\n" + "="*55 + "\n  LiDAR\n" + "="*55)
    files = glob.glob(str(LOG_DIR / "scan_*.csv"))
    files = [f for f in files if "filtered" not in f and "stage" not in f]
    if not files:
        print("  ⚠️  No scan CSV found — skipping LiDAR")
        return None

    path = max(files, key=os.path.getctime)
    print(f"  Loading: {Path(path).name}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = make_timestamp(df)

    # Expand ranges string → numeric columns FIRST (required for S-G filter)
    df = expand_lidar_ranges(df)
    print_stats(df, "LiDAR", "after range expansion")

    df = handle_nan(df, "LiDAR")
    df = remove_duplicates(df, "LiDAR")
    df = sort_by_time(df, "LiDAR")
    if USE_IQR:
        df = remove_outliers(df, "LiDAR")
    save_stage(df, "scan", "stage1_cleaned")

    # S-G filter now works on range_0..range_N numeric columns
    df = sg_filter(df, "LiDAR")
    save_stage(df, "scan", "stage2_sgfiltered")
    print_stats(df, "LiDAR", "after S-G filter")
    return df


def process_cmd_vel():
    """
    Load commanded velocity — the paper's actual control input Ut=[v,ω],
    described as "received from the motion controller", not a sensor
    measurement. Used by ekf_fusion.py for the PREDICT step instead of
    odometry-derived velocity, so odometry isn't used twice (once disguised
    as control input, once as a genuine measurement).
    """
    print("\n" + "="*55 + "\n  CMD_VEL (control input)\n" + "="*55)
    files = glob.glob(str(LOG_DIR / "cmd_vel_*.csv"))
    files = [f for f in files if "filtered" not in f and "stage" not in f]
    if not files:
        print("  ⚠️  No cmd_vel CSV found — skipping. ekf_fusion.py will fall "
              "back to sensor-derived velocity for prediction.")
        return None

    path = max(files, key=os.path.getctime)
    print(f"  Loading: {Path(path).name}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = make_timestamp(df)
    print_stats(df, "CmdVel", "raw")

    df = handle_nan(df, "CmdVel")
    df = remove_duplicates(df, "CmdVel")
    df = sort_by_time(df, "CmdVel")
    # Deliberately NO S-G filtering and NO outlier removal — these are
    # commanded values from the controller/teleop, not noisy sensor readings.
    # Smoothing or clipping the actual control input would distort the very
    # thing the paper says the EKF predict step should use directly.
    save_stage(df, "cmd_vel", "stage1_cleaned")
    print_stats(df, "CmdVel", "cleaned (unfiltered, as-commanded)")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█"*55)
    print("  SENSOR DATA PREPROCESSING")
    print("█"*55)

    # Process each sensor at its native frequency
    imu_df   = process_imu()
    odom_df  = process_odom()
    lidar_df = process_lidar()
    gt_df    = process_ground_truth()
    cmd_df   = process_cmd_vel()

    # ── Build ONE global shared timeline ─────────────────────────────────────
    print("\n" + "="*55)
    print("  GLOBAL TIMELINE SYNCHRONIZATION")
    print("="*55)

    t_grid = build_global_timeline(imu_df, odom_df, lidar_df, gt_df, cmd_df)

    imu_sync   = resample_to_grid(imu_df,   t_grid, "IMU")
    odom_sync  = resample_to_grid(odom_df,  t_grid, "Odom")
    lidar_sync = resample_to_grid(lidar_df, t_grid, "LiDAR") if lidar_df is not None else None
    gt_sync    = resample_to_grid(gt_df,    t_grid, "GroundTruth")
    cmd_sync   = resample_to_grid(cmd_df,   t_grid, "CmdVel", method="zoh") if cmd_df is not None else None

    # Save individual filtered outputs
    imu_out  = LOG_DIR / "imu_filtered.csv"
    odom_out = LOG_DIR / "odom_filtered.csv"
    gt_out   = LOG_DIR / "ground_truth_filtered.csv"
    imu_sync.to_csv(imu_out, index=False)
    odom_sync.to_csv(odom_out, index=False)
    gt_sync.to_csv(gt_out, index=False)
    print(f"\n  Saved: imu_filtered.csv ({len(imu_sync)} rows)")
    print(f"  Saved: odom_filtered.csv ({len(odom_sync)} rows)")
    print(f"  Saved: ground_truth_filtered.csv ({len(gt_sync)} rows)")

    if lidar_sync is not None:
        scan_out = LOG_DIR / "scan_filtered.csv"
        lidar_sync.to_csv(scan_out, index=False)
        print(f"  Saved: scan_filtered.csv ({len(lidar_sync)} rows)")

    if cmd_sync is not None:
        cmd_out = LOG_DIR / "cmd_vel_filtered.csv"
        cmd_sync.to_csv(cmd_out, index=False)
        print(f"  Saved: cmd_vel_filtered.csv ({len(cmd_sync)} rows)")

    # ── Build fusion_ready.csv (all sensors + ground truth on same timeline) ──
    print("\n" + "="*55)
    print("  BUILDING fusion_ready.csv")
    print("="*55)

    # Prefix columns to avoid clashes
    imu_cols  = {c: f"imu_{c}"  for c in imu_sync.columns  if c != "timestamp"}
    odom_cols = {c: f"odom_{c}" for c in odom_sync.columns if c != "timestamp"}

    fusion = imu_sync.rename(columns=imu_cols)
    fusion = fusion.merge(odom_sync.rename(columns=odom_cols), on="timestamp")

    if lidar_sync is not None:
        # Only include LiDAR metadata cols (not all 540 range cols) in fusion_ready
        lidar_meta_cols = [c for c in lidar_sync.columns
                           if not c.startswith("range_")]
        lidar_meta = lidar_sync[lidar_meta_cols].copy()
        lidar_meta_renamed = {c: f"lidar_{c}" for c in lidar_meta_cols
                              if c != "timestamp"}
        lidar_meta = lidar_meta.rename(columns=lidar_meta_renamed)
        fusion = fusion.merge(lidar_meta, on="timestamp")

    # Ground truth columns (gt_x, gt_y, gt_z, gt_orient_*) are already
    # prefixed "gt_" in the raw CSV itself — don't double-prefix them.
    fusion = fusion.merge(gt_sync, on="timestamp")

    if cmd_sync is not None:
        cmd_cols = {c: f"cmd_{c}" for c in cmd_sync.columns if c != "timestamp"}
        fusion = fusion.merge(cmd_sync.rename(columns=cmd_cols), on="timestamp")

    fusion_out = LOG_DIR / "fusion_ready.csv"
    fusion.to_csv(fusion_out, index=False)
    print(f"\n  ✅ fusion_ready.csv saved — {len(fusion)} rows × {len(fusion.columns)} cols")
    print(f"     All sensors + REAL ground truth aligned on one {TARGET_HZ}Hz timeline")
    print(f"     Ready for EKF fusion stage")

    print("\n" + "█"*55)
    print("  ✅ PREPROCESSING COMPLETE")
    print("  Next: python3 ~/ws_mobile/scripts/ekf_fusion.py")
    print("█"*55 + "\n")


if __name__ == "__main__":
    main()