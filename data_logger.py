#!/usr/bin/env python3
"""
data_logger.py — paper-aligned sensor + ground-truth logger, full session logging.

Matches the paper's "Data acquisition" 3-stage description exactly:
  1. Ground truth estimation  — Gazebo's built-in localization module (/tf, with
                                 a model-pose fallback for setups without a GT /tf)
  2. Sensor data logging      — raw LiDAR/IMU/odometry, each at its own native
                                 rate — no resampling, filtering, or clipping here
                                 (that belongs to preprocessing_data.py's S-G stage)
  3. Controlled motion execution — driven externally (teleop/nav); this node only
                                 listens, it never commands the robot

Table 1 (paper) sensor targets — LiDAR 0-270° / 0.5° / 10-250cm, IMU 50Hz,
Odometry 100Hz — are Gazebo/URDF/SDF sensor-plugin settings, NOT something this
script can enforce. This script verifies, at capture time, that whatever you
actually wired up matches those targets, and warns immediately if not.

  4 changes + 2 bug fixes (original):
  1. Ground truth logging (base_link + base_footprint supported)
  2. Full odometry quaternion (x, y, z, w)
  3. Floating-point timestamp
  4. LiDAR extra parameters
  Fix 1: pose_callback indentation correct
  Fix 2: gt_last_source = "pose" inside pose_callback (not "tf")

  Paper Table 1 alignment (previous update):
  5. PAPER_SENSOR_SPECS documents the paper's target sensor configuration
  6. One-time LiDAR geometry check vs Table 1 (angle span / resolution / range)
  7. Periodic IMU/Odometry rate check vs Table 1 (50 Hz / 100 Hz, ±20% tolerance)

  Full session logging (this update):
  8.  Frequency logging  — per-topic Hz persisted to frequency_log_*.csv every
                            5s for all 5 topics (not just printed to console)
  9.  Covariance logging — IMU orientation/angular_velocity/linear_acceleration
                            covariance + Odometry pose/twist covariance, each
                            stored as a ';'-joined string column (same pattern
                            as the existing LiDAR ranges column). Only added to
                            IMU/Odometry — LaserScan and TF carry no covariance
                            field in ROS, so there's nothing to log there.
  10. Frame IDs           — msg.header.frame_id added to imu/odom/scan; both
                            frame_id + child_frame_id added to ground_truth
  11. /cmd_vel logging    — new cmd_vel_*.csv. This is the paper's control
                            input Ut = [vt, ωt], "received from the motion
                            controller" — note this is a DIFFERENT signal from
                            odom/IMU's sensed velocity, which is what
                            ekf_fusion.py currently uses as a stand-in for Ut
  12. Metadata file       — metadata_*.json: session info, paper targets,
                            detected LiDAR geometry, file manifest; rewritten
                            at shutdown with final counts/rates/statistics
  13. Sample statistics   — single-pass (Welford) mean/std/min/max for key
                            signals on every topic, saved into the metadata
                            file and printed at shutdown

  CAVEAT: preprocessing_data.py's resample_to_grid() currently skips any
  column with dtype == object (see its `if df[col].dtype == object: continue`).
  frame_id and the covariance strings are object-dtype, so they'll be captured
  in the raw CSVs here but will NOT automatically survive into *_filtered.csv
  or fusion_ready.csv unless preprocessing_data.py is also updated to parse
  and carry them forward.
"""

import json
import math
import signal
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Imu, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped, Twist, TwistStamped
from tf2_msgs.msg import TFMessage
import csv
import os
from datetime import datetime

# ── Paper Table 1 — target sensor configuration ───────────────────────────────
# Reference only. These live in the Gazebo/URDF/SDF sensor plugins, not here.
PAPER_SENSOR_SPECS = {
    'lidar': {
        'angle_span_deg': 270.0,
        'resolution_deg': 0.5,
        'range_min_m':    0.10,
        'range_max_m':    2.50,
    },
    'imu':      {'output_hz': 50.0},
    'odometry': {'output_hz': 100.0},
}
RATE_TOLERANCE        = 0.20   # ±20% before flagging a mismatch
RATE_CHECK_PERIOD_SEC = 5.0
LIDAR_ANGLE_TOL_DEG   = 5.0
LIDAR_RES_TOL_DEG     = 0.05
LIDAR_RANGE_TOL_M     = 0.05


class RunningStat:
    """Single-pass mean/std/min/max via Welford's algorithm — O(1) memory,
    so session length never affects logger memory use."""
    __slots__ = ('count', 'mean', 'm2', 'min', 'max')

    def __init__(self):
        self.count = 0
        self.mean  = 0.0
        self.m2    = 0.0
        self.min   = None
        self.max   = None

    def update(self, x):
        x = float(x)
        if not math.isfinite(x):
            return
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2   += delta * (x - self.mean)
        self.min = x if self.min is None else min(self.min, x)
        self.max = x if self.max is None else max(self.max, x)

    def summary(self):
        if self.count == 0:
            return {'count': 0}
        variance = self.m2 / self.count
        return {
            'count': self.count,
            'mean':  round(self.mean, 6),
            'std':   round(math.sqrt(max(variance, 0.0)), 6),
            'min':   round(self.min, 6),
            'max':   round(self.max, 6),
        }


class DataLogger(Node):
    def __init__(self):
        super().__init__('data_logger')

        # ── Fix #6/#7: frame names and robot name were hardcoded guesses.
        # Now they're parameters you can verify against your live sim and
        # override with --ros-args -p robot_name:=... instead of editing code.
        self.declare_parameter('robot_name', 'differential_drive_robot')  # confirmed via `gz topic -l`
        self.declare_parameter('gt_child_frame_ids', ['base_footprint', 'base_link'])
        self.robot_name = self.get_parameter('robot_name').value
        self.gt_child_frame_ids = tuple(self.get_parameter('gt_child_frame_ids').value)

        # This node is run standalone (`python3 data_logger.py`), NOT through
        # the launch file that sets use_sim_time:=true for its own nodes — so
        # it never inherits that parameter automatically. Without it,
        # self.get_clock().now() (used to stamp plain Twist/cmd_vel messages,
        # which have no header of their own) returns REAL wall-clock epoch
        # time, while every other sensor's timestamp comes from Gazebo's
        # sim-time header. That clock mismatch silently produces an
        # inverted/empty time range in preprocess_data.py — every downstream
        # file becomes 0 rows with no crash, no error, nothing but a
        # confusing "0 steps" line. Check for it loudly, right now, instead.
        #
        # FIX: whether use_sim_time is already declared depends on rclpy
        # version / how this node was launched — calling get_parameter() on
        # an undeclared parameter throws and would crash startup. Check
        # safely instead of assuming either way.
        try:
            use_sim_time = self.get_parameter('use_sim_time').value
        except Exception:
            self.declare_parameter('use_sim_time', False)
            use_sim_time = self.get_parameter('use_sim_time').value

        if not use_sim_time:
            self.get_logger().error(
                '🛑 use_sim_time is FALSE. cmd_vel timestamps will use REAL '
                'wall-clock time while every other sensor uses simulation '
                'time — this silently breaks the entire preprocessing '
                'pipeline later (0-row files, no error). Restart this '
                'script with: --ros-args -p use_sim_time:=true')

        self.log_dir = os.path.expanduser('~/ws_mobile/sensor_logs')
        os.makedirs(self.log_dir, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self._session_id = ts

        # Open CSV files
        self.imu_file     = open(os.path.join(self.log_dir, f'imu_{ts}.csv'),           'w', newline='')
        self.odom_file    = open(os.path.join(self.log_dir, f'odom_{ts}.csv'),          'w', newline='')
        self.scan_file    = open(os.path.join(self.log_dir, f'scan_{ts}.csv'),          'w', newline='')
        self.gt_file      = open(os.path.join(self.log_dir, f'ground_truth_{ts}.csv'), 'w', newline='')
        self.cmd_vel_file = open(os.path.join(self.log_dir, f'cmd_vel_{ts}.csv'),      'w', newline='')
        self.freq_file    = open(os.path.join(self.log_dir, f'frequency_log_{ts}.csv'),'w', newline='')

        self.imu_writer     = csv.writer(self.imu_file)
        self.odom_writer    = csv.writer(self.odom_file)
        self.scan_writer    = csv.writer(self.scan_file)
        self.gt_writer      = csv.writer(self.gt_file)
        self.cmd_vel_writer = csv.writer(self.cmd_vel_file)
        self.freq_writer    = csv.writer(self.freq_file)

        # Headers
        self.imu_writer.writerow([
            'time_sec', 'time_nsec', 'timestamp', 'frame_id',
            'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
            'linear_acc_x', 'linear_acc_y', 'linear_acc_z',
            'orient_x', 'orient_y', 'orient_z', 'orient_w',
            'orientation_covariance',          # 9 values, ';'-joined
            'angular_velocity_covariance',     # 9 values, ';'-joined
            'linear_acceleration_covariance',  # 9 values, ';'-joined
        ])
        self.odom_writer.writerow([
            'time_sec', 'time_nsec', 'timestamp', 'frame_id',
            'pos_x', 'pos_y', 'pos_z',
            'orient_x', 'orient_y', 'orient_z', 'orient_w',
            'linear_x', 'angular_z',
            'pose_covariance',   # 36 values, ';'-joined
            'twist_covariance',  # 36 values, ';'-joined
        ])
        self.scan_writer.writerow([
            'time_sec', 'time_nsec', 'timestamp', 'frame_id',
            'angle_min', 'angle_max', 'angle_increment',
            'time_increment', 'scan_time',
            'range_min', 'range_max',
            'ranges'
        ])
        self.gt_writer.writerow([
            'time_sec', 'time_nsec', 'timestamp', 'frame_id', 'child_frame_id',
            'gt_x', 'gt_y', 'gt_z',
            'gt_orient_x', 'gt_orient_y', 'gt_orient_z', 'gt_orient_w'
        ])
        self.cmd_vel_writer.writerow([
            'time_sec', 'time_nsec', 'timestamp', 'timestamp_source',
            'linear_x', 'linear_y', 'linear_z',
            'angular_x', 'angular_y', 'angular_z',
        ])
        self.freq_writer.writerow([
            'elapsed_sec', 'imu_hz', 'odom_hz', 'scan_hz', 'cmd_vel_hz', 'gt_hz'
        ])

        # Ground truth state
        self.gt_source_found = False
        self.gt_last_source  = None   # "tf" or "pose" — prevents duplicates

        # Per-topic counters: window (reset every RATE_CHECK_PERIOD_SEC) + total
        topics = ('imu', 'odom', 'scan', 'cmd_vel', 'gt')
        self._window_counts = {k: 0 for k in topics}
        self._total_counts  = {k: 0 for k in topics}
        self._lidar_checked           = False
        self._lidar_geometry_detected = None

        # Sample statistics — O(1) memory regardless of session length
        self.stats = {
            'imu_angular_vel_z': RunningStat(),
            'imu_linear_acc_x':  RunningStat(),
            'odom_linear_x':     RunningStat(),
            'odom_pos_x':        RunningStat(),
            'odom_pos_y':        RunningStat(),
            'cmd_vel_linear_x':  RunningStat(),
            'cmd_vel_angular_z': RunningStat(),
            'lidar_mean_range':  RunningStat(),
            'gt_pos_x':          RunningStat(),
            'gt_pos_y':          RunningStat(),
        }

        self._start_time         = self.get_clock().now()
        self._session_start_wall = datetime.now().isoformat()

        self._rate_timer = self.create_timer(
            RATE_CHECK_PERIOD_SEC, self.log_frequencies)

        # Subscribers
        self.imu_sub     = self.create_subscription(Imu,       '/imu',     self.imu_callback,     10)
        self.odom_sub    = self.create_subscription(Odometry,  '/odom',    self.odom_callback,    10)
        self.scan_sub    = self.create_subscription(LaserScan, '/scan',    self.scan_callback,    10)
        self.tf_sub      = self.create_subscription(TFMessage, '/tf',      self.tf_callback,      10)

        # Fix #3: plain Twist has no header/stamp, so a receipt-time stamp was
        # being written and silently mislabeled as the command time. A topic
        # only has ONE real message type — you can't subscribe to both types
        # on the same name and have it work. So we check what's actually
        # published FIRST, then subscribe with the matching type.
        cmd_vel_type = self._detect_cmd_vel_type()
        if cmd_vel_type == 'stamped':
            self.cmd_vel_sub = self.create_subscription(
                TwistStamped, '/cmd_vel', self.cmd_vel_stamped_callback, 10)
            self.get_logger().info(
                '/cmd_vel is TwistStamped — logging real command timestamps.')
        else:
            self.cmd_vel_sub = self.create_subscription(
                Twist, '/cmd_vel', self.cmd_vel_callback, 10)
            self.get_logger().warn(
                '/cmd_vel is plain Twist (no header) — timestamps for this '
                'topic will be logger-receipt time, NOT true command time. '
                'Marked as "receipt_approx" in the CSV. If you need exact '
                'command timing, republish /cmd_vel as TwistStamped from '
                'your controller.')

        # Fix: this used to be f'/model/{robot_name}/pose' — that's the GZ-side
        # topic name, not the ROS name your bridge_parameters.yaml maps it to.
        # Your bridge entry uses ros_topic_name: "ground_truth_pose", so THAT's
        # the actual ROS topic. Subscribing to the gz-style name meant this
        # never received anything, ever, silently.
        pose_topic = '/ground_truth_pose'
        try:
            self.pose_sub = self.create_subscription(
                PoseStamped, pose_topic, self.pose_callback, 10)
        except Exception:
            self.pose_sub = None
        self._pose_topic = pose_topic

        # Fix #7: don't assume topics exist — check them against what's
        # actually being published and warn loudly, not silently, if any
        # expected topic is missing at startup.
        self._verify_expected_topics()

        self.metadata_path = os.path.join(self.log_dir, f'metadata_{ts}.json')
        self._write_metadata(final=False)

        self.get_logger().info(f'Data logger started. Saving to: {self.log_dir}')
        self.get_logger().info(
            'Paper Table 1 targets — LiDAR: 0-270°/0.5°/10-250cm, '
            'IMU: 50Hz, Odometry: 100Hz. Verifying against live topics...')

    # ── Fix #7: loud topic check instead of silent empty-column failure ──────
    def _verify_expected_topics(self):
        published = dict(self.get_topic_names_and_types())
        expected = ['/imu', '/odom', '/scan', '/tf', '/cmd_vel', self._pose_topic]
        missing = [t for t in expected if t not in published]
        if missing:
            self.get_logger().warn(
                'These expected topics are NOT currently published: '
                + ', '.join(missing) +
                '. Any of these being missing means that column silently '
                'stays empty — it will NOT throw an error. Check your '
                'ros_gz_bridge config / robot_name param before trusting '
                'this session\'s data.')
        else:
            self.get_logger().info('All expected topics present ✔ ' + ', '.join(expected))

        # One-shot check a few seconds in: did ground truth actually arrive?
        self._gt_check_timer = self.create_timer(8.0, self._check_gt_arrived_once)

    def _check_gt_arrived_once(self):
        self._gt_check_timer.cancel()  # create_timer repeats by default — this makes it one-shot
        if not self.gt_source_found:
            self.get_logger().warn(
                f'No ground truth received yet from /tf (child frame in '
                f'{self.gt_child_frame_ids}) or {self._pose_topic}. '
                f'If this stays empty for the whole session, your GT column '
                f'is dead and every downstream error metric is meaningless. '
                f'Fix gt_child_frame_ids / robot_name params or your TF tree.')

    # ── Timestamp helper ──────────────────────────────────────────────────────
    def stamp_to_float(self, stamp):
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _detect_cmd_vel_type(self):
        """Look up /cmd_vel's actual published type before subscribing.
        Falls back to plain Twist assumption if the topic isn't up yet
        (e.g. controller node hasn't started) — in that case you'll get
        the receipt-time warning at runtime instead of a wrong guess here."""
        for name, types in self.get_topic_names_and_types():
            if name == '/cmd_vel':
                if 'geometry_msgs/msg/TwistStamped' in types:
                    return 'stamped'
                if 'geometry_msgs/msg/Twist' in types:
                    return 'plain'
        self.get_logger().warn(
            '/cmd_vel not visible yet at startup — assuming plain Twist. '
            'If your controller actually publishes TwistStamped, restart '
            'this logger after the controller is up.')
        return 'plain'

    @staticmethod
    def _cov_str(cov):
        return ';'.join(str(c) for c in cov)

    @staticmethod
    def _finite_mean(values):
        finite = [v for v in values if math.isfinite(v)]
        return (sum(finite) / len(finite)) if finite else None

    # ── IMU ───────────────────────────────────────────────────────────────────
    def imu_callback(self, msg):
        self._window_counts['imu'] += 1
        self._total_counts['imu']  += 1
        self.stats['imu_angular_vel_z'].update(msg.angular_velocity.z)
        self.stats['imu_linear_acc_x'].update(msg.linear_acceleration.x)

        t = msg.header.stamp
        self.imu_writer.writerow([
            t.sec, t.nanosec, self.stamp_to_float(t), msg.header.frame_id,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            self._cov_str(msg.orientation_covariance),
            self._cov_str(msg.angular_velocity_covariance),
            self._cov_str(msg.linear_acceleration_covariance),
        ])
        self.imu_file.flush()

    # ── Odometry ──────────────────────────────────────────────────────────────
    def odom_callback(self, msg):
        self._window_counts['odom'] += 1
        self._total_counts['odom']  += 1
        p = msg.pose.pose
        self.stats['odom_linear_x'].update(msg.twist.twist.linear.x)
        self.stats['odom_pos_x'].update(p.position.x)
        self.stats['odom_pos_y'].update(p.position.y)

        t = msg.header.stamp
        self.odom_writer.writerow([
            t.sec, t.nanosec, self.stamp_to_float(t), msg.header.frame_id,
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w,
            msg.twist.twist.linear.x, msg.twist.twist.angular.z,
            self._cov_str(msg.pose.covariance),
            self._cov_str(msg.twist.covariance),
        ])
        self.odom_file.flush()

    # ── LiDAR ─────────────────────────────────────────────────────────────────
    def scan_callback(self, msg):
        self._window_counts['scan'] += 1
        self._total_counts['scan']  += 1

        if not self._lidar_checked:
            self.check_lidar_geometry(msg)
            self._lidar_checked = True

        mean_range = self._finite_mean(msg.ranges)
        if mean_range is not None:
            self.stats['lidar_mean_range'].update(mean_range)

        t = msg.header.stamp
        ranges_str = ';'.join([str(r) for r in msg.ranges])
        self.scan_writer.writerow([
            t.sec, t.nanosec, self.stamp_to_float(t), msg.header.frame_id,
            msg.angle_min, msg.angle_max, msg.angle_increment,
            msg.time_increment, msg.scan_time,
            msg.range_min, msg.range_max,
            ranges_str
        ])
        self.scan_file.flush()

    # ── Control input — paper's Ut = [vt, ωt], "received from the motion
    #    controller". Distinct from odom/IMU's SENSED velocity. ────────────────
    def cmd_vel_callback(self, msg):
        self._window_counts['cmd_vel'] += 1
        self._total_counts['cmd_vel']  += 1
        self.stats['cmd_vel_linear_x'].update(msg.linear.x)
        self.stats['cmd_vel_angular_z'].update(msg.angular.z)

        t = self.get_clock().now().to_msg()   # Twist has no header/stamp of its own
        self.cmd_vel_writer.writerow([
            t.sec, t.nanosec, self.stamp_to_float(t), 'receipt_approx',
            msg.linear.x, msg.linear.y, msg.linear.z,
            msg.angular.x, msg.angular.y, msg.angular.z,
        ])
        self.cmd_vel_file.flush()

    def cmd_vel_stamped_callback(self, msg):
        """Same as cmd_vel_callback but for TwistStamped — real command
        timestamp from msg.header.stamp, not logger receipt time."""
        self._window_counts['cmd_vel'] += 1
        self._total_counts['cmd_vel']  += 1
        self.stats['cmd_vel_linear_x'].update(msg.twist.linear.x)
        self.stats['cmd_vel_angular_z'].update(msg.twist.angular.z)

        t = msg.header.stamp
        self.cmd_vel_writer.writerow([
            t.sec, t.nanosec, self.stamp_to_float(t), 'msg_stamp',
            msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z,
            msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z,
        ])
        self.cmd_vel_file.flush()

    # ── Ground truth from /tf — FALLBACK ONLY, this is dead-reckoning, ────────
    # NOT real ground truth. It only writes if the real pose topic has never
    # fired. Once /ground_truth_pose starts producing data, this stops writing
    # entirely and logs a one-time notice instead of silently continuing.
    def tf_callback(self, msg):
        if self.gt_last_source == 'pose':
            if not getattr(self, '_tf_demoted_logged', False):
                self.get_logger().info(
                    'Real ground truth (/ground_truth_pose) is active — '
                    '/tf is now ignored for ground truth (it is dead-reckoning, '
                    'not true pose).')
                self._tf_demoted_logged = True
            return
        for transform in msg.transforms:
            if (transform.header.frame_id == 'odom' and
                    transform.child_frame_id in self.gt_child_frame_ids):
                t  = transform.header.stamp
                tr = transform.transform.translation
                ro = transform.transform.rotation
                self._window_counts['gt'] += 1
                self._total_counts['gt']  += 1
                self.stats['gt_pos_x'].update(tr.x)
                self.stats['gt_pos_y'].update(tr.y)
                self.gt_writer.writerow([
                    t.sec, t.nanosec, self.stamp_to_float(t),
                    transform.header.frame_id, transform.child_frame_id,
                    tr.x, tr.y, tr.z,
                    ro.x, ro.y, ro.z, ro.w
                ])
                self.gt_file.flush()
                if not self.gt_source_found:
                    self.get_logger().warn(
                        f'Ground truth currently from /tf (odom→{transform.child_frame_id}) '
                        f'— this is DEAD-RECKONING, not real ground truth. Waiting for '
                        f'/ground_truth_pose to take over as authoritative source.')
                    self.gt_source_found = True
                    self.gt_last_source  = 'tf'
                break

    # ── Ground truth from Gazebo pose topic — AUTHORITATIVE, real physics ────
    # truth. This ALWAYS writes and ALWAYS overrides /tf, since /tf is only
    # ever dead-reckoning odometry wearing a different name.
    def pose_callback(self, msg):
        t = msg.header.stamp
        p = msg.pose
        self._window_counts['gt'] += 1
        self._total_counts['gt']  += 1
        self.stats['gt_pos_x'].update(p.position.x)
        self.stats['gt_pos_y'].update(p.position.y)
        self.gt_writer.writerow([
            t.sec, t.nanosec, self.stamp_to_float(t),
            msg.header.frame_id, self.robot_name,
            p.position.x, p.position.y, p.position.z,
            p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w
        ])
        self.gt_file.flush()
        if self.gt_last_source != 'pose':
            self.get_logger().info(
                f'Ground truth now authoritative from {self._pose_topic} '
                f'(real physics pose) — this is what error metrics should use.')
            self.gt_source_found = True
            self.gt_last_source  = 'pose'

    # ── Paper Table 1 verification (does not alter logged data) ──────────────
    def check_lidar_geometry(self, msg):
        spec     = PAPER_SENSOR_SPECS['lidar']
        span_deg = math.degrees(msg.angle_max - msg.angle_min)
        res_deg  = math.degrees(msg.angle_increment)

        self._lidar_geometry_detected = {
            'angle_span_deg': round(span_deg, 3),
            'resolution_deg': round(res_deg, 4),
            'range_min_m':    msg.range_min,
            'range_max_m':    msg.range_max,
        }

        issues = []
        if abs(span_deg - spec['angle_span_deg']) > LIDAR_ANGLE_TOL_DEG:
            issues.append(f"angle span {span_deg:.1f}° (paper: {spec['angle_span_deg']:.1f}°)")
        if abs(res_deg - spec['resolution_deg']) > LIDAR_RES_TOL_DEG:
            issues.append(f"resolution {res_deg:.3f}° (paper: {spec['resolution_deg']:.1f}°)")
        if abs(msg.range_min - spec['range_min_m']) > LIDAR_RANGE_TOL_M:
            issues.append(f"range_min {msg.range_min:.3f}m (paper: {spec['range_min_m']:.2f}m)")
        if abs(msg.range_max - spec['range_max_m']) > LIDAR_RANGE_TOL_M:
            issues.append(f"range_max {msg.range_max:.3f}m (paper: {spec['range_max_m']:.2f}m)")

        if issues:
            self.get_logger().warn(
                'LiDAR config differs from paper Table 1 — ' + '; '.join(issues) +
                '. Fix this in the LiDAR SDF/URDF, not in this script.')
        else:
            self.get_logger().info(
                'LiDAR config matches paper Table 1 (0-270°, 0.5°, 10-250cm) ✔')

    def log_frequencies(self):
        """Persist per-topic Hz to frequency_log_*.csv every RATE_CHECK_PERIOD_SEC,
        and compare IMU/Odometry — the only two with paper-stated targets."""
        elapsed_total = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
        hz = {k: v / RATE_CHECK_PERIOD_SEC for k, v in self._window_counts.items()}
        for k in self._window_counts:
            self._window_counts[k] = 0

        self.freq_writer.writerow([
            round(elapsed_total, 2),
            round(hz['imu'], 2), round(hz['odom'], 2), round(hz['scan'], 2),
            round(hz['cmd_vel'], 2), round(hz['gt'], 2),
        ])
        self.freq_file.flush()

        for name, key, target in [
            ('IMU',      'imu',  PAPER_SENSOR_SPECS['imu']['output_hz']),
            ('Odometry', 'odom', PAPER_SENSOR_SPECS['odometry']['output_hz']),
        ]:
            h = hz[key]
            if h == 0.0:
                continue
            deviation = abs(h - target) / target
            if deviation > RATE_TOLERANCE:
                self.get_logger().warn(
                    f'{name} rate {h:.1f} Hz differs from paper target '
                    f'{target:.0f} Hz by {deviation * 100:.0f}% '
                    f'— fix this in the sensor plugin config, not in this script.')
            else:
                self.get_logger().info(
                    f'{name} rate {h:.1f} Hz ✔ (paper target: {target:.0f} Hz)')

    # ── Metadata file ──────────────────────────────────────────────────────────
    def _write_metadata(self, final: bool):
        meta = {
            'session_id':             self._session_id,
            'started_at':             self._session_start_wall,
            'log_dir':                self.log_dir,
            'ground_truth_source':    self.gt_last_source,
            'paper_table1_targets':   PAPER_SENSOR_SPECS,
            'lidar_geometry_detected': self._lidar_geometry_detected,
            'files': {
                'imu':           os.path.basename(self.imu_file.name),
                'odom':          os.path.basename(self.odom_file.name),
                'scan':          os.path.basename(self.scan_file.name),
                'ground_truth':  os.path.basename(self.gt_file.name),
                'cmd_vel':       os.path.basename(self.cmd_vel_file.name),
                'frequency_log': os.path.basename(self.freq_file.name),
            },
        }
        if final:
            elapsed = (self.get_clock().now() - self._start_time).nanoseconds / 1e9
            meta['session_duration_sec'] = round(elapsed, 3)
            meta['message_counts']       = dict(self._total_counts)
            meta['average_rates_hz'] = {
                k: (round(v / elapsed, 3) if elapsed > 0 else None)
                for k, v in self._total_counts.items()
            }
            meta['sample_statistics'] = {
                k: v.summary() for k, v in self.stats.items()
            }
        with open(self.metadata_path, 'w') as f:
            json.dump(meta, f, indent=2)

    # ── Cleanup ───────────────────────────────────────────────────────────────
    def destroy_node(self):
        self._rate_timer.cancel()
        self._write_metadata(final=True)

        self.get_logger().info('── Session sample statistics ──')
        for name, stat in self.stats.items():
            s = stat.summary()
            if s['count'] > 0:
                self.get_logger().info(
                    f'  {name:<20} n={s["count"]:<6} mean={s["mean"]:.4f} '
                    f'std={s["std"]:.4f} min={s["min"]:.4f} max={s["max"]:.4f}')

        self.imu_file.close()
        self.odom_file.close()
        self.scan_file.close()
        self.gt_file.close()
        self.cmd_vel_file.close()
        self.freq_file.close()
        self.get_logger().info(
            f'Data logger stopped. All files + metadata saved to: {self.log_dir}')
        super().destroy_node()


def main(args=None):
    # FIX: rclpy's default SIGINT handler tears down the ROS context
    # immediately on Ctrl+C — before this node's own shutdown summary (all
    # those per-topic stat lines) finishes logging. Every get_logger().info()
    # call after that point fails with "publisher's context is invalid"
    # spam. Disable rclpy's automatic handler and manage SIGINT ourselves so
    # cleanup logging happens while the context is still fully valid.
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = DataLogger()

    def _handle_sigint(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _handle_sigint)

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        # Ctrl+C can trigger rclpy's own signal handler to shut the context
        # down before this finally block runs, causing a spurious
        # "rcl_shutdown already called" crash on exit. Guard it.
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()