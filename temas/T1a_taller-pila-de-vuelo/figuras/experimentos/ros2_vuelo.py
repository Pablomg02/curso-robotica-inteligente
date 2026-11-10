"""PX4 SITL desde ROS 2 (rclpy + px4_msgs + uXRCE-DDS): frecuencias, edad de los mensajes, ACK de un comando y vuelo offboard.
Salida: out/ros2_vuelo.json y out/ros2_traza.csv"""
import time, json, csv, statistics, sys
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import (OffboardControlMode, TrajectorySetpoint, VehicleCommand, VehicleCommandAck,
                          VehicleLocalPosition, VehicleStatus, VehicleOdometry, SensorCombined, TimesyncStatus)
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                 history=HistoryPolicy.KEEP_LAST, depth=1)
OUT = "/work/exp/out"

class Exp(Node):
    def __init__(s):
        super().__init__("taller_exp")
        s.cnt = {}; s.age = {"odom": [], "imu": []}; s.lp = None; s.st = None; s.off = None; s.acks = []; s.traza = []
        s.create_subscription(SensorCombined, "/fmu/out/sensor_combined", s.cb_imu, QOS)
        s.create_subscription(VehicleOdometry, "/fmu/out/vehicle_odometry", s.cb_odom, QOS)
        s.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", s.cb_lp, QOS)
        s.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1", s.cb_st, QOS)
        s.create_subscription(TimesyncStatus, "/fmu/out/timesync_status", s.cb_ts, QOS)
        s.create_subscription(VehicleCommandAck, "/fmu/out/vehicle_command_ack", s.cb_ack, QOS)
        s.p_ocm = s.create_publisher(OffboardControlMode, "/fmu/in/offboard_control_mode", QOS)
        s.p_sp = s.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", QOS)
        s.p_cmd = s.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", QOS)
    def now_us(s): return s.get_clock().now().nanoseconds // 1000
    def tick(s, k): s.cnt[k] = s.cnt.get(k, 0) + 1
    def cb_imu(s, m): s.tick("sensor_combined"); s.age["imu"].append(s.now_us() - m.timestamp)
    def cb_odom(s, m): s.tick("vehicle_odometry"); s.age["odom"].append(s.now_us() - m.timestamp_sample)
    def cb_lp(s, m):
        s.tick("vehicle_local_position"); s.lp = m
        s.traza.append((time.monotonic(), m.x, m.y, m.z, s.sp_actual[0], s.sp_actual[1], s.sp_actual[2]))
    def cb_st(s, m): s.tick("vehicle_status"); s.st = m
    def cb_ts(s, m): s.tick("timesync_status"); s.off = m
    def cb_ack(s, m): s.acks.append((time.perf_counter(), m.command, m.result))
    sp_actual = (float("nan"),) * 3
    def offboard(s, x, y, z):
        o = OffboardControlMode(); o.position = True; o.timestamp = s.now_us(); s.p_ocm.publish(o)
        t = TrajectorySetpoint(); t.position = [x, y, z]; t.velocity = [float("nan")] * 3; t.acceleration = [float("nan")] * 3
        t.yaw = 0.0; t.timestamp = s.now_us(); s.p_sp.publish(t); s.sp_actual = (x, y, z)
    def cmd(s, c, p1=0.0, p2=0.0):
        v = VehicleCommand(); v.command = c; v.param1 = p1; v.param2 = p2; v.target_system = 1; v.target_component = 1
        v.source_system = 1; v.source_component = 1; v.from_external = True; v.timestamp = s.now_us(); s.p_cmd.publish(v)

def spin_for(n, sec, f=None, hz=20.0):
    t_end = time.monotonic() + sec; k = 0; t0 = time.monotonic()
    while time.monotonic() < t_end:
        if f: f()
        k += 1
        while time.monotonic() < t0 + k / hz:
            rclpy.spin_once(n, timeout_sec=0.002)

rclpy.init(); n = Exp(); res = {}
spin_for(n, 10.0)
res["tasas_hz"] = {k: round(v / 10.0, 1) for k, v in n.cnt.items()}
q = lambda a: {"mediana": round(statistics.median(a) / 1e3, 3), "p95": round(statistics.quantiles(a, n=20)[-1] / 1e3, 3)}
res["edad_ms"] = {"sensor_combined (timestamp)": q(n.age["imu"]), "vehicle_odometry (timestamp_sample)": q(n.age["odom"])}
if n.off: res["timesync"] = {"offset_estimado_us": int(n.off.estimated_offset), "rtt_us": int(n.off.round_trip_time)}
# ACK de un comando que atiende commander: cambiar a modo Position (DO_SET_MODE 1/3) 20 veces
lat = []
for i in range(20):
    a = time.perf_counter(); na = len(n.acks); n.cmd(176, 1.0, 3.0)
    while len(n.acks) == na and time.perf_counter() - a < 1.0: rclpy.spin_once(n, timeout_sec=0.0005)
    if len(n.acks) > na: lat.append((n.acks[-1][0] - a) * 1e3)
res["ack_ms"] = {"n": len(lat), "mediana": round(statistics.median(lat), 3), "max": round(max(lat), 3)} if lat else None
# Vuelo offboard: 1 s de consignas, modo offboard (DO_SET_MODE 1/6), armar, subir a 5 m, escalón de 5 m
spin_for(n, 1.0, lambda: n.offboard(0.0, 0.0, -5.0))
t_mode = time.monotonic(); n.cmd(176, 1.0, 6.0); n.cmd(400, 1.0)
spin_for(n, 15.0, lambda: n.offboard(0.0, 0.0, -5.0))
res["tras_subida"] = {"z": round(n.lp.z, 3), "nav_state": int(n.st.nav_state), "arming_state": int(n.st.arming_state)}
t_step = time.monotonic()
spin_for(n, 10.0, lambda: n.offboard(5.0, 0.0, -5.0))
res["tras_escalon"] = {"x": round(n.lp.x, 3), "z": round(n.lp.z, 3)}
n.cmd(21)  # aterrizar (VEHICLE_CMD_NAV_LAND)
spin_for(n, 12.0)
res["t_modo"] = t_mode; res["t_escalon"] = t_step
with open(f"{OUT}/ros2_traza.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["t_mono", "x", "y", "z", "sp_x", "sp_y", "sp_z"]); w.writerows(n.traza)
json.dump(res, open(f"{OUT}/ros2_vuelo.json", "w"), indent=1); print(json.dumps(res, indent=1))
