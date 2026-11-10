"""Qué manda PX4 SITL por cada puerto MAVLink y cuánto tarda en contestar un parámetro (pymavlink).
Uso: python3 mavlink_flujos.py <puerto> <segundos> <salida.json>"""
import sys, time, json, collections, statistics
from pymavlink import mavutil
port, dur, out = int(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
m = mavutil.mavlink_connection(f"udpin:0.0.0.0:{port}", source_system=255, source_component=190)
m.wait_heartbeat(timeout=30)
cnt = collections.Counter(); nbytes = 0
t0 = time.time(); last_hb = 0
while time.time() - t0 < dur:
    if time.time() - last_hb > 1:   # sin latido nuestro PX4 nos daría por perdidos
        m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS, mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0); last_hb = time.time()
    msg = m.recv_match(blocking=True, timeout=0.5)
    if msg is None or msg.get_type() == "BAD_DATA": continue
    cnt[msg.get_type()] += 1; nbytes += len(msg.get_msgbuf())
rates = {k: round(v / dur, 1) for k, v in cnt.most_common()}
# Ida y vuelta de un parámetro
rtt = []
for i in range(100):
    a = time.perf_counter()
    m.mav.param_request_read_send(m.target_system, m.target_component, b"MPC_XY_VEL_MAX", -1)
    r = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=2)
    if r is not None: rtt.append((time.perf_counter() - a) * 1e3)
q = statistics.quantiles(rtt, n=20)
res = {"port": port, "dur_s": dur, "msgs_per_s": round(sum(cnt.values()) / dur, 1), "bytes_per_s": round(nbytes / dur),
       "rates_hz": rates, "param_rtt_ms": {"n": len(rtt), "median": round(statistics.median(rtt), 3), "p5": round(q[0], 3), "p95": round(q[-1], 3), "max": round(max(rtt), 3)}}
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res)[:1500])
