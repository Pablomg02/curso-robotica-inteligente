"""IMU a 1 kHz y paso de física de 1 ms: frecuencias de los lazos de PX4 y factor de tiempo real máximo."""
import subprocess, time, re, json, os, sys
ROOT = "/work/px4"; BIN = f"{ROOT}/build/px4_sitl_default/bin"; RF = f"{ROOT}/build/px4_sitl_default/rootfs"
def sh(c, t=60): return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout
def kill(): sh("pkill -9 -f '[b]in/px4'; pkill -9 -f '[g]z sim'"); time.sleep(2)
def stats(w):
    o = sh(f"gz topic -e -t /world/{w}/stats -n 1", 30)
    m = re.search(r"sim_time \{\s*sec: (\d+)\s*(?:nsec: (\d+))?", o); return int(m.group(1)) + int(m.group(2) or 0) * 1e-9, time.time()
res = []
for modelo, mundo, sf in [("gz_x500_1k", "default_1khz", 1), ("gz_x500_1k", "default_1khz", 100)]:
    kill()
    env = dict(os.environ, HEADLESS="1", PX4_SIM_MODEL=modelo, PX4_GZ_WORLD=mundo, PX4_SYS_AUTOSTART="4001", GZ_IP="127.0.0.1")
    if sf != 1: env["PX4_SIM_SPEED_FACTOR"] = str(sf)
    subprocess.Popen(f"{BIN}/px4 -d", shell=True, cwd=RF, env=env, stdout=open(f"/work/exp/out/imu1k_{sf}.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(22)
    sh(f"cd {RF} && {BIN}/px4-param set NAV_DLL_ACT 0"); time.sleep(1)
    sh(f"cd {RF} && {BIN}/px4-commander arm"); sh(f"cd {RF} && {BIN}/px4-commander takeoff"); time.sleep(12)
    a = stats(mundo); time.sleep(10); b = stats(mundo)
    wq = sh(f"cd {RF} && {BIN}/px4-work_queue status")
    hz = {k: float(m.group(1)) for k in ["mc_rate_control", "vehicle_angular_velocity", "vehicle_imu", "ekf2", "control_allocator", "mc_att_control"]
          for m in [re.search(rf"\)\s+{k}\s+([\d.]+) Hz", wq)] if m}
    z = re.search(r"\n\s*z: (-?[\d.]+)", sh(f"cd {RF} && {BIN}/px4-listener vehicle_local_position -n 1"))
    r = {"modelo": modelo, "mundo": mundo, "pedido": sf, "rtf": round((b[0] - a[0]) / (b[1] - a[1]), 2), "hz": hz, "z": float(z.group(1)) if z else None}
    print(r, flush=True); res.append(r)
kill(); json.dump(res, open(sys.argv[1], "w"), indent=1)
