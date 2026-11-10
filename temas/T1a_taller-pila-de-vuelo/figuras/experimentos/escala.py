"""Factor de tiempo real alcanzado por PX4 SITL + Gazebo (sin interfaz) según el factor pedido y el número de drones.
Cada dron despega a 3 m y se mide en estacionario. Uso: python3 escala.py <salida.json>"""
import subprocess, time, re, json, sys, os
ROOT = "/work/px4"; BIN = f"{ROOT}/build/px4_sitl_default/bin"
def sh(c, **k): return subprocess.run(c, shell=True, capture_output=True, text=True, **k).stdout
def kill():
    sh("pkill -9 -f '[b]in/px4'; pkill -9 -f '[g]z sim'"); time.sleep(2)   # los corchetes evitan que pkill se mate a sí mismo
def stats():
    o = sh("gz topic -e -t /world/default/stats -n 1", timeout=20)
    s = re.search(r"sim_time \{\s*sec: (\d+)\s*(?:nsec: (\d+))?", o); r = re.search(r"real_time \{\s*sec: (\d+)\s*(?:nsec: (\d+))?", o)
    f = lambda m: int(m.group(1)) + int(m.group(2) or 0) * 1e-9
    return f(s), f(r), time.time()
def cpu():
    o = sh("ps -eo pcpu,args | grep -E 'bin/px4|gz sim' | grep -v grep")
    px4 = sum(float(l.split()[0]) for l in o.splitlines() if "bin/px4" in l); gz = sum(float(l.split()[0]) for l in o.splitlines() if "gz sim" in l)
    return round(px4, 1), round(gz, 1)
def run(sf, n):
    kill(); procs = []
    env = dict(os.environ, HEADLESS="1", PX4_SYS_AUTOSTART="4001", PX4_SIM_MODEL="gz_x500", GZ_IP="127.0.0.1",
               PX4_PARAM_NAV_DLL_ACT="0", PX4_PARAM_NAV_RCL_ACT="0", PX4_PARAM_COM_RC_IN_MODE="4")
    if sf != 1: env["PX4_SIM_SPEED_FACTOR"] = str(sf)
    for i in range(n):
        e = dict(env, PX4_GZ_MODEL_POSE=f"0,{2*i}")
        if i > 0: e["PX4_GZ_STANDALONE"] = "1"
        procs.append(subprocess.Popen(f"{BIN}/px4 -i {i} -d", shell=True, cwd=ROOT, env=e, stdout=open(f"/work/exp/out/esc_{i}.log", "w"), stderr=subprocess.STDOUT))
        time.sleep(12 if i == 0 else 4)
    time.sleep(8)
    for i in range(n):
        sh(f"{BIN}/px4-param --instance {i} set NAV_DLL_ACT 0"); time.sleep(1)
        sh(f"{BIN}/px4-commander --instance {i} arm"); sh(f"{BIN}/px4-commander --instance {i} takeoff")
    time.sleep(10)
    a = stats(); c = cpu(); time.sleep(10); b = stats()
    alt = []
    for i in range(n):
        o = sh(f"{BIN}/px4-listener --instance {i} vehicle_local_position -n 1")
        m = re.search(r"\n\s*z: (-?[\d.]+)", o); alt.append(float(m.group(1)) if m else None)
    rtf_gz = (b[0] - a[0]) / (b[1] - a[1]); rtf_wall = (b[0] - a[0]) / (b[2] - a[2])
    r = {"pedido": sf, "drones": n, "rtf": round(rtf_wall, 2), "rtf_gz": round(rtf_gz, 2), "cpu_px4_pct": c[0], "cpu_gz_pct": c[1], "z": alt}
    print(r, flush=True); return r
out = []
for sf in [1, 2, 4, 8, 16, 32, 100]: out.append(run(sf, 1))
for n in [2, 4, 8]: out.append(run(1, n)); out.append(run(100, n))
kill(); json.dump(out, open(sys.argv[1], "w"), indent=1)
