"""Coste del render en Gazebo sin pantalla (EGL): factor de tiempo real y frecuencia de las cámaras, con y sin cámara.
Uso: python3 camaras.py <salida.json>"""
import subprocess, time, re, json, sys, os
ROOT = "/work/px4"; BIN = f"{ROOT}/build/px4_sitl_default/bin"
def sh(c, t=60): return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t).stdout + ""
def kill(): sh("pkill -9 -f '[b]in/px4'; pkill -9 -f '[g]z sim'"); time.sleep(2)
def stats():
    o = sh("gz topic -e -t /world/default/stats -n 1", 20)
    g = lambda k: (lambda m: int(m.group(1)) + int(m.group(2) or 0) * 1e-9)(re.search(k + r" \{\s*sec: (\d+)\s*(?:nsec: (\d+))?", o))
    return g("sim_time"), time.time()
def hz(topic, dur=8):
    o = sh(f"timeout {dur+4} gz topic -f -t {topic} -d {dur}", dur + 10)
    v = re.findall(r"average rate: ([\d.]+)|([\d.]+) Hz", o)
    return o.strip().splitlines()[-3:] if o else None
res = []
for modelo, sf in [("gz_x500", 1), ("gz_x500_mono_cam", 1), ("gz_x500_depth", 1), ("gz_x500_mono_cam", 100), ("gz_x500_depth", 100)]:
    kill()
    env = dict(os.environ, HEADLESS="1", PX4_SIM_MODEL=modelo, GZ_IP="127.0.0.1")
    if sf != 1: env["PX4_SIM_SPEED_FACTOR"] = str(sf)
    p = subprocess.Popen(f"{BIN}/px4 -d", shell=True, cwd=f"{ROOT}/build/px4_sitl_default/rootfs", env=env,
                         stdout=open(f"/work/exp/out/cam_{modelo}_{sf}.log", "w"), stderr=subprocess.STDOUT)
    time.sleep(25)
    tops = [t for t in sh("gz topic -l").split() if "image" in t or "depth" in t]
    a = stats(); time.sleep(10); b = stats()
    rtf = (b[0] - a[0]) / (b[1] - a[1])
    f = {t: hz(t) for t in tops[:3]}
    r = {"modelo": modelo, "pedido": sf, "rtf": round(rtf, 2), "temas": tops, "frecuencia_pared": f}
    print(json.dumps(r, ensure_ascii=False), flush=True); res.append(r)
kill(); json.dump(res, open(sys.argv[1], "w"), indent=1, ensure_ascii=False)
