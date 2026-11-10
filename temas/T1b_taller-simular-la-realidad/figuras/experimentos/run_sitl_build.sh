#!/bin/bash
# Arranca PX4 SITL + Gazebo (sin interfaz) en modo demonio y marca cada línea del registro con el tiempo desde el arranque.
# Uso: run_sitl.sh <fichero_log> [VAR=valor ...]
LOG=$1; shift
pkill -9 -f "bin/px4" 2>/dev/null; pkill -9 -f "gz sim" 2>/dev/null; sleep 1
cd /work/px4/build/${BUILD:-px4_sitl_default}/rootfs
rm -rf log parameters*.bson dataman 2>/dev/null
env HEADLESS=1 PX4_SIM_MODEL=${PX4_SIM_MODEL:-gz_x500} GZ_IP=127.0.0.1 "$@" ../bin/px4 -d 2>&1 | python3 -u -c '
import sys,time
t0=time.time()
for l in sys.stdin:
    print(f"{time.time()-t0:8.3f} {l}",end="",flush=True)' > "$LOG"
