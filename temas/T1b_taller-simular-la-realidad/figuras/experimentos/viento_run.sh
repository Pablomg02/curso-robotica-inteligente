#!/bin/bash
# Dos vuelos: mundo default y mundo windy; guarda el ulog de cada uno en /work/exp/out/viento_<mundo>.ulg
for W in default windy; do
  bash /work/exp/run_sitl.sh /work/exp/boot_viento_$W.log PX4_GZ_WORLD=$W &
  for i in $(seq 1 60); do grep -q "Startup script returned" /work/exp/boot_viento_$W.log && break; sleep 1; done
  sleep 3
  python3 /work/exp/viento.py $W
  sleep 2
  L=$(ls -t /work/px4/build/px4_sitl_default/rootfs/log/*/*.ulg | head -1); cp "$L" /work/exp/out/viento_$W.ulg
  pkill -9 -f "[b]in/px4"; pkill -9 -f "[g]z sim"; sleep 2
done
echo fin
