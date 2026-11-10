#!/bin/bash
# Variante del mundo y del x500 con la IMU a 1 kHz y paso de física de 1 ms (para imu_1khz.py).
# Uso: crear_mundo_1khz.sh /ruta/a/PX4-Autopilot
M=$1/Tools/simulation/gz
sed -e 's#<max_step_size>0.004</max_step_size>#<max_step_size>0.001</max_step_size>#' \
    -e 's#<real_time_update_rate>250</real_time_update_rate>#<real_time_update_rate>1000</real_time_update_rate>#' \
    -e 's#<world name="default">#<world name="default_1khz">#' $M/worlds/default.sdf > $M/worlds/default_1khz.sdf
rm -rf $M/models/x500_base_1k $M/models/x500_1k
cp -r $M/models/x500_base $M/models/x500_base_1k; cp -r $M/models/x500 $M/models/x500_1k
python3 - "$M" <<'PY'
import re, sys
M = sys.argv[1]
p = f"{M}/models/x500_base_1k/model.sdf"; s = open(p).read()
i = s.index('<sensor name="imu_sensor"'); j = s.index('</sensor>', i)
s = s[:i] + s[i:j].replace('<update_rate>250</update_rate>', '<update_rate>1000</update_rate>') + s[j:]
open(p, "w").write(s.replace('<model name="x500_base">', '<model name="x500_base_1k">'))
p = f"{M}/models/x500_1k/model.sdf"; open(p, "w").write(open(p).read().replace("model://x500_base", "model://x500_base_1k"))
for f in [f"{M}/models/x500_base_1k/model.config", f"{M}/models/x500_1k/model.config"]:
    s = open(f).read(); open(f, "w").write(re.sub(r"<name>([^<]*)</name>", lambda m: "<name>" + m.group(1) + "_1k</name>", s, count=1))
PY
echo "Listo: PX4_SIM_MODEL=gz_x500_1k PX4_GZ_WORLD=default_1khz PX4_SYS_AUTOSTART=4001 ./build/px4_sitl_default/bin/px4"
