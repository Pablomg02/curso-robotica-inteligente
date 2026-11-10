"""La red de mc_nn_control (entrenada en Aerial Gym para el X500 V2) vuela el x500 de Gazebo en PX4 SITL (build px4_sitl_neural).
Despegue con el control clásico (MAVSDK), cambio al modo externo «Neural Control» (commander mode ext1) y escalones de
posición publicados en /fmu/in/trajectory_setpoint desde ROS 2. Los datos se leen después del ulog (neural_control)."""
import time, json, subprocess, threading
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import TrajectorySetpoint, VehicleLocalPosition
from mavsdk import Mavsdk, Configuration, ComponentType
from mavsdk.plugins.action import Action
from mavsdk.plugins.telemetry import Telemetry
BIN = "/work/px4/build/px4_sitl_neural/bin"
def px4(c): return subprocess.run(f"cd /work/px4/build/px4_sitl_neural/rootfs && {BIN}/px4-{c}", shell=True, capture_output=True, text=True).stdout
QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.TRANSIENT_LOCAL, history=HistoryPolicy.KEEP_LAST, depth=1)
t0 = time.monotonic(); ev = []
def marca(n): ev.append((n, round(time.monotonic() - t0, 3))); print(n, flush=True)
mav = Mavsdk(Configuration.create_with_component_type(ComponentType.GROUND_STATION))
mav.add_any_connection("udpin://0.0.0.0:14540"); d = mav.first_autopilot(30.0); tel, act = Telemetry(d), Action(d)
while not tel.health_all_ok(): time.sleep(0.1)
print(px4("param set MC_NN_MANL_CTRL 0"))   # la red sigue /fmu/in/trajectory_setpoint en vez de los mandos
act.set_takeoff_altitude(5.0)
for _ in range(20):
    try: act.arm(); break
    except Exception: time.sleep(1.0)
act.takeoff(); marca("despegue")
rclpy.init(); n = Node("taller_neural"); lp = {}
n.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", lambda m: lp.update(x=m.x, y=m.y, z=m.z), QOS)
pub = n.create_publisher(TrajectorySetpoint, "/fmu/in/trajectory_setpoint", QOS)
def spin(sec, sp=None):
    te = time.monotonic() + sec
    while time.monotonic() < te:
        if sp is not None:
            m = TrajectorySetpoint(); m.timestamp = n.get_clock().now().nanoseconds // 1000
            m.position = [float(sp[0]), float(sp[1]), float(sp[2])]; m.velocity = [float("nan")] * 3; m.acceleration = [float("nan")] * 3; m.yaw = 0.0
            pub.publish(m)
        rclpy.spin_once(n, timeout_sec=0.05)
spin(12.0)
x0, y0 = lp["x"], lp["y"]
spin(1.0, (x0, y0, -5.0))
marca("ext1"); print(px4("commander mode ext1"))
spin(6.0, (x0, y0, -5.0)); marca("escalon_x")
spin(8.0, (x0 + 2.0, y0, -5.0)); marca("escalon_y")
spin(8.0, (x0 + 2.0, y0 + 2.0, -5.0)); marca("escalon_z")
spin(8.0, (x0 + 2.0, y0 + 2.0, -7.0)); marca("vuelta")
spin(8.0, (x0, y0, -5.0)); marca("fin_neural")
print(px4("commander mode auto:loiter")); spin(3.0); act.land()
while tel.armed(): time.sleep(0.2)
marca("desarmado")
json.dump({"eventos": ev, "x0": x0, "y0": y0}, open("/work/exp/out/neural.json", "w"), indent=1)
