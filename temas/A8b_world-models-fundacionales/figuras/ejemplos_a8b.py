"""Ejemplos numéricos y datos de las figuras nuevas del tema A8b (desarrollo de la matemática).

Uso:  python3 ejemplos_a8b.py > salida_ejemplos.txt      (numpy, CPU, unos segundos)

Cada ejemplo lleva el número con el que se cita en las hojas («ejemplo N del script»).
Las cifras del experimento del x500 NO se recalculan aquí: se leen de salida_gpu.txt y salida_cpu.txt, que
escribió figuras_a8b.py (GPU, minutos). Lo demás son cuentas pequeñas que se siguen a mano y modelos de juguete
construidos para explicar un mecanismo; no reproducen ningún sistema real.
"""
import math
import re
from pathlib import Path

import numpy as np

np.set_printoptions(precision=4, suppress=True)
AQUI = Path(__file__).resolve().parent
GPU = (AQUI / "salida_gpu.txt").read_text()
CPU = (AQUI / "salida_cpu.txt").read_text()


def titulo(n, texto):
    print("=" * 78)
    print(f"EJEMPLO {n}. {texto}")


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


# --------------------------------------------------------------------------------------------- 1
titulo(1, "Qué se predice: una observación con una parte predecible y otra impredecible")
# o_{t+1} = (u_{t+1}, n_{t+1}): u_{t+1} = u_t + a_t es predecible; n son 3 componentes de ruido puro (sigma = 0,5),
# como hojas que se mueven al viento. Pérdida cuadrática por paso.
rng = np.random.default_rng(1)
N = 200_000
u = rng.normal(0, 1, N); a = rng.uniform(-1, 1, N); sig_n = 0.5
u2 = u + a
n2 = rng.normal(0, sig_n, (N, 3))
# generativo: el mejor predictor de n es su media (0): queda la varianza
l_gen = np.mean((u2 - (u + a)) ** 2 + np.sum(n2 ** 2, 1))
print(f"  generativo (predice o entera; para n, lo mejor es la media 0): pérdida media {l_gen:.4f}"
      f"  (teórica 3 x 0,5² = {3*sig_n**2:.4f}); la predicción de n es 'borrosa': 0 en vez de la hoja")
# predictivo conjunto con E(o) = u: la pérdida puede llegar a 0 y la representación sigue informando de u
l_jepa = np.mean((u2 - (u + a)) ** 2)
print(f"  predictivo con E(o) = u: pérdida {l_jepa:.4f}; correlación de la representación con u: "
      f"{np.corrcoef(u2, u + a)[0,1]:.3f}")
print(f"  predictivo colapsado, E(o) = c = 0,7: pérdida 0 y correlación con u: indefinida (varianza 0)")

# --------------------------------------------------------------------------------------------- 2
titulo(2, "Colapso en un modelo escalar: representación w·x, predictor q, dos vistas x1, x2 del mismo s")
var_s, var_n = 1.0, 0.25
A_, B_ = var_s + var_n, var_s          # E[x1²] = E[x2²] = A; E[x1 x2] = B
print(f"  x1 = s + n1, x2 = s + n2, Var(s) = {var_s}, Var(n) = {var_n}: A = E[x1²] = {A_}, B = E[x1·x2] = {B_}")
print(f"  pérdida L = E[(q·w·x1 − w'·x2)²] = q²w²A − 2·q·w·w'·B + w'²A, con w' el peso del objetivo")
q_opt = B_ / A_
print(f"  sin gradiente detenido (w' = w, se deriva por los dos lados): L = w²(q²A − 2qB + A)")
print(f"    con q óptimo q* = B/A = {q_opt:.2f}: L = w²(A − B²/A) = w² · {A_ - B_**2/A_:.2f}  -> mínimo en w = 0: colapso")
print(f"    primer paso a mano con w = 1, q = 0,8, eta = 0,1: dL/dw = 2w(q²A − 2qB + A) = "
      f"{2*1*(0.8**2*A_ - 2*0.8*B_ + A_):.3f}; w pasa a {1 - 0.1*2*(0.8**2*A_ - 2*0.8*B_ + A_):.3f}")
print(f"  con gradiente detenido (w' = sg(w)): dL/dw = 2q(q·w·A − w'·B); con q = q* y w' = w vale 0: w no se mueve")


def simular(modo, pasos=3000, eta=0.05, tau=0.99, w0=1.0, q0=0.2):
    w, q, wb = w0, q0, w0
    hist = []
    for k in range(pasos + 1):
        if k in (0, 10, 100, 500, 1000, 3000):
            hist.append((k, w, q))
        if modo == "sin sg":
            gw = 2 * w * (q * q * A_ - 2 * q * B_ + A_)
            gq = 2 * w * w * (q * A_ - B_)
        else:
            wt = w if modo == "sg" else wb
            gw = 2 * q * (q * w * A_ - wt * B_)
            gq = 2 * w * (q * w * A_ - wt * B_)
        w, q = w - eta * gw, q - eta * gq
        wb = tau * wb + (1 - tau) * w
    return hist


for modo in ("sin sg", "sg", "sg + EMA"):
    h = simular(modo)
    print(f"  {modo:9s}: " + "; ".join(f"paso {k}: w = {w:.3f}, q = {q:.3f}" for k, w, q in h))
print("  Lectura: sin gradiente detenido, w -> 0 (la representación se apaga). Con él, w se queda en un valor finito")
print("  y q -> B/A: el predictor aprende a predecir y la representación sobrevive. La EMA solo hace el objetivo más lento.")
# datos de la figura: w frente al paso para los tres casos
def tray(modo, pasos=200, eta=0.05, tau=0.99):
    w, q, wb = 1.0, 0.2, 1.0
    W = []
    for k in range(pasos + 1):
        W.append(w)
        if modo == "sin sg":
            gw = 2 * w * (q * q * A_ - 2 * q * B_ + A_); gq = 2 * w * w * (q * A_ - B_)
        else:
            wt = w if modo == "sg" else wb
            gw = 2 * q * (q * w * A_ - wt * B_); gq = 2 * w * (q * w * A_ - wt * B_)
        w, q = w - eta * gw, q - eta * gq
        wb = tau * wb + (1 - tau) * w
    return np.array(W)
X2, Y2 = lin(0, 200, 60, 520), lin(0, 1.4, 230, 30)
for modo in ("sin sg", "sg", "sg + EMA"):
    W = tray(modo)
    ks = np.arange(0, 201, 2)
    print(f"path w {modo}:", path(ks, W[ks], X2, Y2))
    print(f"  {modo}: w a 20 / 50 / 200 pasos = {W[20]:.3f} / {W[50]:.3f} / {W[-1]:.3f}")

# --------------------------------------------------------------------------------------------- 3
titulo(3, "La media móvil exponencial (EMA) de los pesos: cuánto recuerda")
for tau in (0.99, 0.99925):
    print(f"  tau = {tau}: memoria efectiva 1/(1 − tau) = {1/(1-tau):.0f} pasos; semivida ln2/(1 − tau) = {math.log(2)/(1-tau):.0f} pasos")
th = [1.0, 2.0, 2.0, 2.0]
tb = 1.0
for k, t in enumerate(th[1:], 1):
    tb = 0.9 * tb + 0.1 * t
    print(f"  a mano con tau = 0,9: theta pasa de 1 a 2 y se queda; paso {k}: theta_barra = {tb:.3f}")

# --------------------------------------------------------------------------------------------- 4
titulo(4, "V-JEPA 2-AC frente a Cosmos: coste por muestra evaluada (tabla 3 de Assran et al.)")
c_cos = 240 / (80 * 10); c_vj = 16 / (800 * 10)
print(f"  Cosmos: 4 min = 240 s por acción / (80 muestras x 10 iteraciones) = {c_cos:.3f} s por muestra")
print(f"  V-JEPA 2-AC: 16 s / (800 x 10) = {c_vj*1e3:.1f} ms por muestra; cociente {c_cos/c_vj:.0f}")
print(f"  16 s frente a 20 ms (50 Hz): factor {16/0.02:.0f}, log10 = {math.log10(800):.2f} órdenes de magnitud")

# --------------------------------------------------------------------------------------------- 5
titulo(5, "CEM en el espacio de representaciones, a mano (juguete: representación 2-D, acción 1-D)")
zk = np.array([0.0, 0.0]); zg = np.array([1.0, 0.5]); bvec = np.array([1.0, 0.4])
P = lambda a: zk + a * bvec                     # predictor de juguete: P(a; z_k) = z_k + a·b
cand = np.array([-1.0, -0.3, 0.2, 0.8, 1.3, 2.0])
En = np.array([np.abs(P(a) - zg).sum() for a in cand])
for a, e in zip(cand, En):
    zz = P(a)
    print(f"  a = {a:+.1f}: P = ({zz[0]:+.2f}, {zz[1]:+.2f}); |{zz[0]:.2f} − 1| + |{zz[1]:.2f} − 0,5| = {e:.3f}")
el = cand[np.argsort(En)[:2]]
print(f"  dos mejores (élites): {el}; nueva media {el.mean():.3f}, nueva desviación {el.std():.3f}")
grid = np.linspace(-1, 3, 4001); Eg = np.abs(grid[:, None] * bvec - zg).sum(1)
print(f"  mínimo exacto de la energía: a = {grid[Eg.argmin()]:.3f} (energía {Eg.min():.3f})")
X5, Y5 = lin(-1.2, 2.2, 40, 500), lin(-0.6, 1.0, 250, 20)
for a in cand:
    print(f"  figura: a = {a:+.1f} -> punto ({X5(a):.1f}, {Y5(0.4*a):.1f})")
print(f"  figura: z_k ({X5(0):.1f}, {Y5(0):.1f}); z_g ({X5(1):.1f}, {Y5(0.5):.1f}); recta P(a) de ({X5(-1.2):.1f}, {Y5(-0.48):.1f}) a ({X5(2.2):.1f}, {Y5(0.88):.1f})")

# --------------------------------------------------------------------------------------------- 6
titulo(6, "Acciones latentes de Genie: un cuello de botella de 8 códigos (juguete con un sprite 1-D)")
print(f"  |A| = 8 códigos: log2 8 = {math.log2(8):.0f} bits por transición")
bits_frame = 160 * 90 * 3 * 8
print(f"  un fotograma de 160 x 90 en RGB de 8 bits (supuesto): {bits_frame} bits; el código lleva {3/bits_frame:.1e} de eso")
rng = np.random.default_rng(6)
acc = rng.integers(0, 3, 3000) - 1             # izquierda, quieto, derecha
dx = 2.0 * acc + rng.normal(0, 0.3, 3000)       # desplazamiento del sprite entre dos fotogramas (px)
C = np.array([-1.0, 0.5, 1.0])                  # k-means de 3 centros sobre el desplazamiento
for _ in range(20):
    lab = np.argmin(np.abs(dx[:, None] - C[None]), 1)
    C = np.array([dx[lab == j].mean() for j in range(3)])
lab = np.argmin(np.abs(dx[:, None] - C[None]), 1)
orden = np.argsort(C)
acierto = np.mean(orden[acc + 1] == lab)
print(f"  centros aprendidos (px): {np.sort(C)}; el código coincide con la acción real en {acierto*100:.1f} % de las transiciones")

# --------------------------------------------------------------------------------------------- 7
titulo(7, "DIAMOND: el denoiser de EDM y por qué c_skip es el denoiser óptimo de unos datos gaussianos")
sd = 0.5
print(f"  sigma_data = {sd}")
for s in (0.1, 0.5, 1.0, 2.0, 10.0):
    cs = sd**2 / (sd**2 + s**2); co = s * sd / math.sqrt(s**2 + sd**2); ci = 1 / math.sqrt(s**2 + sd**2)
    print(f"  sigma = {s:5.1f}: c_skip = {cs:.3f}, c_out = {co:.3f}, c_in = {ci:.3f}")
x0, eps, s = 0.6, 0.8, 1.0
x = x0 + s * eps
cs = sd**2 / (sd**2 + s**2)
print(f"  a mano: x0 = {x0}, eps = {eps}, sigma = {s}: x = x0 + sigma·eps = {x:.2f}; "
      f"E[x0 | x] = c_skip·x = {cs:.2f} x {x:.2f} = {cs*x:.3f}")
rng = np.random.default_rng(7)
X0 = rng.normal(0, sd, 1_000_000); XX = X0 + s * rng.normal(0, 1, X0.size)
k = np.polyfit(XX, X0, 1)[0]
print(f"  comprobación por Monte Carlo: regresión de x0 sobre x, pendiente {k:.3f} (teórica {cs:.3f})")

# --------------------------------------------------------------------------------------------- 8
titulo(8, "Ordenar políticas sin acertar su éxito: Pearson, orden y error absoluto")
real = np.array([0.70, 0.50, 0.30]); pred = np.array([0.40, 0.30, 0.10])
r = np.corrcoef(real, pred)[0, 1]
print(f"  éxito real {real}, predicho {pred}: Pearson {r:.3f}; mismo orden: {list(np.argsort(-real)) == list(np.argsort(-pred))}; "
      f"error absoluto medio {np.abs(real-pred).mean():.3f}")
pred2 = np.array([0.55, 0.62, 0.30])
r2 = np.corrcoef(real, pred2)[0, 1]
print(f"  predicho {pred2}: Pearson {r2:.3f}; mismo orden: {list(np.argsort(-real)) == list(np.argsort(-pred2))}; "
      f"error absoluto medio {np.abs(real-pred2).mean():.3f}")
mr, mp = real.mean(), pred.mean()
print(f"  a mano (primer caso): medias {mr:.2f} y {mp:.4f}; desviaciones {real-mr} y {pred-mp}")
print(f"    numerador {np.sum((real-mr)*(pred-mp)):.4f}; denominador sqrt({np.sum((real-mr)**2):.4f} x {np.sum((pred-mp)**2):.4f})"
      f" = {math.sqrt(np.sum((real-mr)**2)*np.sum((pred-mp)**2)):.4f}")

# --------------------------------------------------------------------------------------------- 9
titulo(9, "El término de velocidad del coste: (1/λ²)‖v + λe‖² = ‖e + v/λ‖²")
lam = 1.5
print(f"  lambda = {lam} 1/s: 1/lambda = {1/lam:.3f} s")
for v in (0.0, -0.75, -1.5):
    e = 0.5
    print(f"  e = p − p* = +0,50 m, v = {v:+.2f} m/s: e + v/lambda = {e + v/lam:+.3f} m, término = {(e + v/lam)**2:.4f} m²;"
          f" ‖v + lambda·e‖²/lambda² = {(v + lam*e)**2/lam**2:.4f}")
print("  velocidad desde posiciones predichas: v_k = (p_{k+1} − p_k)/0,02 s; p = 1,000 y 1,010 m -> v = 0,50 m/s")

# --------------------------------------------------------------------------------------------- 10
titulo(10, "Por qué la posición sola puede no bastar con 0,5 s: una inclinación sostenida, a mano")
g, om = 9.81, 1.0
for t in (0.1, 0.2, 0.3, 0.4, 0.5):
    print(f"  t = {t:.1f} s: theta = {om*t:.2f} rad, a = g·theta = {g*om*t:.2f} m/s², v = g·om·t²/2 = {g*om*t*t/2:.3f} m/s,"
          f" p = g·om·t³/6 = {g*om*t**3/6:.4f} m")
print(f"  sin(0,5) = {math.sin(0.5):.3f} frente a 0,5: el ángulo pequeño se equivoca un {100*(0.5-math.sin(0.5))/math.sin(0.5):.0f} % al final")
# coste de dos planes de 25 pasos con el objetivo 1 m por delante: flotar o inclinarse
tk = np.arange(1, 26) * 0.02
p_inc = g * om * tk ** 3 / 6; v_inc = g * om * tk ** 2 / 2
e_flo = -np.ones(25); e_inc = p_inc - 1.0
pos_flo, pos_inc = np.sum(e_flo ** 2), np.sum(e_inc ** 2)
vel_flo = np.sum((0 + lam * e_flo) ** 2) / lam ** 2; vel_inc = np.sum((v_inc + lam * e_inc) ** 2) / lam ** 2
print(f"  objetivo 1 m delante, 25 pasos: término de posición: flotar {pos_flo:.2f}, inclinarse {pos_inc:.2f} "
      f"(mejora {pos_flo-pos_inc:.2f}, {100*(pos_flo-pos_inc)/pos_flo:.1f} %)")
print(f"    término de velocidad: flotar {vel_flo:.2f}, inclinarse {vel_inc:.2f} (mejora {vel_flo-vel_inc:.2f}, "
      f"{100*(vel_flo-vel_inc)/vel_flo:.1f} %)")

# --------------------------------------------------------------------------------------------- 11
titulo(11, "MPPI con élites como en figuras_a8b.py: pesos y ruido correlado")
ce = np.array([3.0, 3.4, 4.0, 5.2, 7.0])        # costes de 5 élites (juguete; en el script son 32)
sdev = ce.std()
w = np.exp(-2 * (ce - ce[0]) / sdev); w /= w.sum()
print(f"  costes de las élites {ce}: desviación {sdev:.3f}; temperatura efectiva eta = desviación/2 = {sdev/2:.3f}")
print(f"  exponentes −2(c − c_min)/desv = {-2*(ce-ce[0])/sdev}; pesos {w} (suman {w.sum():.3f})")
rho = 0.9
print(f"  ruido AR(1) con rho = {rho}: e_k = rho·e_(k−1) + sqrt(1 − rho²)·n_k, sqrt(1 − rho²) = {math.sqrt(1-rho**2):.4f}")
print(f"    varianza estacionaria: rho² + (1 − rho²) = 1; correlación a j pasos rho^j: j = 5 (0,1 s) {rho**5:.3f}, "
      f"j = 25 (0,5 s) {rho**25:.3f}")
Hh = 25
var_sum_ind = Hh
var_sum_cor = Hh + 2 * sum((Hh - j) * rho ** j for j in range(1, Hh))
print(f"    desviación de la media de los 25 valores: independiente {math.sqrt(var_sum_ind)/Hh:.3f}, "
      f"correlado {math.sqrt(var_sum_cor)/Hh:.3f} (x {math.sqrt(var_sum_cor/var_sum_ind):.2f})")
print(f"    con sigma = 0,3 (primera iteración): media de la acción en el horizonte ±{0.3*math.sqrt(var_sum_ind)/Hh:.3f} "
      f"frente a ±{0.3*math.sqrt(var_sum_cor)/Hh:.3f}; velocidad angular x4 rad/s: ±{4*0.3*math.sqrt(var_sum_ind)/Hh:.2f} frente a "
      f"±{4*0.3*math.sqrt(var_sum_cor)/Hh:.2f} rad/s")
print(f"    evaluaciones del modelo por paso de control: 256 x 25 x 3 = {256*25*3}")

# --------------------------------------------------------------------------------------------- 12
titulo(12, "Por qué el controlador de los datos se queda bajo con un dron pesado (PD sin integral)")
M0 = 2.0
for mr_ in (0.90, 1.00, 1.10, 1.11):
    ez = g / 2 * (1 - mr_)
    print(f"  m/M0 = {mr_:.2f}: error vertical de equilibrio e_z = (g/2)(1 − m/M0) = {ez*100:+.1f} cm")
print(f"  masas uniformes en (1,07; 1,15): media 1,11 -> {g/2*(1-1.11)*100:+.1f} cm; en (0,85; 0,93): media 0,89 -> {g/2*(1-0.89)*100:+.1f} cm")
# simulación vertical 1-D con el mismo controlador, el mismo retardo del motor y el mismo rozamiento que figuras_a8b.py
rng = np.random.default_rng(12)
def sim_z(m, z0, zr, T=150, dt=0.002, sub=10):
    z, vz, Tr = z0, 0.0, M0 * g
    zs = []
    for k in range(T):
        Tc = M0 * (-2.0 * (z - zr) - 2.5 * vz + g)
        Tc = np.clip(Tc, M0 * g * 0.2, M0 * g * 1.8)
        for _ in range(sub):
            Tr += dt / 0.03 * (Tc - Tr)
            az = Tr / m - g - 0.15 / m * vz
            vz += dt * az; z += dt * vz
        zs.append(z - zr)
    return np.array(zs)
res = {"pesado": [], "ligero": []}
for _ in range(4000):
    mr_ = rng.uniform(0.85, 1.15)
    z0 = rng.uniform(-0.5, 0.5); zr = rng.uniform(-1, 1)
    e = sim_z(M0 * mr_, z0, zr)[100:].mean()
    if mr_ > 1.07: res["pesado"].append(e)
    if mr_ < 0.93: res["ligero"].append(e)
print(f"  simulación vertical 1-D (4000 episodios, último segundo): pesado {np.mean(res['pesado'])*100:+.1f} cm, "
      f"ligero {np.mean(res['ligero'])*100:+.1f} cm")
m_ = re.search(r"controlador de los datos: ídem ([-+\d.]+) cm y ([-+\d.]+) cm", GPU)
print(f"  figuras_a8b.py (3-D, 64 episodios): {m_.group(1)} y {m_.group(2)} cm")

# --------------------------------------------------------------------------------------------- 13
titulo(13, "El planificador elige los errores a su favor: la maldición del optimizador (juguete)")
rng = np.random.default_rng(13)
R = 20000
for n in (1, 32, 256):
    eps = rng.normal(0, 1, (R, n))
    sel = eps[np.arange(R), eps.argmin(1)]
    print(f"  {n:3d} candidatas, todas con el mismo coste real y error del modelo N(0, 1): error de la elegida, media {sel.mean():+.3f}")
# mitad dentro de los datos (error pequeño) y mitad fuera (error grande), mismo coste real
for s_in, s_out in ((0.2, 1.0),):
    n = 256
    sig = np.r_[np.full(n // 2, s_in), np.full(n // 2, s_out)]
    eps = rng.normal(0, 1, (R, n)) * sig
    i = eps.argmin(1)
    print(f"  256 candidatas, 128 cerca de los datos (desv. {s_in}) y 128 lejos (desv. {s_out}): la elegida está lejos en "
          f"{np.mean(i >= n//2)*100:.1f} % de los casos; su error medio {eps[np.arange(R), i].mean():+.3f}")
    eps_in = eps[:, : n // 2]
    print(f"  si solo se proponen candidatas cerca de los datos (residuo acotado): error medio de la elegida "
          f"{eps_in.min(1).mean():+.3f}")

# figura: 16 candidatas, 8 cerca de los datos y 8 lejos, coste real 3 para todas
rng = np.random.default_rng(130)
sig = np.r_[np.full(8, 0.2), np.full(8, 1.0)]
err = rng.normal(0, 1, 16) * sig
c_pred = 3.0 + err
i = int(np.argmin(c_pred))
print(f"  figura (16 candidatas, coste real 3): costes predichos {np.round(c_pred, 2)}; elegida {i} (lejos: {i >= 8}),"
      f" predicho {c_pred[i]:.2f}, error {err[i]:+.2f}")
Yb = lin(0, 5, 230, 30)
print("  figura: alturas y =", [round(float(Yb(c)), 1) for c in c_pred], "; y(3) =", round(Yb(3), 1))

# --------------------------------------------------------------------------------------------- 14
titulo(14, "Error en bucle abierto con cada fuente de acciones (de salida_gpu.txt, sección 4) y su figura")
filas = {}
for l in GPU.splitlines():
    m = re.match(r"\s+(RSSM|latente sin reconstrucción|MLP de un paso)\s+: mediana a 0,1 / 0,5 / 1 s \(cm\): acciones de los datos\s+"
                 r"([\d.]+),\s+([\d.]+),\s+([\d.]+) \| de su mppi\s+([\d.]+),\s+([\d.]+),\s+([\d.]+) \| de su mppi\+politica\s+"
                 r"([\d.]+),\s+([\d.]+),\s+([\d.]+)", l)
    if m:
        v = [float(x) for x in m.groups()[1:]]
        filas[m.group(1)] = (v[0:3], v[3:6], v[6:9])
for nom, (d, p, pp) in filas.items():
    print(f"  {nom:28s}: datos {d}, MPPI puro {p}, MPPI + política {pp}; a 0,5 s, MPPI puro / datos = {p[1]/d[1]:.1f}")
tt = [0.1, 0.5, 1.0]
X14 = lin(0, 1.0, 70, 520)
Y14 = lambda cm: 230 - (math.log10(cm) - math.log10(5)) / (math.log10(1000) - math.log10(5)) * 200
d, p, pp = filas["RSSM"]
for et, vv in (("datos", d), ("MPPI puro", p), ("MPPI + política", pp)):
    print(f"path RSSM {et}:", "M" + " L".join(f"{X14(t):.1f},{Y14(c):.1f}" for t, c in zip(tt, vv)),
          "| puntos", [(round(X14(t), 1), round(Y14(c), 1)) for t, c in zip(tt, vv)])
for c in (5, 10, 30, 100, 300, 1000):
    print(f"  marca eje y {c} cm -> y = {Y14(c):.1f}")

# --------------------------------------------------------------------------------------------- 15
titulo(15, "Latencia por evaluación del modelo y R² de la política imitada (de las salidas de figuras_a8b.py)")
m_ = re.search(r"RSSM\s+: MPPI\s+([\d.]+) ms; MPPI \+ política\s+([\d.]+) ms; política sola\s+([\d.]+) ms", GPU)
mc = re.search(r"RSSM\s+: MPPI\s+([\d.]+) ms por paso", CPU)
tg, tc = float(m_.group(1)), float(mc.group(1))
print(f"  MPPI con el RSSM: GPU {tg} ms / 19200 = {tg/19200*1e3:.3f} µs por paso del modelo;"
      f" CPU {tc} ms / 19200 = {tc/19200*1e3:.2f} µs")
print(f"  presupuesto de 20 ms a 50 Hz: GPU usa {tg/20*100:.0f} %; CPU {tc/20:.1f} veces el presupuesto")
m_ = re.search(r"política imitada desde RSSM: error cuadrático ([\d.]+) \(entrenamiento\), ([\d.]+) \(prueba\); varianza de la acción del maestro ([\d.]+)", GPU)
mse, var = float(m_.group(2)), float(m_.group(3))
print(f"  R² = 1 − ECM/varianza = 1 − {mse}/{var} = {1-mse/var:.3f}")

# --------------------------------------------------------------------------------------------- 16
titulo(16, "Lectura de las curvas de error frente al tiempo (paths de salida_gpu.txt, sección 3)")
# las curvas son la mediana del error con X = lin(0, 3 s, 60, 520) e Y = lin(0, 1 m, 250, 30), recortadas a 1 m
for et in ("controlador de los datos", "física nominal · MPPI", "RSSM · mppi", "RSSM · politica", "RSSM · mppi+politica"):
    m_ = re.search(r"path curva " + re.escape(et) + r": (M[^\n]+)", GPU)
    pts = [tuple(map(float, q.split(","))) for q in m_.group(1)[1:].split(" L")]
    xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
    t = (xs - 60) / 460 * 3.0; e = (250 - ys) / 220
    lec = "; ".join(f"t = {tt:.1f} s: {np.interp(tt, t, e)*100:.0f} cm" for tt in (0.0, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.96))
    print(f"  {et:28s}: {lec}")
    if et == "RSSM · mppi":
        print(f"    primer instante en que la mediana llega a 1 m: {t[np.argmax(e >= 0.999)]:.2f} s")
