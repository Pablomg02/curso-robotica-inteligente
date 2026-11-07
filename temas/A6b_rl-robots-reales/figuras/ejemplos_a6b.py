"""Ejemplos numéricos y datos de las figuras nuevas del tema A6b (desarrollo de la matemática).

Uso:  python3 ejemplos_a6b.py > salida_ejemplos.txt      (numpy, CPU, unos segundos)

Cada ejemplo lleva el número con el que se cita en las hojas («ejemplo N del script»).
Los experimentos con PPO, DAgger y los espacios de acción están en figuras_a6b.py (GPU, minutos)
y no se repiten aquí: este script solo reproduce lo que se sigue a mano y lo que dibujan las
figuras nuevas. Las constantes del x500 (masa 2 kg, brazo 0,25 m, inercias, mapeos de la acción)
son las de figuras_a6b.py.
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)

M0, G = 2.0, 9.81
INER = np.array([0.022, 0.022, 0.040])     # inercias del x500 de figuras_a6b.py (kg m²)
L_ARM, C_TAU = 0.25, 0.016                 # brazo y relación par/empuje de un rotor
DT = 0.01                                  # paso de control: 100 Hz


def titulo(n, texto):
    print("=" * 78)
    print(f"EJEMPLO {n}. {texto}")


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


# =============================================================================== PARTE 1
titulo(1, "ALEATORIZACIÓN DE DOMINIO: QUÉ SE SORTEA EN CADA ENTORNO (figuras_a6b.py)")
print("  factor de masa  m/m0 ~ U(0,8; 1,2)  -> masa de 1,6 a 2,4 kg")
print("  ganancia de empuje k_T ~ U(0,9; 1,1)")
print("  constante del motor tau_m ~ U(20; 60) ms")
xi = np.column_stack([rng.uniform(0.8, 1.2, 4), rng.uniform(0.9, 1.1, 4), rng.uniform(0.02, 0.06, 4)])
for i, (mf, kt, tm) in enumerate(xi):
    print(f"  entorno {i}: masa {M0*mf:.2f} kg, k_T {kt:.3f}, tau_m {tm*1000:.0f} ms")

titulo(2, "UN EMPUJE PARA TODAS LAS MASAS: LA MEJOR ACCIÓN QUE NO VE LA MASA")
# coste c(F, m) = (F - m g)^2 (empuje que sobra o falta para quedarse quieto)
m_lo, m_hi = 1.6, 2.4
Em = (m_lo + m_hi) / 2
Vm = (m_hi - m_lo) ** 2 / 12
F_opt = G * Em
coste_min = G ** 2 * Vm
print(f"  E[m] = {Em:.2f} kg, Var(m) = (0,8)^2/12 = {Vm:.4f} kg^2")
print(f"  F* = g E[m] = {F_opt:.2f} N; coste esperado mínimo g^2 Var(m) = {G**2:.2f} x {Vm:.4f} = {coste_min:.2f} N^2"
      f" (desviación {math.sqrt(coste_min):.2f} N)")
for m in (m_lo, Em, m_hi):
    err = F_opt - m * G
    print(f"  masa {m:.1f} kg: F* - m g = {err:+.2f} N -> aceleración vertical {err/m:+.2f} m/s^2")
print("  quien ve la masa (maestro) elige F = m g: coste 0")
# comprobación por Monte Carlo y gradiente medio sobre un lote de 4096 entornos
ms = rng.uniform(m_lo, m_hi, 4096)
for F in (18.0, F_opt, 21.0):
    grad = np.mean(-2 * (F - ms * G))       # gradiente de -c respecto a F, promediado en el lote
    print(f"  F = {F:5.2f} N: gradiente medio de -coste en 4096 entornos = {grad:+.2f} N"
          f"  (exacto {-2*(F - G*Em):+.2f})")

titulo(3, "UNA LÍNEA BASE QUE VE EL ESTADO OCULTO: MISMA MEDIA, MENOS VARIANZA")
# política gaussiana a ~ N(mu, sigma^2); retorno G = -a^2 + 5 xi, xi = ±1 oculto al actor
mu, sig = 1.0, 0.5
n = 200_000
a = mu + sig * rng.standard_normal(n)
x = rng.choice([-1.0, 1.0], n)
Gr = -a ** 2 + 5 * x
score = (a - mu) / sig ** 2
EG = -(mu ** 2 + sig ** 2)
print(f"  mu = {mu}, sigma = {sig}; gradiente exacto d/dmu E[-a^2] = -2 mu = {-2*mu:.2f}")
for nombre, b in (("sin línea base, b = 0", 0.0 * Gr),
                  ("simétrica, b = E[G] (constante)", np.full(n, EG)),
                  ("asimétrica, b = E[G] + 5 xi", EG + 5 * x)):
    g = score * (Gr - b)
    print(f"  {nombre:34s}: media {g.mean():+.3f}, desviación por muestra {g.std():6.2f}")

titulo(4, "GAE CON lambda = 1 Y CON lambda = 0 CUANDO EL CRÍTICO SE EQUIVOCA")
# episodio de 3 pasos, gamma = 1, recompensas (0, 0, 1); valores verdaderos (1, 1, 1)
r = np.array([0.0, 0.0, 1.0])
for nombre, V in (("crítico exacto", np.array([1.0, 1.0, 1.0, 0.0])),
                  ("crítico con +0,25 en s1", np.array([1.0, 1.25, 1.0, 0.0]))):
    delta = r + V[1:] - V[:-1]
    A1 = delta.sum()                         # lambda = 1: suma de deltas = G0 - V(s0)
    A0 = delta[0]                            # lambda = 0: un paso
    print(f"  {nombre:24s}: deltas {np.round(delta, 2)}, A0 con lambda=1: {A1:+.2f}, con lambda=0: {A0:+.2f}")

titulo(5, "DOS PUERTAS Y UNA PISTA: V(s), V(h) y V(h, s)")
p_pista = 0.5
V_con, V_sin = 1.0, 0.5
Vs = p_pista * V_con + (1 - p_pista) * V_sin
print(f"  V(s=izq) = V(s=der) = 1/2 x 1 + 1/2 x 1/2 = {Vs:.2f}")
print(f"  V(h = con pista) = {V_con:.2f};  V(h = sin pista) = {V_sin:.2f}")
print(f"  E_(s|h)[V(s)] = {Vs:.2f} para las dos historias: error {Vs - V_sin:+.2f} sin pista, {Vs - V_con:+.2f} con pista")
print(f"  V(h, s) = V(h) en este ejemplo: 1 con pista, 0,5 sin ella, cualquiera que sea s")
print(f"  error medio de V(s) frente a V(h): {p_pista*(Vs-V_con) + (1-p_pista)*(Vs-V_sin):+.2f} (se compensa en media, no en cada caso)")

# =============================================================================== PARTE 2
titulo(6, "CLONACIÓN FRENTE A DAgger: CUÁNTO COSTE EXTRA SE ACUMULA")
# modelo peor caso de la clonación: un error y se pierde (coste 1 en cada paso restante)
# DAgger: cada error cuesta como mucho u y el alumno vuelve a la distribución donde aprendió
for T, eps in ((100, 0.01), (1000, 0.001)):
    t = np.arange(1, T + 1)
    p_perdido = 1 - (1 - eps) ** t           # probabilidad de haber fallado ya en el paso t
    bc = p_perdido.sum()
    union = eps * T * (T + 1) / 2
    for u in (3,):
        print(f"  T = {T}, eps = {eps}: clonación exacta {bc:.1f}; cota de la unión eps T(T+1)/2 = {union:.1f};"
              f" T^2 eps = {T*T*eps:.0f}; DAgger u T eps (u = {u}) = {u*T*eps:.0f}")
    if T == 100:
        for tt in (1, 10, 50, 100):
            print(f"     paso {tt:3d}: P(ya se ha equivocado) = {p_perdido[tt-1]:.3f} (cota t eps = {tt*eps:.2f});"
                  f" coste extra acumulado {p_perdido[:tt].sum():.2f}")
        X, Y = lin(0, 100, 60, 520), lin(0, 60, 250, 30)
        print("path bc_exacto:", path(t, np.cumsum(p_perdido), X, Y))
        print("path bc_cota:", path(t, eps * t * (t + 1) / 2, X, Y))
        print("path dagger:", path(t, 3 * eps * t, X, Y))
# Monte Carlo del modelo de cadena
T, eps, u = 100, 0.01, 3
n = 100_000
err = rng.random((n, T)) < eps
primero = np.where(err.any(1), err.argmax(1), T)
extra_bc = (T - primero).mean()
extra_dg = (u * err).sum(1).mean()
print(f"  Monte Carlo ({n} episodios): clonación {extra_bc:.1f}; DAgger con coste u = 3 por error {extra_dg:.2f}")

titulo(7, "BRECHA DE IMITACIÓN: EL ÓPTIMO CUADRÁTICO ES LA MEDIA CONDICIONADA")
for p_izq in (0.5, 0.8, 1.0):
    m = p_izq * (-1) + (1 - p_izq) * (+1)
    var = p_izq * (-1 - m) ** 2 + (1 - p_izq) * (1 - m) ** 2
    print(f"  P(izq | o) = {p_izq:.1f}: E[pi*(s) | o] = {m:+.2f}; pérdida mínima Var = {var:.2f}")
aa = np.linspace(-1, 1, 5)
print("  pérdida E[(a - pi*)^2 | o] con P(izq|o) = 0,5 para a =", aa, "->", 1 + aa ** 2)

# =============================================================================== PARTE 3
titulo(8, "QUÉ HACE CADA ACCIÓN EN EL x500 DE figuras_a6b.py")


def ctbr(a, om=np.zeros(3)):
    F = M0 * G * (1 + 0.8 * a[0])
    om_d = np.array([4.0, 4.0, 2.0]) * a[1:]
    Mp = INER * 20 * (om_d - om) + np.cross(om, INER * om)
    f = np.array([F / 4 - Mp[1] / (2 * L_ARM) + Mp[2] / (4 * C_TAU),
                  F / 4 + Mp[0] / (2 * L_ARM) - Mp[2] / (4 * C_TAU),
                  F / 4 + Mp[1] / (2 * L_ARM) + Mp[2] / (4 * C_TAU),
                  F / 4 - Mp[0] / (2 * L_ARM) - Mp[2] / (4 * C_TAU)])
    return F, om_d, Mp, f


for a in (np.array([0.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.5, 0.0, 0.0]), np.array([0.25, 0.0, 0.0, 0.0])):
    F, om_d, Mp, f = ctbr(a)
    print(f"  CTBR a = {a}: F = {F:.2f} N, omega_d = {om_d} rad/s, M = {np.round(Mp, 3)} N m, f = {np.round(f, 3)} N (suma {f.sum():.2f})")
for a in (np.zeros(4), np.array([0.0, 0.36, 0.0, -0.36])):
    f = M0 * G / 4 * (1 + 0.8 * a)
    Mx = L_ARM * (f[1] - f[3])
    print(f"  SRT  a = {a}: f = {np.round(f, 3)} N, F = {f.sum():.2f} N, M_x = {Mx:.3f} N m")
print(f"  límites: F de {M0*G*0.2:.2f} a {M0*G*1.8:.2f} N; |omega_d| hasta (4, 4, 2) rad/s; cada rotor de 0 a m0 g = {M0*G:.2f} N")

titulo(9, "EL TÉRMINO DE SUAVIDAD EN EL COSTE")
w = 0.1
for da in (np.array([0.1, 0, 0, 0]), np.array([0.5, 0, 0, 0]), np.array([1.0, -1.0, 1.0, -1.0])):
    c = w * np.sum(da ** 2)
    print(f"  Delta a = {da}: 0,1 ||Delta a||^2 = {c:.3f}")
print("  recompensa r = 1 - tanh(coste) - 10 choque; con coste 0,05: r =", f"{1-math.tanh(0.05):.3f}",
      "; con 0,45:", f"{1-math.tanh(0.45):.3f}")

titulo(10, "AUTOVALORES DE da_t/da_{t-1}: QUÉ HACE CADA kappa")
for k in (-0.9, 0.5, 0.95, 1.0):
    seq = k ** np.arange(8)
    print(f"  kappa = {k:+.2f}: a_t = kappa^t a_0 ->", np.round(seq, 3))
print(f"  kappa = -0,9: cambia de signo cada paso -> periodo 2 pasos = {2*DT*1000:.0f} ms -> {1/(2*DT):.0f} Hz (Nyquist a 100 Hz)")
print(f"  kappa = -0,9 tras 20 pasos (200 ms): |a| = {0.9**20:.3f} de la inicial; kappa = -0,99: {0.99**20:.3f}")
for z in (0.32 + 0.05j, -0.26 + 0.30j):
    rho, phi = abs(z), math.atan2(z.imag, z.real)
    print(f"  autovalor {z.real:+.2f}{z.imag:+.2f}j: módulo {rho:.3f}, ángulo {phi:.3f} rad -> giro de {phi/(2*math.pi*DT):.1f} Hz;"
          f" tras 3 pasos queda {rho**3:.3f}")
# datos de la figura: tallos de kappa^t, un panel por kappa (x0 = 40 + 180 p, 15 px por paso; y = 110 - 60 a_t)
for p_, k in enumerate((-0.9, 0.5, 0.95)):
    seq = k ** np.arange(10)
    print(f"stems kappa {k}:", " ".join(f"{40 + 180*p_ + 15*i:.0f},{110 - 60*v:.1f}" for i, v in enumerate(seq)))

titulo(11, "RETARDO DE UN FILTRO PASO BAJO DE PRIMER ORDEN, tau = 20 ms")
tau = 0.020
for f in (1, 5, 10):
    om = 2 * math.pi * f
    ph = math.atan(om * tau)
    print(f"  f = {f:2d} Hz: omega = {om:.1f} rad/s, omega tau = {om*tau:.3f}, desfase atan = {ph:.3f} rad"
          f" ({math.degrees(ph):.1f}°), retardo = {ph/om*1000:.1f} ms, ganancia |H| = {1/math.sqrt(1+(om*tau)**2):.3f}")
print(f"  frecuencia de corte 1/(2 pi tau) = {1/(2*math.pi*tau):.1f} Hz")
print(f"  media móvil de 5 muestras a 100 Hz: retardo (5-1)/2 x 10 ms = {(5-1)/2*10:.0f} ms")

# =============================================================================== PARTE 4
titulo(12, "DEL SIMULADOR (z ARRIBA) A PX4 (NED, FRD), CON UNA ACCIÓN CTBR")
F, om_d, Mp, f = ctbr(np.array([0.0, 0.5, 0.25, 0.5]))
om_frd = om_d * np.array([1, -1, -1])
F_max = 4 * M0 * G
print(f"  acción a = (0; 0,5; 0,25; 0,5): F = {F:.2f} N, omega_d (FLU) = {om_d} rad/s")
print(f"  FLU -> FRD: (x, y, z) -> (x, -y, -z): omega_d (FRD) = {om_frd} rad/s")
print(f"  si el empuje normalizado fuera lineal con F_max = 4 m0 g = {F_max:.2f} N (los límites del simulador):"
      f" F/F_max = {F/F_max:.3f} -> thrust_body = (0, 0, {-F/F_max:.3f})")
p_enu = np.array([1.0, 2.0, 3.0])
print(f"  posición con z arriba (ENU) {p_enu} -> NED (y, x, -z) = {np.array([p_enu[1], p_enu[0], -p_enu[2]])}")
