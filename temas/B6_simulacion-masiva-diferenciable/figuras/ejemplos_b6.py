"""Ejemplos numéricos y datos de las figuras nuevas del tema B6 (desarrollo de la matemática).

Uso:  python3 ejemplos_b6.py > salida_ejemplos.txt      (numpy, CPU, unos segundos)

Cada ejemplo lleva el número con el que se cita en las hojas («ejemplo N del script»).
Las medidas de rendimiento NO se repiten aquí: se leen de salida_cpu.txt y salida_gpu.txt, que
escribió figuras_b6.py en una máquina concreta (i7-14700K y RTX 4060 Ti). Los demás ejemplos son
cuentas pequeñas que se siguen a mano, más el modelo 1-D exacto del muro de figuras_b6.py, sección 3.
"""
import math
import re
from pathlib import Path

import numpy as np

np.set_printoptions(precision=5, suppress=True)
AQUI = Path(__file__).resolve().parent


def titulo(n, texto):
    print("=" * 78)
    print(f"EJEMPLO {n}. {texto}")


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def Phi(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def phi(x):
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def leer_medidas(fichero):
    filas = []
    for l in (AQUI / fichero).read_text().splitlines():
        m = re.match(r"\s+(\d+) entornos:\s+(\d+) pasos de entorno/s.*?;\s+([\d.]+) ms por 100 pasos", l)
        if m:
            filas.append((int(m.group(1)), float(m.group(2)), float(m.group(3)) / 100))  # ms por paso
    return filas


# --------------------------------------------------------------------------------------------- 1
titulo(1, "Modelo de coste del lote: t(N) = t0 + c·N y rendimiento R(N) = N / t(N)")
gpu, cpu = leer_medidas("salida_gpu.txt"), leer_medidas("salida_cpu.txt")
for nombre, filas in (("GPU", gpu), ("CPU", cpu)):
    print(f"  {nombre}: N | ms por paso del lote | µs por paso | pasos de entorno/s | coste marginal por entorno")
    prev = None
    for n, sps, tms in filas:
        marg = "" if prev is None else f"{(tms - prev[1]) * 1e6 / (n - prev[0]):8.1f} ns/entorno entre {prev[0]} y {n}"
        print(f"    {n:8d} | {tms:9.4f} | {tms*1e3:9.1f} | {sps:12.0f} | {marg}")
        prev = (n, tms)
t0 = gpu[0][2]                          # ms por paso con un entorno: el coste fijo
c = (gpu[5][2] - gpu[4][2]) / (gpu[5][0] - gpu[4][0])     # pendiente entre 16 384 y 65 536
print(f"  GPU: coste fijo t0 ≈ {t0*1e3:.0f} µs por paso; pendiente c ≈ {c*1e6:.1f} ns por entorno y paso")
print(f"  techo del modelo 1/c ≈ {1/(c*1e-3):.3g} pasos/s; N de equilibrio t0/c ≈ {t0/c:.0f} entornos")
for n in (1, 16, 256, 4096, 65536):
    print(f"    modelo N = {n:6d}: t = {t0 + c*n:.4f} ms -> R = {n/((t0 + c*n)*1e-3):.3g} pasos/s")
tpy = 1 / 3612 * 1e3
print(f"  Python con numpy: {tpy*1e3:.0f} µs por paso de UN entorno (intérprete); jit en CPU con 1 entorno: "
      f"{cpu[0][2]*1e3:.2f} µs")
for n in (4096, 65536, 262144, 1048576):
    print(f"  estado del lote, {n} entornos × 17 float32: {n*17*4/1e6:.1f} MB")
# figura: µs por paso frente a N, log-log, eje x igual que la de rendimiento
X = lambda n: 60 + math.log10(n) / 6.3 * 460
Y = lambda us: 250 - (math.log10(us) - (-1)) / 6 * 220      # 0,1 µs .. 100 000 µs
print("  figura t(N): path GPU", path([n for n, _, _ in gpu], [t * 1e3 for _, _, t in gpu], X, Y))
print("  figura t(N): path CPU", path([n for n, _, _ in cpu], [t * 1e3 for _, _, t in cpu], X, Y))
ns = np.logspace(0, 6.02, 40)
print("  figura t(N): modelo t0 + cN", path(ns, (t0 + c * ns) * 1e3, X, Y))
print("  y de t0:", f"{Y(t0*1e3):.1f}", "| y python:", f"{Y(tpy*1e3):.1f}")
print("  ticks y (µs): " + " ".join(f"{v}->{Y(v):.1f}" for v in (0.1, 1, 10, 100, 1000, 10000, 100000)))
print("  puntos GPU:", " ".join(f"{X(n):.1f},{Y(t*1e3):.1f}" for n, _, t in gpu))
print("  puntos CPU:", " ".join(f"{X(n):.1f},{Y(t*1e3):.1f}" for n, _, t in cpu))

# --------------------------------------------------------------------------------------------- 2
titulo(2, "El lote de PPO como rejilla entornos × pasos (cifras de Rudin et al.)")
N, T = 4096, 24
print(f"  B = {N} × {T} = {N*T}; minilotes de {N} × 6 = {N*6} -> {T//6} minilotes por pasada")
print(f"  1500 actualizaciones × {N*T} = {1500*N*T:.3g} pasos")
for n in (2048, 4096, 8192, 16384):
    print(f"  B = 98 304 con {n:5d} entornos -> {98304/n:5.1f} pasos por entorno ({98304/n*0.02:.2f} s a 50 Hz)")

# --------------------------------------------------------------------------------------------- 3
titulo(3, "Orden cero a mano: J(θ) = θ², θ = 1, σ = 0,1, una pareja antitética")
th, sg = 1.0, 0.1
for e in (0.5, 1.0, 1.5, -1.0):
    Jp, Jm = (th + sg * e) ** 2, (th - sg * e) ** 2
    print(f"  ε = {e:4.1f}: J(θ+σε) = {Jp:.4f}, J(θ−σε) = {Jm:.4f}, "
          f"estimación (J+ − J−)/(2σ)·ε = {(Jp - Jm) / (2 * sg) * e:.4f}  (= 2θε²)")
rng = np.random.default_rng(0)
e = rng.standard_normal(1_000_000)
est = ((th + sg * e) ** 2 - (th - sg * e) ** 2) / (2 * sg) * e
print(f"  un millón de parejas: media {est.mean():.4f} (verdadero 2), varianza {est.var():.3f} (teoría 2g² = 8)")
# sin antitética, sin línea base: J(θ+σε)ε/σ
est1 = (th + sg * e) ** 2 * e / sg
print(f"  sin antitética, J(θ+σε)·ε/σ: media {est1.mean():.3f}, varianza {est1.var():.1f} "
      f"(el término J(θ)·ε/σ, de media 0, aporta J(θ)²/σ² = {th**4/sg**2:.0f})")
# comprobación de la derivada de la gaussiana
u, s2 = 1.3, 0.04
dens = lambda t: math.exp(-(u - t) ** 2 / (2 * s2)) / math.sqrt(2 * math.pi * s2)
fd = (dens(th + 1e-6) - dens(th - 1e-6)) / 2e-6
print(f"  ∂N(u;θ,σ²)/∂θ en u = 1,3, θ = 1, σ² = 0,04: diferencias finitas {fd:.5f}; N·(u−θ)/σ² = {dens(th)*(u-th)/s2:.5f}")

# --------------------------------------------------------------------------------------------- 4
titulo(4, "Varianza del orden cero en d dimensiones: J lineal, n parejas antitéticas")
for d, n in ((1, 1), (6, 1), (6, 16), (100, 16), (10000, 16)):
    g = np.zeros(d); g[0] = 1.0
    reps = 20000 if d <= 100 else 400
    E = rng.standard_normal((reps, n, d))
    est = ((E @ g)[:, :, None] * E).mean(1)            # (g·ε)ε promediado sobre n
    err = ((est - g) ** 2).sum(1).mean()
    print(f"  d = {d:5d}, n = {n:2d}: E‖ĝ − g‖² medido {err:9.3f}; teoría (d+1)‖g‖²/n = {(d+1)/n:9.3f}; "
          f"desviación relativa {math.sqrt(err):.2f} (teoría {math.sqrt((d+1)/n):.2f})")

# --------------------------------------------------------------------------------------------- 5
titulo(5, "Primer orden a mano: J(θ,ξ) = (θ+ξ)², ξ ~ N(0,1), θ = 1")
xi = rng.standard_normal(1_000_000)
g1 = 2 * (th + xi)
print(f"  E[J] = θ² + 1 -> gradiente verdadero 2θ = 2")
print(f"  primer orden 2(θ+ξ): media {g1.mean():.4f}, varianza {g1.var():.3f} (teoría 4)")
for x in (-1.0, 0.0, 0.5):
    print(f"    muestra ξ = {x:4.1f}: gradiente {2*(th+x):.1f}")
eps = rng.standard_normal(1_000_000)
g0 = ((th + sg * eps + xi) ** 2 - (th - sg * eps + xi) ** 2) / (2 * sg) * eps
print(f"  orden cero, una pareja con el mismo ξ: 2(θ+ξ)ε², media {g0.mean():.4f}, varianza {g0.var():.2f} "
      f"(teoría 3·4·(θ²+1) − 4 = {3*4*(th**2+1)-4:.0f})")

# --------------------------------------------------------------------------------------------- 6
titulo(6, "BPTT a mano: x_{t+1} = x_t + 0,5·u_t, u_t = −θ x_t, θ = 1, x_0 = 1, J = x_1² + x_2² + x_3²")
th, H = 1.0, 3
x = [1.0]
for t in range(H):
    x.append(x[t] + 0.5 * (-th * x[t]))
A = 1 - 0.5 * th            # f_x + f_u·π_x = 1 + 0,5·(−θ)
print(f"  estados x_0..x_3 = {x}; A = f_x + f_u π_x = 1 + 0,5·(−θ) = {A}")
S = [0.0]
for t in range(H):
    S.append(A * S[t] + 0.5 * (-x[t]))     # f_u·π_θ = 0,5·(−x_t)
print(f"  sensibilidades hacia delante S_t = ∂x_t/∂θ: {S}")
gF = sum(2 * x[t] * S[t] for t in range(1, H + 1))
print(f"  dJ/dθ = Σ 2 x_t S_t = {gF}")
lam = [0.0] * (H + 2)
lam[H] = 2 * x[H]
for t in range(H - 1, 0, -1):
    lam[t] = 2 * x[t] + A * lam[t + 1]
print(f"  adjuntos hacia atrás λ_3, λ_2, λ_1 = {lam[3]}, {lam[2]}, {lam[1]}")
gB = sum(lam[t + 1] * 0.5 * (-x[t]) for t in range(H))
print(f"  dJ/dθ = Σ_t λ_(t+1) f_u π_θ|_t = {gB}")
def Jfun(th):
    xx, J = 1.0, 0.0
    for _ in range(H):
        xx = xx + 0.5 * (-th * xx); J += xx ** 2
    return J
print(f"  diferencias finitas: {(Jfun(1 + 1e-6) - Jfun(1 - 1e-6)) / 2e-6:.6f}")
# comprobación de la recursión del adjunto con coste de mando: c = x² + u², u = −θx
def Jfun2(th, H=3):
    xx, J = 1.0, 0.0
    for _ in range(H):
        u = -th * xx; J += xx ** 2 + u ** 2; xx = xx + 0.5 * u
    return J
xs = [1.0]
for t in range(H):
    xs.append(xs[t] + 0.5 * (-th * xs[t]))
# J = Σ_{t=0}^{H-1} c(x_t,u_t); λ_t = ∂(coste desde t)/∂x_t
lamA = [0.0] * (H + 1); lamB = [0.0] * (H + 1)
for t in range(H - 1, -1, -1):
    cx, cu, pix = 2 * xs[t], 2 * (-th * xs[t]), -th
    lamA[t] = cx + pix * cu + A * lamA[t + 1]       # con el término π_x^T c_u
    lamB[t] = cx + A * lamB[t + 1]                  # sin él
gA = sum((2 * (-th * xs[t]) + lamA[t + 1] * 0.5) * (-xs[t]) for t in range(H))
gBm = sum((2 * (-th * xs[t]) + lamB[t + 1] * 0.5) * (-xs[t]) for t in range(H))
fd2 = (Jfun2(1 + 1e-6) - Jfun2(1 - 1e-6)) / 2e-6
print(f"  con coste de mando c = x² + u²: diferencias finitas {fd2:.6f}; adjunto con π_xᵀc_u {gA:.6f}; "
      f"adjunto sin ese término {gBm:.6f}")

# --------------------------------------------------------------------------------------------- 7
titulo(7, "Producto de jacobianos en escalar: x_{t+1} = a x_t + b θ, S_H = b(1 − a^H)/(1 − a)")
print("   a      |  a^10      a^100      a^1000     a^3000   | S_H/b para H = 10, 100, 1000, 3000")
for a in (0.5, 0.9, 0.988, 0.9998, 1.0, 1.01, 1.1):
    pots = [a ** H for H in (10, 100, 1000, 3000)]
    Ss = [H if a == 1.0 else (1 - a ** H) / (1 - a) for H in (10, 100, 1000, 3000)]
    print(f"  {a:7.4f} | " + " ".join(f"{p:9.3g}" for p in pots) + " | " + " ".join(f"{s:9.4g}" for s in Ss))
for rho in (0.98798, 0.99979):
    print(f"  ρ = {rho}: constante de tiempo −Δt/ln ρ = {-0.01/math.log(rho):.2f} s = {-1/math.log(rho):.0f} pasos; "
          f"1/(1−ρ) = {1/(1-rho):.0f} pasos")
# figura: log10 |a^t| frente a t, t = 0..300
Xf = lin(0, 300, 60, 520)
Yf = lin(-3, 3, 250, 30)         # log10 de 10^-3 a 10^3
for a in (0.97, 0.988, 0.9998, 1.02):
    ts = np.arange(0, 301, 5)
    ys = np.clip(ts * math.log10(a), -3, 3)
    print(f"  figura a = {a}: path", path(ts, ys, Xf, Yf), f"| t final dentro del marco: a^300 = {a**300:.3g}")
print("  ticks y: " + " ".join(f"1e{e}->{Yf(e):.0f}" for e in range(-3, 4)))
print("  ticks x: " + " ".join(f"{t}->{Xf(t):.0f}" for t in (0, 100, 200, 300)))

# --------------------------------------------------------------------------------------------- 8
titulo(8, "El muro en 1-D: z al cruzar = θ + e, con e gaussiana que no depende de θ")
DT, SIGZ, NM = 0.01, 20.0, 200
# e = z − θ obedece  ë = −4e − 3ė + w, con e_0 = ė_0 = 0: lineal en las ráfagas, independiente de θ
Ad = np.array([[1 - 4 * DT * DT, DT * (1 - 3 * DT)], [-4 * DT, 1 - 3 * DT]])   # (e, ė) con Euler semiimplícito
Bd = np.array([DT * DT, DT])
Pc = np.zeros((2, 2))
for _ in range(NM):
    Pc = Ad @ Pc @ Ad.T + SIGZ ** 2 * np.outer(Bd, Bd)
s = math.sqrt(Pc[0, 0])
# comprobación por simulación
W = SIGZ * rng.standard_normal((200000, NM))
e_, v_ = np.zeros(200000), np.zeros(200000)
for k in range(NM):
    v_ = v_ + DT * (-4 * e_ - 3 * v_ + W[:, k]); e_ = e_ + DT * v_
print(f"  desviación típica de e tras {NM} pasos: s = {s:.4f} m (simulación con 200 000 trayectorias: {e_.std():.4f})")
print("  θ | (3−θ)/s | P(choque) = Φ((3−θ)/s) | densidad de z en 3 m, φ/s | dE[J]/dθ = 2 − 100·φ/s | 1.er orden")
for th in (3.4, 3.8, 4.2, 4.6):
    zz = (3 - th) / s
    print(f"  {th:.1f} | {zz:7.3f} | {Phi(zz):.4f} | {phi(zz)/s:.4f} | {2 - 100*phi(zz)/s:8.2f} | 2,00")
th = 3.4; zz = (3 - th) / s
print(f"  a mano en θ = 3,4: (3 − 3,4)/{s:.3f} = {zz:.3f}; φ({abs(zz):.3f}) = {phi(zz):.4f}; /s = {phi(zz)/s:.4f}; "
      f"×100 = {100*phi(zz)/s:.2f}; 2 − eso = {2-100*phi(zz)/s:.2f}")
print(f"  (figuras_b6.py da −59,78 por diferencias finitas, Δθ = 0,02, sobre 200 000 trayectorias de float32)")
# error de Monte Carlo de esas diferencias finitas
d = 0.02; p_band = phi(zz) / s * 2 * d
print(f"  ruido de esas diferencias finitas: una muestra vale 2 − 100/(2Δθ) con prob. ≈ {p_band:.4f}; "
      f"desviación por muestra ≈ {100/(2*d)*math.sqrt(p_band):.0f}; con 200 000: ± {100/(2*d)*math.sqrt(p_band)/math.sqrt(2e5):.2f}")
# mínimo de E[J] = 2θ + 100 Φ((3−θ)/s): 100 φ/s = 2
ths = np.linspace(3.0, 5.0, 20001)
EJ = 2 * ths + 100 * np.array([Phi((3 - t) / s) for t in ths])
print(f"  mínimo de E[J] analítico: θ = {ths[EJ.argmin()]:.3f} m, E[J] = {EJ.min():.2f} (figuras_b6.py: 4,15 m y 8,55)")
# orden cero con σ_p = 0,1: estima el gradiente del problema suavizado dos veces
sp = 0.1; s2 = math.sqrt(s * s + sp * sp)
for th in (3.4, 3.8, 4.2, 4.6):
    print(f"  orden cero (σ = 0,1): gradiente de 2θ + 100Φ((3−θ)/{s2:.3f}) en θ = {th}: {2 - 100*phi((3-th)/s2)/s2:7.2f}")
# figura: densidad de z en el muro para θ = 3,4 y para θ + 0,1
Xz = lin(2.0, 4.8, 40, 520)
Yz = lin(0, 1.1, 230, 30)
zs = np.linspace(2.0, 4.8, 141)
for thf in (3.4, 3.5):
    dens = np.exp(-0.5 * ((zs - thf) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    print(f"  figura densidad θ = {thf}: path", path(zs, dens, Xz, Yz))
    zc = zs[zs <= 3.0]
    dc = np.exp(-0.5 * ((zc - thf) / s) ** 2) / (s * math.sqrt(2 * math.pi))
    print(f"  figura área de choque θ = {thf}: path", path(zc, dc, Xz, Yz) + f" L{Xz(3.0):.1f},{Yz(0):.1f} L{Xz(2.0):.1f},{Yz(0):.1f} Z")
print(f"  x(3 m) = {Xz(3.0):.1f}; x(3,4) = {Xz(3.4):.1f}; x(3,5) = {Xz(3.5):.1f}; y(0) = {Yz(0):.1f}; "
      f"y(densidad en 3 m, θ=3,4) = {Yz(phi((3-3.4)/s)/s):.1f}; y(pico) = {Yz(1/(s*math.sqrt(2*math.pi))):.1f}")
print("  ticks x: " + " ".join(f"{v}->{Xz(v):.1f}" for v in (2.0, 2.5, 3.0, 3.5, 4.0, 4.5)))
# esquema del muro: campana de z al cruzar, tumbada junto al muro (z en vertical: y = 270 − 50 z)
zb = np.linspace(3.4 - 3 * s, 3.4 + 3 * s, 61)
xb = 446 + 80 * np.exp(-0.5 * ((zb - 3.4) / s) ** 2)
print("  esquema campana: path", "M" + " L".join(f"{x:.1f},{270 - 50*z:.1f}" for x, z in zip(xb, zb)))
zc = np.linspace(3.4 - 3 * s, 3.0, 25)
xc = 446 + 80 * np.exp(-0.5 * ((zc - 3.4) / s) ** 2)
print("  esquema choque: path", f"M446,{270 - 50*zc[0]:.1f} L" + " L".join(f"{x:.1f},{270 - 50*z:.1f}" for x, z in zip(xc, zc)) + f" L446,{270-150:.1f} Z")

# --------------------------------------------------------------------------------------------- 9
titulo(9, "Muro suavizado: J = 2θ + 100·σ_s((3 − z)/a), gradiente por muestra 2 − (100/a)·σ_s'((3 − z)/a)")
sig = lambda x: 1 / (1 + np.exp(-x))
dsig = lambda x: sig(x) * (1 - sig(x))
print("  ∫ σ_s'(v)² dv =", f"{np.trapezoid(dsig(np.linspace(-40, 40, 400001))**2, np.linspace(-40, 40, 400001)):.5f} (= 1/6)")
print("  máximo de σ_s' = 1/4 -> gradiente por muestra más negativo 2 − 100/(4a)")
print("  a (m) | θ | media exacta | desviación por muestra exacta | aprox. 100·sqrt(p/(6a)) | muestras con |g−2| > 1 | g mín.")
for a in (0.01, 0.1):
    for th in (3.4, 3.8, 4.2):
        zg = np.linspace(th - 8 * s, th + 8 * s, 800001)
        pz = np.exp(-0.5 * ((zg - th) / s) ** 2) / (s * math.sqrt(2 * math.pi))
        g = 2 - 100 / a * dsig((3 - zg) / a)
        m = np.trapezoid(g * pz, zg)
        sd = math.sqrt(np.trapezoid((g - m) ** 2 * pz, zg))
        p3 = phi((3 - th) / s) / s
        frac = np.trapezoid((np.abs(g - 2) > 1) * pz, zg)
        print(f"  {a:.2f} | {th} | {m:8.2f} | {sd:7.1f} | {100*math.sqrt(p3/(6*a)):7.1f} | {100*frac:5.2f} % | {2-100/(4*a):.0f}")
print("  (figuras_b6.py, a = 0,01: −59,35 verdadero y 337,4 por muestra en 3,4 m; 152,4 en 3,8 m; 62,4 en 4,2 m)")
print(f"  con 4096 muestras: error típico de la media = 337/√4096 = {337.4/64:.2f}; muestras en la banda |3 − z| < 3a: "
      f"≈ 4096·p·6a = {4096*phi((3-3.4)/s)/s*0.06:.0f}")
# figura: gradiente por muestra frente a z (a = 0,01), y la densidad de z escalada
Xg = lin(2.90, 3.10, 60, 520)
Yg = lin(-2600, 200, 250, 30)
zg = np.linspace(2.90, 3.10, 201)
print("  figura g(z): path", path(zg, 2 - 100 / 0.01 * dsig((3 - zg) / 0.01), Xg, Yg))
print("  ticks y: " + " ".join(f"{v}->{Yg(v):.1f}" for v in (0, -500, -1000, -1500, -2000, -2500)))
print("  ticks x: " + " ".join(f"{v}->{Xg(v):.1f}" for v in (2.90, 2.95, 3.0, 3.05, 3.10)))

# --------------------------------------------------------------------------------------------- 10
titulo(10, "SHAC: longitud máxima de los productos de jacobianos, ventana h frente a despliegue H")
for a in (0.9998, 1.01):
    print(f"  a = {a}: H = 1000 -> a^1000 = {a**1000:.4g}; h = 32 -> a^32 = {a**32:.4g}; h = 16 -> a^16 = {a**16:.4g}")
print(f"  ventanas de 32 pasos en un episodio de 1000: {1000/32:.1f}; cada una con su propio corte")
g = 0.99
print(f"  peso del crítico con γ = {g}: γ^16 = {g**16:.3f}, γ^32 = {g**32:.3f}")

# --------------------------------------------------------------------------------------------- 11
titulo(11, "Coste de cada estimador en tiempo de reloj (figuras_b6.py, sección 2, H = 1000, 512 estimaciones)")
datos = {"CPU": (3.5, 14.9, 168.6), "GPU": (6.2, 21.3, 15.6)}
for k, (fw, g1, g0) in datos.items():
    print(f"  {k}: simular {fw} ms; 1.er orden {g1} ms (×{g1/fw:.1f}); orden cero {g0} ms (×{g0/fw:.1f})")
Xb = lin(0, 50, 150, 520)
for k, (fw, g1, g0) in datos.items():
    print(f"  barras {k}: 1.er orden ancho {Xb(g1/fw)-150:.1f}; orden cero ancho {Xb(g0/fw)-150:.1f}")
print("  ticks x (×): " + " ".join(f"{v}->{Xb(v):.0f}" for v in (0, 10, 20, 30, 40, 50)))
