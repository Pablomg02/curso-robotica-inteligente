"""Ejemplos numéricos del hito H1 (enunciado y examen).

numpy y la biblioteca estándar, CPU, segundos. Uso:
    python3 ejemplos_h1.py > salida_ejemplos.txt
Las cifras de partida (frecuencias de PX4, 0,1 s y 30° de A5, etc.) son las
que ya citan el hito y sus temas fuente; aquí solo se hacen las cuentas.
"""
import math
from fractions import Fraction
import numpy as np

rng = np.random.default_rng(0)


def cab(n, t):
    print(f"\n=== Ejemplo {n} · {t} ===")


# ------------------------------------------------------------------ 1
cab(1, "Reloj único: periodo de cada bloque en pasos base")
bloques = [("rate", 800), ("IMU", 250), ("actitud", 200), ("posición", 100),
           ("posición visual / profundidad", 20), ("local + minimum snap", 10),
           ("global", 1)]
for fbase in (2000, 4000):
    print(f"reloj a {fbase} Hz (paso {1000/fbase:.2f} ms):")
    for nom, f in bloques:
        r = Fraction(fbase, f)
        ok = "entero" if r.denominator == 1 else "NO entero"
        print(f"   {nom:30s} {f:4d} Hz -> {float(r):7.2f} pasos ({ok})")
per = [Fraction(1, f) for _, f in bloques]          # periodos en s
g = per[0]
for p in per[1:]:
    g = Fraction(math.gcd(g.numerator*p.denominator, p.numerator*g.denominator),
                 g.denominator*p.denominator)
print(f"mayor paso que divide todos los periodos = {float(g)*1e3:.3f} ms -> {1/float(g):.0f} Hz")
print(f"vuelo de 30 s a 4 kHz = {30*4000} pasos; a 2 kHz = {30*2000}")


# ------------------------------------------------------------------ 2
cab(2, "Banda del NEES medio de posición: chi^2 con 150 grados de libertad / 50")


def gammainc_reg(a, x):
    """P(a, x) regularizada (serie para x < a+1, fracción continua si no)."""
    if x <= 0:
        return 0.0
    gln = math.lgamma(a)
    if x < a + 1:
        s = term = 1.0/a
        n = a
        for _ in range(10000):
            n += 1
            term *= x/n
            s += term
            if abs(term) < abs(s)*1e-15:
                break
        return s*math.exp(-x + a*math.log(x) - gln)
    b = x + 1 - a
    c = 1/1e-300
    d = 1/b
    h = d
    for i in range(1, 10000):
        an = -i*(i - a)
        b += 2
        d = an*d + b
        d = 1/d if abs(d) > 1e-300 else 1e300
        c = b + an/c
        c = c if abs(c) > 1e-300 else 1e-300
        delta = d*c
        h *= delta
        if abs(delta - 1) < 1e-15:
            break
    return 1 - math.exp(-x + a*math.log(x) - gln)*h


def chi2_cdf(x, k):
    return gammainc_reg(k/2, x/2)


def chi2_ppf(p, k):
    lo, hi = 0.0, 10*k
    for _ in range(200):
        mid = (lo + hi)/2
        if chi2_cdf(mid, k) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi)/2


n_dim, M = 3, 50
k = n_dim*M
q_lo, q_hi = chi2_ppf(0.025, k), chi2_ppf(0.975, k)
print(f"dimensión n = {n_dim}, vuelos M = {M}: la suma de los NEES sigue chi^2_{k}")
print(f"cuantiles 2,5 % y 97,5 % de chi^2_{k}: {q_lo:.2f} y {q_hi:.2f}")
print(f"divididos por M = {M}: banda [{q_lo/M:.3f}, {q_hi/M:.3f}]  (centro {n_dim})")
# comprobación de Monte Carlo: errores gaussianos con la covarianza que cree el filtro
reps = 20000
e = rng.standard_normal((reps, M, n_dim))         # e ~ N(0, P), blanqueado: P^-1/2 e
nees_medio = (e**2).sum(-1).mean(-1)
dentro = np.mean((nees_medio >= q_lo/M) & (nees_medio <= q_hi/M))
print(f"Monte Carlo ({reps} repeticiones de {M} vuelos): media {nees_medio.mean():.3f}, "
      f"dentro de la banda {100*dentro:.1f} %")
# filtro optimista: cree que la covarianza es la mitad de la real
e2 = rng.standard_normal((reps, M, n_dim))*np.sqrt(2.0)
print(f"filtro con P la mitad de la real: NEES medio {((e2**2).sum(-1).mean(-1)).mean():.2f} "
      f"(fuera de la banda)")
# curva de densidad de y = chi^2_150/50 para la figura
ys = np.linspace(1.8, 4.4, 53)
x = M*ys
logpdf = (k/2 - 1)*np.log(x) - x/2 - (k/2)*math.log(2) - math.lgamma(k/2)
pdf = M*np.exp(logpdf)
print(f"densidad máxima {pdf.max():.3f} en y = {ys[np.argmax(pdf)]:.2f}")
# trazado: y de 1,8 a 4,4 en x [50, 550]; densidad 0..1,3 en y [200, 20]
X = 50 + (ys - 1.8)/2.6*500
Y = 200 - pdf/1.3*180
print("path densidad:", "M" + " L".join(f"{a:.1f} {b:.1f}" for a, b in zip(X, Y)))
for v in (q_lo/M, 3.0, q_hi/M):
    print(f"  x({v:.2f}) = {50 + (v-1.8)/2.6*500:.1f}")


# ------------------------------------------------------------------ 3
cab(3, "Distancia de parada del criterio 7 (fórmula y cifras de A5)")
gval = 9.81
a = gval*math.tan(math.radians(30))
print(f"a = g tan 30° = {a:.2f} m/s^2")
for tl in (0.10, 0.13):
    for v in (3.0, 6.0):
        d = v*tl + v**2/(2*a)
        print(f"  t_l = {tl:.2f} s, v = {v:.0f} m/s: d = v t_l + v^2/2a = {v*tl:.2f} + "
              f"{v**2/(2*a):.2f} = {d:.2f} m")


# ------------------------------------------------------------------ 4
cab(4, "Comprobación de las cuentas del examen")
print(f"P1: 2^(3/2) = {2**1.5:.3f}")
m, acc = 2.0, 3.0
th = math.degrees(math.atan(acc/gval))
T = m*math.hypot(gval, acc)
print(f"P2: theta = {th:.2f}°, T = {T:.2f} N, hover = {m*gval:.2f} N, +{100*(T/(m*gval)-1):.1f} %")
print(f"P3: omega_c tau = {2*math.pi*10*0.005:.4f} rad = {math.degrees(2*math.pi*10*0.005):.1f}°")
wi = 20.0
for kk in (5.0, 20.0):
    z = 0.5*math.sqrt(wi/kk)
    os_ = math.exp(-math.pi*z/math.sqrt(1-z*z))*100 if z < 1 else 0.0
    print(f"P4: k = {kk:.0f}: omega_n = {math.sqrt(wi*kk):.2f} rad/s, zeta = {z:.2f}, "
          f"sobrepaso = {os_:.1f} %")
q, r = 1e-2, 1e-4
w0 = (q/r)**0.25
print(f"P8: omega_0 = {w0:.3f} rad/s ({w0/(2*math.pi):.2f} Hz), K = ({math.sqrt(2)*w0:.2f}, {w0**2:.2f})")
print(f"P9: banda del NEES medio con n = 3 y 50 vuelos = [{q_lo/M:.2f}, {q_hi/M:.2f}]")
print(f"P10: f omega t_d = {450*2*0.005:.2f} px")
al = 0.8
print(f"P12: alpha^-7 = {al**-7:.2f}; alpha^-2 = {al**-2:.4f} -> {3*al**-2:.2f} m/s^2; "
      f"T_min = 10 sqrt(3/4) = {10*math.sqrt(3/4):.2f} s")
