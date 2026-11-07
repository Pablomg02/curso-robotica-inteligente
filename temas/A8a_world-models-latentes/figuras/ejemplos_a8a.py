"""Ejemplos numéricos y datos de las figuras nuevas de A8a (model-based RL y world models latentes).

Cada número de un ejemplo «a mano» de la presentación sale de aquí. Las cifras del RSSM del x500
(error en bucle abierto, sondeos, muestras) siguen saliendo de figuras_a8a.py (GPU), que no hace
falta relanzar. Los hiperparámetros reales son los que ya cita el tema (Dreamer: H = 15, γ = 0,99,
λ = 0,95; DreamerV3: β = 1 / 1 / 0,1 y 1 nat de bits libres; TD-MPC: H = 5, λ = 0,5; TD-MPC2: V = 8).
Los valores marcados como SUPUESTO son números pequeños elegidos aquí para que se sigan a mano.
Ejecutar:  python3 ejemplos_a8a.py > salida_ejemplos.txt   (numpy, CPU, segundos)
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def titulo(n, s):
    print("\n" + "=" * 72 + f"\nEjemplo {n}. {s}\n" + "=" * 72)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def logN(x, m, v):
    return -0.5 * math.log(2 * math.pi * v) - (x - m) ** 2 / (2 * v)


def kl_g(mq, vq, mp, vp):
    """KL(N(mq, vq) || N(mp, vp)) en 1-D, con varianzas."""
    return 0.5 * math.log(vp / vq) + (vq + (mq - mp) ** 2) / (2 * vp) - 0.5


# ---------------------------------------------------------------------------
titulo(1, "PETS: incertidumbre aleatoria y epistémica de un conjunto de 5 redes (SUPUESTO)")
# Cada red b predice el cambio de altura del dron en el siguiente paso como N(mu_b, s2_b), en cm.
casos = {
    "cerca de los datos": (np.array([10.0, 12.0, 9.0, 11.0, 13.0]), np.array([4.0, 4.0, 5.0, 3.0, 4.0])),
    "lejos de los datos": (np.array([10.0, 20.0, 0.0, 15.0, 5.0]), np.array([4.0, 4.0, 5.0, 3.0, 4.0])),
}
for nombre, (mu, s2) in casos.items():
    alea = s2.mean()
    epi = ((mu - mu.mean()) ** 2).mean()
    # comprobación por Monte Carlo: muestrear red al azar y luego su gaussiana
    rng = np.random.default_rng(0)
    b = rng.integers(0, 5, 2_000_000)
    x = mu[b] + np.sqrt(s2[b]) * rng.standard_normal(b.size)
    print(f"  {nombre}: medias {mu}, varianzas {s2}")
    print(f"    media del conjunto {mu.mean():.2f} cm; aleatoria = media de varianzas = {alea:.2f} cm²;"
          f" epistémica = varianza de medias = {epi:.2f} cm²; total = {alea+epi:.2f} cm²")
    print(f"    Monte Carlo (2 M muestras): media {x.mean():.3f}, varianza {x.var():.3f}")
# figura: y = 220 - 6 (v + 5); barras de ±sigma de cada red y de ±sigma total
Yp = lin(-5, 25, 220, 40)
for nombre, x0 in (("cerca de los datos", 70), ("lejos de los datos", 310)):
    mu, s2 = casos[nombre]
    tot = math.sqrt(s2.mean() + ((mu - mu.mean()) ** 2).mean())
    print(f"  figura {nombre}: " + "; ".join(f"x {x0+30*i} y {Yp(m):.1f} ± {6*math.sqrt(v):.1f}" for i, (m, v) in enumerate(zip(mu, s2)))
          + f"; total x {x0+155} y {Yp(mu.mean()):.1f} ± {6*tot:.1f}")
print(f"  marcas: 0 -> {Yp(0):.0f}, 10 -> {Yp(10):.0f}, 20 -> {Yp(20):.0f}")

# ---------------------------------------------------------------------------
titulo(2, "Error compuesto escalar: e_{t+1} = L e_t + delta (SUPUESTO delta = 1 cm)")
delta = 1.0
for L in (0.9, 1.0, 1.1):
    e, es = 0.0, []
    for k in range(1, 21):
        e = L * e + delta
        es.append(e)
    cerrada = [(L ** k - 1) / (L - 1) if L != 1 else k for k in range(1, 21)]
    print(f"  L = {L}: e_1..e_5 = {[round(v, 3) for v in es[:5]]}; e_10 = {es[9]:.3f}; e_20 = {es[19]:.3f};"
          f" fórmula cerrada e_5 = {cerrada[4]:.3f}; límite delta/(1-L) = "
          + (f"{delta/(1-L):.1f}" if L < 1 else "no hay"))
# figura: e_k frente a k, k = 0..20
X = lin(0, 20, 60, 520)
Y = lin(0, 60, 250, 30)
for L in (0.9, 1.0, 1.1):
    ks = np.arange(0, 21)
    ek = np.array([(L ** k - 1) / (L - 1) if L != 1 else k for k in ks], float)
    print(f"  path L={L}: {path(ks, ek, X, Y)}")
print(f"  marcas y: 0 -> {Y(0):.1f}, 20 -> {Y(20):.1f}, 40 -> {Y(40):.1f}, 60 -> {Y(60):.1f};"
      f" x: 5 -> {X(5):.1f}, 10 -> {X(10):.1f}, 15 -> {X(15):.1f}, 20 -> {X(20):.1f}")

# ---------------------------------------------------------------------------
titulo(3, "Acoplamiento posición-velocidad: error de aceleración b en un doble integrador")
DT = 0.02          # 50 Hz, el control del x500 del tema
b = 0.5            # SUPUESTO: el modelo se equivoca 0,5 m/s² en la aceleración (≈ 5 % de g)
ep, ev = 0.0, 0.0
filas = []
for k in range(1, 51):
    ep = ep + DT * ev           # la posición integra la velocidad (errónea) del paso anterior
    ev = ev + DT * b            # la velocidad acumula el error de aceleración
    filas.append((k, ev, ep))
for k, ev_, ep_ in filas[:4]:
    print(f"  k = {k}: error de velocidad {ev_*100:.1f} cm/s; error de posición {ep_*100:.2f} cm")
for k in (5, 25, 50):
    ev_, ep_ = filas[k - 1][1], filas[k - 1][2]
    print(f"  k = {k} ({k*DT:.1f} s): error de velocidad {ev_*100:.1f} cm/s; posición {ep_*100:.2f} cm;"
          f" fórmula DT² b k(k-1)/2 = {DT**2*b*k*(k-1)/2*100:.2f} cm; ½ b t² = {0.5*b*(k*DT)**2*100:.2f} cm")
print("  En forma matricial: e_{k+1} = A e_k + (0, DT b), A = [[1, DT], [0, 1]]: el 1 de la diagonal hace que la"
      " velocidad no olvide su error, y el DT fuera de la diagonal lo pasa a la posición.")

# ---------------------------------------------------------------------------
titulo(4, "Cota de variación total en una cadena de 2 estados (SUPUESTO)")
T = np.array([[0.90, 0.10], [0.20, 0.80]])     # filas: estado actual; columnas: siguiente
Th = np.array([[0.85, 0.15], [0.20, 0.80]])
eps = max(0.5 * np.abs(T[s] - Th[s]).sum() for s in range(2))
d = np.array([1.0, 0.0]); dh = d.copy()
print(f"  epsilon = max_s TV(T̂(·|s), T(·|s)) = {eps:.3f}")
for t in range(1, 6):
    d, dh = d @ T, dh @ Th
    tv = 0.5 * np.abs(d - dh).sum()
    print(f"  t = {t}: d_t = {d}, d̂_t = {dh}, TV = {tv:.4f} <= t·eps = {t*eps:.2f}")
for t in (10, 20, 50):
    d = np.array([1.0, 0.0]) @ np.linalg.matrix_power(T, t)
    dh = np.array([1.0, 0.0]) @ np.linalg.matrix_power(Th, t)
    print(f"  t = {t}: TV = {0.5*np.abs(d-dh).sum():.4f} (cota {t*eps:.2f})")
# figura: TV(t) y la cota t·eps para t = 0..20 (eje y hasta 0,5)
Xc = lin(0, 20, 60, 520)
Yc = lin(0, 0.5, 220, 20)
tvs = []
for t in range(21):
    d = np.array([1.0, 0.0]) @ np.linalg.matrix_power(T, t)
    dh = np.array([1.0, 0.0]) @ np.linalg.matrix_power(Th, t)
    tvs.append(0.5 * np.abs(d - dh).sum())
print(f"  path TV: {path(range(21), tvs, Xc, Yc)}")
print(f"  path cota (t = 0..10): {path(range(11), [t*eps for t in range(11)], Xc, Yc)}")
print(f"  marcas y: 0,1 -> {Yc(0.1):.1f}, 0,25 -> {Yc(0.25):.1f}, 0,5 -> {Yc(0.5):.1f}; x 5 -> {Xc(5):.0f}, 10 -> {Xc(10):.0f}, 20 -> {Xc(20):.0f}")
print("  Caso en que la cota casi se alcanza: el modelo cae con prob. eps a un estado absorbente que el mundo nunca visita")
for t in (1, 5, 10, 20):
    print(f"    t = {t}: TV = 1 - (1 - eps)^t = {1-(1-eps)**t:.4f} frente a t·eps = {t*eps:.2f}")
print("  La cota t·eps es el peor caso; esta cadena se mezcla y el TV se estanca (estacionarias 2/3 y 4/7 en A).")

# ---------------------------------------------------------------------------
titulo(5, "Cota del retorno: R_max sum_{t=1}^k gamma^t t eps")
for eps_, gam, Rm in ((0.01, 0.99, 1.0),):
    for k in (1, 5, 15, 50):
        cota = Rm * sum(gam ** t * t * eps_ for t in range(1, k + 1))
        print(f"  eps = {eps_}, gamma = {gam}, R_max = {Rm}, k = {k}: cota = {cota:.4f}"
              f"  (con gamma = 1: eps k(k+1)/2 = {eps_*k*(k+1)/2:.4f})")
print(f"  retorno máximo posible con k = 50: sum_(t=0)^50 0,99^t = {sum(0.99**t for t in range(51)):.2f}")
print("  Desglose para k = 5: " + " + ".join(f"{gam**t*t*eps_:.4f}" for t in range(1, 6)))

# ---------------------------------------------------------------------------
titulo(6, "Jensen: el log de la media es mayor que la media de los log (figura)")
w = np.array([1.0, 4.0])
print(f"  w = {w}: log(media) = log({w.mean():.2f}) = {math.log(w.mean()):.4f};"
      f" media(log) = ({math.log(1):.4f} + {math.log(4):.4f})/2 = {np.log(w).mean():.4f}")
X = lin(0, 5, 50, 330)
Y = lin(-0.5, 1.75, 200, 20)
xs = np.linspace(0.6, 5.0, 45)
print(f"  path log: {path(xs, np.log(xs), X, Y)}")
for v, nm in ((1, "w1"), (4, "w2"), (2.5, "media")):
    print(f"  punto {nm}: x {X(v):.1f}, y(log) {Y(math.log(v)):.1f}")
print(f"  y de media(log) = {Y(np.log(w).mean()):.1f}; y de 0 = {Y(0):.1f}; y de 1 = {Y(1):.1f}")

# ---------------------------------------------------------------------------
titulo(7, "KL entre dos gaussianas 1-D, y su asimetría")
mq, sq, mp, sp = 1.0, 0.5, 0.0, 1.0
k1 = kl_g(mq, sq ** 2, mp, sp ** 2)
k2 = kl_g(mp, sp ** 2, mq, sq ** 2)
print(f"  KL(N(1, 0,5²) || N(0, 1)) = ln(1/0,5) + (0,25 + 1)/2 - 1/2 = {math.log(2):.4f} + {1.25/2:.4f} - 0.5 = {k1:.4f} nat")
print(f"  KL(N(0, 1) || N(1, 0,5²)) = ln(0,5/1) + (1 + 1)/(2·0,25) - 1/2 = {k2:.4f} nat")
rng = np.random.default_rng(1)
z = mq + sq * rng.standard_normal(2_000_000)
mc = np.mean(np.array([logN(v, mq, sq**2) - logN(v, mp, sp**2) for v in z[:200000]]))
print(f"  Monte Carlo de E_q[ln q - ln p] con 200 000 muestras: {mc:.4f}")

# ---------------------------------------------------------------------------
titulo(8, "Cota de evidencia en un modelo gaussiano 1-D: p(z) = N(0, 1), p(o|z) = N(z, 0,5²), o = 1")
vp, vo, o = 1.0, 0.25, 1.0
v_post = 1 / (1 / vp + 1 / vo)
m_post = v_post * o / vo
logp = logN(o, 0, vp + vo)
print(f"  evidencia exacta: p(o) = N(1; 0, 1,25), ln p(o) = {logp:.4f}")
print(f"  posterior exacto: varianza 1/(1 + 4) = {v_post:.3f}, media {v_post:.1f}·4·1 = {m_post:.3f}")


def elbo(mq, vq):
    e2 = (o - mq) ** 2 + vq                     # E_q[(o - z)²]
    rec = -0.5 * math.log(2 * math.pi * vo) - e2 / (2 * vo)
    kl = kl_g(mq, vq, 0.0, vp)
    return rec, kl, rec - kl, e2


for nombre, (mq_, vq_) in {"q = posterior exacto N(0,8; 0,2)": (m_post, v_post),
                           "q = N(0,5; 0,2) (peor media)": (0.5, 0.2),
                           "q = N(1; 0,01) (copia la observación)": (1.0, 0.01),
                           "q = prior N(0; 1) (ignora la observación)": (0.0, 1.0)}.items():
    rec, kl, el, e2 = elbo(mq_, vq_)
    hueco = kl_g(mq_, vq_, m_post, v_post)
    print(f"  {nombre}: E_q[(o-z)²] = {e2:.3f}; reconstrucción {rec:.4f}; KL {kl:.4f}; ELBO {el:.4f};"
          f" ln p(o) - ELBO = {logp-el:.4f} = KL(q || posterior) = {hueco:.4f}")
print(f"  constante de la reconstrucción: -½ ln(2π·0,25) = {-0.5*math.log(2*math.pi*vo):.4f}")

# ---------------------------------------------------------------------------
titulo(9, "KL partido con gradiente detenido (DreamerV3) y KL equilibrado (DreamerV2)")
# q = N(mq, 1), p = N(mp, 1): KL = (mq - mp)²/2; dKL/dmp = -(mq - mp), dKL/dmq = (mq - mp)
mq, mp, lr = 2.0, 0.0, 0.1                    # SUPUESTO
kl = (mq - mp) ** 2 / 2
g = mq - mp
print(f"  q = N({mq}, 1), p = N({mp}, 1): KL = {kl:.2f} nat; |dKL/dmu_p| = |dKL/dmu_q| = {g:.1f}")
for nombre, bd, br in (("DreamerV3 (β_dyn = 1, β_rep = 0,1)", 1.0, 0.1), ("DreamerV2 (α = 0,8)", 0.8, 0.2),
                       ("KL sin partir, peso 1", 1.0, 1.0)):
    mp1 = mp + lr * bd * g
    mq1 = mq - lr * br * g
    valor = kl if nombre.startswith("KL sin") else (bd + br) * kl
    print(f"  {nombre}: valor de la pérdida {valor:.2f}; paso de gradiente (lr {lr}): mu_p {mp} -> {mp1:.2f},"
          f" mu_q {mq} -> {mq1:.2f}; el prior se mueve {bd/br:.0f} veces más")
mq2 = 1.0
kl2 = (mq2 - mp) ** 2 / 2
print(f"  bits libres: con mu_q = {mq2}, KL = {kl2:.2f} nat < 1: max(1, KL) = 1 es constante y su gradiente es 0")
X = lin(-3, 5, 30, 410)
Y = lin(0, 0.42, 150, 20)
zs = np.linspace(-3, 5, 81)
for m, nm in ((0.0, "p"), (2.0, "q")):
    dens = np.exp(-(zs - m) ** 2 / 2) / math.sqrt(2 * math.pi)
    print(f"  path {nm}: {path(zs, dens, X, Y)}")
print(f"  x de 0: {X(0):.1f}; x de 0,2: {X(0.2):.1f}; x de 1,98: {X(1.98):.1f}; x de 2: {X(2):.1f}; y del pico: {Y(1/math.sqrt(2*math.pi)):.1f}")

# ---------------------------------------------------------------------------
titulo(10, "Straight-through de DreamerV2: una variable categórica de 3 clases (SUPUESTO)")
logits = np.array([1.0, 0.0, -1.0])
probs = np.exp(logits) / np.exp(logits).sum()
muestra = np.array([1.0, 0.0, 0.0])
print(f"  logits {logits} -> probs {probs}; muestra one-hot {muestra}")
print(f"  hacia delante: muestra + probs - sg(probs) = {muestra + probs - probs} (la muestra)")
print("  hacia atrás: d/dlogits de (muestra + probs - sg(probs)) = d probs/dlogits (jacobiano del softmax):")
Jsm = np.diag(probs) - np.outer(probs, probs)
print(Jsm)
print("  DreamerV2: 32 variables de 32 clases -> vector de 1024 con 32 unos")

# ---------------------------------------------------------------------------
titulo(11, "symlog y symexp de DreamerV3; normalización del retorno por max(1, S)")
for x in (0.01, 1.0, 1000.0, -1000.0):
    y = math.copysign(math.log(abs(x) + 1), x)
    print(f"  symlog({x}) = {y:.4f}; symexp de vuelta = {math.copysign(math.exp(abs(y)) - 1, y):.4f}")
for p5, p95 in ((2.0, 52.0), (0.1, 0.5)):
    S = p95 - p5
    print(f"  percentiles 5 y 95 del retorno: {p5} y {p95}: S = {S:.1f}; se divide por max(1, S) = {max(1, S):.1f}")

# ---------------------------------------------------------------------------
titulo(12, "Retorno lambda con 3 pasos imaginados: definición y recursión (SUPUESTO)")
gam, lam, H = 0.9, 0.5, 3
r = [1.0, 0.0, 2.0]            # r_0, r_1, r_2
v = {1: 5.0, 2: 4.0, 3: 3.0}   # crítico en s_1, s_2, s_3


def VN(tau, k):
    h = min(tau + k, H)
    return sum(gam ** (n - tau) * r[n] for n in range(tau, h)) + gam ** (h - tau) * v[h]


V = {n: VN(0, n) for n in (1, 2, 3)}
for n in (1, 2, 3):
    print(f"  V^{n}_N(s_0) = {V[n]:.4f}")
pesos = [(1 - lam) * lam ** (n - 1) for n in (1, 2)] + [lam ** (H - 1)]
Vl = sum(p * V[n] for p, n in zip(pesos, (1, 2, 3)))
print(f"  pesos (1-λ), (1-λ)λ, λ²: {pesos} (suman {sum(pesos):.2f}); V_λ(s_0) por la definición = {Vl:.5f}")
Vr = v[3]
print(f"  recursión: V_λ(s_3) = v(s_3) = {Vr}")
for tau in (2, 1, 0):
    Vr = r[tau] + gam * ((1 - lam) * v[tau + 1] + lam * Vr)
    print(f"  V_λ(s_{tau}) = r_{tau} + γ((1-λ)v(s_{tau+1}) + λ V_λ(s_{tau+1})) = {Vr:.5f}")
print(f"  coinciden: {abs(Vr - Vl) < 1e-12}")
print("  comprobación aleatoria con H = 15, γ = 0,99, λ = 0,95:")
rng = np.random.default_rng(3)
H2 = 15
r2 = rng.normal(size=H2); v2 = rng.normal(size=H2 + 1)


def VN2(tau, k):
    h = min(tau + k, H2)
    return sum(0.99 ** (n - tau) * r2[n] for n in range(tau, h)) + 0.99 ** (h - tau) * v2[h]


lam2 = 0.95
defi = (1 - lam2) * sum(lam2 ** (n - 1) * VN2(0, n) for n in range(1, H2)) + lam2 ** (H2 - 1) * VN2(0, H2)
rec = v2[H2]
for tau in range(H2 - 1, -1, -1):
    rec = r2[tau] + 0.99 * ((1 - lam2) * v2[tau + 1] + lam2 * rec)
print(f"    definición {defi:.6f}, recursión {rec:.6f}")
pes = [(1 - lam2) * lam2 ** (n - 1) for n in range(1, H2)] + [lam2 ** (H2 - 1)]
print(f"  pesos de Dreamer (λ = 0,95, H = 15): V^1 {pes[0]:.3f}, V^2 {pes[1]:.4f}, V^14 {pes[13]:.4f}, V^15 {pes[14]:.3f}; suma {sum(pes):.3f}")

# ---------------------------------------------------------------------------
titulo(13, "TD-MPC: pesos temporales, objetivo de TD y el colapso que evitan recompensa y valor")
lam_t, Ht = 0.5, 5
print(f"  pesos λ^(i-t) con λ = {lam_t}, i = t..t+{Ht}: {[lam_t**i for i in range(Ht+1)]}")
ri, g_, Qn, Q = 1.0, 0.99, 10.0, 10.0       # SUPUESTO
y = ri + g_ * Qn
print(f"  objetivo TD: r + γ Q⁻ = {ri} + {g_}·{Qn} = {y:.2f}; con Q = {Q}: error² = {(Q-y)**2:.4f}")
# colapso: latente constante
rs = np.array([1.0, 0.0])
print(f"  dos estados con recompensas {rs}: si el codificador da el mismo z a los dos, la consistencia vale 0"
      f" pero la mejor predicción de recompensa es {rs.mean():.1f} y su error cuadrático medio {((rs-rs.mean())**2).mean():.2f}:"
      " solo distinguirlos baja esa pérdida.")

# ---------------------------------------------------------------------------
titulo(14, "SimNorm de TD-MPC2 con grupos de V = 4 (el artículo usa V = 8) y τ = 1 (SUPUESTO)")
zz = np.array([2.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, 0.0])
g = zz.reshape(-1, 4)
sn = np.exp(g) / np.exp(g).sum(1, keepdims=True)
print(f"  z = {zz} -> grupos {g.tolist()}")
print(f"  SimNorm: {sn.round(3).tolist()}; cada grupo suma {sn.sum(1)}")
print(f"  norma del latente: antes {np.linalg.norm(zz):.3f}; después {np.linalg.norm(sn):.3f} (siempre entre sqrt(L/V) = {math.sqrt(2/4):.3f} y sqrt(L) = {math.sqrt(2):.3f})")

# ---------------------------------------------------------------------------
titulo(15, "MuZero: objetivo de valor de n pasos z_t (SUPUESTO n = 3)")
u = [1.0, 0.0, 2.0]; nu = 3.0; gm = 0.9
zt = sum(gm ** i * u[i] for i in range(3)) + gm ** 3 * nu
print(f"  z_t = u_(t+1) + γ u_(t+2) + γ² u_(t+3) + γ³ ν_(t+3) = 1 + 0 + 0,81·2 + 0,729·3 = {zt:.3f} (= V^3 del ejemplo 12)")
