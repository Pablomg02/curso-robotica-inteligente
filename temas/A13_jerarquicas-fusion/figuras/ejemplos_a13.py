"""Ejemplos numéricos y datos de las figuras nuevas del tema A13 (CPU, menos de un segundo).

Uso:  uv run --with numpy python ejemplos_a13.py > salida_ejemplos.txt

Complementa a derivaciones_a13.py (fase de un retardo, Padé, pesos de ACT, mando mantenido) con las cuentas de las
hojas desplegadas: margen de fase con retardo, de dónde sale la cota de cruce, opciones, FeUdal, HIRO, RTC, ACT,
BID, RL residual, el MPPI y el PD del experimento, y los trazados de las figuras nuevas.
Cada ejemplo lleva su número, que es el que citan las hojas.
"""
import math

import numpy as np

deg = math.degrees


def cab(n, titulo):
    print(f"\nEJEMPLO {n}. {titulo}")


# ---------------------------------------------------------------- parte 1: el retardo
cab(1, "Desfase de un retardo de 100 ms a distintas frecuencias")
tau = 0.1
for f in (0.5, 1.0, 2.0, 5.0):
    w = 2 * math.pi * f
    print(f"  f = {f} Hz: omega = {w:.2f} rad/s; desfase omega*tau = {w * tau:.3f} rad = {deg(w * tau):.0f} grados; "
          f"en tiempo, siempre {tau * 1e3:.0f} ms = {tau * f:.2f} periodos")

cab(2, "Margen de fase de un lazo L(s) = (omega_c/s) e^{-s tau} con cruce a 2 Hz")
wc = 2 * math.pi * 2.0
print(f"  omega_c = {wc:.2f} rad/s; |L(i omega_c)| = omega_c/omega_c = 1 (el retardo no cambia la ganancia)")
for tau in (0.0, 0.02, 0.1, 0.25):
    fase = -90 - deg(wc * tau)
    print(f"  tau = {tau * 1e3:.0f} ms: arg L(i omega_c) = -90 - {deg(wc * tau):.1f} = {fase:.1f} grados; "
          f"margen phi_m = 180 + arg = {180 + fase:.1f} grados")
# trazados del diagrama de Bode: omega de 1 a 100 rad/s en x de 70 a 530; ganancia de +30 a -30 dB en y de 20 a 140;
# fase de -60 a -270 grados en y de 175 a 315
X = lambda w: 70 + (math.log10(w) - 0) / 2 * 460
YG = lambda db: 20 + (30 - db) / 60 * 120
YF = lambda f: 175 + (-60 - f) / 210 * 140
ws = np.logspace(0, 2, 41)
print("  path ganancia:", "M" + " L".join(f"{X(w):.1f},{YG(20 * math.log10(wc / w)):.1f}" for w in ws))
for tau in (0.0, 0.02, 0.1):
    pts = [(w, -90 - deg(w * tau)) for w in ws]
    pts = [(w, f) for w, f in pts if f >= -270]
    print(f"  path fase tau = {tau * 1e3:.0f} ms:", "M" + " L".join(f"{X(w):.1f},{YF(f):.1f}" for w, f in pts))
print(f"  x de omega_c = {X(wc):.1f}; y de -180 grados = {YF(-180):.1f}; y de -90 = {YF(-90):.1f}; "
      f"y de -104,4 = {YF(-104.4):.1f}; y de -162 = {YF(-162):.1f}; y de 0 dB = {YG(0):.1f}")

cab(3, "Aproximación de Padé de primer orden de e^{-x}, x = s tau")
for x in (0.1, 0.2, 0.5, 1.0):
    print(f"  x = {x}: e^-x = {math.exp(-x):.4f}; (1 - x/2)/(1 + x/2) = {(1 - x / 2) / (1 + x / 2):.4f}")
print("  series: e^-x = 1 - x + x^2/2 - x^3/6 + ...; (1 - x/2)/(1 + x/2) = 1 - x + x^2/2 - x^3/4 + ... (coinciden hasta x^2)")
tau = 0.1
print(f"  con tau = {tau * 1e3:.0f} ms: cero en s = +2/tau = {2 / tau:.0f} rad/s, polo en s = -2/tau = {-2 / tau:.0f} rad/s")
for w in (1.0, 5.0, 10.0, 20.0, 40.0):
    print(f"  omega = {w:.0f} rad/s: módulo de Padé = 1; fase exacta -omega tau = {-deg(w * tau):.1f} grados; "
          f"Padé -2 arctan(omega tau/2) = {-2 * deg(math.atan(w * tau / 2)):.1f} grados")
# figura: fase (grados) frente a omega tau de 0 a 4 rad, x de 60 a 500, y de 0 a -240 grados en y de 30 a 230
XP = lambda u: 60 + u / 4 * 440
YP = lambda f: 30 + (-f) / 240 * 200
us = np.linspace(0, 4, 41)
print("  path fase exacta:", "M" + " L".join(f"{XP(u):.1f},{YP(-deg(u)):.1f}" for u in us if deg(u) <= 240))
print("  path fase Padé:", "M" + " L".join(f"{XP(u):.1f},{YP(-2 * deg(math.atan(u / 2))):.1f}" for u in us))
print(f"  y de -180 grados = {YP(-180):.1f}; asíntota de Padé -180; x de omega tau = 1 -> {XP(1):.1f}; 2 -> {XP(2):.1f}")

cab(4, "De dónde sale la cota de cruce (Åström y Murray, teorema 14.4 y ejemplo 14.3)")
print("  desigualdad: fase del pasa-todo en el cruce phi_ap(omega_gc) <= pi - phi_m + n_gc pi/2 = phi_ap_max")
for phim, ngc in ((60, -1), (45, -0.5), (30, -1)):
    print(f"  phi_m = {phim} grados, n_gc = {ngc}: phi_ap_max = 180 - {phim} + ({ngc})*90 = {180 - phim + ngc * 90:.0f} grados")
print(f"  cero z: phi_ap = 2 arctan(omega/z) <= phi_ap_max  ->  omega_gc <= z tan(phi_ap_max/2)")
for a in (30, 60, 90):
    print(f"    phi_ap_max = {a} grados: omega_gc <= {math.tan(math.radians(a / 2)):.3f} z")
print(f"  con phi_ap_max = 60 grados (pi/3): tan(30) = {math.tan(math.pi / 6):.4f}, que el libro redondea a 0,6")
print(f"  Padé: z = 2/tau -> omega_gc <= 2 tan(pi/6)/tau = {2 * math.tan(math.pi / 6):.3f}/tau (con el 0,6 redondeado, 1,2/tau)")
print(f"  retardo exacto: phi_ap = omega tau <= pi/3 -> omega_gc <= {math.pi / 3:.3f}/tau")
print(f"  margen que queda con un lazo tipo integrador (fase -90) en omega_c = 1,2/tau: 90 - {deg(1.2):.1f} = {90 - deg(1.2):.1f} grados;"
      f" en omega_c = (pi/3)/tau: 90 - 60 = 30 grados; con 45 grados (A2), omega_c = (pi/4)/tau = {math.pi / 4:.3f}/tau")
for tau in (0.02, 0.1, 0.25, 0.4, 0.8):
    a, b, c = 1.2 / tau, (math.pi / 3) / tau, (math.pi / 4) / tau
    print(f"  tau = {tau * 1e3:.0f} ms: 1,2/tau = {a:.2f} rad/s ({a / 2 / math.pi:.2f} Hz); (pi/3)/tau = {b:.2f} rad/s "
          f"({b / 2 / math.pi:.2f} Hz); (pi/4)/tau = {c:.2f} rad/s ({c / 2 / math.pi:.2f} Hz)")
# figura: cruce máximo en Hz frente al retardo (escala log en las dos), retardo de 10 a 1000 ms en x de 70 a 510,
# f de 0,1 a 20 Hz en y de 220 a 30
XT = lambda t: 70 + (math.log10(t) - math.log10(0.01)) / 2 * 440
YH = lambda f: 220 - (math.log10(f) - math.log10(0.1)) / math.log10(200) * 190
ts = np.logspace(-2, 0, 21)
for nombre, k in (("1,2", 1.2), ("pi/3", math.pi / 3), ("pi/4", math.pi / 4)):
    print(f"  path cota {nombre}/tau:", "M" + " L".join(f"{XT(t):.1f},{YH(k / t / 2 / math.pi):.1f}" for t in ts))
for t in (0.02, 0.1, 0.25, 0.4, 0.8):
    print(f"    x(tau = {t * 1e3:.0f} ms) = {XT(t):.1f}; y(1,2/tau) = {YH(1.2 / t / 2 / math.pi):.1f}")
for f in (0.1, 1, 10):
    print(f"    y({f} Hz) = {YH(f):.1f}")

# ---------------------------------------------------------------- parte 2: jerarquías
cab(5, "Valor de una opción en el semi-MDP")
g, r, V = 0.9, [-1.0, -1.0, -1.0], 10.0
k = len(r)
ret = sum(g ** i * ri for i, ri in enumerate(r))
print(f"  opción de k = {k} pasos con recompensas {r} y gamma = {g}: sum gamma^i r = {ret:.3f}")
print(f"  al terminar, V(s_t+k) = {V}: Q(s, o) = {ret:.3f} + gamma^k V = {ret:.3f} + {g ** k:.3f}*{V} = {ret + g ** k * V:.3f}")
print(f"  como tres pasos primitivos con la ecuación de Bellman de siempre: "
      f"-1 + 0,9(-1 + 0,9(-1 + 0,9*10)) = {-1 + 0.9 * (-1 + 0.9 * (-1 + 0.9 * 10)):.3f}")

cab(6, "Option-Critic: hacia dónde empuja el gradiente de terminación")
for Q, Vv in ((4.0, 5.0), (6.0, 5.0)):
    A = Q - Vv
    print(f"  Q_Omega(s', w) = {Q}, V_Omega(s') = {Vv}: A = {A:+.1f}; el gradiente de U es -dbeta * A, "
          f"así que subir beta {'mejora' if A < 0 else 'empeora'} el objetivo: {'terminar más' if A < 0 else 'seguir con la opción'}")

cab(7, "FeUdal: similitud coseno y recompensa intrínseca")
dcos = lambda a, b: float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
gdir = np.array([1.0, 0.0])
for mov in ((0.9, 0.3), (0.0, 1.0), (-1.0, 0.0), (3.0, 1.0)):
    print(f"  objetivo g = {gdir.tolist()}, movimiento {list(mov)}: d_cos = {dcos(np.array(mov), gdir):.3f}")
# recompensa intrínseca con c = 3: estados latentes de juguete y los objetivos de los tres pasos anteriores
S = {0: np.array([0.0, 0.0]), 1: np.array([0.3, 0.0]), 2: np.array([0.6, 0.2]), 3: np.array([1.0, 0.2])}
G = {0: np.array([1.0, 0.0]), 1: np.array([1.0, 0.0]), 2: np.array([0.8, 0.6])}
t, c = 3, 3
terms = []
for i in range(1, c + 1):
    d = S[t] - S[t - i]
    terms.append(dcos(d, G[t - i]))
    print(f"  i = {i}: s_3 - s_{t - i} = {np.round(d, 2).tolist()}, g_{t - i} = {G[t - i].tolist()}: d_cos = {terms[-1]:.3f}")
print(f"  r^I_3 = (1/{c}) * suma = {np.mean(terms):.3f}")

cab(8, "HIRO: objetivo relativo, transición del objetivo y recompensa del nivel bajo")
s = [np.array([0.0, 0.0]), np.array([1.0, 0.5]), np.array([2.5, 0.5])]
gg = np.array([5.0, 0.0])
print(f"  s_0 = {s[0].tolist()}, g_0 = {gg.tolist()}: punto deseado s_0 + g_0 = {(s[0] + gg).tolist()}")
for k in range(2):
    rr = -np.linalg.norm(s[k] + gg - s[k + 1])
    gn = s[k] + gg - s[k + 1]
    print(f"  paso {k}: s_{k + 1} = {s[k + 1].tolist()}; r = -||s_{k} + g_{k} - s_{k + 1}|| = {rr:.2f}; "
          f"g_{k + 1} = h(...) = {np.round(gn, 2).tolist()}; punto deseado s_{k + 1} + g_{k + 1} = {(s[k + 1] + gn).tolist()}")
    gg = gn

cab(9, "HIRO: reetiquetar el objetivo con la política baja de ahora (1D, c = 3)")
# política vieja: a = 0,25 g; política de ahora: a = 0,5 g; dinámica s_{t+1} = s_t + a_t; transición g' = s + g - s'
s0, g0, k_viejo, k_nuevo = 0.0, 4.0, 0.25, 0.5
ss, aa, gs = [s0], [], [g0]
for i in range(3):
    a = k_viejo * gs[-1]
    aa.append(a)
    ss.append(ss[-1] + a)
    gs.append(ss[-2] + gs[-1] - ss[-1])
print(f"  datos viejos (política a = {k_viejo} g, objetivo g_t = {g0}): estados {np.round(ss, 4).tolist()}, "
      f"acciones {np.round(aa, 4).tolist()}, objetivos intermedios {np.round(gs[:3], 4).tolist()}")


def logp(gt):
    gi, tot, mus = gt, 0.0, []
    for i in range(3):
        mu = k_nuevo * gi
        mus.append(mu)
        tot += (aa[i] - mu) ** 2
        gi = ss[i] + gi - ss[i + 1]
    return -0.5 * tot, mus


cands = [("original g_t", g0), ("s_t+c - s_t", ss[3] - ss[0]), ("muestra 1", 1.8), ("muestra 2", 3.2)]
for nom, gt in cands:
    lp, mus = logp(gt)
    print(f"  candidato {nom} = {gt:.4f}: la política de ahora haría {np.round(mus, 4).tolist()} -> "
          f"log mu_lo = -1/2 sum (a - mu)^2 = {lp:.4f}")
best = max(cands, key=lambda c_: logp(c_[1])[0])
print(f"  gana {best[0]} = {best[1]:.4f}; el óptimo continuo sería g = {2 * np.mean([aa[0], aa[1] + k_nuevo * 1.0, aa[2] + k_nuevo * 1.75]):.4f}")

# ---------------------------------------------------------------- parte 3 y 4: lento y rápido, asincronía
cab(10, "HiRT, tabla 4: periodo de control y cada cuánto se refresca el VLM")
for f, cada in ((1.97, 1), (13.42, 6)):
    print(f"  {f} Hz: un paso cada {1 / f * 1e3:.0f} ms; VLM cada {cada} pasos = cada {cada / f * 1e3:.0f} ms")

cab(11, "RTC: retardo en pasos y qué parte del trozo se congela")
for dt, delta, H in ((0.02, 0.3, 50), (0.02, 0.08, 16), (0.01, 0.4, 50)):
    d = math.floor(delta / dt)
    print(f"  Delta t = {dt * 1e3:.0f} ms, delta = {delta * 1e3:.0f} ms: d = floor(delta/Delta t) = {d}; con H = {H}, "
          f"s debe cumplir d <= s <= H - d: {d} <= s <= {H - d}" + ("" if d <= H - d else " (imposible: hace falta H >= 2d)"))
print("  figura 3 de Black et al.: d = 4, s = 5, H = 16 -> congeladas a_0..a_3, intermedias a_4..a_10 (H - d - s = 7), nuevas a_11..a_15")

cab(12, "Ensamblado temporal de ACT, a mano, con m = 0,1")
preds = np.array([0.50, 0.44, 0.40, 0.38])
m = 0.1
wi = np.exp(-m * np.arange(4))
for i in range(4):
    print(f"  i = {i}: A_t[{i}] = {preds[i]:.2f}; w_{i} = e^(-{m}*{i}) = {wi[i]:.4f}; w*A = {wi[i] * preds[i]:.4f}")
print(f"  suma de w*A = {np.sum(wi * preds):.4f}; suma de w = {wi.sum():.4f}; a_t = {np.sum(wi * preds) / wi.sum():.4f}")
for m in (0.01, 0.1, 0.5, 2.0):
    wi = np.exp(-m * np.arange(4))
    wn = wi / wi.sum()
    print(f"  m = {m}: pesos normalizados {np.round(wn, 3).tolist()}; a_t = {np.sum(wn * preds):.4f}; "
          f"peso de la más nueva / más antigua = {wn[3] / wn[0]:.3f}")
# barras: 4 pesos normalizados para m = 0,01, 0,1, 0,5; altura máxima 0,5 -> 150 px
for m in (0.01, 0.1, 0.5):
    wn = np.exp(-m * np.arange(4))
    wn = wn / wn.sum()
    print(f"  alturas de barras m = {m}:", [round(float(x) * 300, 1) for x in wn])

cab(13, "BID, a mano, con acciones 1D, trozos de 3 pasos solapados y rho = 0,5 (sin el recorte a K)")
prev = np.array([0.40, 0.42, 0.44])
A = {"c1": np.array([0.41, 0.43, 0.46]), "c2": np.array([0.10, 0.05, 0.00]), "c3": np.array([0.47, 0.52, 0.58])}
Aneg = {"n1": np.array([0.30, 0.25, 0.20]), "n2": np.array([0.60, 0.70, 0.80])}
rho = 0.5
res = {}
for k_, a in A.items():
    LB = float(np.sum(rho ** np.arange(3) * np.abs(a - prev)))
    pos = [np.sum(np.abs(a - b)) for kk, b in A.items() if kk != k_]
    neg = [np.sum(np.abs(a - b)) for b in Aneg.values()]
    LF = float(np.mean(pos) - np.mean(neg))
    res[k_] = (LB, LF)
    print(f"  {k_} = {a.tolist()}: L_B = sum rho^tau |a - a_prev| = {LB:.4f}; distancia media a A+ = {np.mean(pos):.4f}, "
          f"a A- = {np.mean(neg):.4f}; L_F = {LF:.4f}; total = {LB + LF:.4f}")
print(f"  elegido: {min(res, key=lambda k_: sum(res[k_]))}")

# ---------------------------------------------------------------- parte 5 y 6: fusión y experimento
cab(14, "RL residual: lo que un PD deja sin corregir del arrastre del experimento")
cd, kp = 0.05, 4.0
for v in (1.5, 3.0, 6.0):
    arr = cd * v * v
    print(f"  v = {v} m/s: arrastre c v^2 = {arr:.4f} m/s^2; error de posición en régimen del PD solo, c v^2 / k_p = {arr / kp:.3f} m;"
          f" un residuo que aprenda +c|v|v lo anula")

cab(15, "El seguidor PD del experimento sobre un doble integrador")
kp, kd = 4.0, 3.0
wn = math.sqrt(kp)
z = kd / (2 * wn)
print(f"  e'' + k_d e' + k_p e = 0 con k_p = {kp}, k_d = {kd}: omega_n = sqrt(k_p) = {wn:.2f} rad/s ({wn / 2 / math.pi:.2f} Hz), "
      f"zeta = k_d/(2 sqrt(k_p)) = {z:.2f}")
print(f"  raíces: {np.round(np.roots([1, kd, kp]), 3).tolist()}; tiempo característico 1/(zeta omega_n) = {1 / (z * wn):.2f} s")
print(f"  constante de tiempo de la aceleración 0,08 s frente al periodo del PD 0,01 s: 8 pasos de control")

cab(16, "Pesos del MPPI del experimento (lambda = 2)")
lam = 2.0
for dJ in (0.5, 2.0, 10.0, 16.0, 800.0):
    print(f"  coste {dJ} por encima del mejor: peso relativo e^(-{dJ}/{lam}) = {math.exp(-dJ / lam):.3g}")
print(f"  meterse 0,2 m en el margen de una copa durante un paso: 400 * 0,2^2 = {400 * 0.04:.0f}; más de 0,5 m: +400*2 = 800 por paso")
print(f"  seguir a la persona: (d - 1)^2 a d = 3 m: {(3 - 1) ** 2:.0f}; esfuerzo 0,01 |a|^2 a 6 m/s^2: {0.01 * 36:.2f}")
print(f"  horizonte 20 pasos de 0,1 s = 2 s; ruido de exploración sigma = 3 m/s^2, recortado a +-6")

cab(17, "Escena 0 del experimento, para el dibujo (misma semilla que figuras_a13.py)")
T_FIN, DT, V_PER = 20.0, 0.01, 1.5


def escenas(n, sem):  # copia literal de figuras_a13.py
    rng = np.random.default_rng(sem)
    E = []
    for _ in range(n):
        obs = []
        while len(obs) < 8:
            c, r = rng.uniform(4, 36, 2), rng.uniform(1.5, 3.0)
            if all(np.linalg.norm(c - o[:2]) > r + o[2] + 2.0 for o in obs):
                obs.append(np.array([*c, r]))
        obs = np.array(obs)
        while True:
            p0 = rng.uniform(5, 35, 2)
            if np.all(np.linalg.norm(obs[:, :2] - p0, axis=1) > obs[:, 2] + 1.5):
                break
        g0 = p0 + rng.uniform(-3, 3, 2)
        n_t = int(T_FIN / DT) + int(2 / DT) + 2
        th = rng.uniform(0, 2 * np.pi)
        g = np.zeros((n_t, 2))
        vg = np.zeros((n_t, 2))
        g[0], w = g0, 0.0
        for k in range(1, n_t):
            w += -w * DT / 1.0 + rng.normal(0, 0.8) * math.sqrt(DT)
            th += w * DT
            v = V_PER * np.array([math.cos(th), math.sin(th)])
            q = g[k - 1] + v * DT
            for i in range(2):
                if not 2 < q[i] < 38:
                    v[i] = -v[i]
                    th = math.atan2(v[1], v[0])
                    q[i] = g[k - 1][i] + v[i] * DT
            g[k], vg[k] = q, v
        vg[0] = vg[1]
        E.append(dict(obs=obs, p0=p0, g=g, vg=vg))
    return E


e = escenas(1, 0)[0]
S_ = 6.0  # px por metro: 40 m -> 240 px, origen en (20, 20), y hacia arriba
P = lambda x, y: (20 + x * S_, 260 - y * S_)
for o in e["obs"]:
    cx, cy = P(o[0], o[1])
    print(f"  copa en ({o[0]:.1f}, {o[1]:.1f}) m, radio {o[2]:.2f} m -> circle cx={cx:.1f} cy={cy:.1f} r={o[2] * S_:.1f}")
idx = list(range(0, 2001, 50))
print("  path persona 0-20 s:", "M" + " L".join("{:.1f},{:.1f}".format(*P(*e["g"][k])) for k in idx))
print("  persona en 0 s: ({:.1f}, {:.1f}) px; en 20 s: ({:.1f}, {:.1f}) px".format(*P(*e["g"][0]), *P(*e["g"][2000])))
print("  dron en 0 s: ({:.1f}, {:.1f}) m -> ({:.1f}, {:.1f}) px".format(*e["p0"], *P(*e["p0"])))
recorrido = np.sum(np.linalg.norm(np.diff(e["g"][:2001], axis=0), axis=1))
print(f"  la persona recorre {recorrido:.1f} m en 20 s")

cab(18, "Separación de escalas en los sistemas del tema (cocientes de frecuencias)")
print(f"  Loquercio et al.: MPC/red = 100/24,7 = {100 / 24.7:.1f}; bajo nivel/MPC = 2000/100 = {2000 / 100:.0f}")
print(f"  PX4: lazo de velocidad angular a 800 Hz frente a la red de Loquercio a 24,7 Hz: {800 / 24.7:.0f} veces")
for lento, rapido in ((10, 120), (7, 200)):
    print(f"  lento a {lento} Hz y rápido a {rapido} Hz: cada salida del lento sirve {rapido / lento:.1f} pasos del rápido "
          f"({1e3 / lento:.0f} ms)")
