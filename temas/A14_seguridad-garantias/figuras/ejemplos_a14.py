"""Ejemplos numéricos y datos de las figuras hechas a mano del tema A14 (seguridad y garantías).

Cada número de un ejemplo de la presentación sale de aquí (sustituye a derivaciones_a14.py, cuyos cuatro ejemplos son
ahora los ejemplos 5, 6, 8 y 9). Las cifras del experimento (200 escenas, filtro CBF con ruido, viento y poca
aceleración) siguen saliendo de figuras_a14.py, que no hace falta relanzar: su salida está en salida.txt.

Ejecutar:  python3 ejemplos_a14.py > salida_ejemplos.txt     (solo numpy, CPU, unos segundos)
"""
import math

import numpy as np

np.set_printoptions(precision=4, suppress=True)


def titulo(s):
    print("\n" + "=" * 78 + "\n" + s + "\n" + "=" * 78)


def lin(d0, d1, p0, p1):
    return lambda x: p0 + (x - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


# ===========================================================================================================
titulo("1. Un paso del filtro con respaldo (monitor predictivo): dron en línea recta hacia un poste")
A, DT = 6.0, 0.1          # frenada máxima (m/s2) y periodo del filtro, grande para que se vea a mano
u_nom = 2.0               # la política pide acelerar 2 m/s2 hacia el poste
for s, v in ((3.0, 4.0), (1.6, 4.0)):
    v1 = v + u_nom * DT
    s1 = s - v * DT - 0.5 * u_nom * DT ** 2
    frenada = v1 ** 2 / (2 * A)
    ok = frenada < s1
    print(f"  s = {s} m al borde, v = {v} m/s, u_nom = {u_nom} m/s2 hacia el poste, paso de {DT} s:")
    print(f"     tras el paso: v' = {v1:.2f} m/s, s' = {s:.2f} - {v * DT:.2f} - {0.5 * u_nom * DT ** 2:.2f} = {s1:.2f} m;"
          f" frenada del respaldo v'^2/(2·{A:.0f}) = {frenada:.2f} m -> {'deja pasar u_nom' if ok else 'INTERVIENE: frenar'}")

# ===========================================================================================================
titulo("2. La función h de un obstáculo circular (r = 1 m, centro en el origen)")
r = 1.0
for dist in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    print(f"  |p - c| = {dist:.1f} m: h = {dist ** 2 - r ** 2:6.2f} m^2, |grad h| = 2|p - c| = {2 * dist:.1f} m")
print("  curvas de nivel h = 0, 1.25, 3, 8 -> radios sqrt(h + r^2) =", [round(math.sqrt(hh + r ** 2), 3) for hh in (0, 1.25, 3, 8)])

# ===========================================================================================================
titulo("3. Derivadas de Lie en los dos modelos de dron")
p, c = np.array([0.0, 0.0]), np.array([2.0, 1.0])
d = p - c
print(f"  integrador simple (u = velocidad): p = {p.tolist()}, c = {c.tolist()}: grad h = 2(p - c) = {(2 * d).tolist()}; "
      f"f = 0 -> Lf h = 0; g = I -> Lg h = {(2 * d).tolist()}")
p, v, c = np.array([4.0, 0.0]), np.array([3.0, 0.0]), np.array([10.0, 0.0])
d = p - c
print(f"  doble integrador (u = aceleración): x = (p, v), p = {p.tolist()}, v = {v.tolist()}, c = {c.tolist()}: "
      f"grad h = (2d, 0) = ({(2 * d).tolist()}, [0, 0]); f = (v, 0) -> Lf h = 2 d·v = {2 * d @ v:.0f}; g = (0, I) -> Lg h = (0, 0)")

# ===========================================================================================================
titulo("4. Velocidad máxima de acercamiento que permite la CBF (integrador simple, de frente, r = 1 m)")
print("  s_max = alfa (|d|^2 - r^2) / (2 |d|)   (cerca del borde, |d| = r + delta: s_max ~ alfa·delta)")
for alfa in (0.5, 1.0, 2.0):
    fila = [f"{alfa * (D ** 2 - 1) / (2 * D):.2f}" for D in (1.0, 1.1, 1.5, 2.0, 3.0, 4.0)]
    print(f"  alfa = {alfa}: |d| = 1, 1.1, 1.5, 2, 3, 4 m -> s_max =", ", ".join(fila), "m/s")
X, Y = lin(1.0, 4.0, 60, 400), lin(0.0, 4.0, 220, 30)
Ds = np.linspace(1.0, 4.0, 31)
for alfa in (0.5, 1.0, 2.0):
    print(f"  curva alfa = {alfa}:", path(Ds, alfa * (Ds ** 2 - 1) / (2 * Ds), X, Y))

# ===========================================================================================================
titulo("5. Cota exponencial: con hdot >= -alfa h, h(t) >= h(0) exp(-alfa t)")
for alfa in (0.5, 2.0):
    print(f"  alfa = {alfa}, h(0) = 1: h(1 s) >= {math.exp(-alfa):.3f}, h(3 s) >= {math.exp(-3 * alfa):.4f}  (positivo siempre)")
for alfa in (0.5, 1.0, 2.0):
    print(f"  alfa = {alfa}: h baja al 1 % de h(0) como pronto en ln(100)/alfa = {math.log(100) / alfa:.2f} s")
# trayectoria real: integrador simple de frente desde |d| = 3, r = 1, u_nom = 1 m/s hacia el centro, alfa = 1
alfa, dt = 1.0, 0.001
D, t, hs, ts, activo = 3.0, 0.0, [], [], None
h0 = D ** 2 - 1
while t <= 4.0 + 1e-9:
    h = D ** 2 - 1
    smax = alfa * h / (2 * D)
    s = min(1.0, smax)
    if activo is None and s < 1.0:
        activo = (t, D, h)
    if round(t / dt) % 100 == 0:
        hs.append(h)
        ts.append(t)
    D -= s * dt
    t += dt
print(f"  trayectoria (|d|0 = 3, u_nom = 1 m/s, alfa = 1): h(0) = {h0:.0f}; el filtro empieza a actuar en t = {activo[0]:.3f} s, "
      f"|d| = {activo[1]:.3f} m (= 1 + sqrt 2), h = {activo[2]:.3f}")
for tt in (0.0, 0.5, 1.0, 2.0, 3.0, 4.0):
    k = int(round(tt * 10))
    print(f"     t = {tt:.1f} s: h = {hs[k]:.3f}, cota h(0) e^(-t) = {h0 * math.exp(-tt):.3f}")
X, Y = lin(0.0, 4.0, 60, 460), lin(0.0, 8.0, 210, 30)
tg = np.linspace(0, 4, 41)
for a_ in (0.5, 1.0, 2.0):
    print(f"  curva cota alfa = {a_}:", path(tg, h0 * np.exp(-a_ * tg), X, Y))
print("  curva trayectoria:", path(ts, hs, X, Y))

# ===========================================================================================================
titulo("6. CBF-QP con una restricción: solución cerrada y geometría (dron mandado en velocidad)")
p, c, r, alfa = np.array([0.0, 0.0]), np.array([2.0, 1.0]), 1.0, 1.0
d = p - c
h = d @ d - r ** 2
gh = 2 * d
u_nom = np.array([3.0, 0.0])
print(f"  escena: p = {p.tolist()}, poste c = {c.tolist()}, r = {r}, alfa = {alfa}, u_nom = {u_nom.tolist()} m/s")
print(f"  h = {h:.0f}, grad h = {gh.tolist()}, hdot con u_nom = {gh @ u_nom:.0f}, -alfa h = {-alfa * h:.0f} -> viola la condición")
print(f"  restricción alfa h + grad h·u >= 0:  {alfa * h:.0f} + ({gh[0]:.0f}) u_x + ({gh[1]:.0f}) u_y >= 0;  dividida por 2: 2 - 2 u_x - u_y >= 0")
a, b = 2.0, np.array([-2.0, -1.0])     # la misma restricción dividida por 2 (no cambia la solución)
viol = a + b @ u_nom
lam = max(0.0, -viol) / (b @ b)
u = u_nom + lam * b
print(f"  a = {a}, b = {b.tolist()}: a + b·u_nom = {viol:.2f} (< 0: inseguro); |b|^2 = {b @ b:.0f}")
print(f"  lambda = max(0, -(a + b·u_nom)) / |b|^2 = {lam:.3f}; u* = u_nom + lambda b = {np.round(u, 3).tolist()}; a + b·u* = {a + b @ u:.3f}")
print(f"  con la restricción sin dividir: a = 4, b = (-4, -2): lambda = 8/20 = {8 / 20:.2f}, u* = {(u_nom + 0.4 * np.array([-4, -2])).tolist()} (la misma)")
print(f"  u_nom - u* = {np.round(u_nom - u, 3).tolist()} = 0.8·(2, 1): perpendicular a la recta frontera; |u_nom - u*| = {np.linalg.norm(u_nom - u):.3f}")
print(f"  hdot con u* = {gh @ u:.2f} = -alfa h: justo en el límite")
# figura del plano de entradas, misma escala en los dos ejes (60 px por m/s): u_x de -1 a 4 -> 40..340, u_y de 2.5 a -2.5 -> 20..320
X, Y = lin(-1.0, 4.0, 40, 340), lin(-2.5, 2.5, 320, 20)
for nombre, (ux, uy) in (("u_nom", u_nom), ("u*", u), ("origen", (0, 0))):
    print(f"  px plano {nombre}: {X(ux):.1f},{Y(uy):.1f}")
for uy in (-2.5, 2.5):
    print(f"  px plano recta 2 u_x + u_y = 2 en u_y = {uy}: u_x = {(2 - uy) / 2:.2f} -> {X((2 - uy) / 2):.1f},{Y(uy):.1f}")
# escena vista desde arriba: x -> 80 + 70 x, y -> 190 - 70 y; velocidades a 50 px por m/s
Xe, Ye = (lambda x: 80 + 70 * x), (lambda y: 190 - 70 * y)
print(f"  px escena: dron {Xe(0):.0f},{Ye(0):.0f}; poste {Xe(2):.0f},{Ye(1):.0f} radio {70 * r:.0f}; "
      f"punta u_nom {Xe(0) + 50 * u_nom[0]:.0f},{Ye(0) - 50 * u_nom[1]:.0f}; punta u* {Xe(0) + 50 * u[0]:.0f},{Ye(0) - 50 * u[1]:.0f}; "
      f"-grad h unitario x 40 px: {Xe(0) + 40 * gh[0] / np.linalg.norm(gh):.1f},{Ye(0) - 40 * gh[1] / np.linalg.norm(gh):.1f}")

# ===========================================================================================================
titulo("7. Varias restricciones: el método de Hildreth (ascenso por coordenadas en el dual)")
G = np.array([[2.0, 1.0], [0.0, -1.0]])        # G u <= g: 2 u_x + u_y <= 2 (obstáculo) y -u_y <= 0.5 (límite |u_y| <= 0.5)
g = np.array([2.0, 0.5])
u = u_nom.copy()
lamv = np.zeros(2)
it = [u.copy()]
print(f"  restricciones: 2 u_x + u_y <= 2 (la de la hoja anterior) y u_y >= -0.5 (límite lateral); u_nom = {u_nom.tolist()}")
for barrido in range(4):
    for j in range(2):
        rj = G[j] @ u - g[j]
        nuevo = max(0.0, lamv[j] + rj / (G[j] @ G[j]))
        u = u - (nuevo - lamv[j]) * G[j]
        print(f"   barrido {barrido + 1}, restricción {j + 1}: violación {rj:+.4f} -> lambda_{j + 1} = {nuevo:.4f}, u = {np.round(u, 4).tolist()}")
        lamv[j] = nuevo
        it.append(u.copy())
print(f"  tras 4 barridos: u = {np.round(u, 4).tolist()}, lambda = {np.round(lamv, 4).tolist()}")
# solución exacta con las dos restricciones activas: G u = g y u_nom - u = G^T lambda
u_ex = np.linalg.solve(G, g)
lam_ex = np.linalg.solve(G.T, u_nom - u_ex)
print(f"  solución exacta (las dos activas): u* = {np.round(u_ex, 4).tolist()}, lambda = {np.round(lam_ex, 4).tolist()} (ambas >= 0: KKT)")
print(f"  KKT: u_nom - u* = {np.round(u_nom - u, 4).tolist()} = lambda_1 (2, 1) + lambda_2 (0, -1) = {np.round(lamv[0] * G[0] + lamv[1] * G[1], 4).tolist()}")
# figura ampliada, 100 px por m/s en los dos ejes: u_x 0..3.5 -> 30..380, u_y 1..-1.5 -> 20..270
X, Y = lin(0.0, 3.5, 30, 380), lin(1.0, -1.5, 20, 270)
print("  px iteraciones:", " ".join(f"{X(a_):.1f},{Y(b_):.1f}" for a_, b_ in it[:7]), f"| final {X(u[0]):.1f},{Y(u[1]):.1f}")
for uy in (1.0, -1.5):
    print(f"  px recta 1 en u_y = {uy}: {X((2 - uy) / 2):.1f},{Y(uy):.1f}")
print(f"  px recta 2 (u_y = -0.5): y = {Y(-0.5):.1f}; corte de las dos rectas {X(1.25):.1f},{Y(-0.5):.1f}")

# ===========================================================================================================
titulo("8. Grado relativo 2: el dron que manda aceleraciones (doble integrador) frente a un poste")
c, rs = np.array([10.0, 0.0]), 1.3
p, v = np.array([4.0, 0.0]), np.array([3.0, 0.0])
d = p - c
h = d @ d - rs ** 2
hd = 2 * d @ v
print(f"  p = {p.tolist()}, v = {v.tolist()} m/s, poste {c.tolist()}, r = {rs}: h = {h:.2f}, hdot = 2 d·v = {hd:.2f}")
print(f"  orden 1 (todo o nada): la condición hdot >= -alfa h no contiene u:")
for alfa in (1.0, 2.0):
    print(f"     alfa = {alfa}: hdot = {hd:.2f} frente a -alfa h = {-alfa * h:.2f} -> {'se cumple sola' if hd >= -alfa * h else 'falla y ninguna u la arregla en este instante'}")
for alfa in (1.0, 2.0):
    rhs = 2 * v @ v + 2 * alfa * hd + alfa ** 2 * h
    ux_max = rhs / (-2 * d[0])
    psi1 = hd + alfa * h
    print(f"  orden 2, alfa = {alfa}: psi_1 = hdot + alfa h = {psi1:.2f}; 2|v|^2 + 2 alfa hdot + alfa^2 h = {2 * v @ v:.0f} + {2 * alfa * hd:.0f} + {alfa ** 2 * h:.2f} = {rhs:.2f};"
          f" 2 d·u = {2 * d[0]:.0f} u_x -> u_x <= {ux_max:.2f} m/s2")
dist_borde = abs(d[0]) - rs
print(f"  distancia al borde {dist_borde:.1f} m; frenada física a 6 m/s2 desde 3 m/s: {9 / 12:.2f} m")

# ===========================================================================================================
titulo("8b. Plano de fases de grado relativo 2 (pared: h = s, distancia; v = velocidad de acercamiento)")
print("  hdot = -v, hddot = -u (u = aceleración hacia la pared); psi_1 = -v + alfa s; condición: u <= alfa^2 s - 2 alfa v")
X, Y = lin(0.0, 10.0, 60, 470), lin(0.0, 7.0, 250, 20)
for alfa, amax in ((1.0, 6.0), (2.0, 6.0), (2.0, 2.0)):
    s, vv, dt = 10.0, 0.0, 0.001
    S, V = [s], [vv]
    vmax_alc, imposibles = 0.0, 0
    for k in range(12000):
        un = 3.0
        lim = alfa ** 2 * s - 2 * alfa * vv
        u_ = max(-amax, min(un, lim, amax))
        imposibles += lim < -amax
        vv += u_ * dt
        s -= vv * dt
        vmax_alc = max(vmax_alc, vv)
        if k % 100 == 99:
            S.append(s)
            V.append(vv)
    print(f"  alfa = {alfa}, a_max = {amax}: tras 12 s, s = {s:.4f} m, v = {vv:.4f} m/s; velocidad máxima {vmax_alc:.2f} m/s; s mínimo {min(S):.4f}; pasos imposibles {imposibles}")
    print(f"  trayectoria alfa = {alfa}, a_max = {amax}:", path(S, V, X, Y))
for alfa in (1.0, 2.0):
    print(f"  recta psi_1 = 0 (v = alfa s), alfa = {alfa}: px {X(0):.1f},{Y(0):.1f} a {X(min(10, 7 / alfa)):.1f},{Y(alfa * min(10, 7 / alfa)):.1f}")
# figura de la actuación limitada (s de -1.5 a 10 -> 50..470; v de 0 a 6 -> 250..20)
Xa, Ya = lin(-1.5, 10.0, 50, 470), lin(0.0, 6.0, 250, 20)
alfa, amax = 2.0, 2.0
s, vv, dt = 10.0, 0.0, 0.001
S, V, smin = [s], [vv], (s, 0)
for k in range(12000):
    lim = alfa ** 2 * s - 2 * alfa * vv
    u_ = max(-amax, min(3.0, lim, amax))
    vv += u_ * dt
    s -= vv * dt
    if s < smin[0]:
        smin = (s, vv)
    if k % 100 == 99 and vv >= 0:
        S.append(s)
        V.append(vv)
print(f"  actuación limitada alfa = 2, a_max = 2: s mínimo {smin[0]:.3f} m")
print("  trayectoria actuación:", path(S, V, Xa, Ya))
sa = np.linspace(0, 10, 81)
va = np.sqrt(2 * 2.0 * sa)
m = va <= 6.0
print("  parábola actuación a_max = 2:", path(sa[m], va[m], Xa, Ya))
print(f"  px actuación: origen {Xa(0):.1f},{Ya(0):.1f}; recta v = 2 s hasta {Xa(3):.1f},{Ya(6):.1f}; cruce (1, 2) {Xa(1):.1f},{Ya(2):.1f}; s = 10: {Xa(10):.1f}")
ss = np.linspace(0, 10, 41)
for amax in (2.0, 6.0):
    vv = np.sqrt(2 * amax * ss)
    m = vv <= 7.0
    print(f"  parábola de frenada v = sqrt(2·{amax:.0f}·s):", path(ss[m], vv[m], X, Y))

# ===========================================================================================================
titulo("9. Muestreo: la condición solo se impone en t_k (integrador simple contra una pared, h = distancia)")
print("  u_k = alfa h_k (lo máximo que permite la condición) mantenida durante dt: h_{k+1} = h_k - u_k dt = (1 - alfa dt) h_k")
dtm = 0.05
for alfa in (2.0, 10.0, 25.0):
    f = 1 - alfa * dtm
    seq = [1.0 * f ** k for k in range(6)]
    print(f"  dt = {dtm} s, alfa = {alfa}: factor 1 - alfa dt = {f:+.2f}; h_k =", ", ".join(f"{x:+.3f}" for x in seq))
for v_, dt in ((6.0, 0.01), (6.0, 0.05)):
    print(f"  a {v_} m/s con paso de {dt * 1e3:.0f} ms, en un paso se recorren {v_ * dt * 100:.0f} cm")
X, Y = lin(0.0, 5.0, 60, 460), lin(-0.4, 1.0, 200, 30)
for alfa in (10.0, 25.0):
    f = 1 - alfa * dtm
    pts = []
    for k in range(5):
        hk = f ** k
        pts += [(k, hk), (k + 1, hk * f)]
    print(f"  px alfa = {alfa}:", " ".join(f"{X(a_):.1f},{Y(b_):.1f}" for a_, b_ in pts))
print(f"  px eje h = 0: y = {Y(0):.1f}; h = 1: y = {Y(1):.1f}")

# ===========================================================================================================
titulo("10. Ruido de estimación: el filtro cree estar fuera y está dentro")
sig = 0.3
for marg in (0.2, 0.6):
    print(f"  sigma = {sig} m, estimado a {marg} m del borde: P(el real está dentro) = 1 - Phi({marg / sig:.2f}) = {1 - Phi(marg / sig):.3f}")
Xg, Yg = lin(-1.0, 1.4, 40, 460), lin(0.0, 1.4, 200, 40)
xs = np.linspace(-1.0, 1.4, 97)
for mu in (0.2, 0.6):
    pdf = np.exp(-(xs - mu) ** 2 / (2 * sig ** 2)) / (sig * math.sqrt(2 * math.pi))
    print(f"  gaussiana mu = {mu}:", path(xs, pdf, Xg, Yg))
    m = xs <= 0
    print(f"  cola mu = {mu}:", path(xs[m], pdf[m], Xg, Yg) + f" L{Xg(0):.1f},{Yg(0):.1f} L{Xg(-1):.1f},{Yg(0):.1f} Z")
print(f"  px ruido: borde x = {Xg(0):.1f}; mu 0.2 x = {Xg(0.2):.1f}; mu 0.6 x = {Xg(0.6):.1f}; base y = {Yg(0):.1f}; pico y = {Yg(1 / (sig * math.sqrt(2 * math.pi))):.1f}")
print(f"  inflar 2 sigma = {2 * sig} m: si el estimado está justo en el borde inflado, P(dentro) = 1 - Phi(2) = {1 - Phi(2):.3f}; P(|N| < 2 sigma) = {2 * Phi(2) - 1:.3f}")
rs, eps = 1.3, 0.6
print(f"  desigualdad triangular: |p - c| >= |p_hat - c| - |p - p_hat|; con r = {rs}, eps = {eps}: exigir |p_hat - c| >= {rs + eps:.1f} -> |p - c| >= {rs + eps - eps:.1f}")
# margen de Lipschitz (estilo Dean et al.) para el integrador simple: cond(p, u) = 2 d·u + alfa (|d|^2 - r^2)
uu, dd, al, ep = 1.0, 2.0, 1.0, 0.3
L = 2 * uu + 2 * al * dd
print(f"  Lipschitz: grad_p cond = 2u + 2 alfa d; con |u| = {uu}, |d| = {dd}, alfa = {al}: L = 2|u| + 2 alfa |d| = {L:.0f}; eps = {ep}: margen L eps = {L * ep:.1f}"
      f" (frente a alfa h = {al * (dd ** 2 - 1):.0f} con r = 1)")

# ===========================================================================================================
titulo("11. Viento: el término que el filtro no ve y su peor caso")
p, v, c, rs = np.array([4.0, 0.0]), np.array([3.0, 0.0]), np.array([10.0, 0.0]), 1.3
d = p - c
h = d @ d - rs ** 2
hd = 2 * d @ v
alfa = 2.0
rhs = 2 * v @ v + 2 * alfa * hd + alfa ** 2 * h
for wm in (0.0, 1.0, 2.5):
    lim = (rhs - 2 * np.linalg.norm(d) * wm) / (-2 * d[0])
    print(f"  ejemplo de la parte 3, alfa = 2, w_max = {wm}: 2|d| w_max = {2 * np.linalg.norm(d) * wm:.1f}; u_x <= {lim:.3f} m/s2")
print(f"  |d| = 2 m, w_max = 1: 2 d·w >= -2|d| w_max = {-2 * 2 * 1.0:.0f}")

# ===========================================================================================================
titulo("12. Retardo: lo que se mueve el dron mientras la orden viaja")
for v_, tau in ((6.0, 0.05), (3.0, 0.05), (6.0, 0.1)):
    print(f"  v = {v_} m/s, retardo {tau * 1e3:.0f} ms: el dron recorre {v_ * tau * 100:.0f} cm sin que el filtro lo sepa")
tau, w = 0.05, 1.0
print(f"  predicción con las órdenes ya enviadas: exacta sin perturbación; con viento desconocido w = {w} m/s2: error de posición "
      f"w tau^2 / 2 = {0.5 * w * tau ** 2 * 1000:.2f} mm, de velocidad w tau = {w * tau * 100:.0f} cm/s")
# predicción paso a paso como en figuras_a14.py (k_ret pasos de 10 ms)
DTp, k = 0.01, 5
pe, ve = 0.0, 4.0
us = [-2.0] * k
for j in range(k):
    ve = ve + us[j] * DTp
    pe = pe + ve * DTp
print(f"  predicción a mano: v = 4 m/s, 5 órdenes de -2 m/s2 en vuelo (50 ms): v_pred = {ve:.2f} m/s, avance predicho {pe * 100:.1f} cm (a velocidad constante serían 20 cm)")

# ===========================================================================================================
titulo("13. Actuación limitada: la CBF de orden 2 frente a la frenada real (pared)")
print("  zona que deja la CBF: v <= alfa s (psi_1 >= 0); zona de la que se puede frenar: v <= sqrt(2 a_max s)")
for alfa, amax in ((2.0, 6.0), (2.0, 2.0), (1.0, 2.0)):
    s_c, v_c = 2 * amax / alfa ** 2, 2 * amax / alfa
    print(f"  alfa = {alfa}, a_max = {amax}: se cruzan en s = 2 a_max / alfa^2 = {s_c:.2f} m, v = {v_c:.2f} m/s; "
          f"sobre la recta, la condición pide frenar alfa^2 s, que supera a_max a partir de s = a_max/alfa^2 = {amax / alfa ** 2:.2f} m")
alfa, amax, s, vv = 2.0, 2.0, 1.0, 2.0
print(f"  ejemplo: alfa = 2, a_max = 2, s = 1 m, v = 2 m/s: la condición pide u <= alfa^2 s - 2 alfa v = {alfa ** 2 * s - 2 * alfa * vv:.0f} (frenar 4 m/s2) y solo hay 2;"
      f" frenando 2 m/s2 se necesitan v^2/(2 a) = {vv ** 2 / (2 * amax):.1f} m: llega justo a la pared")

# ===========================================================================================================
titulo("14. Alcanzabilidad: la función de valor de la pared, analítica y con la ecuación de Bellman de la seguridad")
a = 6.0
print("  g(s, v) = s (fallo si s <= 0). Mejor control: frenar a tope. V(s, v) = min_t s(t) = s - v^2 / (2 a) si v > 0; s si v <= 0")
for s, v in ((2.0, 4.0), (2.0, 6.0), (3.0, 6.0)):
    print(f"  s = {s} m, v = {v} m/s, a = {a}: V = {s} - {v ** 2 / (2 * a):.3f} = {s - v ** 2 / (2 * a):+.3f} -> {'seguro' if s - v ** 2 / (2 * a) > 1e-9 else ('justo en la frontera: frenando a tope roza la pared' if abs(s - v ** 2 / (2 * a)) < 1e-9 else 'en el BRT: choque inevitable')}")
for T in (0.25, 0.5, 1.0):
    print(f"  horizonte T = {T} s: frontera del BRT: v = sqrt(2 a s) para s <= a T^2/2 = {a * T * T / 2:.3f} m, después v = (s + a T^2/2)/T")
X, Y = lin(0.0, 4.0, 60, 470), lin(0.0, 8.0, 250, 20)
ss = np.linspace(0, 4, 81)
for T in (0.25, 0.5, 1.0, None):
    if T is None:
        vb = np.sqrt(2 * a * ss)
    else:
        vb = np.where(ss <= a * T * T / 2, np.sqrt(2 * a * ss), (ss + a * T * T / 2) / T)
    m = vb <= 8.0
    print(f"  frontera T = {T}:", path(ss[m], vb[m], X, Y))
# iteración de valor en una rejilla 2D: V_k = min(g, max_u V_{k+1}(f(x, u))), con interpolación bilineal
sg, vg = np.linspace(-3, 8, 221), np.linspace(-2, 8, 201)
S_, V_ = np.meshgrid(sg, vg, indexing="ij")
gfun = S_.copy()
Vk = gfun.copy()
dtv = 0.01


def interp(Vm, s, v):
    i = np.clip((s - sg[0]) / (sg[1] - sg[0]), 0, len(sg) - 1.001)
    j = np.clip((v - vg[0]) / (vg[1] - vg[0]), 0, len(vg) - 1.001)
    i0, j0 = np.floor(i).astype(int), np.floor(j).astype(int)
    fi, fj = i - i0, j - j0
    return (Vm[i0, j0] * (1 - fi) * (1 - fj) + Vm[i0 + 1, j0] * fi * (1 - fj) + Vm[i0, j0 + 1] * (1 - fi) * fj + Vm[i0 + 1, j0 + 1] * fi * fj)


for k in range(300):                       # 3 s de horizonte
    mejor = None
    for uu in (-a, 0.0, a):                # u: aceleración hacia la pared; frenar es u = -a
        vn = V_ + uu * dtv
        sn = S_ - V_ * dtv - 0.5 * uu * dtv ** 2
        val = interp(Vk, sn, vn)
        mejor = val if mejor is None else np.maximum(mejor, val)
    Vk = np.minimum(gfun, mejor)
for s, v in ((2.0, 4.0), (2.0, 6.0), (3.0, 6.0)):
    print(f"  rejilla 221 x 201, dt = 0.01 s, 3 s de horizonte: V({s}, {v}) = {interp(Vk, np.array(s), np.array(v)):+.3f} (analítico {s - v * v / (2 * a):+.3f})")
print("  coste de una rejilla con 100 puntos por eje (float32, 4 bytes):")
for n in (2, 4, 6, 10, 12):
    pts = 100.0 ** n
    print(f"     n = {n:2d}: 100^{n} = {pts:.0e} puntos, {pts * 4:.0e} bytes")

# ===========================================================================================================
titulo("15. Certificados aprendidos: la garantía entre muestras (argumento de Lipschitz)")
for L, sep in ((10.0, 0.1), (10.0, 0.02), (50.0, 0.1)):
    tau = sep / 2
    print(f"  L = {L:.0f}, separación entre puntos 2 tau = {sep} -> la condición debe cumplirse con margen L tau = {L * tau:.2f} en cada punto")
# juguete 1D: c(x) = -0.8 + 0.3 sin(3x) en [0, 3] (c <= 0 es "se cumple"); L = 0.9; puntos cada 0.5 -> tau = 0.25
xs = np.linspace(0, 3, 121)
cx = -0.8 + 0.3 * np.sin(3 * xs)
Lj, tau = 0.9, 0.25
xj = np.arange(0, 3.01, 0.5)
cj = -0.8 + 0.3 * np.sin(3 * xj)
print(f"  juguete: c(x_j) = {np.round(cj, 3).tolist()}; máximo {cj.max():.3f} <= -L tau = {-Lj * tau:.3f}: {'garantizado entre puntos' if cj.max() <= -Lj * tau else 'no basta'}; máximo real de c = {cx.max():.3f}")
Xl, Yl = lin(0.0, 3.0, 50, 450), lin(-1.4, 0.4, 210, 30)
print("  curva c:", path(xs, cx, Xl, Yl))
tiendas = []
for x0, c0 in zip(xj, cj):
    a0, b0 = max(0, x0 - tau), min(3, x0 + tau)
    tiendas.append(f"M{Xl(a0):.1f},{Yl(c0 + Lj * (x0 - a0)):.1f} L{Xl(x0):.1f},{Yl(c0):.1f} L{Xl(b0):.1f},{Yl(c0 + Lj * (b0 - x0)):.1f}")
print("  tiendas:", " ".join(tiendas))
print("  px puntos:", " ".join(f"{Xl(a_):.1f},{Yl(b_):.1f}" for a_, b_ in zip(xj, cj)), f"| c = 0 en y = {Yl(0):.1f}; c = -L tau en y = {Yl(-Lj * tau):.1f}")
for n in (2, 4, 6):
    print(f"  rejilla de [-5, 5]^n con separación 0,1: 100^{n} = {100.0 ** n:.0e} puntos que verificar")

# ===========================================================================================================
titulo("16. Experimento: el primer paso de la política nominal")
META, KP, KD, AMAX = np.array([28.0, 10.0]), 2.0, 2.2, 6.0
p0, v0 = np.array([2.0, 10.0]), np.array([0.0, 0.0])
un = KP * (META - p0) - KD * v0
print(f"  p0 = {p0.tolist()}, v0 = 0, meta = {META.tolist()}: Kp (meta - p) - Kd v = {un.tolist()} -> saturada a {np.clip(un, -AMAX, AMAX).tolist()} m/s2")
for p1 in (np.array([20.0, 10.0]), np.array([24.0, 10.0])):
    v1 = np.array([6.0, 0.0])
    un = KP * (META - p1) - KD * v1
    print(f"  p = {p1.tolist()}, v = {v1.tolist()}: Kp (meta - p) - Kd v = {np.round(un, 2).tolist()} m/s2 ({'aún acelera' if un[0] > 0 else 'ya frena'})")
print("  restricciones por paso: 8 de seguridad (7 obstáculos + zona) + 4 de entrada (±u_x, ±u_y) = 12")
