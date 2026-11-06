"""Datos de las figuras y tablas del tema A4a (filtrado bayesiano: Bayes, KF, EKF, UKF, PF y ESKF).

Uso:  python3 figuras_a4a.py > salida.txt      (numpy; menos de un minuto)
"""
import math
import numpy as np

rng0 = np.random.default_rng(0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


print("=" * 70)
print("1. FILTRO DE BAYES EN REJILLA: robot en un pasillo de 20 m con puertas en 3, 7 y 15 m")
xs = np.linspace(0, 20, 401)
dx = xs[1] - xs[0]
puertas = np.array([3.0, 7.0, 15.0])
def p_puerta(x):   # probabilidad de que el sensor diga «puerta» en x
    d = np.min(np.abs(x[:, None] - puertas[None, :]), axis=1)
    return 0.05 + 0.85 * np.exp(-0.5 * (d / 0.4) ** 2)
kern = np.exp(-0.5 * ((np.arange(-80, 81) * dx - 1.0) / 0.15) ** 2); kern /= kern.sum()   # avanza 1 m, ruido 0,15 m
bel = np.ones_like(xs) / 20.0                    # belief uniforme
x_real = 1.0
snap, obs_seq = {}, []
for paso in range(12):
    ve = p_puerta(np.array([x_real]))[0] > 0.5   # medida más probable (determinista, para la figura)
    obs_seq.append(int(ve))
    bel = bel * (p_puerta(xs) if ve else 1 - p_puerta(xs)); bel /= bel.sum() * dx
    snap[paso] = (bel.copy(), x_real)
    bel = np.convolve(bel, kern, mode="same")    # predicción: convolución con el modelo de movimiento
    x_real += 1.0
print(f"  observaciones (1 = puerta) en x = 1, 2, ..., 12: {obs_seq}")
for paso in (2, 6, 11):
    b, xr = snap[paso]
    picos = [round(xs[i], 1) for i in range(1, len(xs) - 1) if b[i] > b[i - 1] and b[i] >= b[i + 1] and b[i] > 0.2 * b.max()]
    print(f"  paso {paso} (x real {xr:.0f} m): picos del belief en {picos}; masa a menos de 0,5 m del real {np.sum(b[np.abs(xs-xr)<0.5])*dx:.2f}")
X1 = lin(0, 20, 40, 500)
for i, paso in enumerate((2, 6, 11)):
    b, xr = snap[paso]
    Y1 = lin(0, b.max() * 1.05, 70 + 75 * i, 14 + 75 * i)
    print(f"path bayes {paso}:", path(xs, b, X1, Y1))
    print(f"  paso {paso}: x real -> {X1(xr):.1f}; base y = {70 + 75*i}")
print("  puertas en x =", [round(float(X1(p)), 1) for p in puertas])

print("=" * 70)
print("2. KALMAN: velocidad constante, posición medida a 10 Hz; consistencia (NEES) con 200 ensayos")
dt = 0.1
F = np.array([[1, dt], [0, 1]]); Hm = np.array([[1.0, 0.0]])
q_real = 0.5                                     # densidad espectral de la aceleración (m^2/s^3)
Qr = q_real * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
Rr = np.array([[0.25]])                          # sigma de 0,5 m
def kf_run(q_filtro, rng, T=100):
    Qf = q_filtro * np.array([[dt**3 / 3, dt**2 / 2], [dt**2 / 2, dt]])
    P = np.diag([1.0, 1.0]); xh = np.array([0.0, 1.0])
    x = xh + rng.multivariate_normal([0, 0], P)      # error inicial con la covarianza inicial del filtro
    nees = []; err = []
    for k in range(T):
        x = F @ x + rng.multivariate_normal([0, 0], Qr)
        y = Hm @ x + rng.normal(0, 0.5, 1)
        xh = F @ xh; P = F @ P @ F.T + Qf
        S = Hm @ P @ Hm.T + Rr; Kg = P @ Hm.T @ np.linalg.inv(S)
        xh = xh + Kg @ (y - Hm @ xh)
        IKH = np.eye(2) - Kg @ Hm
        P = IKH @ P @ IKH.T + Kg @ Rr @ Kg.T          # forma de Joseph
        e = x - xh
        nees.append(e @ np.linalg.solve(P, e)); err.append(e[0])
    return np.array(nees), np.array(err)
res2 = {}
for nombre, qf in (("ajustado", q_real), ("Q 100 veces menor", q_real / 100)):
    rng = np.random.default_rng(1)
    N = np.array([kf_run(qf, rng)[0] for _ in range(200)])
    m = N.mean(0)
    res2[nombre] = m
    print(f"  {nombre:18s}: NEES medio (últimos 50 pasos) {m[50:].mean():6.2f} (ideal 2)")
# intervalo del 95 % de la media de 200 chi2 con 2 grados de libertad: chi2(400)/200
lo, hi = 2 * (1 - 2 / (9 * 400) - 1.96 * math.sqrt(2 / (9 * 400))) ** 3, 2 * (1 - 2 / (9 * 400) + 1.96 * math.sqrt(2 / (9 * 400))) ** 3
print(f"  (intervalo calculado por Wilson-Hilferty: {lo:.2f}–{hi:.2f})")
X2 = lin(0, 10, 60, 500)
Y2 = lin(0, 8, 250, 30)
tt = dt * np.arange(1, 101)
for nombre in res2:
    print(f"path nees {nombre}:", path(tt, np.minimum(res2[nombre], 8), X2, Y2))
print(f"  líneas: y(2) = {Y2(2):.1f}, y(1,72) = {Y2(lo):.1f}, y(2,30) = {Y2(hi):.1f}")

print("=" * 70)
print("3. NO LINEAL: modelo de prueba clásico (x' = x/2 + 25x/(1+x^2) + 8cos(1,2k) + w;  y = x^2/20 + v)")
Qn, Rn, T3 = 10.0, 1.0, 60
f3 = lambda x, k: 0.5 * x + 25 * x / (1 + x**2) + 8 * math.cos(1.2 * k)
df3 = lambda x: 0.5 + 25 * (1 - x**2) / (1 + x**2) ** 2
h3 = lambda x: x**2 / 20
def corre(rng):
    x = 0.1; xs_ = []; ys_ = []
    for k in range(1, T3 + 1):
        x = f3(x, k) + rng.normal(0, math.sqrt(Qn)); xs_.append(x); ys_.append(h3(x) + rng.normal(0, math.sqrt(Rn)))
    return np.array(xs_), np.array(ys_)
def ekf(ys):
    m, P = 0.1, 2.0; out = []
    for k, y in enumerate(ys, 1):
        Fk = df3(m); m = f3(m, k); P = Fk * P * Fk + Qn
        Hk = m / 10; S = Hk * P * Hk + Rn; Kk = P * Hk / S
        m = m + Kk * (y - h3(m)); P = (1 - Kk * Hk) * P; out.append(m)
    return np.array(out)
def ukf(ys, alpha=1.0, beta=0.0, kappa=2.0):
    n = 1; lam = alpha**2 * (n + kappa) - n
    wm = np.array([lam / (n + lam), 0.5 / (n + lam), 0.5 / (n + lam)])
    wc = wm.copy(); wc[0] += 1 - alpha**2 + beta
    m, P = 0.1, 2.0; out = []
    for k, y in enumerate(ys, 1):
        s = math.sqrt((n + lam) * P); X = np.array([m, m + s, m - s])
        Xp = np.array([f3(xi, k) for xi in X]); m = wm @ Xp; P = wc @ (Xp - m) ** 2 + Qn
        s = math.sqrt((n + lam) * P); X = np.array([m, m + s, m - s])
        Yp = h3(X); yh = wm @ Yp; S = wc @ (Yp - yh) ** 2 + Rn; C = wc @ ((X - m) * (Yp - yh))
        Kk = C / S; m = m + Kk * (y - yh); P = P - Kk * S * Kk; out.append(m)
    return np.array(out)
def pf(ys, rng, Np=500):
    x = rng.normal(0.1, math.sqrt(2.0), Np); out = []; ess_min = Np
    for k, y in enumerate(ys, 1):
        x = 0.5 * x + 25 * x / (1 + x**2) + 8 * math.cos(1.2 * k) + rng.normal(0, math.sqrt(Qn), Np)
        lw = -0.5 * (y - x**2 / 20) ** 2 / Rn; w = np.exp(lw - lw.max()); w /= w.sum()
        out.append(w @ x)
        ess_min = min(ess_min, 1 / np.sum(w**2))
        idx = np.searchsorted(np.cumsum(w), (rng.random() + np.arange(Np)) / Np)  # remuestreo sistemático
        x = x[np.minimum(idx, Np - 1)]
    return np.array(out), ess_min
rmse = {"EKF": [], "UKF": [], "PF (500)": [], "PF (50)": []}
for s in range(50):
    rng = np.random.default_rng(100 + s)
    xt, ys = corre(rng)
    rmse["EKF"].append(np.sqrt(np.mean((ekf(ys) - xt) ** 2)))
    rmse["UKF"].append(np.sqrt(np.mean((ukf(ys) - xt) ** 2)))
    rmse["PF (500)"].append(np.sqrt(np.mean((pf(ys, rng)[0] - xt) ** 2)))
    rmse["PF (50)"].append(np.sqrt(np.mean((pf(ys, rng, 50)[0] - xt) ** 2)))
for kname, v in rmse.items():
    print(f"  {kname:9s}: RMSE medio {np.mean(v):5.2f} (mediana {np.median(v):5.2f}) en 50 ensayos")
rng = np.random.default_rng(7)
xt, ys = corre(rng)
e_ekf, e_ukf = ekf(ys), ukf(ys)
e_pf, essmin = pf(ys, rng)
print(f"  ejemplo: tamaño efectivo mínimo del PF {essmin:.0f} de 500")
X3 = lin(1, T3, 60, 500)
Y3 = lin(-25, 25, 250, 30)
kk = np.arange(1, T3 + 1)
print("path nl real:", path(kk, xt, X3, Y3))
print("path nl ekf:", path(kk, np.clip(e_ekf, -25, 25), X3, Y3))
print("path nl pf:", path(kk, e_pf, X3, Y3))

print("=" * 70)
print("4. ESKF DE ACTITUD con sesgo del giróscopo: acelerómetro + magnetómetro frente a solo acelerómetro")
def skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
def Exp(phi):
    a = np.linalg.norm(phi)
    if a < 1e-12:
        return np.eye(3) + skew(phi)
    K = skew(phi / a)
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * K @ K
def Log(Rm):
    c = max(-1.0, min(1.0, (np.trace(Rm) - 1) / 2)); a = math.acos(c)
    w = np.array([Rm[2, 1] - Rm[1, 2], Rm[0, 2] - Rm[2, 0], Rm[1, 0] - Rm[0, 1]])
    return w / 2 if a < 1e-9 else a / (2 * math.sin(a)) * w
dtg, Tg = 0.005, 120.0                        # IMU a 200 Hz, 2 minutos
g_w = np.array([0, 0, 9.81]); m_w = np.array([0.4, 0.0, -0.9]); m_w /= np.linalg.norm(m_w)
sg, sa, sm, sbg = 0.005, 0.1, 0.02, 1e-5       # ruido del giróscopo (rad/s), acel. (m/s^2), mag. (normalizado), paseo del sesgo
bias = np.array([0.02, -0.015, 0.01])          # rad/s
def eskf(con_mag, seed=3):
    rng = np.random.default_rng(seed)
    Rt = np.eye(3); Rh = Exp(np.radians([3.0, -3.0, 10.0])); bh = np.zeros(3)   # error inicial: 3°, -3° y 10° de guiñada
    P = np.diag([0.05, 0.05, 0.05, 1e-3, 1e-3, 1e-3])
    Qd = np.diag([sg**2 * dtg] * 3 + [sbg**2 * dtg] * 3)
    t_out, e_yaw, s_yaw, e_tilt = [], [], [], []
    n = int(Tg / dtg)
    for i in range(n):
        t = i * dtg
        w = np.array([0.6 * math.sin(0.7 * t), 0.5 * math.sin(0.5 * t + 1), 0.8 * math.sin(0.3 * t + 2)])
        Rt = Rt @ Exp(w * dtg)
        wm = w + bias + rng.normal(0, sg / math.sqrt(dtg), 3)
        # predicción (estado nominal y error en el tangente, perturbación por la derecha)
        we = wm - bh
        Rh = Rh @ Exp(we * dtg)
        Fx = np.eye(6); Fx[0:3, 0:3] = Exp(-we * dtg); Fx[0:3, 3:6] = -np.eye(3) * dtg
        P = Fx @ P @ Fx.T + Qd
        if i % 4 == 0:                           # actualizaciones a 50 Hz
            meds = [(g_w, Rt.T @ g_w + rng.normal(0, sa, 3), sa)]
            if con_mag:
                meds.append((m_w, Rt.T @ m_w + rng.normal(0, sm, 3), sm))
            for v, y, s in meds:
                yh = Rh.T @ v
                Hk = np.zeros((3, 6)); Hk[:, 0:3] = skew(yh)
                S = Hk @ P @ Hk.T + s**2 * np.eye(3); Kk = P @ Hk.T @ np.linalg.inv(S)
                dxe = Kk @ (y - yh)
                IKH = np.eye(6) - Kk @ Hk
                P = IKH @ P @ IKH.T + Kk @ (s**2 * np.eye(3)) @ Kk.T
                Rh = Rh @ Exp(dxe[0:3]); bh = bh + dxe[3:6]        # inyección y reinicio (G ~ I)
        if i % 200 == 0:
            # error de guiñada en ejes mundo: rotación de Rh respecto a Rt alrededor de z del mundo
            Ew = Rh @ Rt.T
            e_yaw.append(math.degrees(math.atan2(Ew[1, 0], Ew[0, 0])))
            zt = Rt.T @ np.array([0, 0, 1.0]); zh = Rh.T @ np.array([0, 0, 1.0])   # vertical vista desde el cuerpo: no depende de la guiñada
            e_tilt.append(math.degrees(math.acos(max(-1, min(1, zt @ zh)))))
            # sigma de guiñada en el mundo: proyección de la covarianza del error (cuerpo) sobre z del mundo
            u = Rh.T @ np.array([0, 0, 1.0])
            s_yaw.append(math.degrees(math.sqrt(u @ P[0:3, 0:3] @ u)))
            t_out.append(t)
    return np.array(t_out), np.array(e_yaw), np.array(s_yaw), np.array(e_tilt), bh
res4 = {}
for nombre, cm in (("acel + mag", True), ("solo acel", False)):
    t_, ey, sy, et, bh = eskf(cm)
    res4[nombre] = (t_, ey, sy)
    print(f"  {nombre:10s}: inclinación error máx. (t > 10 s) {et[10:].max():.2f}°; guiñada: error final {abs(ey[-1]):5.2f}°,"
          f" sigma del filtro final {sy[-1]:5.2f}° (inicial {sy[0]:.1f}°); sesgo estimado {np.round(bh,4)} (real {bias})")
print("  Sensibilidad a la semilla (6 ensayos de 120 s cada uno):")
for nombre, cm in (("acel + mag", True), ("solo acel", False)):
    errs, sigs = [], []
    for sd in range(6):
        _, ey_s, sy_s, _, _ = eskf(cm, seed=sd)
        errs.append(abs(ey_s[-1])); sigs.append(sy_s[-1])
    print(f"    {nombre:10s}: error final de guiñada {min(errs):5.2f}–{max(errs):5.2f}°; sigma final {min(sigs):4.2f}–{max(sigs):4.2f}°")
X4 = lin(0, Tg, 60, 500)
Y4 = lin(0, 15, 250, 30)
for nombre in res4:
    t_, ey, sy = res4[nombre]
    print(f"path yaw err {nombre}:", path(t_, np.minimum(np.abs(ey), 15), X4, Y4))
    print(f"path yaw sig {nombre}:", path(t_, np.minimum(2 * sy, 15), X4, Y4))

print("=" * 70)
print("5. UMBRALES chi^2 PARA RECHAZAR MEDIDAS (prueba de innovación con NIS)")
def chi2_cdf(x, k):
    # función gamma incompleta inferior regularizada P(k/2, x/2) por serie
    a, z = k / 2, x / 2
    s, term, n = 0.0, 1.0 / a, 0
    while term > 1e-14 * s or n < 5:
        s += term; n += 1; term *= z / (a + n)
    return math.exp(-z + a * math.log(z) - math.lgamma(a)) * s
def chi2_ppf(p, k):
    lo, hi = 0.0, 200.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if chi2_cdf(mid, k) < p else (lo, mid)
    return (lo + hi) / 2
for k in (1, 2, 3, 6):
    print(f"  {k} grados de libertad: 95 % -> {chi2_ppf(0.95, k):.2f}; 99 % -> {chi2_ppf(0.99, k):.2f}; 99,9 % -> {chi2_ppf(0.999, k):.2f}")

print("=" * 70)
print("6. FILTRO COMPLEMENTARIO = KALMAN ESTACIONARIO (posición medida + acelerómetro como entrada)")
# p' = v, v' = a_m + w (w: ruido del acelerómetro, densidad q); y = p + v_med (densidad r, tiempo continuo)
for q, r in ((0.35**2, 0.5**2), (0.35**2, 0.05**2)):
    A = np.array([[0, 1.0], [0, 0]]); L = np.array([[0.0], [1.0]]); C = np.array([[1.0, 0]])
    P = np.eye(2)
    for _ in range(200000):                      # integrar la Riccati continua hasta el estacionario
        dP = A @ P + P @ A.T + q * L @ L.T - P @ C.T @ C @ P / r
        P = P + 1e-4 * dP
    K = P @ C.T / r
    w0 = (q / r) ** 0.25
    print(f"  q = {q:.4f}, r = {r:.4f}: K = ({K[0,0]:.3f}, {K[1,0]:.3f}); fórmula: (sqrt(2) w0, w0^2) = ({math.sqrt(2)*w0:.3f}, {w0**2:.3f});"
          f" frecuencia de cruce w0 = {w0:.2f} rad/s = {w0/2/math.pi:.3f} Hz")
