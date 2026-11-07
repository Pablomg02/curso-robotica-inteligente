"""Ejemplos numéricos y datos de figuras de M1 (herramientas matemáticas).

Cada número que aparece en un ejemplo de la presentación sale de aquí.
Ejecutar:  python3 ejemplos_m1.py   (solo necesita numpy)
"""
import numpy as np

np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(0)
LN2 = np.log(2.0)


def titulo(s):
    print("\n" + "=" * 70 + "\n" + s + "\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Herramientas de SO(3)
# ---------------------------------------------------------------------------
def hat(w):
    return np.array([[0.0, -w[2], w[1]], [w[2], 0.0, -w[0]], [-w[1], w[0], 0.0]])


def vee(W):
    return np.array([W[2, 1], W[0, 2], W[1, 0]])


def Exp(phi):
    th = np.linalg.norm(phi)
    if th < 1e-12:
        return np.eye(3) + hat(phi)
    K = hat(phi / th)
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


def Log(R):
    th = np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))
    if th < 1e-12:
        return vee(R - R.T) / 2
    return vee(th / (2 * np.sin(th)) * (R - R.T))


def expm_serie(A, n=40):
    out, term = np.eye(3), np.eye(3)
    for k in range(1, n):
        term = term @ A / k
        out = out + term
    return out


def Rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def Ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def Jr(phi):
    th = np.linalg.norm(phi)
    P = hat(phi)
    return np.eye(3) - (1 - np.cos(th)) / th**2 * P + (th - np.sin(th)) / th**3 * P @ P


def Vleft(phi):
    th = np.linalg.norm(phi)
    P = hat(phi)
    return np.eye(3) + (1 - np.cos(th)) / th**2 * P + (th - np.sin(th)) / th**3 * P @ P


def qmul(q, p):  # Hamilton, (w, x, y, z)
    qw, qv, pw, pv = q[0], q[1:], p[0], p[1:]
    return np.concatenate([[qw * pw - qv @ pv], qw * pv + pw * qv + np.cross(qv, pv)])


def ang_err_deg(R1, R2):
    return np.degrees(np.linalg.norm(Log(R1.T @ R2)))


# ---------------------------------------------------------------------------
titulo("1. Matriz de rotación: dron con guiñada de 90°")
R = Rz(np.pi / 2)
print("R_WB =\n", R)
print("columna 1 (morro del dron en ejes mundo):", R[:, 0])
pB = np.array([1.0, 0, 0])
print("punto 1 m delante del dron, en mundo:", R @ pB)
print("R^T R = I:", np.allclose(R.T @ R, np.eye(3)), " det =", np.linalg.det(R))

titulo("2. Ángulos de Euler ZYX: con cabeceo 90° solo cuenta phi - psi")
def euler_zyx(phi, th, psi):
    return Rz(psi) @ Ry(th) @ Rx(phi)
A = euler_zyx(np.radians(30), np.pi / 2, np.radians(10))
B = euler_zyx(np.radians(50), np.pi / 2, np.radians(30))
print("(phi,psi)=(30,10):\n", A)
print("(phi,psi)=(50,30) igual:", np.allclose(A, B))
d = np.radians(20)
print("forma cerrada [[0, s, c],[0, c, -s],[-1,0,0]] con phi-psi=20°:",
      np.allclose(A, np.array([[0, np.sin(d), np.cos(d)], [0, np.cos(d), -np.sin(d)], [-1, 0, 0]])))
for thd in [0, 60, 80, 89, 89.9]:
    print(f"  cabeceo {thd:5.1f}°: 1/cos = {1/np.cos(np.radians(thd)):8.1f}")

phi_, th_, psi_ = 0.3, -0.4, 1.0
dphi_, dth_, dpsi_ = 0.2, 0.5, -0.7
hh = 1e-6
Rn = lambda a, b, c: euler_zyx(a, b, c)
Rdot = (Rn(phi_+hh*dphi_, th_+hh*dth_, psi_+hh*dpsi_) - Rn(phi_-hh*dphi_, th_-hh*dth_, psi_-hh*dpsi_)) / (2*hh)
w_body = vee(Rn(phi_, th_, psi_).T @ Rdot)
M = np.array([[1, 0, -np.sin(th_)], [0, np.cos(phi_), np.sin(phi_)*np.cos(th_)], [0, -np.sin(phi_), np.cos(phi_)*np.cos(th_)]])
print("omega = M(eta) eta_dot:", np.allclose(w_body, M @ np.array([dphi_, dth_, dpsi_]), atol=1e-8), " det M = cos(theta):", np.isclose(np.linalg.det(M), np.cos(th_)))

titulo("3. hat: el producto vectorial como matriz")
w, v = np.array([1.0, 2, 3]), np.array([0.0, 0, 1])
print("w x v =", np.cross(w, v), "  [w]x v =", hat(w) @ v)
u = np.array([1.0, 2, 2]) / 3
print("[u]x^2 == u u^T - I:", np.allclose(hat(u) @ hat(u), np.outer(u, u) - np.eye(3)))
print("[u]x^3 == -[u]x:", np.allclose(hat(u) @ hat(u) @ hat(u), -hat(u)))

titulo("4. Cinemática: R(t) = Rz(r t) cumple dR/dt = R [w]x con w = (0,0,r)")
r, t, h = 0.7, 1.3, 1e-6
dR = (Rz(r * (t + h)) - Rz(r * (t - h))) / (2 * h)
print("dR/dt == R [w]x:", np.allclose(dR, Rz(r * t) @ hat([0, 0, r]), atol=1e-8))

titulo("5. Exp como solución de dR/dt = R[w]x con w constante")
w = np.array([0.4, -0.3, 1.1])
Rk = np.eye(3)
n = 20000
for _ in range(n):
    Rk = Rk @ (np.eye(3) + hat(w) / n)  # Euler muy fino
print("Euler fino tras 1 s vs Exp(w):", ang_err_deg(Rk, Exp(w)), "grados")
print("serie de la exponencial vs Rodrigues:", np.abs(expm_serie(hat(w)) - Exp(w)).max())

titulo("6. Rodrigues con números")
print("Exp([0,0,pi/2]) =\n", Exp(np.array([0, 0, np.pi / 2])))
u = np.ones(3) / np.sqrt(3)
R120 = Exp(2 * np.pi / 3 * u)
print("Exp(120° · (1,1,1)/sqrt3) =\n", R120)
print("cos120 =", np.cos(2 * np.pi / 3), " sin120 =", np.sin(2 * np.pi / 3), " 1-cos =", 1 - np.cos(2 * np.pi / 3))
K = hat(u)
print("[u]x =\n", K, "\n[u]x^2 =\n", K @ K)
print("Rodrigues aplicado a e_x:", R120 @ np.array([1.0, 0, 0]))
print("Log: tr(R) =", np.trace(R120), " theta =", np.degrees(np.arccos((np.trace(R120) - 1) / 2)))
print("(R - R^T) =\n", R120 - R120.T)
print("Log(R) =", Log(R120), " = 2pi/3 * u =", 2 * np.pi / 3 * u)
# descomposición geométrica
vv = np.array([1.0, 0, 0])
par = u * (u @ vv); perp = vv - par
print("v_par =", par, " v_perp =", perp, " u x v =", np.cross(u, vv))
print("v_par + cos v_perp + sin u x v =", par + np.cos(2*np.pi/3) * perp + np.sin(2*np.pi/3) * np.cross(u, vv))

titulo("7. Integrar la actitud: primer orden frente a Exp")
w = np.array([0.0, 2 * np.pi, 0.0])  # 360 °/s en cabeceo: una voltereta por segundo
for f in [250, 1000]:
    dt = 1.0 / f
    R1, R2 = np.eye(3), np.eye(3)
    for _ in range(f):
        R1 = R1 + R1 @ hat(w) * dt
        R2 = R2 @ Exp(w * dt)
    Rtrue = Exp(w * 1.0)
    print(f"{f} Hz, 1 s: ||R^T R - I||_F primer orden = {np.linalg.norm(R1.T@R1-np.eye(3)):.3f},"
          f" Exp = {np.linalg.norm(R2.T@R2-np.eye(3)):.1e};"
          f" escala de la columna 1 = {np.linalg.norm(R1[:,0]):.4f};"
          f" error angular Exp = {ang_err_deg(R2, Rtrue):.1e}°")
    # por paso
    E = np.eye(3) + hat(w) * dt
    print("   por paso: (I+[w]dt)^T(I+[w]dt) - I = -[w]^2 dt^2, norma de columna^2 = 1 + (|w|dt)^2 =",
          1 + (np.linalg.norm(w) * dt) ** 2, np.allclose(E.T @ E - np.eye(3), -hat(w) @ hat(w) * dt**2))

titulo("8. Cuaterniones: 90° sobre z aplicado a e_x")
q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
qc = q * np.array([1, -1, -1, -1])
vq = np.array([0.0, 1, 0, 0])
a = qmul(q, vq)
print("q =", q, " q (x) (0,v) =", a, " ... (x) q* =", qmul(a, qc))
print("-q da lo mismo:", qmul(qmul(-q, vq), -qc))
q30x = np.array([np.cos(np.radians(15)), np.sin(np.radians(15)), 0, 0])
print("composición q90z (x) q30x =", qmul(q, q30x))

titulo("9. Perturbación local frente a global (X = Rz(90°), tau = 30° sobre x)")
X = Rz(np.pi / 2)
tau = np.array([np.radians(30), 0, 0])
L, G = X @ Exp(tau), Exp(tau) @ X
print("local  X Exp(tau): eje z del cuerpo en mundo =", L[:, 2], " morro =", L[:, 0])
print("global Exp(tau) X: eje z del cuerpo en mundo =", G[:, 2], " morro =", G[:, 0])
print("Ad: X Exp(tau) == Exp(X tau) X:", np.allclose(L, Exp(X @ tau) @ X), "  X tau =", X @ tau)
Rr = Exp(np.array([0.3, -0.5, 0.8])); tt = np.array([0.2, 0.1, -0.4]); vv = np.array([1.0, -2, 0.5])
print("R[t]x R^T == [R t]x:", np.allclose(Rr @ hat(tt) @ Rr.T, hat(Rr @ tt)))

titulo("10. Jacobiana de R Exp(theta) p respecto a theta")
R = Rz(np.pi / 2); p = np.array([1.0, 0, 0])
J = -R @ hat(p)
print("-R[p]x =\n", J)
h = 1e-6
Jn = np.column_stack([(R @ Exp(h * e) @ p - R @ Exp(-h * e) @ p) / (2 * h) for e in np.eye(3)])
print("diferencias finitas coinciden:", np.allclose(J, Jn, atol=1e-8))
print("theta = (0,0,0.01): punto se mueve", J @ np.array([0, 0, 0.01]), " theta=(0,0.01,0):", J @ np.array([0, 0.01, 0]))

sg = 0.1
Sp = J @ (sg**2 * np.eye(3)) @ J.T
print("Sigma_theta = 0,1^2 I  ->  Sigma_p = J Sigma J^T =\n", Sp, "\n desviaciones:", np.sqrt(np.diag(Sp)))
th_s = sg * rng.standard_normal((200000, 3))
pts = np.array([R @ Exp(t) @ p for t in th_s[:20000]])
print(" Monte Carlo (20000): sd por eje", pts.std(0))

titulo("11. Jacobiano derecho")
phi = np.array([0.3, -0.2, 0.5]); dphi = 1e-3 * np.array([1.0, 2.0, -1.0])
e_sin = ang_err_deg(Exp(phi + dphi), Exp(phi) @ Exp(dphi))
e_con = ang_err_deg(Exp(phi + dphi), Exp(phi) @ Exp(Jr(phi) @ dphi))
print(f"error sin J_r = {e_sin:.2e}°   con J_r = {e_con:.2e}°")
# J_r como integral de Exp(-s phi)
ss = np.linspace(0, 1, 20001)
Jint = np.trapezoid(np.array([Exp(-s * phi) for s in ss]), ss, axis=0)
print("J_r == int_0^1 Exp(-s phi) ds:", np.allclose(Jint, Jr(phi), atol=1e-8))
Vint = np.trapezoid(np.array([Exp(s * phi) for s in ss]), ss, axis=0)
print("V == int_0^1 Exp(s phi) ds:", np.allclose(Vint, Vleft(phi), atol=1e-8))
print("J_r(phi) =\n", Jr(phi))

titulo("12. SE(3): avanzar 1 m/s girando 90° en 1 s")
phi = np.array([0, 0, np.pi / 2]); rho = np.array([1.0, 0, 0])
t = Vleft(phi) @ rho
print("t = V rho =", t, " 2/pi =", 2 / np.pi)
# integración fina de p' = R v
n = 100000; dt = 1 / n; pp = np.zeros(3)
for k in range(n):
    pp = pp + Exp((k + 0.5) * dt * phi) @ rho * dt
print("integración fina:", pp)
print("SO(3)xR3 con la misma 'tau' (traslación directa):", rho)
# composición de poses
T1 = np.eye(4); T1[:3, :3] = Rz(np.pi / 2); T1[:3, 3] = [2, 0, 0]
pB = np.array([1.0, 0, 0, 1])
print("T = [Rz90 | (2,0,0)] aplicado al punto (1,0,0) del cuerpo:", (T1 @ pB)[:3])

titulo("13. El plátano: avance 5 m (sd 0,1) con rumbo sd 0,35 rad")
N = 200000
d = 5 + 0.1 * rng.standard_normal(N); psi = 0.35 * rng.standard_normal(N)
x, y = d * np.cos(psi), d * np.sin(psi)
print(f"media x muestreo = {x.mean():.3f} m  (exacta 5 exp(-s^2/2) = {5*np.exp(-0.35**2/2):.3f});  linealizada = 5")
print(f"sd y muestreo = {y.std():.3f} m;  linealizada 5*0,35 = {5*0.35:.3f} m")
print(f"sd x muestreo = {x.std():.3f} m;  linealizada = 0,1 m")
# fracción fuera de la elipse 2 sigma linealizada (eje x 0,1, eje y 1,75)
m2 = ((x - 5) / 0.1) ** 2 + (y / 1.75) ** 2
print(f"muestras dentro de la elipse 2σ linealizada: {(m2 <= 4).mean()*100:.1f} %  (gaussiana 2D: {100*(1-np.exp(-2)):.1f} %)")

titulo("14. Error de actitud de Lee et al.: e_R = sin(theta) u")
for deg in [10, 90, 170]:
    th = np.radians(deg); Rd = np.eye(3); Rm = Exp(th * np.array([0, 0, 1.0]))
    eR = vee(0.5 * (Rd.T @ Rm - Rm.T @ Rd))
    print(f"error {deg:3d}° en guiñada: e_R = {eR}  (sin = {np.sin(th):.4f}, theta = {th:.4f} rad)")

titulo("15. Marcos: ENU/FLU -> NED/FRD")
T_ne = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])   # p_NED = T_ne p_ENU
T_bf = np.diag([1.0, -1, -1])                            # p_FRD = T_bf p_FLU
pENU = np.array([3.0, 4.0, 10.0])  # 3 m al este, 4 al norte, 10 arriba
print("p_ENU (3 E, 4 N, 10 arriba) -> p_NED =", T_ne @ pENU)
R_enu_flu = Rz(np.radians(30))  # guiñada 30° en ENU (desde el este, hacia el norte)
R_ned_frd = T_ne @ R_enu_flu @ T_bf.T
print("R_ENU_FLU guiñada 30° -> R_NED_FRD =\n", R_ned_frd, "\n   guiñada NED =",
      np.degrees(np.arctan2(R_ned_frd[1, 0], R_ned_frd[0, 0])), "° (desde el norte, hacia el este)")
print("det T_ne =", np.linalg.det(T_ne), " det T_bf =", np.linalg.det(T_bf),
      " (las dos son rotaciones de 180°: T_ne sobre (1,1,0)/sqrt2, T_bf sobre x)")
print("eje de T_ne:", np.linalg.eig(T_ne)[1][:, np.isclose(np.linalg.eig(T_ne)[0], 1)].ravel())

# ---------------------------------------------------------------------------
titulo("16. Filtro de Bayes discreto: objetivo en 4 celdas")
b0 = np.full(4, 0.25)
T = np.zeros((4, 4))  # T[i, j] = p(s'=i | s=j)
for j in range(4):
    if j < 3:
        T[j, j] = 0.8; T[j + 1, j] = 0.2
    else:
        T[j, j] = 1.0
bbar = T @ b0
print("prior b0 =", b0, "\npredicción bbar =", bbar)
pd, pfa = 0.8, 0.1
lik_no = np.array([1 - pfa, 1 - pd, 1 - pfa, 1 - pfa])  # el dron mira la celda 2 y NO detecta
un = lik_no * bbar
print("p(no detecta | s) =", lik_no, "\nsin normalizar =", un, " suma (1/eta) =", un.sum())
print("posterior =", un / un.sum())
lik_si = np.array([pfa, pd, pfa, pfa])
print("si hubiera detectado: posterior =", lik_si * bbar / (lik_si * bbar).sum())

titulo("17. Condicionar una gaussiana escalar: altura con barómetro")
mx, Pxx, Rn, yobs = 10.0, 4.0, 1.0, 12.0
Sxy, Syy = Pxx, Pxx + Rn
print(f"Sigma_xy = {Sxy}, Sigma_yy = {Syy}, rho = {Sxy/np.sqrt(Pxx*Syy):.3f}")
print(f"media posterior = {mx + Sxy/Syy*(yobs-mx):.2f}, varianza posterior = {Pxx - Sxy**2/Syy:.2f}, K = {Sxy/Syy}")
# comprobación por Monte Carlo
xs = mx + 2 * rng.standard_normal(2_000_000); ys = xs + rng.standard_normal(2_000_000)
sel = np.abs(ys - yobs) < 0.02
print(f"Monte Carlo (y≈12): media {xs[sel].mean():.3f}, var {xs[sel].var():.3f}, n = {sel.sum()}")

# Figura: gaussiana conjunta (x, y) y corte en y = 12
X0, X1, Y0, Y1 = 4.0, 16.0, 4.0, 16.0
PX0, PX1, PY0, PY1 = 60.0, 440.0, 330.0, 20.0
sx = lambda v: PX0 + (v - X0) / (X1 - X0) * (PX1 - PX0)
sy = lambda v: PY0 + (v - Y0) / (Y1 - Y0) * (PY1 - PY0)
Sj = np.array([[4.0, 4.0], [4.0, 5.0]]); Lc = np.linalg.cholesky(Sj)
for k in (1, 2):
    tt = np.linspace(0, 2 * np.pi, 73)
    pts = np.array([10, 10])[:, None] + k * Lc @ np.vstack([np.cos(tt), np.sin(tt)])
    print(f"FIG elipse {k}σ: M" + " L".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in pts.T) + " Z")
xx = np.linspace(X0, X1, 121)
cond = np.exp(-0.5 * (xx - 11.6) ** 2 / 0.8) / np.sqrt(2 * np.pi * 0.8)
prior = np.exp(-0.5 * (xx - 10) ** 2 / 4) / np.sqrt(2 * np.pi * 4)
print("FIG condicional (sobre y=12, 150 px por unidad de densidad): M" + " L".join(f"{sx(a):.1f},{sy(12) - 150*c:.1f}" for a, c in zip(xx, cond)))
print("FIG prior de x (sobre el eje, 150 px por unidad): M" + " L".join(f"{sx(a):.1f},{PY0 - 150*c:.1f}" for a, c in zip(xx, prior)))
print("FIG marcas x:", {v: round(sx(v), 1) for v in (4, 6, 8, 10, 11.6, 12, 14, 16)}, " y:", {v: round(sy(v), 1) for v in (4, 8, 10, 12, 16)})
print("FIG recta de regresión E[x|y] = 10 + 0,8 (y-10): de", (round(sx(10 + 0.8 * (5 - 10)), 1), round(sy(5), 1)), "a", (round(sx(10 + 0.8 * 6), 1), round(sy(16), 1)))

titulo("18. Kalman con dos estados: altura y velocidad vertical")
x0 = np.array([10.0, 0.5]); P0 = np.eye(2)
F = np.array([[1, 1.0], [0, 1]]); Q = np.diag([0.0, 0.2])
xb = F @ x0; Pb = F @ P0 @ F.T + Q
print("predicción: x =", xb, "\nP =\n", Pb)
H = np.array([[1.0, 0]]); Rm = np.array([[0.5]]); y = np.array([11.5])
S = H @ Pb @ H.T + Rm; Kg = Pb @ H.T @ np.linalg.inv(S); innov = y - H @ xb
xh = xb + (Kg @ innov); Pp = (np.eye(2) - Kg @ H) @ Pb
print("S =", S, " K =", Kg.ravel(), " innovación =", innov)
print("corrección: x =", xh, "\nP+ =\n", Pp)
# misma cuenta con la fórmula de condicionamiento conjunto
Sig = np.block([[Pb, Pb @ H.T], [H @ Pb, S]])
print("covarianza conjunta [x; y]:\n", Sig)

titulo("19. KL discreta: asimetría")
p = np.array([0.5, 0.5]); q = np.array([0.9, 0.1])
kl = lambda a, b: np.sum(a * np.log(a / b))
print(f"KL(q||p) = {kl(q,p):.3f} nats   KL(p||q) = {kl(p,q):.3f} nats")
print("términos KL(q||p):", q * np.log(q / p), "  KL(p||q):", p * np.log(p / q))

titulo("20. Ajuste de una gaussiana a la mezcla bimodal (figura de KL)")
xg = np.linspace(-12, 12, 24001); dx = xg[1] - xg[0]
npdf = lambda x, m, s: np.exp(-0.5 * ((x - m) / s) ** 2) / (s * np.sqrt(2 * np.pi))
pm = 0.5 * npdf(xg, -2, 0.6) + 0.5 * npdf(xg, 2, 0.6)
print("KL directa (moment matching): media 0, sd =", np.sqrt(0.6**2 + 4))
best = None
for m in np.linspace(1.5, 2.5, 101):
    for s in np.linspace(0.4, 1.0, 121):
        qq = npdf(xg, m, s)
        val = np.sum(qq * (np.log(qq + 1e-300) - np.log(pm + 1e-300))) * dx
        if best is None or val < best[0]:
            best = (val, m, s)
print(f"KL inversa: media {best[1]:.3f}, sd {best[2]:.3f}, KL = {best[0]:.3f} nats (≈ ln 2 = {LN2:.3f})")

titulo("21. ELBO en un modelo lineal gaussiano: z~N(0,1), x = z + e, e~N(0,1), x = 2")
xo, s2 = 2.0, 1.0
logpx = -0.5 * np.log(2 * np.pi * (1 + s2)) - xo**2 / (2 * (1 + s2))
mpost, vpost = xo / (1 + s2), s2 / (1 + s2)
print(f"log p(x) = {logpx:.4f};  posterior N({mpost}, {vpost})")
def elbo(m, v):
    rec = -0.5 * np.log(2 * np.pi * s2) - ((xo - m) ** 2 + v) / (2 * s2)
    klp = 0.5 * (v + m**2 - 1 - np.log(v))
    return rec, klp, rec - klp
def klg(m1, v1, m2, v2):
    return 0.5 * (np.log(v2 / v1) + (v1 + (m1 - m2) ** 2) / v2 - 1)
for (m, v, nombre) in [(0.0, 1.0, "q = prior"), (0.5, 0.75, "q intermedia"), (1.0, 0.5, "q = posterior")]:
    rec, klp, e = elbo(m, v)
    print(f"{nombre:14s} N({m},{v}): reconstrucción {rec:.4f}, KL al prior {klp:.4f}, ELBO {e:.4f},"
          f" brecha {logpx-e:.4f}, KL(q||posterior) {klg(m, v, mpost, vpost):.4f}")
# Monte Carlo del ELBO con q = prior
zs = rng.standard_normal(1_000_000)
mc = np.mean(-0.5*np.log(2*np.pi) - (xo - zs)**2/2 - 0.5*np.log(2*np.pi) - zs**2/2 + 0.5*np.log(2*np.pi) + zs**2/2)
print(f"Monte Carlo ELBO(q=prior) = {mc:.4f}")

titulo("22. Reparametrización frente a función de puntuación: f(z) = z^2, z~N(mu, s^2)")
mu, s = 1.0, 1.0
eps = rng.standard_normal(2_000_000); z = mu + s * eps
g_rep = 2 * z
g_sco = z**2 * (z - mu) / s**2
print(f"gradiente exacto d/dmu (mu^2+s^2) = {2*mu}")
print(f"reparametrización: media {g_rep.mean():.3f}, varianza {g_rep.var():.2f}")
print(f"función de puntuación: media {g_sco.mean():.3f}, varianza {g_sco.var():.2f}")
# varianzas exactas con mu = s = 1: reparam Var(2(1+e)) = 4;
# puntuación g = (1+e)^2 e = e + 2e^2 + e^3, E[g^2] = 1 + 12 + 15 + 6 = 34, Var = 34 - 4 = 30
print("varianza exacta: reparametrización 4, función de puntuación 30")

titulo("23. KL entre gaussianas: ejemplo escalar y vectorial")
print(f"KL(N(0,1) || N(1,0.5)) = {klg(0,1,1,0.5):.4f}")
m1, S1 = np.array([1.0, 0]), np.diag([0.5, 2.0])
m2, S2 = np.zeros(2), np.eye(2)
kv = 0.5 * (np.trace(np.linalg.inv(S2) @ S1) + (m2-m1) @ np.linalg.inv(S2) @ (m2-m1) - 2 + np.log(np.linalg.det(S2)/np.linalg.det(S1)))
kd = 0.5 * np.sum(np.diag(S1) + m1**2 - 1 - np.log(np.diag(S1)))
print(f"vectorial N((1,0), diag(0.5,2)) || N(0,I): fórmula general {kv:.4f}, fórmula diagonal {kd:.4f}")
print("términos por componente:", 0.5 * (np.diag(S1) + m1**2 - 1 - np.log(np.diag(S1))))

# ---------------------------------------------------------------------------
titulo("24. Entropías")
Hb = lambda pp: -np.sum([a * np.log2(a) for a in pp if a > 0])
print("moneda 0,5:", Hb([0.5, 0.5]), "bits;  moneda 0,9:", round(Hb([0.9, 0.1]), 3), "bits;  8 celdas:", Hb([1/8]*8), "bits")
print("belief de la celda tras no detectar (ej. 16):", np.round(un / un.sum(), 4), "H =", round(Hb(un / un.sum()), 3), "bits; prior H =", Hb(b0))
hg = lambda sd: 0.5 * np.log2(2 * np.pi * np.e * sd**2)
print(f"gaussiana sd 1 m: {hg(1):.3f} bits; sd 0,1 m: {hg(0.1):.3f} bits; diferencia {hg(1)-hg(0.1):.3f} = log2(10)")
print(f"misma sd 0,1 m expresada en cm (sd 10): {hg(10):.3f} bits")

titulo("25. Información mutua discreta: detector")
pX = np.array([0.5, 0.5])  # [presente, ausente]
pY_X = np.array([[0.8, 0.2], [0.1, 0.9]])  # filas X, columnas Y=[detecta, no]
pXY = pX[:, None] * pY_X; pY = pXY.sum(0)
HX = Hb(pX); HXgY = sum(pY[j] * Hb(pXY[:, j] / pY[j]) for j in range(2))
print("p(Y) =", pY, " H(X) =", HX, " H(X|Y) =", round(HXgY, 4), " I =", round(HX - HXgY, 4), "bits")
print("p(X|detecta) =", pXY[:, 0] / pY[0], "  p(X|no) =", pXY[:, 1] / pY[1])

titulo("26. Información mutua gaussiana: varianza a priori 4, ruido 1")
P, Rn = 4.0, 1.0
Ppost = 1 / (1 / P + 1 / Rn)
ig = 0.5 * np.log(P / Ppost)
print(f"P+ = {Ppost}, IG = {ig:.3f} nats = {ig/LN2:.3f} bits;  quiz (4, 4): {0.5*np.log(2):.3f} nats = {0.5:.2f} bits")

titulo("27. Percepción activa: qué medir")
P = np.diag([4.0, 1.0]); R1 = np.array([[1.0]])
for nombre, Ha in [("medir x", np.array([[1.0, 0]])), ("medir y", np.array([[0, 1.0]]))]:
    IG = 0.5 * np.log(np.linalg.det(np.eye(1) + np.linalg.inv(R1) @ Ha @ P @ Ha.T))
    print(f"P = diag(4,1), {nombre}: IG = {IG:.3f} nats = {IG/LN2:.3f} bits")
Pc = np.array([[4.0, 1.8], [1.8, 1.0]])
for nombre, Ha in [("medir x", np.array([[1.0, 0]])), ("medir y", np.array([[0, 1.0]])), ("medir x - 2y", np.array([[1.0, -2.0]]))]:
    IG = 0.5 * np.log(np.linalg.det(np.eye(1) + Ha @ Pc @ Ha.T))
    Pp = np.linalg.inv(np.linalg.inv(Pc) + Ha.T @ Ha)
    print(f"P correlada, {nombre}: HPH^T = {(Ha@Pc@Ha.T).item():.2f}, IG = {IG:.3f} nats; var x tras medir = {Pp[0,0]:.3f}")

titulo("28. Fisher y Cramér-Rao")
sig, Nn = 2.0, 25
print(f"media de N={Nn} muestras con sd {sig}: I = N/s^2 = {Nn/sig**2}, cota = {sig**2/Nn}, sd mínima = {sig/np.sqrt(Nn)}")
reps = 200000
est = (sig * rng.standard_normal((reps, Nn))).mean(1)
print(f"varianza de la media muestral (Monte Carlo) = {est.var():.4f}")
Ifish = np.array([[1.0, 1.0], [1.0, 1.0]]) / 0.5**2
print("y = th1 + th2 + ruido (sd 0,5): I =\n", Ifish, " det =", np.linalg.det(Ifish),
      " autovalores =", np.linalg.eigvalsh(Ifish), " dirección no identificable (1,-1)/sqrt2")
