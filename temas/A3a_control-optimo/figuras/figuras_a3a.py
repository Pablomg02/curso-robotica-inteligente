"""Datos de las figuras y tablas del tema A3a (control óptimo: LQR, MPC, NMPC, RTI, MPCC).

Uso:  python3 figuras_a3a.py > salida.txt
Requiere numpy. Imprime cifras y trazados SVG ('path etiqueta: M...') escalados al viewBox de cada figura.
Planta: x500 de PX4 en Gazebo (A1/A2): m = 2,064 kg, J = diag(0,0239; 0,0239; 0,044), brazo 0,174 m,
k_T = 8,54858e-6, Omega_max = 1000 rad/s. Convenio ENU/FLU.
"""
import math
import time
import numpy as np

G = 9.81
M = 2.0 + 4 * 0.016076923
JX = JY = 0.02389
JZ = 0.0440
ARM = 0.174
KT = 8.54858e-6
TMAX = KT * 1000.0**2            # empuje máximo por motor, N
T0 = M * G / 4                   # empuje por motor en estacionario


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def logmap(d0, d1, p0, p1):
    return lambda v: p0 + (math.log10(v) - math.log10(d0)) * (p1 - p0) / (math.log10(d1) - math.log10(d0))


def c2d(A, B, dt):
    """Discretización exacta con retenedor de orden cero (exponencial de la matriz aumentada)."""
    n, m = B.shape
    Maug = np.zeros((n + m, n + m))
    Maug[:n, :n] = A * dt
    Maug[:n, n:] = B * dt
    E = expm(Maug)
    return E[:n, :n], E[:n, n:]


def expm(Mx):
    # Padé con escalado y cuadrado (suficiente para matrices pequeñas)
    nrm = np.linalg.norm(Mx, 1)
    s = max(0, int(math.ceil(math.log2(nrm))) + 1) if nrm > 0.5 else 0
    X = Mx / 2**s
    n = X.shape[0]
    c = [1.0, 0.5, 0.1171875, 0.01953125, 0.002034505208333333, 0.0001627604166666667,
         8.138020833333333e-06, 1.937624007936508e-07]
    I = np.eye(n)
    U = np.zeros_like(X); V = np.zeros_like(X)
    Xp = I.copy()
    for k, ck in enumerate(c):
        if k % 2 == 0:
            V = V + ck * Xp
        else:
            U = U + ck * Xp
        Xp = Xp @ X
    E = np.linalg.solve(V - U, V + U)
    for _ in range(s):
        E = E @ E
    return E


def dare(A, B, Q, R, it=20000, tol=1e-12):
    P = Q.copy()
    for k in range(it):
        K = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ K
        if np.max(np.abs(Pn - P)) < tol * max(1.0, np.max(np.abs(P))):
            return Pn, K, k
        P = Pn
    return P, K, it


print("=" * 70)
print("1. LQR DEL x500 EN ESTACIONARIO (12 estados: p, v, ángulos pequeños, omega; entradas dT, tau)")
A = np.zeros((12, 12)); B = np.zeros((12, 4))
A[0:3, 3:6] = np.eye(3)
A[3, 7] = G            # vx' = g * theta
A[4, 6] = -G           # vy' = -g * phi
A[6:9, 9:12] = np.eye(3)
B[5, 0] = 1 / M
B[9, 1] = 1 / JX; B[10, 2] = 1 / JY; B[11, 3] = 1 / JZ
dt = 0.01
Ad, Bd = c2d(A, B, dt)
# Regla de Bryson: Q = diag(1/x_max^2), R = diag(1/u_max^2)
xmax = np.array([0.5, 0.5, 0.5, 1.0, 1.0, 1.0, 0.3, 0.3, 0.5, 2.0, 2.0, 2.0])
umax = np.array([4 * min(TMAX - T0, T0), 1.0, 1.0, 0.1])
Q = np.diag(1 / xmax**2); R = np.diag(1 / umax**2)
P, K, its = dare(Ad, Bd, Q, R)
ev = np.linalg.eigvals(Ad - Bd @ K)
s_ev = np.log(ev.astype(complex)) / dt
print(f"  dt = {dt} s; DARE por iteración de Riccati: {its} pasos hasta converger")
print("  ganancias de la ley u = -K x (eje x: tau_y frente a x, vx, theta, q):"
      f" {K[2,0]:.2f}, {K[2,3]:.2f}, {K[2,7]:.2f}, {K[2,10]:.3f}")
lat = sorted(s_ev, key=lambda z: abs(z))
print("  polos de lazo cerrado en tiempo continuo (rad/s), ordenados por módulo:")
for z in lat:
    print(f"    {z.real:8.2f} {z.imag:+8.2f}j   |s| = {abs(z):7.2f}")

print("=" * 70)
print("2. CONVERGENCIA DE LA RECURSIÓN DE RICCATI CON EL HORIZONTE (eje x: 4 estados)")
idx = [0, 3, 7, 10]
Ax = Ad[np.ix_(idx, idx)]; Bx = Bd[np.ix_(idx, [2])]
Qx = Q[np.ix_(idx, idx)]; Rx = R[np.ix_([2], [2])]
Pinf, Kinf, _ = dare(Ax, Bx, Qx, Rx)
Pk = np.zeros((4, 4))   # coste terminal nulo
gaps = []
for k in range(1, 401):
    Kk = np.linalg.solve(Bx.T @ Pk @ Bx + Rx, Bx.T @ Pk @ Ax)
    Pk = Qx + Ax.T @ Pk @ Ax - Ax.T @ Pk @ Bx @ Kk
    gaps.append(np.linalg.norm(Kk - Kinf) / np.linalg.norm(Kinf))
gaps = np.array(gaps)
for frac in (0.5, 0.1, 0.01):
    k = int(np.argmax(gaps < frac)) + 1
    print(f"  la ganancia del primer paso está a menos del {100*frac:g} % de la LQR infinita con N = {k} ({k*dt:.2f} s)")
X2 = lin(0, 400, 60, 500)
Y2 = logmap(1e-4, 1.0, 250, 30)
ks = np.arange(1, 401)
keep = gaps > 1e-4
print("path riccati:", path(ks[keep], np.maximum(gaps[keep], 1e-4), X2, Y2))

print("=" * 70)
print("3. MPC LINEAL CON LÍMITE DE INCLINACIÓN FRENTE A LQR SATURADO (eje x, salto de 1 m)")
SALTO = 1.0
dtm = 0.02
Am = np.zeros((4, 4)); Bm = np.zeros((4, 1))
Am[0, 1] = 1; Am[1, 2] = G; Am[2, 3] = 1; Bm[3, 0] = 1 / JY
Amd, Bmd = c2d(Am, Bm, dtm)
TH_MAX = math.radians(20.0)                       # límite de inclinación
TAU_MAX = 2 * ARM * 2 * min(TMAX - T0, T0)        # par de cabeceo máximo desde el estacionario (2 motores suben, 2 bajan)
Qm = np.diag([10.0, 1.0, 1.0, 0.01]); Rm = np.array([[0.05]])
Pm, Km, _ = dare(Amd, Bmd, Qm, Rm)
N = 50                                            # horizonte de 1 s
# Forma condensada: X = Phi x0 + Gam U
n, m = 4, 1
Phi = np.zeros((N * n, n)); Gam = np.zeros((N * n, N * m))
Ak = np.eye(n)
for k in range(N):
    Ak = Amd @ Ak
    Phi[k * n:(k + 1) * n] = Ak
    for j in range(k + 1):
        Gam[k * n:(k + 1) * n, j * m:(j + 1) * m] = np.linalg.matrix_power(Amd, k - j) @ Bmd
Qbar = np.kron(np.eye(N), Qm); Qbar[-n:, -n:] = Pm       # coste terminal = LQR (DARE)
Rbar = np.kron(np.eye(N), Rm)
H = Gam.T @ Qbar @ Gam + Rbar
Sel = np.zeros((N, N * n)); Sel[np.arange(N), np.arange(N) * n + 2] = 1   # filas de theta
Ac = np.vstack([np.eye(N), Sel @ Gam])


def qp_admm(Hq, f, Acn, l, u, z0=None, rho=10.0, sigma=1e-6, iters=1500):
    """ADMM al estilo de OSQP para min 1/2 x'Hx + f'x  s.a. l <= Ax <= u."""
    nv = Hq.shape[0]
    Kkt = np.linalg.inv(Hq + sigma * np.eye(nv) + rho * Acn.T @ Acn)
    x = np.zeros(nv) if z0 is None else z0.copy()
    z = np.clip(Acn @ x, l, u); y = np.zeros(Acn.shape[0])
    for _ in range(iters):
        x = Kkt @ (sigma * x - f + Acn.T @ (rho * z - y))
        zn = np.clip(Acn @ x + y / rho, l, u)
        y = y + rho * (Acn @ x - zn)
        z = zn
    return x


def lazo(ley, T=4.0):
    x = np.array([-SALTO, 0.0, 0.0, 0.0])
    xs, us = [x.copy()], []
    Uw = None
    t0 = time.perf_counter()
    for k in range(int(T / dtm)):
        if ley == "lqr":
            u = float(np.clip(-(Km @ x)[0], -TAU_MAX, TAU_MAX))
        else:
            f = Gam.T @ Qbar @ (Phi @ x)
            th0 = (Sel @ Phi @ x)
            l = np.concatenate([-TAU_MAX * np.ones(N), -TH_MAX - th0])
            uu = np.concatenate([TAU_MAX * np.ones(N), TH_MAX - th0])
            Uw = qp_admm(H, f, Ac, l, uu, z0=None if Uw is None else np.append(Uw[1:], Uw[-1]))
            u = float(Uw[0])
        x = Amd @ x + Bmd[:, 0] * u
        xs.append(x.copy()); us.append(u)
    return np.array(xs), np.array(us), time.perf_counter() - t0


xl, ul, _ = lazo("lqr")
xm, um, tmpc = lazo("mpc")
tt = np.arange(len(xl)) * dtm
for nm_, xs in (("LQR saturado", xl), ("MPC", xm)):
    fuera = np.where(np.abs(xs[:, 0]) > 0.05 * SALTO)[0]
    t5 = tt[fuera[-1] + 1] if len(fuera) and fuera[-1] + 1 < len(tt) else float("nan")
    print(f"  {nm_:12s}: inclinación máxima {math.degrees(np.max(np.abs(xs[:,2]))):5.1f}° (límite 20°);"
          f" se queda en el 5 % del salto desde {t5:.2f} s; sobrepaso de posición {100*max(0, xs[:,0].max())/SALTO:.1f} %")
# con un salto de 2 m, el LQR saturado diverge en el modelo lineal
x = np.array([-2.0, 0, 0, 0])
for k in range(300):
    x = Amd @ x + Bmd[:, 0] * float(np.clip(-(Km @ x)[0], -TAU_MAX, TAU_MAX))
print(f"  salto de 2 m con LQR saturado: |x| a los 6 s = {abs(x[0]):.0f} m (diverge)")
print(f"  par máximo disponible {TAU_MAX:.2f} N m; horizonte N = {N} ({N*dtm:.1f} s); QP de {N} variables y {2*N} restricciones")
X3 = lin(0, 4, 60, 500)
Yx = lin(-1.2, 0.3, 130, 20)
Yt = lin(-60, 60, 270, 160)
print("path pos lqr:", path(tt, xl[:, 0], X3, Yx))
print("path pos mpc:", path(tt, xm[:, 0], X3, Yx))
print("path th lqr:", path(tt, np.degrees(xl[:, 2]), X3, Yt))
print("path th mpc:", path(tt, np.degrees(xm[:, 2]), X3, Yt))
print(f"  líneas de límite: y(+20°) = {Yt(20):.1f}, y(-20°) = {Yt(-20):.1f}; y(0 m) = {Yx(0):.1f}")

print("=" * 70)
print("4. COSTE DE RESOLVER EL QP: Riccati (disperso) frente a condensado y denso (n = 12, m = 4)")
nn, mm = 12, 4
Ns = np.array([5, 10, 20, 40, 80, 160])
ric = Ns * (7 / 3 * nn**3 + 4 * nn**2 * mm + 2 * nn * mm**2 + mm**3 / 3)
cond = Ns**3 * nn * mm**2 / 3 + Ns**3 * mm**3 / 3
fact = Ns**3 * mm**3 / 3
for Nn, a, b, c in zip(Ns, ric, cond, fact):
    print(f"  N = {Nn:3d}: Riccati {a/1e3:9.1f} kflop; condensar y factorizar {b/1e3:11.1f} kflop (x{b/a:6.1f});"
          f" solo factorizar {c/1e3:9.1f} kflop (x{c/a:5.1f})")
n1, m1 = 4, 1
for Nn in (20, 80, 160):
    r1 = Nn * (7 / 3 * n1**3 + 4 * n1**2 * m1 + 2 * n1 * m1**2 + m1**3 / 3)
    c1 = Nn**3 * n1 * m1**2 / 3 + Nn**3 * m1**3 / 3
    print(f"  subsistema del eje x (n = 4, m = 1), N = {Nn}: condensado / Riccati = x{c1/r1:.1f}")

print("=" * 70)
print("5. NMPC CON GAUSS-NEWTON: RTI (una iteración por paso) FRENTE A SQP CONVERGIDO")
# Cuadricóptero plano (y, z, phi, vy, vz, omega); entradas T1, T2 (cada una = 2 motores del x500)
LP = ARM
U_H = M * G / 2
dts, NH = 0.02, 50                         # control a 50 Hz, horizonte de 1 s con la misma malla
Qs = np.diag([20.0, 20.0, 1.0, 1.0, 1.0, 0.1]); Rs = np.diag([0.02, 0.02])


def f_pq(s, u):
    y, z, ph, vy, vz, om = s.T
    T = u[:, 0] + u[:, 1]
    return np.stack([vy, vz, om, -T * np.sin(ph) / M, T * np.cos(ph) / M - G, LP * (u[:, 0] - u[:, 1]) / JX], axis=1)


def rk4(s, u, h=dts):
    k1 = f_pq(s, u); k2 = f_pq(s + h / 2 * k1, u); k3 = f_pq(s + h / 2 * k2, u); k4 = f_pq(s + h * k3, u)
    return s + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def jac(S, U, eps=1e-6):
    """Jacobianos de la discretización RK4 en todos los nodos a la vez (diferencias centradas)."""
    Nn = S.shape[0]
    Aj = np.zeros((Nn, 6, 6)); Bj = np.zeros((Nn, 6, 2))
    for i in range(6):
        d = np.zeros(6); d[i] = eps
        Aj[:, :, i] = (rk4(S + d, U) - rk4(S - d, U)) / (2 * eps)
    for i in range(2):
        d = np.zeros(2); d[i] = eps
        Bj[:, :, i] = (rk4(S, U + d) - rk4(S, U - d)) / (2 * eps)
    return Aj, Bj


def ref(t):
    w = 1.2
    y, z = 3 * np.sin(w * t), 1.0 * np.sin(2 * w * t)
    vy, vz = 3 * w * np.cos(w * t), 2 * w * np.cos(2 * w * t)
    return np.stack([y, z, 0 * t, vy, vz, 0 * t], axis=-1)


def rollout(x0, U):
    S = np.zeros((NH + 1, 6)); S[0] = x0
    for k in range(NH):
        S[k + 1] = rk4(S[k:k + 1], U[k:k + 1])[0]
    return S


def gn_paso(x0, U, Sref):
    """Una iteración de Gauss-Newton (pasada atrás tipo Riccati + pasada adelante)."""
    S = rollout(x0, U)
    Aj, Bj = jac(S[:-1], U)
    Vx = Qs @ (S[-1] - Sref[-1]); Vxx = Qs.copy()
    kk = np.zeros((NH, 2)); KK = np.zeros((NH, 2, 6))
    for k in range(NH - 1, -1, -1):
        Ak, Bk = Aj[k], Bj[k]
        Qx = Qs @ (S[k] - Sref[k]) + Ak.T @ Vx
        Qu = Rs @ (U[k] - U_H) + Bk.T @ Vx
        Qxx = Qs + Ak.T @ Vxx @ Ak; Quu = Rs + Bk.T @ Vxx @ Bk; Qux = Bk.T @ Vxx @ Ak
        kk[k] = -np.linalg.solve(Quu, Qu); KK[k] = -np.linalg.solve(Quu, Qux)
        Vx = Qx + KK[k].T @ Quu @ kk[k] + KK[k].T @ Qu + Qux.T @ kk[k]
        Vxx = Qxx + KK[k].T @ Quu @ KK[k] + KK[k].T @ Qux + Qux.T @ KK[k]
        Vxx = 0.5 * (Vxx + Vxx.T)
    return S, kk, KK


def adelante(x0, S, U, kk, KK):
    Un = U.copy(); xb = x0.copy()
    for k in range(NH):
        Un[k] = U[k] + kk[k] + KK[k] @ (xb - S[k])
        xb = rk4(xb[None], Un[k:k + 1])[0]
    return Un


def simula_nmpc(modo, T=8.0, empujon=3.0):
    x = ref(np.array([0.0]))[0].copy()
    U = np.full((NH, 2), U_H)
    S_pred = None
    errs, its, tcalc, umin, umax = [], [], 0.0, np.inf, 0.0
    for i in range(int(T / dts)):
        t = i * dts
        Sref = ref(t + dts * np.arange(NH + 1))
        t0 = time.perf_counter()
        if modo == "rti":
            # preparación: linealizar alrededor de la predicción (sin conocer aún x)
            x_pred = x if S_pred is None else S_pred
            S, kk, KK = gn_paso(x_pred, U, Sref)
            # realimentación: en cuanto llega x, un producto matriz-vector
            U = adelante(x, S, U, kk, KK)
            nit = 1
        else:
            nit = 0
            for nit in range(1, 31):
                S, kk, KK = gn_paso(x, U, Sref)
                Un = adelante(x, S, U, kk, KK)
                dU = np.max(np.abs(Un - U)); U = Un
                if dU < 1e-6:
                    break
        tcalc += time.perf_counter() - t0
        u = U[0].copy(); umin = min(umin, u.min())
        x = rk4(x[None], u[None])[0]
        S_pred = x.copy()                                 # predicción del modelo para la siguiente preparación
        if abs(t + dts - empujon) < dts / 2:
            x[5] += 2.0                                   # empujón (no previsto): +2 rad/s de velocidad angular
        umax = max(umax, u.max())
        U = np.vstack([U[1:], U[-1:]])                    # desplazar (transición)
        errs.append(np.linalg.norm(x[:2] - ref(np.array([t + dts]))[0][:2]))
        its.append(nit)
    return np.array(errs), np.array(its), tcalc, umin, umax


e_rti, it_rti, t_rti, umin_rti, umax_rti = simula_nmpc("rti")
e_sqp, it_sqp, t_sqp, umin_sqp, umax_sqp = simula_nmpc("sqp")
tt5 = dts * np.arange(1, len(e_rti) + 1)
k0 = tt5 > 1.0
print(f"  error de posición RMS (t > 1 s): RTI {100*np.sqrt(np.mean(e_rti[k0]**2)):.2f} cm,"
      f" SQP convergido {100*np.sqrt(np.mean(e_sqp[k0]**2)):.2f} cm")
print(f"  pico tras el empujón: RTI {100*e_rti[tt5>3].max():.1f} cm, SQP {100*e_sqp[tt5>3].max():.1f} cm;"
      f" diferencia máxima entre ambos {100*np.max(np.abs(e_rti-e_sqp)):.2f} cm")
print(f"  iteraciones del SQP: media {it_sqp.mean():.1f}, máximo {it_sqp.max()} (tras el empujón: {it_sqp[tt5>3][:5].tolist()})")
print(f"  tiempo de cálculo total (Python): RTI {t_rti:.1f} s, SQP {t_sqp:.1f} s -> x{t_sqp/t_rti:.1f}")
print(f"  empuje pedido por par de motores: RTI {umin_rti:.2f}-{umax_rti:.2f} N, SQP {umin_sqp:.2f}-{umax_sqp:.2f} N (límite 0-{2*TMAX:.1f} N)")
X5 = lin(0, 8, 60, 500)
Y5 = lin(0, 8, 250, 30)
print("path err rti:", path(tt5, np.minimum(100 * e_rti, 8), X5, Y5))
print("path err sqp:", path(tt5, np.minimum(100 * e_sqp, 8), X5, Y5))
print(f"  marca del empujón: x = {X5(3.0):.1f}")
# Figura de coste: x log N 5..160 -> 60..500; y log kflop 1..1e6 -> 250..30
X4 = logmap(5, 160, 60, 500)
Y4 = logmap(1, 1e6, 250, 30)
print("path cost ric:", path(Ns, ric / 1e3, X4, Y4))
print("path cost cond:", path(Ns, cond / 1e3, X4, Y4))
print(f"  marca N = 20: x = {X4(20):.1f}")
