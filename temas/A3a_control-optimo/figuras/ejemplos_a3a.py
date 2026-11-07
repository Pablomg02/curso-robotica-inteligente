"""Ejemplos numéricos de las hojas del tema A3a (control óptimo), para seguir a mano.

Uso:  python3 ejemplos_a3a.py > salida_ejemplos.txt
Requiere numpy; corre en CPU en unos segundos. Cada ejemplo lleva su número, que es el que citan
las hojas («ejemplo 4 del script»). Los trazados SVG se imprimen como 'path etiqueta: M...'
escalados al viewBox de la figura correspondiente.

Las cifras del x500 (masa, inercias, pesos de Bryson) son las de figuras_a3a.py.
"""
import itertools
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)
G = 9.81
M = 2.0 + 4 * 0.016076923
JY = 0.02389


def cab(n, titulo):
    print("=" * 72)
    print(f"EJEMPLO {n}. {titulo}")


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def riccati_escalar(a, b, q, r, pf, N):
    """Recursión de Riccati escalar hacia atrás. Devuelve listas Pi[0..N], K[0..N-1]."""
    Pi = [0.0] * (N + 1); K = [0.0] * N
    Pi[N] = pf
    for k in range(N - 1, -1, -1):
        K[k] = a * b * Pi[k + 1] / (r + b * b * Pi[k + 1])
        Pi[k] = q + a * a * Pi[k + 1] - (a * b * Pi[k + 1]) ** 2 / (r + b * b * Pi[k + 1])
    return Pi, K


def qp_enum(H, f, Gm=None, h=None):
    """min 1/2 U'HU + f'U  s.a.  G U <= h, enumerando conjuntos activos (exacto, solo para QP pequeños).
    Devuelve (U, lambdas, conjunto activo)."""
    nv = H.shape[0]
    if Gm is None or len(h) == 0:
        return np.linalg.solve(H, -f), np.zeros(0), ()
    nc = Gm.shape[0]
    mejor = None
    for na in range(0, min(nv, nc) + 1):
        for act in itertools.combinations(range(nc), na):
            act = list(act)
            Ga = Gm[act]
            if na:
                KKT = np.block([[H, Ga.T], [Ga, np.zeros((na, na))]])
                try:
                    sol = np.linalg.solve(KKT, np.concatenate([-f, h[act]]))
                except np.linalg.LinAlgError:
                    continue
                U, lam_a = sol[:nv], sol[nv:]
            else:
                U, lam_a = np.linalg.solve(H, -f), np.zeros(0)
            if np.all(Gm @ U <= h + 1e-9) and np.all(lam_a >= -1e-9):
                lam = np.zeros(nc); lam[act] = lam_a
                cost = 0.5 * U @ H @ U + f @ U
                if mejor is None or cost < mejor[3] - 1e-12:
                    mejor = (U, lam, tuple(act), cost)
        if mejor is not None:
            return mejor[:3]
    return mejor[:3] if mejor else (None, None, None)


def condensar_escalar(a, b, q, r, pf, N):
    """Matrices de la forma condensada para x_{k+1} = a x_k + b u_k (estado y mando escalares)."""
    Phi = np.array([a ** (k + 1) for k in range(N)])
    Gam = np.zeros((N, N))
    for k in range(N):
        for j in range(k + 1):
            Gam[k, j] = a ** (k - j) * b
    Qb = np.diag([q] * (N - 1) + [pf])
    Rb = r * np.eye(N)
    H = Gam.T @ Qb @ Gam + Rb
    return Phi, Gam, Qb, Rb, H


# ---------------------------------------------------------------------------------------------
cab(1, "LQR ESCALAR A MANO: x_{k+1} = x_k + u_k, q = r = 1, P_f = 0, N = 3")
a, b, q, r = 1.0, 1.0, 1.0, 1.0
Pi, K = riccati_escalar(a, b, q, r, 0.0, 3)
for k in range(3, -1, -1):
    extra = f",  K_{k} = {K[k]:.4f}" if k < 3 else ""
    print(f"  Pi_{k} = {Pi[k]:.4f}{extra}")
x = 1.0; coste = 0.0
for k in range(3):
    u = -K[k] * x
    print(f"  k = {k}: x = {x:.4f}, u = -K_{k} x = {u:.4f}, coste de etapa x^2 + u^2 = {x*x + u*u:.4f}")
    coste += x * x + u * u
    x = a * x + b * u
print(f"  x_3 = {x:.4f}; coste total = {coste:.4f}; Pi_0 * x_0^2 = {Pi[0]:.4f}  (coinciden)")
# comprobación por fuerza bruta: minimizar el coste sobre (u0, u1, u2) en una rejilla fina
g = np.linspace(-1, 1, 201)
U0, U1, U2 = np.meshgrid(g, g, g, indexing="ij")
X1 = 1 + U0; X2 = X1 + U1
Jb = 1 + U0**2 + X1**2 + U1**2 + X2**2 + U2**2
i = np.unravel_index(np.argmin(Jb), Jb.shape)
print(f"  fuerza bruta en rejilla de paso 0,01: mínimo {Jb[i]:.4f} en u = ({g[i[0]]:.2f}, {g[i[1]]:.2f}, {g[i[2]]:.2f})")

# ---------------------------------------------------------------------------------------------
cab(2, "DARE ESCALAR: la recursión converge al punto fijo; detectabilidad")
Pi, K = riccati_escalar(a, b, q, r, 0.0, 12)
seq = [Pi[12 - j] for j in range(13)]
print("  Pi hacia atrás (j pasos desde el final):", ", ".join(f"{v:.4f}" for v in seq))
phi = (1 + 5 ** 0.5) / 2
print(f"  DARE escalar con a = b = q = r = 1: pi^2 - pi - 1 = 0 -> pi = (1 + sqrt 5)/2 = {phi:.4f}")
print(f"  K_inf = pi/(1 + pi) = {phi/(1+phi):.4f}; polo en lazo cerrado a - b K = {1 - phi/(1+phi):.4f}")
# detectabilidad: a = 2, q = 0
a2, q2 = 2.0, 0.0
print("  a = 2, b = r = 1, q = 0: soluciones de la DARE pi = 0 y pi = (a^2 - 1) r / b^2 =", (a2**2 - 1))
for p in (0.0, 3.0):
    Kp = a2 * p / (1 + p)
    print(f"    pi = {p:.0f}: K = {Kp:.2f}, polo a - bK = {a2 - Kp:.2f}")
Pi2, _ = riccati_escalar(a2, 1.0, q2, 1.0, 0.0, 30)
print(f"    la recursión desde P_f = 0 se queda en Pi = {Pi2[0]:.2f}: no hacer nada cuesta 0 y el óptimo no estabiliza")
X = lin(0, 12, 60, 500); Y = lin(0, 2, 230, 30)
print("path pi:", path(range(13), seq, X, Y))
print(f"  y(phi) = {Y(phi):.1f}")

# ---------------------------------------------------------------------------------------------
cab(3, "PESOS DE BRYSON Y CADENA DE INTEGRADORES DEL EJE x DEL x500")
for nombre, xm in (("x (0,5 m)", 0.5), ("v_x (1 m/s)", 1.0), ("theta (0,3 rad)", 0.3), ("q (2 rad/s)", 2.0)):
    print(f"  Q_ii para {nombre}: 1/x_max^2 = {1/xm**2:.2f}; un error igual al admisible cuesta 1")
print(f"  R para el par de cabeceo (1 N m): {1/1.0**2:.2f}")
print(f"  cadena: tau_y --1/J_y = {1/JY:.1f}--> q' ; q --> theta' ; theta --g = {G}--> v_x' ; v_x --> x'")

# ---------------------------------------------------------------------------------------------
cab(4, "MÁRGENES DEL LQR: Nyquist del eje x del x500 en continuo y en discreto")
Ac = np.zeros((4, 4)); Ac[0, 1] = 1; Ac[1, 2] = G; Ac[2, 3] = 1
Bc = np.zeros((4, 1)); Bc[3, 0] = 1 / JY
Qx = np.diag([1 / 0.5**2, 1 / 1.0**2, 1 / 0.3**2, 1 / 2.0**2]); Rx = np.array([[1.0]])


def care(A, B, Q, R):
    n = A.shape[0]
    Hm = np.block([[A, -B @ np.linalg.solve(R, B.T)], [-Q, -A.T]])
    w, V = np.linalg.eig(Hm)
    Vs = V[:, w.real < 0]
    P = np.real(Vs[n:] @ np.linalg.inv(Vs[:n]))
    return 0.5 * (P + P.T)


Pc = care(Ac, Bc, Qx, Rx)
Kc = np.linalg.solve(Rx, Bc.T @ Pc)
ws = np.logspace(-2, 3, 4000)
Lc = np.array([(Kc @ np.linalg.solve(1j * w * np.eye(4) - Ac, Bc))[0, 0] for w in ws])
print(f"  continuo: min |1 + L(jw)| = {np.min(np.abs(1 + Lc)):.4f}  (la teoría dice >= 1); cota 2 arcsin(1/2) = {2*math.degrees(math.asin(0.5)):.1f}°")
icross = np.argmin(np.abs(np.abs(Lc) - 1))
pm = 180 + math.degrees(np.angle(Lc[icross]))
print(f"  continuo: |L| = 1 en w = {ws[icross]:.2f} rad/s, margen de fase = {pm:.1f}°")


def expm(Mx):
    w, V = np.linalg.eig(Mx)
    return np.real(V @ np.diag(np.exp(w)) @ np.linalg.inv(V))


def c2d(A, B, dt):
    n, m = B.shape
    Ma = np.zeros((n + m, n + m)); Ma[:n, :n] = A * dt; Ma[:n, n:] = B * dt
    # serie de Taylor larga (las matrices son nilpotentes o pequeñas)
    E = np.eye(n + m); T = np.eye(n + m)
    for k in range(1, 30):
        T = T @ Ma / k; E = E + T
    return E[:n, :n], E[:n, n:]


def dare(A, B, Q, R):
    P = Q.copy()
    for _ in range(200000):
        Kd = np.linalg.solve(B.T @ P @ B + R, B.T @ P @ A)
        Pn = Q + A.T @ P @ A - A.T @ P @ B @ Kd
        if np.max(np.abs(Pn - P)) < 1e-11 * max(1, np.max(np.abs(P))):
            return Pn, Kd
        P = Pn
    return P, Kd


for dt in (0.01, 0.05, 0.1):
    Ad, Bd = c2d(Ac, Bc, dt)
    Pd, Kd = dare(Ad, Bd, Qx * dt, Rx * dt)
    wd = np.linspace(1e-3, math.pi / dt, 6000)
    Ld = np.array([(Kd @ np.linalg.solve(np.exp(1j * w * dt) * np.eye(4) - Ad, Bd))[0, 0] for w in wd])
    ic = np.argmin(np.abs(np.abs(Ld) - 1))
    pmd = 180 + math.degrees(np.angle(Ld[ic]))
    gam = np.min(np.abs(1 + Ld))
    print(f"  discreto, T = {dt:.2f} s: gamma = min |1 + L| = {gam:.3f}, cota 2 arcsin(gamma/2) = {2*math.degrees(math.asin(gam/2)):.1f}°, margen de fase real = {pmd:.1f}°")
# trazado de Nyquist (continuo): escala 1 unidad = 70 px, centro (-1, 0) en (190, 150)
sel = (ws > 0.3)
Lp = Lc[sel]
Xn = lambda re: 260 + 70 * re
Yn = lambda im: 150 - 70 * im
keep = (np.abs(Lp.real) < 3.2) & (np.abs(Lp.imag) < 2.0)
keep = np.where(keep)[0][::12]
print("path nyquist:", path(Lp.real[keep], Lp.imag[keep], Xn, Yn))
print("path nyquist espejo:", path(Lp.real[keep], -Lp.imag[keep], Xn, Yn))
print(f"  punto |L| = 1: ({Xn(Lc[icross].real):.1f}, {Yn(Lc[icross].imag):.1f})")

# ---------------------------------------------------------------------------------------------
cab(5, "HORIZONTE DESLIZANTE: integrador escalar con |u| <= 0,5, N = 4, P_f = Pi_inf, desde x = 2")
N = 4; pinf = phi
Phi, Gam, Qb, Rb, H = condensar_escalar(1.0, 1.0, 1.0, 1.0, pinf, N)
Gm = np.vstack([np.eye(N), -np.eye(N)]); hm = 0.5 * np.ones(2 * N)
x = 2.0
planes = []
for t in range(4):
    f = Gam.T @ Qb @ Phi * x
    U, lam, act = qp_enum(H, f, Gm, hm)
    Xp = np.concatenate([[x], Phi * x + Gam @ U])
    planes.append((t, Xp.copy(), U.copy()))
    print(f"  t = {t}: x(t) = {x:.4f}; plan U = {np.round(U,4).tolist()}; estados predichos = {np.round(Xp,4).tolist()}; se aplica u_0 = {U[0]:.4f}")
    x = x + U[0]
# figura: una fila por instante; en cada fila x de 0 a 2 ocupa 56 px; tiempo 0..7 -> 90..560
Xh = lin(0, 7, 90, 560)
for t, Xp, U in planes[:3]:
    top = 22 + 82 * t
    Yh = lin(0, 2, top + 56, top)
    print(f"fila t = {t} (y de {top} a {top+56}):")
    print(f"  path plan t{t}:", path(range(t, t + N + 1), Xp, Xh, Yh))
    print("  puntos:", " ".join(f"({Xh(t+j):.1f},{Yh(v):.1f})" for j, v in enumerate(Xp)))

# ---------------------------------------------------------------------------------------------
cab(6, "FORMA CONDENSADA, N = 2: integrador escalar, q = r = 1, P_f = 0, x_0 = 1")
Phi, Gam, Qb, Rb, H = condensar_escalar(1.0, 1.0, 1.0, 1.0, 0.0, 2)
print("  Phi =", Phi.tolist(), "  Gamma =", Gam.tolist())
print("  Qbar = diag", np.diag(Qb).tolist(), "  Rbar = diag", np.diag(Rb).tolist())
print("  H = Gamma' Qbar Gamma + Rbar =", H.tolist())
f = Gam.T @ Qb @ Phi * 1.0
print("  término lineal Gamma' Qbar Phi x_0 =", f.tolist())
U = np.linalg.solve(H, -f)
print("  U* = -H^{-1} f =", U.tolist())
Pi, K = riccati_escalar(1, 1, 1, 1, 0, 2)
print(f"  Riccati con N = 2: K_0 = {K[0]:.2f}, K_1 = {K[1]:.2f} -> u_0 = {-K[0]:.2f}, x_1 = {1-K[0]:.2f}, u_1 = {-K[1]*(1-K[0]):.2f}")

# ---------------------------------------------------------------------------------------------
cab(7, "FORMA DISPERSA Y KKT, el mismo problema: variables (u0, x1, u1, x2), multiplicadores (lam1, lam2)")
# coste J = x0^2 + u0^2 + x1^2 + u1^2 + 0*x2^2 ; restricciones x1 - x0 - u0 = 0, x2 - x1 - u1 = 0
Hs = 2 * np.diag([1.0, 1.0, 1.0, 0.0])
Ceq = np.array([[-1.0, 1.0, 0.0, 0.0], [0.0, -1.0, -1.0, 1.0]])
deq = np.array([1.0, 0.0])                        # x0 = 1 pasa al lado derecho
KKT = np.block([[Hs, -Ceq.T], [Ceq, np.zeros((2, 2))]])
sol = np.linalg.solve(KKT, np.concatenate([np.zeros(4), deq]))
print("  matriz KKT (orden u0, x1, u1, x2, lam1, lam2):")
for fila in KKT:
    print("   ", " ".join(f"{v:5.1f}" for v in fila))
print("  solución:", np.round(sol, 4).tolist())
Pi, K = riccati_escalar(1, 1, 1, 1, 0, 2)
print(f"  comprobación lam_k = 2 Pi_k x_k: lam1 = 2*{Pi[1]:.0f}*{sol[1]:.2f} = {2*Pi[1]*sol[1]:.2f}, lam2 = 2*{Pi[2]:.0f}*x2 = 0")
# patrón de la KKT para N = 3 en el orden por etapas (u0, lam1, x1, u1, lam2, x2, u2, lam3, x3)
nombres = ["u0", "λ1", "x1", "u1", "λ2", "x2", "u2", "λ3", "x3"]
Np = 3
idx = {nm: i for i, nm in enumerate(nombres)}
Pat = np.zeros((9, 9), dtype=int)
for k in range(Np):
    Pat[idx[f"u{k}"], idx[f"u{k}"]] = 1
    if k > 0:
        Pat[idx[f"x{k}"], idx[f"x{k}"]] = 1
    # dinámica x_{k+1} = x_k + u_k
    for v in ([f"u{k}", f"x{k+1}"] + ([f"x{k}"] if k > 0 else [])):
        Pat[idx[f"λ{k+1}"], idx[v]] = Pat[idx[v], idx[f"λ{k+1}"]] = 1
Pat[idx["x3"], idx["x3"]] = 1
print("  patrón de no nulos de la KKT con N = 3 (orden por etapas):")
for nm, fila in zip(nombres, Pat):
    print(f"    {nm:3s}", "".join("■" if v else "·" for v in fila))
print("  rects kkt (celda 24 px, origen (64, 40)):", " ".join(f"{64+24*j},{40+24*i}" for i in range(9) for j in range(9) if Pat[i, j]))
bw = max(abs(i - j) for i in range(9) for j in range(9) if Pat[i, j])
print(f"  semiancho de banda = {bw}: no crece con N")

# ---------------------------------------------------------------------------------------------
cab(8, "KKT CON UNA DESIGUALDAD Y LAS TRES FAMILIAS DE SOLVERS: min 1/2 (u - 2)^2  s.a.  u <= 1")
print("  sin restricción: u = 2 (viola u <= 1)")
print("  conjunto activo: se supone activa -> u = 1; estacionariedad (u - 2) + lam = 0 -> lam = 1 >= 0: óptimo")
for mu in (1.0, 0.1, 0.01, 0.001):
    # (u - 2) + mu/(1 - u) = 0  ->  u^2 - 3u + 2 - mu = 0, raíz con u < 1
    uu = (3 - math.sqrt(9 - 4 * (2 - mu))) / 2
    print(f"  punto interior, mu = {mu:g}: u(mu) = {uu:.4f}; multiplicador estimado mu/(1 - u) = {mu/(1-uu):.4f}")
Xq = lin(-0.5, 2.5, 50, 470); Yq = lin(0, 3, 230, 30)
us_ = np.linspace(-0.5, 2.5, 61)
print("path parabola:", path(us_, 0.5 * (us_ - 2) ** 2, Xq, Yq))
print(f"  u = 1 en x = {Xq(1):.1f}; u = 2 en x = {Xq(2):.1f}; f(1) = 0,5 en y = {Yq(0.5):.1f}; f(2) = 0 en y = {Yq(0):.1f}")
for mu in (1.0, 0.1):
    uu = (3 - math.sqrt(9 - 4 * (2 - mu))) / 2
    print(f"  punto interior mu = {mu:g}: ({Xq(uu):.1f}, {Yq(0.5*(uu-2)**2):.1f})")
rho = 1.0; uu = 0.0; zz = 0.0; yy = 0.0
fila = []
for it in range(1, 9):
    uu = (2 + rho * zz - yy) / (1 + rho)
    zz = min(1.0, uu + yy / rho)
    yy = yy + rho * (uu - zz)
    fila.append(f"it {it}: u = {uu:.4f}, z = {zz:.4f}, y = {yy:.4f}")
print("  ADMM con rho = 1 (u libre, z proyectada en z <= 1, y multiplicador):")
for s_ in fila:
    print("   ", s_)

# ---------------------------------------------------------------------------------------------
cab(9, "ESTABILIDAD: desplazar y añadir con números (integrador, |u| <= 0,5, V_f = 2 x^2, N = 2)")
pf = 2.0; Kinf = phi / (1 + phi)
dec = pf * (1 - Kinf) ** 2 - pf
ell = 1 + Kinf ** 2
print(f"  condición terminal con kappa_f = -K_inf x: V_f(x+) - V_f(x) = {dec:.4f} x^2 <= -l = -{ell:.4f} x^2 -> {'se cumple' if dec <= -ell else 'no se cumple'}")
print(f"  (se cumple para todo p >= {ell/(1-(1-Kinf)**2):.4f} = Pi_inf)")
xf = 0.5 / Kinf
print(f"  X_f = [-{xf:.4f}, {xf:.4f}]: ahí |K_inf x| <= 0,5 y el lazo x+ = {1-Kinf:.4f} x lo mantiene dentro")
N = 2
Phi, Gam, Qb, Rb, H = condensar_escalar(1.0, 1.0, 1.0, 1.0, pf, N)
# restricciones: |u_k| <= 0,5 y |x_N| <= xf  (x_N = Phi[-1] x + Gam[-1] U)
def mpc(x):
    Gm = np.vstack([np.eye(N), -np.eye(N), Gam[-1:], -Gam[-1:]])
    hm = np.concatenate([0.5 * np.ones(2 * N), [xf - Phi[-1] * x, xf + Phi[-1] * x]])
    f = Gam.T @ Qb @ Phi * x
    U, lam, act = qp_enum(H, f, Gm, hm)
    Xp = Phi * x + Gam @ U
    V = x * x + sum(U ** 2) + sum(Xp[:-1] ** 2) + pf * Xp[-1] ** 2
    return U, Xp, V
x = 1.0
U, Xp, V = mpc(x)
print(f"  en x = 1: U* = {np.round(U,4).tolist()}, x predichos = {np.round(Xp,4).tolist()}, V*(x) = {V:.4f}")
l0 = x * x + U[0] ** 2
xp = x + U[0]
ut = np.array([U[1], -Kinf * Xp[-1]])
xs_t = [xp, xp + ut[0], xp + ut[0] + ut[1]]
Vt = xs_t[0] ** 2 + ut[0] ** 2 + xs_t[1] ** 2 + ut[1] ** 2 + pf * xs_t[2] ** 2
print(f"  l(x, u0*) = {l0:.4f}; x+ = {xp:.4f}")
print(f"  secuencia desplazada en x+: (u1*, kappa_f(x2*)) = ({ut[0]:.4f}, {ut[1]:.4f}); estados {np.round(xs_t,4).tolist()}; coste {Vt:.4f}")
print(f"  V*(x) - l(x, u0*) = {V - l0:.4f} >= coste desplazado {Vt:.4f}")
U2, Xp2, V2 = mpc(xp)
print(f"  óptimo en x+: U* = {np.round(U2,4).tolist()}, V*(x+) = {V2:.4f} <= {Vt:.4f} <= {V - l0:.4f}")
xx = 1.0; Vs = []
for k in range(6):
    Uk, Xk, Vk = mpc(xx); Vs.append(Vk); xx = xx + Uk[0]
print("  V* a lo largo del lazo cerrado desde x = 1:", ", ".join(f"{v:.4f}" for v in Vs))

# ---------------------------------------------------------------------------------------------
cab(10, "MPC EXPLÍCITO: integrador, |u| <= 0,5, P_f = Pi_inf; regiones del espacio de estados")
for N in (1, 2, 3, 4, 5):
    Phi, Gam, Qb, Rb, H = condensar_escalar(1.0, 1.0, 1.0, 1.0, phi, N)
    Gm = np.vstack([np.eye(N), -np.eye(N)]); hm = 0.5 * np.ones(2 * N)
    xs = np.linspace(-6, 6, 4801)
    acts = []; us = []
    for x in xs:
        U, lam, act = qp_enum(H, Gam.T @ Qb @ Phi * x, Gm, hm)
        acts.append(act); us.append(U[0])
    cambios = [i for i in range(1, len(xs)) if acts[i] != acts[i - 1]]
    fronteras = [0.5 * (xs[i] + xs[i - 1]) for i in cambios]
    print(f"  N = {N}: {len(set(acts))} conjuntos activos, {len(cambios)+1} regiones en [-6, 6]; fronteras en x = {[round(float(v),3) for v in fronteras]}")
    if N == 3:
        us3 = np.array(us); xs3 = xs; fr3 = fronteras
        # comprobar que en cada región u0 es afín: pendiente constante
        for lo, hi in zip([-6] + fr3, fr3 + [6]):
            mk = (xs3 > lo + 0.02) & (xs3 < hi - 0.02)
            p = np.polyfit(xs3[mk], us3[mk], 1)
            print(f"    región ({lo:.3f}, {hi:.3f}): u0 = {p[0]:.4f} x + {p[1]:.4f}")
Xe = lin(-4, 4, 60, 500); Ye = lin(-0.6, 0.6, 210, 30)
mk = np.abs(xs3) <= 4
print("path explicito N3:", path(xs3[mk][::20], us3[mk][::20], Xe, Ye))
print("  fronteras (px):", [round(float(Xe(v)), 1) for v in fr3 if abs(v) <= 4])

# ---------------------------------------------------------------------------------------------
cab(11, "SEGUIMIENTO SIN ERROR ESTACIONARIO: x+ = x + u + d con d = 0,2 desconocida, r = 0")
d = 0.2; Kl = Kinf
x = 0.0
for k in range(60):
    x = x + (-Kl * x) + d
print(f"  sin modelo de perturbación (u = -K x, K = {Kl:.4f}): x se queda en {x:.4f} = d/K = {d/Kl:.4f}")
x = 0.0; dh = 0.0; Lg = 0.5
traza = []
for k in range(40):
    us_, xs_ = -dh, 0.0                                  # cálculo de objetivo: x_s = r = 0, u_s = -d_hat
    u = us_ - Kl * (x - xs_)
    xn = x + u + d                                       # planta real
    dh = dh + Lg * (xn - (x + u + dh))                   # estimador: corrige d_hat con el error de predicción
    x = xn
    if k in (0, 1, 2, 4, 9, 19, 39):
        traza.append(f"k = {k+1}: x = {x:.4f}, d_hat = {dh:.4f}")
print("  con d estimada (ganancia del estimador 0,5) y objetivo u_s = -d_hat:")
for s_ in traza:
    print("   ", s_)

# ---------------------------------------------------------------------------------------------
cab(12, "RESTRICCIÓN ESTRECHADA CON LA INCERTIDUMBRE DEL ESTIMADOR")
for sig in (1.0, 2.0, 4.0):
    print(f"  theta_max = 20°, sigma_theta = {sig:.0f}°: |theta_hat| <= {20 - 2*sig:.0f}°  (margen de 2 sigma)")

# ---------------------------------------------------------------------------------------------
cab(13, "DISPARO SIMPLE FRENTE A MÚLTIPLE CON UN SISTEMA INESTABLE: x+ = 2 x + u")
for N in (5, 10, 20):
    print(f"  N = {N}: dx_N/du_0 = 2^(N-1) = {2**(N-1)}  (disparo simple); en disparo múltiple cada nodo solo ve un paso: ds_(k+1)/dq_k = 1")
print("  una desviación de 1e-3 en u_0 mueve x_20 en", 1e-3 * 2 ** 19)

# ---------------------------------------------------------------------------------------------
cab(14, "GAUSS-NEWTON FRENTE A NEWTON: r(w) = (w - 1, 0,5 (w^2 - 4)), min 1/2 |r|^2")
def r_(w): return np.array([w - 1.0, 0.5 * (w * w - 4.0)])
def J_(w): return np.array([1.0, w])
def H2(w): return 0.5 * 1.0 * 2           # d2 r2 / dw2 = 1 ; se multiplica por r2
w_ = 3.0
# óptimo de referencia
ws_ = np.linspace(0, 3, 300001); fs = 0.5 * ((ws_ - 1) ** 2 + 0.25 * (ws_ ** 2 - 4) ** 2)
wopt = ws_[np.argmin(fs)]
for _ in range(50):
    g_ = J_(wopt) @ r_(wopt); hh = J_(wopt) @ J_(wopt) + r_(wopt)[1] * 1.0
    wopt -= g_ / hh
print(f"  óptimo w* = {wopt:.6f}; residuos en el óptimo r(w*) = {np.round(r_(wopt),4).tolist()}")
for nombre, exacta in (("Gauss-Newton", False), ("Newton", True)):
    w = 3.0; fila = []
    for it in range(6):
        Jw = J_(w); rw = r_(w)
        hh = Jw @ Jw + (rw[1] * 1.0 if exacta else 0.0)
        w = w - (Jw @ rw) / hh
        fila.append(f"{abs(w - wopt):.2e}")
    print(f"  {nombre:12s}: |w_k - w*| =", ", ".join(fila))
w = wopt
print(f"  en w*: J'J = {J_(w)@J_(w):.4f}; término olvidado r2 * d2r2/dw2 = {r_(w)[1]:.4f}; razón = {r_(w)[1]/(J_(w)@J_(w)):.3f} (tasa lineal de GN)")

# ---------------------------------------------------------------------------------------------
cab(15, "POR QUÉ BASTA UNA ITERACIÓN: seguir la solución de w^3 + w = p(t) con un paso de Newton por muestra")
for T in (0.4, 0.2, 0.1, 0.05):
    ts = np.arange(0, 20, T)
    p = 2 * np.sin(0.5 * ts)
    w = 0.0                                                         # solución exacta en t = 0 (p = 0)
    errs = []; desp = []
    for k in range(1, len(ts)):
        # exacta (para medir el error): Newton convergido
        we = w
        for _ in range(50):
            we -= (we ** 3 + we - p[k]) / (3 * we ** 2 + 1)
        # RTI: un solo paso desde la solución anterior
        w_prev = w
        w = w - (w ** 3 + w - p[k]) / (3 * w ** 2 + 1)
        errs.append(abs(w - we))
        desp.append(abs(we - w_prev))
    print(f"  T = {T:.2f} s: la solución se mueve hasta {max(desp):.4f} por muestra; error tras un paso: máx {max(errs):.2e}, medio {np.mean(errs):.2e}")

# ---------------------------------------------------------------------------------------------
cab(16, "GEOMETRÍA DEL MPCC: camino circular de radio 1, p_d(th) = (cos th, sin th)")
th = 0.8
pd = np.array([math.cos(th), math.sin(th)]); tg = np.array([-math.sin(th), math.cos(th)])
p = np.array([0.6, 1.0])
e = p - pd; el = tg @ e; ec = e - el * tg
print(f"  vartheta = {th}: p_d = {np.round(pd,4).tolist()}, t = {np.round(tg,4).tolist()}; dron en p = {p.tolist()}")
print(f"  p - p_d = {np.round(e,4).tolist()}; e_l = t'(p - p_d) = {el:.4f}; e_c = {np.round(ec,4).tolist()}, |e_c| = {np.linalg.norm(ec):.4f}")
print(f"  comprobación: e_l^2 + |e_c|^2 = {el**2 + ec@ec:.4f} = |p - p_d|^2 = {e@e:.4f}; t'e_c = {tg@ec:.1e}")
ths = math.atan2(p[1], p[0]); dist = np.linalg.norm(p) - 1
print(f"  proyección exacta: vartheta* = atan2(p_y, p_x) = {ths:.4f}; distancia al camino = |p| - 1 = {dist:.4f}")
for th2 in (ths - 0.05, ths, ths + 0.2):
    pd2 = np.array([math.cos(th2), math.sin(th2)]); tg2 = np.array([-math.sin(th2), math.cos(th2)])
    e2 = p - pd2; el2 = tg2 @ e2; ec2 = e2 - el2 * tg2
    print(f"    con vartheta = {th2:.4f}: e_l = {el2:+.4f}, |e_c| = {np.linalg.norm(ec2):.4f}")
v_, lat = 20.0, 0.025
print(f"  seguimiento con retraso: a {v_:.0f} m/s ({v_*3.6:.0f} km/h), {1000*lat:.0f} ms de retraso son {v_*lat:.2f} m de error a lo largo del camino")

# ---------------------------------------------------------------------------------------------
cab(17, "COSTE DE RICCATI FRENTE A CONDENSADO: de dónde salen los recuentos y escala de la figura")
nn, mm = 12, 4
for Nn in (20, 40, 160):
    ric = Nn * (7 / 3 * nn**3 + 4 * nn**2 * mm + 2 * nn * mm**2 + mm**3 / 3)
    cond = Nn**3 * nn * mm**2 / 3 + Nn**3 * mm**3 / 3
    print(f"  N = {Nn}: Riccati {ric/1e6:.3f} Mflop, condensado {cond/1e6:.3f} Mflop, x{cond/ric:.1f}")
    print(f"     QP condensado: {Nn*mm} variables; Cholesky de una matriz p x p ~ p^3/3 = {(Nn*mm)**3/3/1e6:.3f} Mflop")
Y4 = lambda kf: 250 + (math.log10(kf) - 0) * (30 - 250) / 6
for e10 in (3, 5, 7, 9):
    print(f"  marca de 10^{e10} flop en la figura (eje en kflop de 1 a 10^6): y = {Y4(10**(e10-3)):.1f}")
