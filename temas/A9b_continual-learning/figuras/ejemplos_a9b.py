"""Ejemplos numéricos y datos de las figuras nuevas del tema A9b (desarrollo de la matemática).

Uso:  python3 ejemplos_a9b.py > salida_ejemplos.txt      (numpy, CPU, unos segundos)

Cada ejemplo lleva el número con el que se cita en las hojas («ejemplo N del script»).
Los experimentos con la red de la aerodinámica (olvido y plasticidad) están en figuras_a9b.py
(JAX en GPU, unos 10 minutos) y no se repiten aquí: sus cifras están en salida_gpu.txt. Este
script solo reproduce lo que se sigue a mano y lo que dibujan las figuras nuevas. La fuerza
residual, los regímenes y la red son los de figuras_a9b.py.
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)
G = 9.81


def titulo(n, texto):
    print("=" * 78)
    print(f"EJEMPLO {n}. {texto}")


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


# =============================================================================== PARTE 1
titulo(1, "TAMAÑO DE LA RED DEL EXPERIMENTO Y DE SU MATRIZ DE FISHER (7 -> 64 -> 64 -> 3)")
capas = [(7, 64), (64, 64), (64, 3)]
P = sum(a * b + b for a, b in capas)
for a, b in capas:
    print(f"  capa {a:2d} -> {b:2d}: {a}·{b} pesos + {b} sesgos = {a*b+b}")
print(f"  parámetros P = {P}")
print(f"  Fisher completa: P² = {P*P:,} números ({P*P*4/1e6:.1f} MB en float32)")
print(f"  Fisher diagonal: P = {P} números ({P*4/1e3:.1f} kB en float32)")

titulo(2, "MATRIZ DE RESULTADOS 3×3 HECHA A MANO (ACIERTOS EN %, CONVENIO DE GEM)")
R = np.array([[90, 40, 20], [70, 85, 30], [60, 75, 88]], float)
b = np.array([10, 10, 10], float)
T = 3
print("  R (fila i: tras entrenar la tarea i; columna j: acierto en la tarea j):")
for i in range(T):
    print("    " + "  ".join(f"{R[i, j]:4.0f}" for j in range(T)))
print("  b̄ (red sin entrenar):", b)
acc = R[-1].mean()
bwt_terms = [R[-1, i] - R[i, i] for i in range(T - 1)]
fwt_terms = [R[i - 1, i] - b[i] for i in range(1, T)]
print(f"  ACC = ({' + '.join(f'{x:.0f}' for x in R[-1])})/3 = {acc:.2f}")
print(f"  BWT = ({' + '.join(f'({x:+.0f})' for x in bwt_terms)})/2 = {np.mean(bwt_terms):+.1f}")
print(f"  FWT = ({' + '.join(f'({x:+.0f})' for x in fwt_terms)})/2 = {np.mean(fwt_terms):+.1f}")

titulo(3, "LAS MÉTRICAS CON ERRORES: LA MATRIZ DEL AJUSTE FINO DE figuras_a9b.py (N²)")
nombres = ["flotar", "agresivo", "avance", "sub./baj.", "lateral"]
R5 = np.array([[0.0001, 0.9018, 0.0267, 0.0137, 0.4735],
               [0.0034, 0.0059, 0.0038, 0.0025, 0.0038],
               [0.0002, 0.1370, 0.0000, 0.0051, 0.0467],
               [0.0000, 0.1150, 0.0019, 0.0000, 0.0961],
               [0.0012, 0.2371, 0.0149, 0.0169, 0.0001]])
var5 = np.array([0.01, 2.70, 0.04, 0.13, 0.25])      # varianza de la fuerza en cada régimen (salida_gpu.txt, redondeada)
T5 = 5
print("  ACC = media de la última fila = (" + " + ".join(f"{x:.4f}" for x in R5[-1]) + f") / 5 = {R5[-1].mean():.4f}")
d = [R5[-1, i] - R5[i, i] for i in range(T5 - 1)]
for i in range(T5 - 1):
    print(f"    BWT, {nombres[i]:9s}: R_5,{i+1} - R_{i+1},{i+1} = {R5[-1, i]:.4f} - {R5[i, i]:.4f} = {d[i]:+.4f}")
print(f"  BWT = suma / 4 = {sum(d):.4f} / 4 = {np.mean(d):+.4f}   (positivo: el error sube, olvido)")
f = [var5[i] - R5[i - 1, i] for i in range(1, T5)]
for i in range(1, T5):
    print(f"    FWT, {nombres[i]:9s}: b̄ - R_{i},{i+1} = {var5[i]:.2f} - {R5[i-1, i]:.4f} = {f[i-1]:+.4f}")
print(f"  FWT = suma / 4 = {sum(f):.4f} / 4 = {np.mean(f):+.4f}   (positivo: lo aprendido antes ayuda)")
print("  (figuras_a9b.py da +0,0660 y +0,5265: usa las varianzas sin redondear y la matriz sin redondear)")

titulo(4, "DOS TAREAS EN EL ESPACIO DE DOS PARÁMETROS: AJUSTE FINO, EWC Y CONJUNTO")
# modelo lineal f_theta(x) = theta1 x1 + theta2 x2, pérdida L_k = ½ E[(f - y)²] (= -log p con varianza 1, sin cte.)
thA, thB = np.array([2.0, 0.0]), np.array([0.5, 1.0])
HA, HB = np.array([1.0, 0.01]), np.array([0.09, 1.0])   # E[x1²], E[x2²] en cada tarea (entradas independientes de media 0)
LA = lambda th: 0.5 * np.sum(HA * (th - thA) ** 2)
LB = lambda th: 0.5 * np.sum(HB * (th - thB) ** 2)
print("  Tarea A: x1 ~ N(0, 1), x2 ~ N(0, 0,1²), y = 2 x1          -> θ*_A = (2, 0),   H_A = diag(1, 0,01)")
print("  Tarea B: x1 ~ N(0, 0,3²), x2 ~ N(0, 1), y = 0,5 x1 + x2    -> θ*_B = (0,5, 1), H_B = diag(0,09, 1)")
print(f"  L_A(θ*_A) = 0 | L_B(θ*_A) = ½(0,09·1,5² + 1·1²) = {LB(thA):.4f}")
print(f"  L_A(θ*_B) = ½(1·1,5² + 0,01·1²) = {LA(thB):.4f} | L_B(θ*_B) = 0")
print("  EWC con la Fisher exacta (F = H_A, porque el modelo es lineal y gaussiano), coordenada a coordenada:")
print("     θ_i = (H_B,i θ*_B,i + λ F_i θ*_A,i) / (H_B,i + λ F_i)")
for lam in (0.0, 0.1, 1.0, 10.0, 100.0):
    th = (HB * thB + lam * HA * thA) / (HB + lam * HA)
    print(f"    λ = {lam:6.1f}: θ = ({th[0]:.4f}, {th[1]:.4f}) | L_A = {LA(th):.4f} | L_B = {LB(th):.4f} | L_A + L_B = {LA(th)+LB(th):.4f}")
thJ = (HB * thB + HA * thA) / (HB + HA)
print(f"  Conjunto (min L_A + L_B) = EWC con λ = 1: θ = ({thJ[0]:.4f}, {thJ[1]:.4f})")
# región buena: L <= 0,1  -> elipse de semiejes sqrt(2·0,1/H_i)
for nom, H in (("A", HA), ("B", HB)):
    print(f"  región L_{nom} <= 0,1: semiejes sqrt(0,2/H) = ({math.sqrt(0.2/H[0]):.3f}, {math.sqrt(0.2/H[1]):.3f})")
# trayectorias de descenso por gradiente desde θ*_A, paso 0,5
X = lin(-0.2, 2.8, 50, 410); Y = lin(-0.4, 1.6, 290, 50)
for nom, lam in (("ajuste fino", 0.0), ("EWC λ=1", 1.0)):
    th = thA.copy(); tr = [th.copy()]
    for k in range(120):
        th = th - 0.5 * (HB * (th - thB) + lam * HA * (th - thA))
        tr.append(th.copy())
    tr = np.array(tr)
    sel = list(range(0, 10)) + list(range(10, 121, 5))
    print(f"  trayectoria {nom}: tras 120 pasos θ = ({tr[-1,0]:.4f}, {tr[-1,1]:.4f}); paso 4: ({tr[4,0]:.3f}, {tr[4,1]:.3f})")
    print(f"path {nom}:", path(tr[sel, 0], tr[sel, 1], X, Y))
print("  centro A px", f"{X(2):.1f},{Y(0):.1f}", "| centro B px", f"{X(0.5):.1f},{Y(1):.1f}", "| conjunto px", f"{X(thJ[0]):.1f},{Y(thJ[1]):.1f}")
print("  semiejes px A", f"{math.sqrt(0.2/HA[0])*120:.1f}, {math.sqrt(0.2/HA[1])*120:.1f}", "| B", f"{math.sqrt(0.2/HB[0])*120:.1f}, {math.sqrt(0.2/HB[1])*120:.1f}")
print("  ejes px: θ1 = 0 en x =", f"{X(0):.1f}", "; θ1 = 1, 2 en", f"{X(1):.1f}, {X(2):.1f}", "; θ2 = 0, 1 en y =", f"{Y(0):.1f}, {Y(1):.1f}")

titulo(5, "LAPLACE EN EL EJEMPLO: EL POSTERIOR DE A COMO GAUSSIANA")
NA = 100
print(f"  Con N_A = {NA} datos de A y ruido de varianza 1, la precisión del posterior es N_A·F = diag({NA*HA[0]:.0f}, {NA*HA[1]:.0f})")
print(f"  desviaciones típicas: θ1 ± {1/math.sqrt(NA*HA[0]):.2f}, θ2 ± {1/math.sqrt(NA*HA[1]):.2f}  (θ2 apenas lo fija A)")
print("  El log-posterior de un modelo lineal y gaussiano es exactamente cuadrático: aquí Laplace es exacta.")
print(f"  Penalización de EWC en θ*_B con λ = 1: ½(1·(0,5-2)² + 0,01·(1-0)²) = {LA(thB):.4f} = L_A(θ*_B)")

titulo(6, "FISHER CON EL JACOBIANO FRENTE A FISHER CON EL GRADIENTE DEL ERROR")
rng = np.random.default_rng(0)
N = 1000
XA = np.column_stack([rng.normal(0, 1, N), rng.normal(0, 0.1, N)])
for sig in (0.0, 0.1):
    yA = XA @ thA + sig * rng.normal(0, 1, N)
    th_hat = np.linalg.lstsq(XA, yA, rcond=None)[0]
    r = yA - XA @ th_hat                                # residuo en los datos
    J = XA                                              # jacobiano de f respecto a θ: la propia entrada
    F_jac = np.mean(J ** 2, 0)
    F_emp = np.mean((r[:, None] * J) ** 2, 0)           # cuadrado del gradiente del error, por dato
    th_ewc = (HB * thB + F_emp * thA) / (HB + F_emp)
    print(f"  ruido σ = {sig}: θ̂_A = ({th_hat[0]:.4f}, {th_hat[1]:.4f}) | residuo cuadrático medio {np.mean(r**2):.4f}")
    print(f"     F con el jacobiano          = ({F_jac[0]:.4f}, {F_jac[1]:.5f})")
    print(f"     F con el gradiente del error = ({F_emp[0]:.6f}, {F_emp[1]:.7f})")
    print(f"     EWC (λ = 1) con esta última: θ = ({th_ewc[0]:.4f}, {th_ewc[1]:.4f}) | L_A = {LA(th_ewc):.4f} (ajuste fino: {LA(thB):.4f})")

titulo(7, "REPETICIÓN COMO ESTIMADOR DE LA PÉRDIDA CONJUNTA")
print("  En figuras_a9b.py, al entrenar el régimen i: pérdida del lote = 0,5·(lote nuevo) + 0,5·(lote de la memoria),")
print("  y la memoria guarda 200 puntos de cada régimen terminado; el lote de memoria se sortea uniforme entre todos.")
for i in range(2, 6):
    print(f"    régimen {i}: peso del nuevo 0,5; de cada uno de los {i-1} anteriores 0,5/{i-1} = {0.5/(i-1):.3f}"
          f" | conjunto: 1/{i} = {1/i:.3f} cada uno")
print(f"  memoria: 200 puntos × (7 entradas + 3 salidas) × 4 B = {200*10*4/1e3:.0f} kB por régimen; 5 regímenes, {5*200*10*4/1e3:.0f} kB")
# estimador sin sesgo: media de gradientes de lotes mezclados = gradiente de ½L_B + ½L_A
th0 = thA.copy()
g_exact = 0.5 * HB * (th0 - thB) + 0.5 * HA * (th0 - thA)
XB = lambda n: np.column_stack([rng.normal(0, 0.3, n), rng.normal(0, 1, n)])
gs = []
for _ in range(20000):
    xb = XB(8); xa = XA[rng.integers(0, 200, 8)]
    eb = xb @ th0 - xb @ thB; ea = xa @ th0 - xa @ thA
    gs.append(0.5 * (xb * eb[:, None]).mean(0) + 0.5 * (xa * ea[:, None]).mean(0))
gs = np.array(gs)
print(f"  en θ*_A, gradiente exacto de ½L_B + ½L_A = ({g_exact[0]:.4f}, {g_exact[1]:.4f})")
print(f"  media de 20000 gradientes de lotes de 8 + 8 (memoria de 200) = ({gs[:,0].mean():.4f}, {gs[:,1].mean():.4f});"
      f" un lote suelto: desv. típica ({gs[:,0].std():.3f}, {gs[:,1].std():.3f})")
for m in (5, 20, 200):
    xm = XA[:m]
    Hm = xm.T @ xm / m                                  # curvatura de la memoria (completa, no diagonal)
    Hb = np.diag(HB)
    th = np.linalg.solve(Hb + Hm, Hb @ thB + Hm @ thA)
    print(f"    memoria de {m:3d} puntos de A: θ = ({th[0]:.4f}, {th[1]:.4f}) | L_A = {LA(th):.4f} | L_B = {LB(th):.4f}")
print(f"    memoria infinita (= conjunto):  θ = ({thJ[0]:.4f}, {thJ[1]:.4f}) | L_A = {LA(thJ):.4f} | L_B = {LB(thJ):.4f}")

# =============================================================================== PARTE 2
def Phi(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def phi(z):
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


titulo(8, "UNIDADES ACTIVAS, LATENTES Y MUERTAS: UNA CAPA DE CUATRO UNIDADES ReLU")
unidades = [("1", 0.5, 1.0), ("2", 1.0, 0.5), ("3", -0.9, 0.6), ("4", -4.0, 0.8)]   # preactivación z ~ N(mu, sd) sobre los datos
rng = np.random.default_rng(1)
Eh, Eh_n = [], []
for nom, mu, sd in unidades:
    e = mu * Phi(mu / sd) + sd * phi(mu / sd)          # E[max(0, z)] para z gaussiana
    z = rng.normal(mu, sd, 1024)
    Eh.append(e); Eh_n.append(np.maximum(z, 0).mean())
    print(f"  unidad {nom}: z ~ N({mu:+.1f}, {sd:.1f}²) | activa (z > 0) el {100*Phi(mu/sd):6.3f} % de las veces | "
          f"E|h| = {e:.4f} (con 1024 entradas: {np.maximum(z,0).mean():.4f}, activas {np.sum(z>0)})")
Eh_n = np.array(Eh_n)
s = Eh_n / Eh_n.mean()
print(f"  media de la capa (1024 entradas) = {Eh_n.mean():.4f}")
for (nom, mu, sd), si in zip(unidades, s):
    print(f"    s_{nom} = {si:.4f}  -> τ = 0,1: {'latente' if si <= 0.1 else 'activa'} | τ = 0: {'muerta' if si <= 0 else 'no muerta'}")
# figura: densidades de z de las cuatro unidades
Xz = lin(-6.5, 3.5, 30, 430)
for k, (nom, mu, sd) in enumerate(unidades):
    zs = np.linspace(mu - 3.2 * sd, mu + 3.2 * sd, 40)
    dens = np.exp(-0.5 * ((zs - mu) / sd) ** 2)        # altura relativa (máximo 1)
    base = 70 + 62 * k
    print(f"path densidad unidad {nom}:", path(zs, dens, Xz, lin(0, 1, base, base - 42)))
print("  z = 0 en x =", f"{Xz(0):.1f}", "| z = -4, -2, 2 en", f"{Xz(-4):.1f}, {Xz(-2):.1f}, {Xz(2):.1f}")
# por qué no se recupera
w = np.array([0.5, -0.3]); b0 = -3.0
xx = rng.normal(0, 1, (10000, 2)); zz = xx @ w + b0
print(f"  unidad con w = (0,5, -0,3), b = -3, entradas N(0, I): z ~ N(-3, {math.sqrt(0.34):.3f}²); z > 0 en {np.sum(zz>0)} de 10000 entradas")
print("  -> dh/dz = 0 en todas: los gradientes de w, b y de sus pesos de salida son exactamente 0 (Adam no la mueve)")

titulo(9, "ENCOGER Y PERTURBAR: θ <- 0,6 θ + 0,1 θ_nueva SOBRE LA NORMA DE LOS PESOS")
rng = np.random.default_rng(2)
th = rng.normal(0, 1, P); th *= 15.9 / np.linalg.norm(th)
tn = rng.normal(0, 1, P); tn *= 15.9 / np.linalg.norm(tn)
nuevo = 0.6 * th + 0.1 * tn
print(f"  ‖θ‖ = 15,9, ‖θ_nueva‖ = 15,9 (vectores independientes de {P} componentes)")
print(f"  ‖0,6 θ + 0,1 θ_nueva‖ = {np.linalg.norm(nuevo):.2f};  aproximación sqrt(0,6² + 0,1²)·15,9 = {math.sqrt(0.37)*15.9:.2f}")
print(f"  coseno entre θ y el resultado: {np.dot(th, nuevo)/np.linalg.norm(th)/np.linalg.norm(nuevo):.3f}  (la dirección se conserva casi entera)")
print("  en figuras_a9b.py la norma final con encoger y perturbar es 6,1 (salida_gpu.txt)")

titulo(10, "RETROPROPAGACIÓN CONTINUA: UTILIDAD, MADUREZ Y TASA DE REEMPLAZO")
eta = 0.99
c = 0.5 * (0.2 + 0.1 + 0.3)
print(f"  contribución de una unidad: |h| = 0,5 y pesos de salida (0,2, -0,1, 0,3): c = 0,5·(0,2 + 0,1 + 0,3) = {c:.2f}")
for n in (1, 10, 100, 300, 500):
    print(f"    tras {n:3d} pasos con c constante, desde u = 0: u = c(1 - 0,99^n) = {c*(1-eta**n):.4f}")
print("  dos unidades maduras (|h| medio, suma de |pesos de salida|):")
cand = [("1", 0.80, 0.9), ("2", 0.40, 0.05)]
for nom, h, ws in cand:
    print(f"    unidad {nom}: {h:.2f} · {ws:.2f} = {h*ws:.3f}")
print("    -> se reinicializa la 2: activa, pero casi no influye en la salida")
rho, H = 1e-4, 64
print(f"  reemplazos por paso y capa: ρ·H = {rho}·{H} = {rho*H:.4f} -> uno cada {1/(rho*H):.0f} pasos")
print(f"  por tarea de 500 pasos: {500*rho*H:.1f} por capa; en 300 tareas: {300*500*rho*H:.0f} por capa ({300*500*rho*H/H:.0f} veces la anchura)")
print(f"  en 600 tareas: {600*500*rho*H:.0f} por capa")

# =============================================================================== PARTE 4
titulo(11, "QUÉ ENTRA Y QUÉ SALE DEL MODELO: LA FUERZA RESIDUAL DE figuras_a9b.py")


def residual(v, T, zb, vw=np.zeros(3), dm=0.0):
    vr = vw - v
    term = {}
    term["resistencia"] = 0.15 * vr + 0.03 * np.linalg.norm(vr) * vr
    vperp = vr - np.dot(vr, zb) * zb
    term["arrastre de rotor"] = 0.04 * T * vperp
    term["pérdida de empuje"] = -0.06 * T ** 2 * zb
    term["carga"] = -dm * G * np.array([0, 0, 1.0])
    return sum(term.values()), term


casos = [("flotar sin viento", np.zeros(3), 1.0, np.array([0, 0, 1.0]), np.zeros(3), 0.0),
         ("avance lento, 2 m/s en x", np.array([2.0, 0, 0]), 1.0, np.array([0, 0, 1.0]), np.zeros(3), 0.0),
         ("flotar con viento de 5 m/s en x", np.zeros(3), 1.0, np.array([0, 0, 1.0]), np.array([5.0, 0, 0]), 0.0),
         ("flotar con 0,3 kg de carga", np.zeros(3), 1.0, np.array([0, 0, 1.0]), np.zeros(3), 0.3)]
for nom, v, T_, zb, vw, dm in casos:
    f, term = residual(v, T_, zb, vw, dm)
    print(f"  {nom}: entrada (v, T, z_b) = ({v}, {T_}, {zb}) | viento {vw}, carga {dm} kg")
    for k, val in term.items():
        print(f"      {k:18s} = {val}")
    print(f"      fuerza residual   = {f}  N")
print("  regímenes (figuras_a9b.py): velocidad = centro + escala·N(0, I), inclinación del eje y empuje:")
reg = [("flotar", (0, 0, 0), (0.3, 0.3, 0.2), 0.12, 0.1), ("agresivo", (0, 0, 0), (3.0, 3.0, 1.5), 0.6, 0.4),
       ("avance lento", (2, 0, 0), (0.6, 0.3, 0.2), 0.12, 0.1), ("subidas y bajadas", (0, 0, 0), (0.3, 0.3, 1.5), 0.12, 0.1),
       ("lateral rápido", (0, 4.5, 0), (0.4, 1.0, 0.2), 0.3, 0.1)]
for nom, cen, esc, tilt, sT in reg:
    print(f"    {nom:18s}: centro {cen} m/s, escala {esc} m/s, inclinación ±{tilt}, empuje 1 ± {sT}")

titulo(12, "ERROR NORMALIZADO DEL FLUJO DE TAREAS")
for e in (0.007, 0.016, 0.045, 0.159):
    print(f"  error normalizado {e:.3f}: el error es el {100*e:.1f} % de la varianza de la fuerza (R² = {1-e:.3f})")
