"""Ejemplos numéricos y datos de las figuras nuevas de A6a (numpy, CPU, segundos).

Uso:  python3 ejemplos_a6a.py > salida_ejemplos.txt

Cada ejemplo lleva un número que se cita en las hojas («ejemplos_a6a.py, ejemplo N»).
Las cifras del PPO en JAX con el x500 (tiempos, errores, semillas, KL) salen de figuras_a6a.py,
que usa la GPU y no es determinista entre ejecuciones: no se relanza aquí.
"""
import math
import numpy as np

np.set_printoptions(precision=4, suppress=True)


def path(xs, ys, X, Y):
    return "M" + " L".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))


def lin(d0, d1, p0, p1):
    return lambda v: p0 + (v - d0) * (p1 - p0) / (d1 - d0)


def cab(n, titulo):
    print("=" * 72)
    print(f"EJEMPLO {n}. {titulo}")


sig = lambda x: 1 / (1 + math.exp(-x))

# ---------------------------------------------------------------------------
cab(1, "TRUCO DEL LOGARITMO EN UN BANDIDO DE DOS ACCIONES, CON Y SIN LÍNEA BASE")
th = 0.0
p = sig(th)
R = {1: 10.0, 0: 8.0}                       # derecha = 1, izquierda = 0
score = {1: 1 - p, 0: -p}                   # d/dθ log π(a), con π(1) = σ(θ)
g_true = p * (1 - p) * (R[1] - R[0])
print(f"  π(der) = σ({th}) = {p}; R(der) = {R[1]}, R(izq) = {R[0]}")
print(f"  J(θ) = σ(θ)·10 + (1-σ(θ))·8; dJ/dθ = σ(1-σ)(10-8) = {g_true}")
print(f"  score: d log π(der)/dθ = 1-σ = {score[1]}; d log π(izq)/dθ = -σ = {score[0]}")
for b in (0.0, 9.0):
    m = {a: score[a] * (R[a] - b) for a in (0, 1)}
    media = p * m[1] + (1 - p) * m[0]
    var = p * m[1] ** 2 + (1 - p) * m[0] ** 2 - media ** 2
    print(f"  b = {b}: muestra si sale der = {m[1]:+.2f}, si sale izq = {m[0]:+.2f}; media {media:.3f}; "
          f"desviación {math.sqrt(var):.3f}; error típico con 64 muestras {math.sqrt(var/64):.3f}")
rng = np.random.default_rng(1)
for b in (0.0, 9.0):
    est = []
    for _ in range(2000):
        a = (rng.random(64) < p).astype(int)
        est.append(np.mean(np.where(a == 1, score[1] * (R[1] - b), score[0] * (R[0] - b))))
    est = np.array(est)
    print(f"  Monte Carlo, 2000 lotes de 64 tiradas, b = {b}: media {est.mean():.3f}, desviación entre lotes {est.std():.3f}")
# política gaussiana del x500 (una componente)
sd, xi = 0.6, 0.5
a_mu = sd * xi
print(f"  gaussiana: σ = {sd}, a - μ = σ·ξ = {a_mu:.2f} (ξ = {xi}); ∂ log N/∂μ = (a-μ)/σ² = {a_mu/sd**2:.3f};"
      f" ∂ log N/∂ log σ = (a-μ)²/σ² - 1 = {a_mu**2/sd**2 - 1:.3f}")

# ---------------------------------------------------------------------------
cab(2, "GAE A MANO EN UNA TRAYECTORIA DE 3 PASOS (TERMINA EN s3)")
g, lam = 0.9, 0.5
r = np.array([1.0, 0.0, 2.0])
V = np.array([2.0, 1.5, 1.0, 0.0])          # V(s3) = 0: estado terminal
d = r + g * V[1:] - V[:-1]
print(f"  γ = {g}, λ = {lam}, γλ = {g*lam}; r = {r}; V(s0..s3) = {V}")
for t in range(3):
    print(f"  δ{t} = r{t} + γV(s{t+1}) - V(s{t}) = {r[t]} + {g}·{V[t+1]} - {V[t]} = {d[t]:+.4f}")
A = np.zeros(3); acc = 0.0
for t in reversed(range(3)):
    acc = d[t] + g * lam * acc
    A[t] = acc
    print(f"  recursión hacia atrás: Â{t} = δ{t} + γλ·Â{t+1} = {A[t]:+.4f}")
suma = sum((g * lam) ** l * d[l] for l in range(3))
print(f"  suma directa Â0 = δ0 + (γλ)δ1 + (γλ)²δ2 = {d[0]:.4f} + {g*lam:.2f}·({d[1]:.2f}) + {(g*lam)**2:.4f}·{d[2]:.2f} = {suma:.4f}")
Ak = [sum(g ** l * d[l] for l in range(k)) for k in (1, 2, 3)]
print(f"  ventajas a k pasos desde t = 0: Â(1) = {Ak[0]:.4f}, Â(2) = {Ak[1]:.4f}, Â(3) = {Ak[2]:.4f} (= retorno - V(s0))")
G0 = r[0] + g * r[1] + g * g * r[2]
print(f"  retorno G0 = 1 + 0,9·0 + 0,81·2 = {G0:.4f}; G0 - V(s0) = {G0 - V[0]:.4f}")
w = [(1 - lam), (1 - lam) * lam, lam ** 2]
print(f"  media exponencial: pesos (1-λ), (1-λ)λ y el resto λ² = {w}: {w[0]:.2f}·{Ak[0]:.2f} + {w[1]:.2f}·({Ak[1]:.2f}) + {w[2]:.2f}·{Ak[2]:.2f} = {sum(wi*ai for wi, ai in zip(w, Ak)):.4f}")
for lm in (0.0, 0.5, 0.9, 1.0):
    acc = 0.0
    for t in reversed(range(3)):
        acc = d[t] + g * lm * acc
    print(f"  λ = {lm}: Â0 = {acc:.4f}")

# ---------------------------------------------------------------------------
cab(3, "HORIZONTE DE GAE Y REPARTO ENTORNOS x PASOS (γ = 0,99, λ = 0,95, 100 Hz)")
g, lam = 0.99, 0.95
gl = g * lam
print(f"  γλ = {gl:.4f}; horizonte efectivo de GAE 1/(1-γλ) = {1/(1-gl):.1f} pasos = {1/(1-gl)*0.01*1000:.0f} ms")
print(f"  horizonte del descuento 1/(1-γ) = {1/(1-g):.0f} pasos = {1/(1-g)*0.01:.1f} s")
print(f"  peso del residuo δ a 17 pasos (170 ms): (γλ)^17 = {gl**17:.3f}")
for T in (8, 32, 128, 512):
    print(f"  T = {T:3d} pasos ({T*10} ms): peso (γλ)^T de lo que queda tras el tramo = {gl**T:.3f}")
N, T, mini, ep, upd = 4096, 32, 4, 5, 600
B = N * T
print(f"  lote B = N·T = {N}·{T} = {B}; minilote {B//mini}; pasos de Adam por actualización {ep*mini}; "
      f"total {upd} actualizaciones = {B*upd} pasos de entorno y {ep*mini*upd} pasos de Adam")

# ---------------------------------------------------------------------------
cab(4, "COCIENTE DE PROBABILIDADES Y ESTIMADOR DE LA KL")
for po, pn in ((0.5, 0.6), (0.5, 0.4), (0.5, 0.75)):
    rr = pn / po
    print(f"  π_viejo(a) = {po}, π_θ(a) = {pn}: r = {rr:.3f}; log r = {math.log(rr):+.4f}; (r-1) - log r = {(rr-1)-math.log(rr):.4f}")
# gaussiana
mo, mn, sd, a = 0.0, 0.1, 0.6, 0.3
lr_ = (-(a - mn) ** 2 + (a - mo) ** 2) / (2 * sd ** 2)
print(f"  gaussiana σ = {sd}: a = {a}, μ_viejo = {mo}, μ = {mn}: r = exp({lr_:.4f}) = {math.exp(lr_):.4f}")
print(f"  en θ = θ_viejo, ∇_μ r = r·∂log π/∂μ = 1·(a-μ)/σ² = {(a-mo)/sd**2:.4f}")

# ---------------------------------------------------------------------------
cab(5, "LOS SEIS CASOS DEL OBJETIVO RECORTADO (ε = 0,2)")
eps = 0.2
for Aa in (2.0, -2.0):
    for rr in (0.7, 1.0, 1.5):
        sin = rr * Aa
        cl = min(max(rr, 1 - eps), 1 + eps) * Aa
        mn_ = min(sin, cl)
        activo = (mn_ == sin) and not (rr != min(max(rr, 1 - eps), 1 + eps) and sin == cl)
        # derivada respecto a r del mínimo
        dr = Aa if (sin < cl or (sin == cl and 1 - eps < rr < 1 + eps)) else 0.0
        solo_clip_dr = Aa if 1 - eps < rr < 1 + eps else 0.0
        print(f"  A = {Aa:+.0f}, r = {rr}: r·A = {sin:+.2f}; clip(r)·A = {cl:+.2f}; min = {mn_:+.2f}; "
              f"d(min)/dr = {dr:+.0f}; solo con el recorte (sin min) d/dr = {solo_clip_dr:+.0f}")
# figura de dos paneles: x de 0,4 a 1,8; y de la función L/|A|
rs = np.array([0.4, 0.8, 1.0, 1.2, 1.8])          # la función es lineal a trozos: basta con sus vértices
for Aa, nombre in ((1.0, "A>0"), (-1.0, "A<0")):
    X = lin(0.4, 1.8, 40, 290)
    Y = lin(-1.9, 1.9, 200, 20)
    L = np.minimum(rs * Aa, np.clip(rs, 1 - eps, 1 + eps) * Aa)
    print(f"  panel {nombre}: path min:", path(rs, L, X, Y))
    print(f"  panel {nombre}: path rA:", path(rs, rs * Aa, X, Y))
    print(f"  panel {nombre}: x(0,7)={X(0.7):.1f} x(0,8)={X(0.8):.1f} x(1)={X(1):.1f} x(1,2)={X(1.2):.1f} x(1,5)={X(1.5):.1f}; "
          f"y(0)={Y(0):.1f}; y(0,7A)={Y(0.7*Aa):.1f}; y(0,8A)={Y(0.8*Aa):.1f}; y(1,2A)={Y(1.2*Aa):.1f}; y(1,5A)={Y(1.5*Aa):.1f}")

# ---------------------------------------------------------------------------
cab(6, "NORMALIZACIÓN DE VENTAJAS EN UN MINILOTE")
Am = np.array([0.2825, -0.15, 1.0, 0.35])
mu, s = Am.mean(), Am.std()
print(f"  Â = {Am}; media {mu:.4f}; desviación {s:.4f}; normalizadas {(Am - mu) / (s + 1e-8)}")
print("  (multiplicar todas las ventajas por k > 0 no cambia el resultado: la escala de la recompensa deja de importar para la política)")

# ---------------------------------------------------------------------------
cab(7, "DISTRIBUCIÓN INICIAL DE LA ACCIÓN DEL x500: GANANCIA 0,01 FRENTE A √2 EN LA ÚLTIMA CAPA")
rng = np.random.default_rng(0)


def ortho(shape, gain):
    a = rng.normal(size=(max(shape), min(shape)))
    q, rr_ = np.linalg.qr(a)
    q = q * np.sign(np.diag(rr_))
    if shape[0] < shape[1]:
        q = q.T
    return gain * q[:shape[0], :shape[1]]


def rot_aleatoria(n):
    eje = rng.normal(size=(n, 3)); eje /= np.linalg.norm(eje, axis=1, keepdims=True)
    ang = rng.uniform(0, math.radians(30), n)
    w = np.cos(ang / 2); xyz = np.sin(ang / 2)[:, None] * eje
    x, y, z = xyz.T
    Rm = np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                   2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                   2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], 1)
    return Rm


n = 4096
O = np.concatenate([rng.uniform(-1.5, 1.5, (n, 3)), rng.uniform(-1, 1, (n, 3)), rot_aleatoria(n),
                    rng.uniform(-1, 1, (n, 3)), np.zeros((n, 4))], 1)
print(f"  observaciones de arranque: {O.shape[1]} = 3 (error de posición) + 3 (velocidad) + 9 (rotación) + 3 (vel. angular) + 4 (acción anterior)")
for gain in (0.01, math.sqrt(2)):
    res = []
    for rep in range(5):
        W1, W2, W3 = ortho((22, 64), math.sqrt(2)), ortho((64, 64), math.sqrt(2)), ortho((64, 4), gain)
        mu_a = np.tanh(np.tanh(O @ W1) @ W2) @ W3
        res.append((np.abs(mu_a).mean(), (np.abs(mu_a[:, 0]) > 1).mean(), np.abs(mu_a[:, 0]).mean()))
    res = np.array(res).mean(0)
    print(f"  ganancia {gain:.3f}: |media de la acción| media {res[0]:.3f}; |media del empuje a0| {res[2]:.3f}; "
          f"fracción de arranques con |a0| > 1 (empuje saturado) {res[1]*100:.1f} %  (media de 5 inicializaciones)")
s0 = math.exp(-0.5)
print(f"  σ inicial = exp(-0,5) = {s0:.4f}; empuje T = m g (1 + 0,8 a0): una σ son ±{0.8*s0*100:.0f} % del peso; "
      f"velocidad angular deseada (4, 4, 2)·a: una σ son ±{4*s0:.2f}, ±{4*s0:.2f} y ±{2*s0:.2f} rad/s")

# ---------------------------------------------------------------------------
cab(8, "SAC: POLÍTICA DE BOLTZMANN, TEMPERATURA Y ESCALA DE LA RECOMPENSA")
Q = np.array([1.0, 0.0])
for al in (0.1, 1.0, 10.0):
    pi = np.exp(Q / al) / np.exp(Q / al).sum()
    H = -(pi * np.log(pi)).sum()
    Vs = al * math.log(np.exp(Q / al).sum())
    print(f"  Q = (1, 0), α = {al:5.1f}: π = ({pi[0]:.4f}, {pi[1]:.4f}); entropía {H:.4f} nats (máx ln 2 = {math.log(2):.4f}); "
          f"valor suave α log Σ exp(Q/α) = {Vs:.4f}")
pi_a = np.exp(np.array([5.0, 0.0]) / 1.0); pi_a /= pi_a.sum()
pi_b = np.exp(np.array([1.0, 0.0]) / 0.2); pi_b /= pi_b.sum()
print(f"  recompensa ×5 con α = 1 (Q = (5, 0)): π = {pi_a}; recompensa ×1 con α = 0,2: π = {pi_b} (idénticas)")
for sd in (0.6, 0.1, 0.089):
    h1 = 0.5 * math.log(2 * math.pi * math.e * sd ** 2)
    print(f"  entropía de una gaussiana σ = {sd}: ½ ln(2πe σ²) = {h1:.4f} nats por dimensión; 4 dimensiones: {4*h1:.3f}")
sd_obj = math.sqrt(math.exp(-2) / (2 * math.pi * math.e))
print(f"  entropía objetivo -dim(A) = -4 con 4 dimensiones ⇔ σ = {sd_obj:.4f} por dimensión (antes de la tanh)")
H, Hb = 4 * 0.5 * math.log(2 * math.pi * math.e * 0.36), -4.0
print(f"  ∂J(α)/∂α = H(π) - H̄ = {H:.3f} - ({Hb}) = {H - Hb:.3f} > 0: el descenso por gradiente baja α")
for u in (0.0, 1.0, 2.0, 3.0):
    a = math.tanh(u)
    print(f"  tanh: u = {u}: a = {a:.4f}; 1 - tanh² u = {1-a*a:.5f}; corrección -log(1 - tanh² u) = {-math.log(1-a*a):.4f}")

# ---------------------------------------------------------------------------
cab(9, "TD3: SOBRESTIMACIÓN AL MAXIMIZAR UN CRÍTICO CON RUIDO, Y EL DOBLE CRÍTICO RECORTADO")
rng = np.random.default_rng(2)
M = 200000
print("  todas las acciones valen de verdad Q = 0; cada crítico se equivoca con ruido N(0, 1) independiente")
print(f"  E[max de 2 gaussianas N(0,1)] analítico = 1/√π = {1/math.sqrt(math.pi):.4f}")
for K in (2, 10):
    Q1 = rng.normal(size=(M, K)); Q2 = rng.normal(size=(M, K))
    k = Q1.argmax(1)
    q1 = Q1[np.arange(M), k]; q2 = Q2[np.arange(M), k]
    print(f"  K = {K:2d} acciones: un crítico, max Q1 = {q1.mean():+.3f}; segundo crítico en esa acción Q2(a*) = {q2.mean():+.3f}; "
          f"mínimo de los dos min(Q1, Q2)(a*) = {np.minimum(q1, q2).mean():+.3f}")
# suavizado de la política objetivo
pi_obj, xi = 0.9, 0.6
xi_c = max(-0.5, min(0.5, xi))
print(f"  suavizado: π_φ'(s') = {pi_obj}, ξ = {xi} → recortado a {xi_c} → {pi_obj+xi_c:.1f} → recortado a la cota de la acción 1,0")
tau = 0.005
print(f"  redes objetivo con τ = {tau}: peso de los parámetros de hace n pasos (1-τ)^n: n = 200 → {(1-tau)**200:.3f}; "
      f"n = 1000 → {(1-tau)**1000:.4f}; memoria media 1/τ = {1/tau:.0f} pasos")

# ---------------------------------------------------------------------------
cab(10, "COSTE DEL x500 Y LAS DOS RECOMPENSAS")


def coste(ep, v, om, da):
    return np.linalg.norm(ep) + 0.1 * np.linalg.norm(v) + 0.05 * np.linalg.norm(om) + 0.1 * np.sum(np.square(da))


casos = {"arranque típico": ((1.0, 1.0, 0.5), (0.6, 0.8, 0.0), (0.3, 0.4, 0.0), (0, 0, 0, 0)),
         "casi en la consigna": ((0.03, 0.04, 0.0), (0.03, 0.04, 0.0), (0.0, 0.0, 0.05), (0, 0, 0, 0)),
         "mando brusco": ((0.03, 0.04, 0.0), (0.03, 0.04, 0.0), (0.0, 0.0, 0.05), (0.5, 0.5, 0, 0))}
for nom, (ep, v, om, da) in casos.items():
    c = coste(np.array(ep), np.array(v), np.array(om), np.array(da))
    print(f"  {nom:20s}: ‖e_p‖ = {np.linalg.norm(ep):.3f}, ‖v‖ = {np.linalg.norm(v):.3f}, ‖ω‖ = {np.linalg.norm(om):.3f}, "
          f"‖Δa‖² = {np.sum(np.square(da)):.2f} → c = {c:.4f}; ingenua 1 - c = {1-c:+.4f}; acotada 1 - tanh c = {1-math.tanh(c):.4f}")
cs = np.linspace(0, 3, 31)
X = lin(0, 3, 50, 330); Y = lin(-2.2, 1.2, 230, 20)
print("  figura: path ingenua:", path(cs, 1 - cs, X, Y))
print("  figura: path acotada:", path(cs, 1 - np.tanh(cs), X, Y))
print(f"  ejes: x(0)={X(0):.1f} x(1)={X(1):.1f} x(1,1)={X(1.1):.1f} x(1,625)={X(1.625):.1f} x(2)={X(2):.1f} x(3)={X(3):.1f}; "
      f"y(1)={Y(1):.1f} y(0)={Y(0):.1f} y(-0,1)={Y(-0.1):.1f} y(-1)={Y(-1):.1f} y(-2)={Y(-2):.1f}; "
      f"y(1-1,625)={Y(1-1.625):.1f} y(1-tanh 1,625)={Y(1-math.tanh(1.625)):.1f}")

# ---------------------------------------------------------------------------
cab(11, "CUÁNDO COMPENSA TERMINAR: γ = 0,99, P = 10")
g, P = 0.99, 10.0
print(f"  umbral: r̄ < -P(1-γ) = {-P*(1-g):.3f}")
for nom, rb in (("ingenua al principio", -0.625), ("acotada al principio", 1 - math.tanh(1.625)), ("acotada en régimen", 0.94)):
    print(f"  {nom:22s}: r̄ = {rb:+.4f}; seguir vale r̄/(1-γ) = {rb/(1-g):+.2f}; estrellarse vale -P = {-P:.0f} → "
          f"{'estrellarse' if rb/(1-g) < -P else 'seguir'}")
for H in (100, 500):
    print(f"  con H = {H} pasos restantes y sin arranque del valor: seguir vale r̄(1-γ^H)/(1-γ); factor 1-γ^H = {1-g**H:.4f}")
print(f"  ingenua, valor de seguir con H = 500: {-0.625*(1-g**500)/(1-g):.2f}")

# ---------------------------------------------------------------------------
cab(12, "MOLDEADO POR POTENCIAL: LA SUMA TELESCÓPICA EN DOS CAMINOS")
g = 0.95
for nom, ds in (("directo", [3, 2, 1, 0]), ("con rodeo", [3, 4, 3, 2, 1, 0])):
    Tn = len(ds) - 1
    phi = [-x for x in ds]
    phi[-1] = 0.0                                  # Φ(meta) = 0 (terminal)
    F = [g * phi[t + 1] - phi[t] for t in range(Tn)]
    Gs = g ** (Tn - 1) * 10.0                      # 10 al llegar, en el último paso
    SF = sum(g ** t * F[t] for t in range(Tn))
    print(f"  {nom:9s}: distancias {ds}; F_t = γΦ(s_t+1) - Φ(s_t) = {[round(f, 4) for f in F]}; Σ γ^t F_t = {SF:.4f} (= -Φ(s0) = 3)")
    print(f"             retorno escaso G = γ^{Tn-1}·10 = {Gs:.4f}; moldeado G̃ = G + 3 = {Gs+SF:.4f}")
print("  la diferencia entre caminos es la misma con y sin moldeado: el orden de las políticas no cambia")
print(f"  bonificación ingenua: quedarse junto a la meta vale Σ γ^t·1 = 1/(1-γ) = {1/(1-g):.0f} > 10 de llegar")
print(f"  si se dejara Φ(meta) = -5 en vez de 0: queda el término γ^T Φ(s_T) = {g**3*(-5):.3f} (directo, T = 3) frente a {g**5*(-5):.3f} (rodeo, T = 5): ya no es igual para todos los caminos")

# ---------------------------------------------------------------------------
cab(13, "OBSERVACIÓN Y ACCIÓN DEL x500")
M_, G_ = 2.0, 9.81
print(f"  peso m g = {M_}·{G_} = {M_*G_:.2f} N; a0 ∈ [-1, 1] → T = m g (1 + 0,8 a0) ∈ [{M_*G_*0.2:.2f}, {M_*G_*1.8:.2f}] N; a0 = 0 → estacionario")
print(f"  periodo de control 1/100 Hz = 10 ms; física 500 Hz = 2 ms → 5 subpasos por acción; retardo de primer orden τ = 30 ms = 3 pasos de control")
print(f"  episodio 5 s = 500 pasos; γ = 0,99 → horizonte 100 pasos = 1 s")
