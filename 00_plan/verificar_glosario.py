"""Comprueba el glosario (aside#glosario, tecla G) de un tema. CONTINUAR.md §6, paso 6.
Uso, desde la raíz del repo y con `python3 -m http.server 8765` en marcha:
    uv run --with playwright python 00_plan/verificar_glosario.py <carpeta_del_tema> [--shots]
Con --shots guarda capturas del panel (claro, oscuro, final de la lista y móvil) en
<tmp>/capturas_glosario/. Usa el Chrome del sistema (channel="chrome").
"""
import sys, re, asyncio, tempfile, html as H
from pathlib import Path
from playwright.async_api import async_playwright

REPO = Path(__file__).resolve().parent.parent
OUT = Path(tempfile.gettempdir()) / "capturas_glosario"
OUT.mkdir(exist_ok=True)
folder = sys.argv[1]
shots = "--shots" in sys.argv
port = next((a.split("=",1)[1] for a in sys.argv if a.startswith("--port=")), "8765")
src = (REPO / "temas" / folder / "index.html").read_text()

# ---------- Comprobación estática: ¿aparece el símbolo en las hojas citadas? ----------
deck = src.split('<main class="deck">', 1)[1].split("</main>", 1)[0]
parts = re.split(r'(?=<section class="slide)', deck)
slides = [p for p in parts if p.startswith('<section class="slide')]
m = re.search(r'<aside id="glosario"[^>]*>(.*?)</aside>', src, re.S)
print(f"[{folder}] hojas: {len(slides)}")
if not m:
    print("FALTA <aside id=\"glosario\">"); sys.exit(1)
g = m.group(1)
pairs = re.findall(r"<dt>(.*?)</dt>\s*<dd([^>]*)>(.*?)</dd>", g, re.S)
print("entradas:", len(pairs), "· grupos:", len(re.findall(r"<h3>", g)))
def squash(t):
    t = H.unescape(t)
    t = re.sub(r"\\[()\[\]]", "", t)
    return re.sub(r"\s+", "", t)
slide_txt = [squash(s) for s in slides]
dupes = {}
miss, nohojas = [], []
for dt, attrs, dd in pairs:
    hs = re.search(r'data-hojas="([^"]*)"', attrs)
    key = squash(dt)
    dupes[key] = dupes.get(key, 0) + 1
    # un <dt> puede llevar varios símbolos hermanos: \(R_k\), \(R_{k+1}\); se comprueba cada uno
    segs = [squash(x) for x in re.findall(r"\\\((.*?)\\\)", dt, re.S)] or [re.sub(r"\s*\(.*?\)\s*", "", squash(dt))]
    if not hs:
        nohojas.append(dt); continue
    nums = [int(x) for x in re.split(r"[\s,]+", hs.group(1)) if x]
    for n in nums:
        if not (1 <= n <= len(slides)):
            miss.append((dt, n, "FUERA DE RANGO")); continue
        # el símbolo se busca sin espacios; se aceptan variantes \mathbf{x} / \boldsymbol{x} solo si coinciden literal
        falta = [x for x in segs if x and x not in slide_txt[n - 1]]
        if falta and n == nums[0]:
            miss.append((dt, n, "no aparece literal: " + " ".join(falta)))
print("sin data-hojas:", len(nohojas), nohojas[:8])
print("revisar (símbolo no literal en la hoja citada; puede ser falso positivo):")
for x in miss: print("   ", x)
print("símbolos repetidos (bien si son acepciones distintas):", [k for k, v in dupes.items() if v > 1])

# ---------- Navegador ----------
async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel="chrome")
        pg = await b.new_page(viewport={"width": 1440, "height": 900})
        msgs = []
        pg.on("console", lambda mm: msgs.append((mm.type, mm.text)) if mm.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: msgs.append(("pageerror", str(e))))
        url = f"http://localhost:{port}/temas/{folder}/index.html"
        await pg.goto(url)
        await pg.wait_for_function("window.MathJax && MathJax.startup && MathJax.startup.promise", timeout=20000)
        await pg.evaluate("MathJax.startup.promise"); await pg.wait_for_timeout(1200)
        print("botón:", await pg.evaluate("!!document.getElementById('btn-glos')"))
        for theme in ["light", "dark"]:
            await pg.evaluate(f"document.documentElement.dataset.theme='{theme}'")
            print(theme, "checkOverflow:", await pg.evaluate("curso.checkOverflow()"))
        await pg.keyboard.press("g"); await pg.wait_for_timeout(300)
        print("panel abierto con G:", await pg.evaluate("!document.querySelector('.glos-panel').hidden"))
        print("merror en glosario:", await pg.evaluate("document.querySelectorAll('.glos-panel mjx-merror').length"),
              "· merror total:", await pg.evaluate("document.querySelectorAll('mjx-merror').length"))
        print("TeX sin convertir en glosario:", await pg.evaluate(
            r"[...document.querySelectorAll('.glos-row')].filter(r=>/\\\(|\\\[/.test(r.textContent)).map(r=>r.textContent.slice(0,40))"))
        print("filas que desbordan en horizontal:", await pg.evaluate(
            "[...document.querySelectorAll('.glos-row dt, .glos-row dd')].filter(e=>e.scrollWidth>e.clientWidth+2).map(e=>e.textContent.slice(0,40))"))
        await pg.fill(".glos-q", "theta"); await pg.wait_for_timeout(100)
        print("búsqueda 'theta' →", await pg.evaluate("[...document.querySelectorAll('.glos-row')].filter(r=>!r.hidden).length"), "filas")
        await pg.fill(".glos-q", "")
        await pg.keyboard.press("Escape"); await pg.wait_for_timeout(100)
        print("cerrado con Esc:", await pg.evaluate("document.querySelector('.glos-panel').hidden"))
        if shots:
            for theme in ["light", "dark"]:
                await pg.evaluate(f"document.documentElement.dataset.theme='{theme}'")
                await pg.keyboard.press("g"); await pg.wait_for_timeout(300)
                await pg.screenshot(path=str(OUT / f"{folder}_{theme}.png"))
                await pg.evaluate("document.querySelector('.glos-panel').scrollTop = 1e6"); await pg.wait_for_timeout(150)
                await pg.screenshot(path=str(OUT / f"{folder}_{theme}_fin.png"))
                await pg.evaluate("document.querySelector('.glos-panel').scrollTop = 0")
                await pg.keyboard.press("Escape")
        mob = await b.new_page(viewport={"width": 375, "height": 800})
        await mob.goto(url)
        await mob.evaluate("MathJax.startup.promise"); await mob.wait_for_timeout(1200)
        print("móvil: hojas con desborde horizontal:", await mob.evaluate(
            "[...document.querySelectorAll('.slide')].map((s,i)=>[i+1, s.scrollWidth - s.clientWidth]).filter(x=>x[1]>2)"))
        print("móvil: barra desborda:", await mob.evaluate("(()=>{const b=document.querySelector('.bar');return b.scrollWidth>b.clientWidth+1})()"))
        await mob.click("#btn-glos"); await mob.wait_for_timeout(300)
        print("móvil: glosario desborda en horizontal:", await mob.evaluate(
            "(()=>{const p=document.querySelector('.glos-panel');return p.scrollWidth-p.clientWidth})()"))
        if shots:
            await mob.screenshot(path=str(OUT / f"{folder}_movil.png"))
        if shots:
            print("capturas en", OUT)
        print("consola:", [x for x in msgs if "glosario" in x[1] or x[0] != "warning"] [:10])
        await b.close()
asyncio.run(main())
