/* Curso de robótica inteligente: motor de presentaciones.
   Modos: presentación (una hoja escalada) y lectura (hojas apiladas).
   Barra: ⌂ enlace al índice del curso.
   Teclas: ← → espacio RePág AvPág Inicio Fin · I índice · G glosario · L modo · T tema */
(function () {
  "use strict";

  const store = {
    get(k) { try { return localStorage.getItem("curso:" + k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem("curso:" + k, v); } catch (e) { /* sin almacenamiento */ } },
  };

  const body = document.body;
  const deck = document.querySelector(".deck");
  if (!deck) return;
  const slides = Array.from(deck.querySelectorAll(":scope > .slide"));
  const N = slides.length;
  const deckId = body.dataset.deckId || "";
  const deckTitle = body.dataset.deckTitle || document.title;
  let cur = 0;

  /* ---------- Secciones y títulos ---------- */
  let section = "";
  slides.forEach((s, i) => {
    if (s.dataset.section) section = s.dataset.section;
    s.dataset.sectionResolved = section;
    s.id = s.id || "s" + (i + 1);
    const h = s.querySelector("h1, h2");
    s.dataset.titleResolved = s.dataset.title || (h ? h.textContent.trim().replace(/\s+/g, " ") : "Diapositiva " + (i + 1));
    const c = document.createElement("div");
    c.className = "cartouche";
    c.setAttribute("aria-hidden", "true");
    c.innerHTML =
      '<span class="c-id"></span><span class="c-sec"></span><span class="c-n"></span>';
    c.querySelector(".c-id").textContent = deckId;
    c.querySelector(".c-sec").textContent = section;
    c.querySelector(".c-n").textContent = String(i + 1).padStart(2, "0") + " / " + String(N).padStart(2, "0");
    s.appendChild(c);
  });

  /* ---------- Barra ---------- */
  const bar = document.createElement("header");
  bar.className = "bar";
  bar.innerHTML =
    '<a class="home" title="Índice del curso">⌂ <span>Curso</span></a>' +
    '<span class="deck-id"></span><span class="deck-title"></span>' +
    '<button type="button" id="btn-prev" aria-label="Diapositiva anterior">←</button>' +
    '<span class="counter" id="counter"></span>' +
    '<button type="button" id="btn-next" aria-label="Diapositiva siguiente">→</button>' +
    '<button type="button" id="btn-index">Índice</button>' +
    '<button type="button" id="btn-mode"></button>' +
    '<button type="button" id="btn-theme">Tema</button>';
  const jsTag = document.querySelector('script[src*="curso.js"]');
  bar.querySelector(".home").href = jsTag ? new URL("../index.html", jsTag.src).href : "../../index.html";
  bar.querySelector(".deck-id").textContent = deckId;
  bar.querySelector(".deck-title").textContent = deckTitle;
  body.prepend(bar);
  const prog = document.createElement("div");
  prog.className = "progress";
  prog.innerHTML = "<span></span>";
  bar.after(prog);
  const counter = bar.querySelector("#counter");
  const btnMode = bar.querySelector("#btn-mode");

  /* ---------- Índice ---------- */
  const panel = document.createElement("nav");
  panel.className = "index-panel";
  panel.hidden = true;
  panel.setAttribute("aria-label", "Índice de diapositivas");
  let html = "";
  let lastSec = null;
  slides.forEach((s, i) => {
    const sec = s.dataset.sectionResolved;
    if (sec !== lastSec) {
      if (lastSec !== null) html += "</ol>";
      html += "<h4></h4><ol>";
      lastSec = sec;
    }
    html += '<li><a href="#' + s.id + '" data-i="' + i + '"><span class="n">' + (i + 1) + "</span><span></span></a></li>";
  });
  html += "</ol>";
  panel.innerHTML = html;
  let hIdx = 0;
  let lastS = null;
  const h4s = panel.querySelectorAll("h4");
  slides.forEach((s) => {
    if (s.dataset.sectionResolved !== lastS) {
      h4s[hIdx].textContent = s.dataset.sectionResolved || deckId;
      hIdx++;
      lastS = s.dataset.sectionResolved;
    }
  });
  panel.querySelectorAll("a").forEach((a) => {
    a.lastElementChild.textContent = slides[+a.dataset.i].dataset.titleResolved;
    a.addEventListener("click", (e) => {
      e.preventDefault();
      go(+a.dataset.i);
      panel.hidden = true;
    });
  });
  body.appendChild(panel);

  /* ---------- Glosario (tecla G) ----------
     Fuente en la página: <aside id="glosario" hidden> con <h3> por grupo y
     <dl> de pares <dt>símbolo</dt><dd data-hojas="12,30">qué es</dd>.
     Se lee aquí, antes de que MathJax (defer) convierta el TeX, para poder
     buscar también por el código del símbolo (\theta, \Sigma…). */
  const gSrc = document.getElementById("glosario");
  let gPanel = null;
  let gInput = null;
  if (gSrc) {
    const norm = (t) => t.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/\s+/g, " ");
    gPanel = document.createElement("aside");
    gPanel.className = "index-panel glos-panel";
    gPanel.hidden = true;
    gPanel.setAttribute("aria-label", "Glosario de símbolos");
    gPanel.innerHTML =
      '<div class="glos-head"><h3>Glosario · <span></span></h3>' +
      '<button type="button" class="glos-close" aria-label="Cerrar glosario">×</button></div>' +
      '<input type="search" class="glos-q" placeholder="Buscar: theta, matriz, covarianza…" aria-label="Buscar en el glosario">' +
      '<p class="glos-empty" hidden>Nada coincide.</p>';
    gPanel.querySelector(".glos-head span").textContent = deckId;
    gInput = gPanel.querySelector(".glos-q");
    const empty = gPanel.querySelector(".glos-empty");
    Array.from(gSrc.children).forEach((el) => gPanel.insertBefore(el, empty));
    gSrc.remove();
    const rows = [];
    gPanel.querySelectorAll("dl").forEach((dl) => {
      Array.from(dl.querySelectorAll(":scope > dt")).forEach((dt) => {
        const dd = dt.nextElementSibling;
        if (!dd || dd.tagName !== "DD") return;
        const row = document.createElement("div");
        row.className = "glos-row";
        dt.before(row);
        row.append(dt, dd);
        rows.push({ row, dl, text: norm(dt.textContent + " " + dd.textContent) });
        const hs = (dd.dataset.hojas || "").split(/[\s,]+/).filter(Boolean).map(Number);
        if (hs.length) {
          const w = document.createElement("span");
          w.className = "glos-hojas";
          hs.forEach((h) => {
            if (!(h >= 1 && h <= N)) { console.warn("[curso] glosario: hoja fuera de rango:", h, dt.textContent); return; }
            const a = document.createElement("a");
            a.href = "#" + slides[h - 1].id;
            a.textContent = String(h);
            a.title = slides[h - 1].dataset.titleResolved;
            a.addEventListener("click", (e) => { e.preventDefault(); go(h - 1); gPanel.hidden = true; });
            w.appendChild(a);
          });
          w.dataset.label = w.children.length > 1 ? "hojas" : "hoja";
          if (w.children.length) dd.appendChild(w);
        }
      });
    });
    gInput.addEventListener("input", () => {
      const q = norm(gInput.value.trim()).replace(/\\/g, "");
      let shown = 0;
      rows.forEach((r) => {
        const ok = !q || r.text.replace(/\\/g, "").includes(q);
        r.row.hidden = !ok;
        if (ok) shown++;
      });
      gPanel.querySelectorAll("dl").forEach((dl) => {
        const any = rows.some((r) => r.dl === dl && !r.row.hidden);
        dl.hidden = !any;
        const h = dl.previousElementSibling;
        if (h && h.tagName === "H3") h.hidden = !any;
      });
      empty.hidden = shown > 0;
    });
    gPanel.querySelector(".glos-close").addEventListener("click", () => toggleGlos(false));
    body.appendChild(gPanel);
    const btnG = document.createElement("button");
    btnG.type = "button";
    btnG.id = "btn-glos";
    btnG.textContent = "Glosario";
    btnG.setAttribute("aria-label", "Glosario de símbolos (tecla G)");
    btnG.addEventListener("click", () => toggleGlos());
    bar.querySelector("#btn-index").after(btnG);
  }
  function toggleGlos(show) {
    if (!gPanel) return;
    const open = show === undefined ? gPanel.hidden : show;
    gPanel.hidden = !open;
    if (!open && document.activeElement === gInput) gInput.blur();
    if (open) {
      panel.hidden = true;
      /* En pantallas táctiles no se enfoca el buscador: abriría el teclado */
      if (!window.matchMedia || window.matchMedia("(hover: hover)").matches) { gInput.focus({ preventScroll: true }); gInput.select(); }
    }
  }

  /* ---------- Referencias ---------- */
  const refItems = Array.from(document.querySelectorAll("ol.refs > li[id^='ref-']"));
  const refNum = {};
  refItems.forEach((li, i) => { refNum[li.id.slice(4)] = i + 1; });
  document.querySelectorAll("a.cite[data-ref]").forEach((a) => {
    const keys = a.dataset.ref.split(",").map((k) => k.trim());
    const nums = keys.map((k) => refNum[k] || "?");
    keys.forEach((k) => { if (!refNum[k]) console.warn("[curso] referencia sin entrada:", k); });
    a.textContent = "[" + nums.join(",") + "]";
    a.href = "#ref-" + keys[0];
    const li = document.getElementById("ref-" + keys[0]);
    if (li) a.title = li.textContent.trim().replace(/\s+/g, " ").slice(0, 220);
    a.addEventListener("click", (e) => {
      const target = document.getElementById("ref-" + keys[0]);
      if (!target) return;
      e.preventDefault();
      const s = target.closest(".slide");
      if (s) go(slides.indexOf(s));
      if (body.classList.contains("mode-read")) target.scrollIntoView({ block: "center" });
      target.setAttribute("tabindex", "-1");
      target.style.background = "var(--tint-time)";
      setTimeout(() => { target.style.background = ""; }, 2400);
    });
  });

  /* ---------- Tests ---------- */
  document.querySelectorAll(".quiz[data-answer]").forEach((q) => {
    const ans = +q.dataset.answer;
    const btns = Array.from(q.querySelectorAll(".opts button"));
    const exp = q.querySelector(".explain");
    if (exp) exp.hidden = true;
    btns.forEach((b, i) => {
      b.addEventListener("click", () => {
        btns.forEach((x) => x.classList.remove("is-ok", "is-bad"));
        b.classList.add(i === ans ? "is-ok" : "is-bad");
        if (i !== ans) btns[ans].classList.add("is-ok");
        if (exp) exp.hidden = false;
      });
    });
  });

  /* ---------- Navegación ---------- */
  function update() {
    counter.textContent = String(cur + 1).padStart(2, "0") + " / " + String(N).padStart(2, "0");
    prog.firstElementChild.style.width = (N > 1 ? (cur / (N - 1)) * 100 : 100) + "%";
    panel.querySelectorAll("a").forEach((a) => a.classList.toggle("is-current", +a.dataset.i === cur));
  }
  function go(i, fromScroll) {
    cur = Math.max(0, Math.min(N - 1, i));
    slides.forEach((s, k) => s.classList.toggle("is-current", k === cur));
    if (body.classList.contains("mode-read") && !fromScroll) {
      slides[cur].scrollIntoView({ block: "start" });
    }
    try { history.replaceState(null, "", "#" + slides[cur].id); } catch (e) { /* marco restringido */ }
    update();
  }

  function fit() {
    const vw = window.innerWidth;
    const vh = window.innerHeight - bar.offsetHeight - 3;
    const s = Math.min((vw - 24) / 1280, (vh - 20) / 720);
    document.documentElement.style.setProperty("--scale", String(Math.max(0.2, s)));
  }

  function setMode(m) {
    body.classList.toggle("mode-present", m === "present");
    body.classList.toggle("mode-read", m === "read");
    btnMode.textContent = m === "present" ? "Lectura" : "Presentación";
    btnMode.setAttribute("aria-label", m === "present" ? "Cambiar a modo lectura" : "Cambiar a modo presentación");
    store.set("mode", m);
    fit();
    go(cur);
    checkOverflow();
  }

  function narrow() { return window.innerWidth < 900 || window.innerHeight < 520; }

  /* Señala las hojas cuyo contenido no cabe en 1280×720 */
  function checkOverflow() {
    if (!body.classList.contains("mode-present")) return [];
    const over = [];
    slides.forEach((s, i) => {
      const was = s.style.display;
      s.style.display = "flex";
      s.style.visibility = s.classList.contains("is-current") ? "" : "hidden";
      s.classList.remove("is-overflowing");
      let o = s.scrollHeight > s.clientHeight + 2 || s.scrollWidth > s.clientWidth + 2;
      /* Contenido recortado dentro de la hoja: tablas, bloques .body y fórmulas en bloque */
      if (!o) {
        o = Array.from(s.querySelectorAll(".table-wrap, .body, .fig-scroll, mjx-container[display='true']"))
          .some((e) => e.scrollHeight > e.clientHeight + 2 || e.scrollWidth > e.clientWidth + 2);
      }
      if (o) { s.classList.add("is-overflowing"); over.push(i + 1); }
      s.style.display = was;
      s.style.visibility = "";
    });
    if (over.length) console.warn("[curso] hojas que no caben en 1280×720:", over.join(", "));
    return over;
  }

  /* Tema */
  function applyTheme(t) {
    if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
    else delete document.documentElement.dataset.theme;
  }
  applyTheme(store.get("theme"));
  bar.querySelector("#btn-theme").addEventListener("click", () => {
    const attr = document.documentElement.dataset.theme;
    const sysDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
    const isDark = attr ? attr === "dark" : sysDark;
    const next = isDark ? "light" : "dark";
    applyTheme(next);
    store.set("theme", next);
  });

  bar.querySelector("#btn-prev").addEventListener("click", () => go(cur - 1));
  bar.querySelector("#btn-next").addEventListener("click", () => go(cur + 1));
  function toggleIndex() {
    panel.hidden = !panel.hidden;
    if (!panel.hidden && gPanel) gPanel.hidden = true;
  }
  bar.querySelector("#btn-index").addEventListener("click", toggleIndex);
  btnMode.addEventListener("click", () => setMode(body.classList.contains("mode-present") ? "read" : "present"));

  document.addEventListener("keydown", (e) => {
    if (gInput && e.target === gInput) {
      if (e.key === "Escape") { e.preventDefault(); toggleGlos(false); }
      return;
    }
    if (e.target.closest("input, textarea, select")) return;
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    const k = e.key;
    const present = body.classList.contains("mode-present");
    if (present && (k === "ArrowRight" || k === "PageDown" || k === " ")) { e.preventDefault(); go(cur + 1); }
    else if (present && (k === "ArrowLeft" || k === "PageUp")) { e.preventDefault(); go(cur - 1); }
    else if (present && k === "Home") { e.preventDefault(); go(0); }
    else if (present && k === "End") { e.preventDefault(); go(N - 1); }
    else if (k === "i" || k === "I") { toggleIndex(); }
    else if ((k === "g" || k === "G") && gPanel) { e.preventDefault(); toggleGlos(); }
    else if (k === "l" || k === "L") { btnMode.click(); }
    else if (k === "t" || k === "T") { bar.querySelector("#btn-theme").click(); }
    else if (k === "Escape") { panel.hidden = true; toggleGlos(false); }
  });

  /* En modo lectura, el contador sigue al scroll */
  if ("IntersectionObserver" in window) {
    const io = new IntersectionObserver((entries) => {
      if (!body.classList.contains("mode-read")) return;
      entries.forEach((en) => {
        if (en.isIntersecting) {
          const i = slides.indexOf(en.target);
          if (i >= 0 && i !== cur) go(i, true);
        }
      });
    }, { rootMargin: "-45% 0px -50% 0px" });
    slides.forEach((s) => io.observe(s));
  }

  window.addEventListener("resize", fit);

  /* Arranque */
  const fromHash = slides.findIndex((s) => "#" + s.id === location.hash);
  cur = fromHash >= 0 ? fromHash : 0;
  const saved = store.get("mode");
  setMode(narrow() ? "read" : saved === "read" ? "read" : "present");
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(checkOverflow);
  window.addEventListener("load", () => {
    if (window.MathJax && MathJax.startup && MathJax.startup.promise) {
      MathJax.startup.promise.then(checkOverflow);
    } else {
      checkOverflow();
    }
  });

  window.curso = { go, checkOverflow, slides };
})();
