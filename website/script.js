/* SONIX marketing site — vanilla JS, no build step, no framework.
   Pieces: nav (incl. accessible dropdown, scroll-spy, theme toggle), the hero
   canvas animation, animated stat counters, the pipeline walkthrough, the
   server-reachability check, the model picker, the EMBEDDED live-capture
   widget (mic -> WebSocket -> pairing -> risk band, all on this page), the
   real upload-and-score flow, an illustrative threshold slider, and small
   conveniences (copy buttons, back-to-top). Served by realtime/server.py at
   "/", so the API is same-origin. */

(() => {
  "use strict";

  // Same origin when served by realtime/server.py; falls back to the documented
  // port when the file is opened directly.
  const SERVER = location.protocol.startsWith("http")
    ? location.origin.replace(/\/$/, "")
    : "http://localhost:8000";
  const WS_URL = SERVER.replace(/^http/, "ws") + "/ws";
  const AMBER_AT = 0.35, RED_AT = 0.65; // canonical thresholds — see realtime/miccapture.py

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

  /* ------------------------------------------------------------ nav */
  function initNav() {
    const dropdown = $("#extDropdown");
    const btn = $("#extDropdownBtn");
    const menu = $("#extMenu");
    const nav = $("#nav");
    const mobileToggle = $("#mobileToggle");

    function openMenu() {
      dropdown.dataset.open = "true";
      btn.setAttribute("aria-expanded", "true");
    }
    function closeMenu(focusBtn) {
      dropdown.dataset.open = "false";
      btn.setAttribute("aria-expanded", "false");
      if (focusBtn) btn.focus();
    }

    btn.addEventListener("click", () => {
      dropdown.dataset.open === "true" ? closeMenu(false) : openMenu();
    });

    btn.addEventListener("keydown", (e) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        openMenu();
        $("a[role='menuitem']", menu)?.focus();
      }
      if (e.key === "Escape") closeMenu(true);
    });

    menu.addEventListener("keydown", (e) => {
      const items = $$("a[role='menuitem']", menu);
      const i = items.indexOf(document.activeElement);
      if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length]?.focus(); }
      if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length]?.focus(); }
      if (e.key === "Escape") closeMenu(true);
      if (e.key === "Tab" && i === items.length - 1 && !e.shiftKey) closeMenu(false);
    });

    document.addEventListener("click", (e) => {
      if (!dropdown.contains(e.target)) closeMenu(false);
    });

    // Any anchor click (menu item or otherwise) closes the dropdown so it
    // doesn't stay open floating over the section it just navigated to.
    menu.addEventListener("click", () => closeMenu(false));

    // Mobile hamburger — same nav markup, just a different open state.
    mobileToggle.addEventListener("click", () => {
      const open = nav.dataset.mobileOpen === "true";
      nav.dataset.mobileOpen = open ? "false" : "true";
      mobileToggle.setAttribute("aria-expanded", String(!open));
    });
    $$("#navlinks a").forEach((a) => a.addEventListener("click", () => {
      nav.dataset.mobileOpen = "false";
      mobileToggle.setAttribute("aria-expanded", "false");
    }));
  }

  /* ------------------------------------------------------------ theme toggle
     Three states — auto (follows the OS), light, dark — cycled by one button
     and remembered per browser. Auto clears the override so the existing
     prefers-color-scheme rules in styles.css take over again. */
  function initThemeToggle() {
    const btn = $("#themeToggle");
    if (!btn) return;
    const root = document.documentElement;
    const ORDER = ["auto", "light", "dark"];
    let mode = "auto";
    try { mode = localStorage.getItem("sonix-theme") || "auto"; } catch (e) { /* private mode etc. */ }

    function apply(m) {
      mode = m;
      if (m === "auto") root.removeAttribute("data-theme");
      else root.setAttribute("data-theme", m);
      btn.dataset.mode = m;
      btn.setAttribute("aria-label", `Colour theme: ${m}. Click to change.`);
      try { localStorage.setItem("sonix-theme", m); } catch (e) { /* ignore */ }
    }

    apply(mode);
    btn.addEventListener("click", () => {
      apply(ORDER[(ORDER.indexOf(mode) + 1) % ORDER.length]);
    });
  }

  /* ------------------------------------------------------------ scroll-spy */
  function initScrollSpy() {
    const links = $$("#navlinks a[data-nav]");
    if (!links.length) return;
    const map = new Map(links.map((a) => [a.dataset.nav, a]));
    const sections = [...map.keys()]
      .map((id) => document.getElementById(id))
      .filter(Boolean);
    if (!sections.length) return;

    const setActive = (id) => {
      links.forEach((a) => a.removeAttribute("aria-current"));
      map.get(id)?.setAttribute("aria-current", "true");
    };

    const io = new IntersectionObserver((entries) => {
      const visible = entries
        .filter((e) => e.isIntersecting)
        .sort((a, b) => b.intersectionRatio - a.intersectionRatio);
      if (visible[0]) setActive(visible[0].target.id);
    }, { rootMargin: "-40% 0px -50% 0px", threshold: [0, .25, .5, .75, 1] });

    sections.forEach((s) => io.observe(s));
  }

  /* ------------------------------------------------------------ back to top */
  function initToTop() {
    const btn = $("#toTop");
    if (!btn) return;
    window.addEventListener("scroll", () => {
      btn.dataset.show = window.scrollY > 700 ? "true" : "false";
    }, { passive: true });
    btn.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
  }

  /* ------------------------------------------------------------ animated counters
     Real, already-verified numbers (see docs/RESULTS_TABLE.md) — this only
     animates the reveal, it never changes what's displayed. */
  function initCounters() {
    const els = $$(".count-up");
    if (!els.length) return;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function animate(el) {
      const target = parseFloat(el.dataset.target);
      const decimals = parseInt(el.dataset.decimals || "0", 10);
      const suffix = el.dataset.suffix || "";
      if (!isFinite(target)) return;
      if (reduceMotion) {
        el.textContent = (decimals ? target.toFixed(decimals) : Math.round(target).toLocaleString()) + suffix;
        return;
      }
      const dur = 1000;
      const t0 = performance.now();
      function tick(now) {
        const p = Math.min(1, (now - t0) / dur);
        const eased = 1 - Math.pow(1 - p, 3);
        const val = target * eased;
        el.textContent = (decimals ? val.toFixed(decimals) : Math.round(val).toLocaleString()) + suffix;
        if (p < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    }

    const io = new IntersectionObserver((entries, obs) => {
      entries.forEach((e) => {
        if (e.isIntersecting) { animate(e.target); obs.unobserve(e.target); }
      });
    }, { threshold: .6 });
    els.forEach((el) => io.observe(el));
  }

  /* ------------------------------------------------------------ pipeline walkthrough */
  function initPipelineWalkthrough() {
    const btn = $("#walkBtn");
    const label = $("#walkLabel");
    const svg = $("#pipeline-svg");
    if (!btn || !svg) return;

    const order = ["step-capture", "step-resample", "step-consent", "step-window",
                    "step-vad", "step-embed", "step-head", "step-band", "step-audit"];
    const nodes = order.map((id) => document.getElementById(id)).filter(Boolean);

    let idx = -1, timer = null, playing = false;

    function highlight(i) {
      nodes.forEach((n) => n.classList.remove("walk-active"));
      const n = nodes[i];
      if (!n) { label.textContent = "Click a step, or press play"; return; }
      n.classList.add("walk-active");
      label.textContent = n.dataset.stepTitle || "";
    }

    function step() {
      idx = (idx + 1) % nodes.length;
      highlight(idx);
    }

    function play() {
      playing = true;
      btn.textContent = "⏸ Pause walkthrough";
      btn.setAttribute("aria-pressed", "true");
      step();
      timer = setInterval(step, 1700);
    }
    function pause() {
      playing = false;
      btn.textContent = "▶ Walk through the pipeline";
      btn.setAttribute("aria-pressed", "false");
      if (timer) clearInterval(timer);
      timer = null;
    }

    btn.addEventListener("click", () => (playing ? pause() : play()));

    nodes.forEach((n, i) => {
      n.addEventListener("click", () => {
        pause();
        idx = i;
        highlight(i);
      });
    });

    svg.addEventListener("mouseleave", () => { if (!playing) highlight(-1); });
  }

  /* ------------------------------------------------------------ copy buttons */
  function initCopyButtons() {
    $$("code.copyable").forEach((el) => {
      el.addEventListener("click", async () => {
        try {
          await navigator.clipboard.writeText(el.textContent.replace(/\s*copy(?:ied)?\s*$/i, ""));
          el.dataset.copied = "true";
          setTimeout(() => { delete el.dataset.copied; }, 1400);
        } catch (e) { /* clipboard unavailable — fine, it's just a convenience */ }
      });
    });
  }

  /* ------------------------------------------------------------ illustrative threshold slider */
  function initGaugeDemo() {
    const slider = $("#gaugeSlider");
    const valueEl = $("#gaugeSliderValue");
    const bandEl = $("#gaugeSliderBand");
    if (!slider) return;
    const descriptions = {
      g: "Green — consistent with a real voice",
      a: "Amber — uncertain, treat with caution",
      r: "Red — likely synthetic, verify another way",
    };
    function render() {
      const v = Number(slider.value) / 100;
      const band = bandFor(v);
      valueEl.textContent = `${slider.value}%`;
      valueEl.style.color = band.css;
      bandEl.textContent = descriptions[band.cls];
      bandEl.style.color = band.css;
    }
    slider.addEventListener("input", render);
    render();
  }

  /* ------------------------------------------------------------ hero canvas
     A mouse-repel particle field with a soft animated waveform layered
     underneath, both in one canvas. Lightweight on purpose. */
  function initHeroCanvas() {
    const canvas = $("#hero-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const hero = canvas.parentElement;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    let w = 0, h = 0, dpr = Math.min(window.devicePixelRatio || 1, 2);
    let particles = [];
    let raf = null;
    let running = false;
    let t = 0;
    const mouse = { x: -9999, y: -9999, active: false };

    const MAX_PARTICLES = 70;

    function tone() {
      const dark = getComputedStyle(document.documentElement).getPropertyValue("--bg").trim() !== "#ffffff";
      return {
        particle: dark ? "rgba(238,244,244,0.5)" : "rgba(22,23,26,0.35)",
        line: dark ? "rgba(238,244,244,0.10)" : "rgba(22,23,26,0.08)",
        wave: dark ? "rgba(238,244,244,0.14)" : "rgba(22,23,26,0.10)",
      };
    }

    function resize() {
      const rect = hero.getBoundingClientRect();
      w = rect.width; h = rect.height;
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
      canvas.style.width = w + "px";
      canvas.style.height = h + "px";
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

      const count = Math.min(MAX_PARTICLES, Math.round((w * h) / 16000));
      particles = Array.from({ length: count }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * 0.18,
        vy: (Math.random() - 0.5) * 0.18,
        r: 1 + Math.random() * 1.6,
      }));
    }

    function drawStatic() {
      resize();
      const c = tone();
      ctx.clearRect(0, 0, w, h);
      ctx.fillStyle = c.particle;
      particles.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      });
    }

    function step() {
      t += 1;
      const c = tone();
      ctx.clearRect(0, 0, w, h);

      ctx.lineWidth = 1.2;
      ctx.strokeStyle = c.wave;
      for (let band = 0; band < 3; band++) {
        ctx.beginPath();
        const amp = 14 + band * 8;
        const freq = 0.006 + band * 0.003;
        const phase = t * (0.006 + band * 0.004) + band * 2;
        const baseY = h * (0.62 + band * 0.09);
        for (let x = 0; x <= w; x += 6) {
          const y = baseY + Math.sin(x * freq + phase) * amp;
          x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      particles.forEach((p) => {
        if (mouse.active) {
          const dx = p.x - mouse.x, dy = p.y - mouse.y;
          const dist2 = dx * dx + dy * dy;
          const radius = 120;
          if (dist2 < radius * radius) {
            const dist = Math.sqrt(dist2) || 1;
            const force = (1 - dist / radius) * 1.6;
            p.vx += (dx / dist) * force * 0.06;
            p.vy += (dy / dist) * force * 0.06;
          }
        }
        p.vx *= 0.96; p.vy *= 0.96;
        p.x += p.vx; p.y += p.vy;

        if (p.x < -10) p.x = w + 10; if (p.x > w + 10) p.x = -10;
        if (p.y < -10) p.y = h + 10; if (p.y > h + 10) p.y = -10;
      });

      ctx.strokeStyle = c.line;
      ctx.lineWidth = 1;
      for (let i = 0; i < particles.length; i++) {
        for (let j = i + 1; j < particles.length; j++) {
          const a = particles[i], b = particles[j];
          const dx = a.x - b.x, dy = a.y - b.y;
          const d2 = dx * dx + dy * dy;
          if (d2 < 95 * 95) {
            ctx.beginPath();
            ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
            ctx.stroke();
          }
        }
      }

      ctx.fillStyle = c.particle;
      particles.forEach((p) => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
        ctx.fill();
      });

      if (running) raf = requestAnimationFrame(step);
    }

    function start() {
      if (running || reduceMotion) return;
      running = true;
      raf = requestAnimationFrame(step);
    }
    function stop() {
      running = false;
      if (raf) cancelAnimationFrame(raf);
      raf = null;
    }

    resize();
    if (reduceMotion) {
      drawStatic();
    } else {
      start();
    }

    window.addEventListener("resize", () => {
      resize();
      if (reduceMotion) drawStatic();
    });

    hero.addEventListener("mousemove", (e) => {
      const rect = hero.getBoundingClientRect();
      mouse.x = e.clientX - rect.left;
      mouse.y = e.clientY - rect.top;
      mouse.active = true;
    });
    hero.addEventListener("mouseleave", () => { mouse.active = false; });

    document.addEventListener("visibilitychange", () => {
      if (reduceMotion) return;
      document.hidden ? stop() : start();
    });

    const themeBtn = $("#themeToggle");
    themeBtn?.addEventListener("click", () => { if (reduceMotion) drawStatic(); });
    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
        if (reduceMotion) drawStatic();
      });
    }
  }

  /* ------------------------------------------------------------ file:// guard
     "No server detected" has two very different causes: the server really
     isn't running, or this page was opened by double-clicking index.html
     (file://) instead of through the server at http://localhost:8000/ — in
     which case even a running server can't be reached, because the fetch
     below is cross-origin from a null file:// origin and the browser blocks
     it before any status pill logic even runs. Say that plainly instead of
     leaving it looking identical to "server is down". */
  function initFileProtocolWarning() {
    if (location.protocol !== "file:") return;
    const bar = document.createElement("div");
    bar.className = "file-warning";
    bar.innerHTML = `<div class="wrap"><strong>You opened this file directly.</strong>
      That's why nothing here can reach the server — start it, then open
      <code>http://localhost:8000/</code> in your browser instead of double-clicking
      this file. Command: <code class="copyable">python -m realtime.server --ws-port 8000 --mode webrtc</code></div>`;
    document.body.prepend(bar);
  }

  /* ------------------------------------------------------------ server reachability */
  async function checkServer(timeoutMs = 1800) {
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await fetch(`${SERVER}/api/status`, { signal: ctrl.signal });
      clearTimeout(timer);
      if (!res.ok) return { up: false };
      const data = await res.json();
      return { up: true, data };
    } catch (e) {
      clearTimeout(timer);
      return { up: false };
    }
  }

  function setPill(pillId, textId, state, text) {
    const pill = document.getElementById(pillId);
    const label = document.getElementById(textId);
    if (!pill || !label) return;
    pill.dataset.state = state;
    label.textContent = text;
  }

  function bandFor(score) {
    if (score >= RED_AT) return { name: "Red", cls: "r", css: "var(--crit)" };
    if (score >= AMBER_AT) return { name: "Amber", cls: "a", css: "var(--warn)" };
    return { name: "Green", cls: "g", css: "var(--good)" };
  }

  function renderBandInto(verdictEl, whyEl, score, scoringAvailable) {
    if (!verdictEl || !whyEl) return;
    if (!scoringAvailable) {
      verdictEl.textContent = "Scoring unavailable";
      verdictEl.style.color = "var(--ink-3)";
      whyEl.textContent = "Server is running without a trained head (--ckpt). Capture is live; no verdict is shown.";
      return;
    }
    if (score === null || score === undefined) {
      verdictEl.textContent = "—";
      verdictEl.style.color = "var(--ink-3)";
      whyEl.textContent = "Waiting for the first 4-second window.";
      return;
    }
    const band = bandFor(score);
    verdictEl.textContent = `${band.name} · ${Math.round(score * 100)}%`;
    verdictEl.style.color = band.css;
    const action = { g: "Consistent with a real voice.", a: "Uncertain — treat with caution.", r: "Likely synthetic — verify another way." }[band.cls];
    whyEl.textContent = `${action} (Amber ≥ 35%, Red ≥ 65% — provisional thresholds)`;
  }

  let captureIsRunning = false; // shared with initLiveStatus so it doesn't clobber a live session's readout

  let lastKnownUp = null; // undefined until the first check resolves

  async function refreshServerStatus() {
    const result = await checkServer();

    if (!result.up) {
      setPill("micStatus", "micStatusText", "down", "No local server detected — start it, then Start capture below will work");
      setPill("uploadStatus", "uploadStatusText", "down", "No local server detected — uploads will not send anywhere");
      if (!captureIsRunning) {
        const why = $("#miniWhy");
        if (why) why.textContent = "Server not reachable on localhost:8000 from this page right now.";
      }
      lastKnownUp = false;
      return;
    }

    const scoringOn = !!result.data.scoring_available;
    setPill("micStatus", "micStatusText", "up",
      scoringOn ? "Server running — scoring is live" : "Server running — no trained head loaded yet");
    setPill("uploadStatus", "uploadStatusText", "up",
      scoringOn ? "Server running — uploads will be scored for real" : "Server running — scoring is switched off (no --ckpt)");

    // The server only just came up since the last check (e.g. it was started
    // after this page was already open) — the pill would otherwise have sat
    // on "down" forever, since nothing used to re-check it after page load.
    if (lastKnownUp === false && !captureIsRunning) {
      const why = $("#miniWhy");
      if (why) why.textContent = "Server just came online — press Start capture above to begin.";
    }
    lastKnownUp = true;
  }

  function initLiveStatus() {
    refreshServerStatus();
    // Same-origin GET to /api/status, a few hundred bytes, every 4s — cheap
    // enough to just keep polling so the pills self-heal without a refresh.
    setInterval(refreshServerStatus, 4000);
  }

  /* ------------------------------------------------------------ model picker
     Fills a model <select> from /api/models — only heads present on disk.
     Left disabled (just "Server default") when the server is unreachable or
     in mock mode. Shared between the Upload picker and the Live Voice
     Detection picker so both offer every trained head the same way. */
  let modelsCache = null;
  async function fetchModels() {
    if (modelsCache) return modelsCache;
    try {
      const res = await fetch(`${SERVER}/api/models`);
      if (!res.ok) return null;
      modelsCache = await res.json();
      return modelsCache;
    } catch (e) { return null; }
  }

  async function populateModelSelect(sel, note) {
    if (!sel) return null;
    const data = await fetchModels();
    if (!data) return null;

    if (data.mock || !data.models || !data.models.length) {
      if (note) note.textContent = data.mock
        ? "Server is in mock mode — no real heads to choose from."
        : "";
      return data;
    }

    const byKey = {};
    for (const m of data.models) {
      byKey[m.key] = m;
      const opt = document.createElement("option");
      opt.value = m.key;
      opt.textContent = m.label + (m.key === data.default ? "  (server default)" : "") + (m.exists ? "" : "  — file missing");
      opt.disabled = !m.exists;
      opt.title = m.note || "";
      if (m.key === data.default) opt.selected = true;
      sel.appendChild(opt);
    }
    if (sel.options.length > 1) sel.disabled = false;

    const reflect = () => {
      const m = byKey[sel.value] || byKey[data.default];
      if (note) note.textContent = m ? m.note : "";
    };
    sel.addEventListener("change", reflect);
    reflect();
    return data;
  }

  async function initModelPicker() {
    await populateModelSelect($("#modelSelect"), $("#modelNote"));
  }

  /* ------------------------------------------------------------ score gauge (upload result) */
  function setGauge(score) {
    const fill = $("#gaugeFill"), valueEl = $("#gaugeValue");
    if (!fill || !valueEl) return;
    const total = fill.getTotalLength();
    fill.style.strokeDasharray = `${total}`;
    if (score === null || score === undefined) {
      fill.style.strokeDashoffset = `${total}`;
      fill.style.stroke = "var(--ink-3)";
      valueEl.textContent = "—";
      return;
    }
    const clamped = Math.max(0, Math.min(1, score));
    fill.style.strokeDashoffset = `${total * (1 - clamped)}`;
    fill.style.stroke = bandFor(clamped).css;
    valueEl.textContent = `${Math.round(clamped * 100)}%`;
  }

  /* ------------------------------------------------------------ upload flow
     Real POST to /api/score-file — the same endpoint the demo and the
     dashboard use. Degrades to a clear explanation if unreachable. */
  function initUpload() {
    const dz = $("#dropzone");
    const input = $("#fileInput");
    const dzMain = $("#dzMain");
    const resultBox = $("#uploadResult");
    if (!dz || !input) return;
    setGauge(null);

    const openPicker = () => input.click();
    dz.addEventListener("click", openPicker);
    dz.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); openPicker(); }
    });

    ["dragenter", "dragover"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
    ["dragleave", "drop"].forEach((ev) =>
      dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
    dz.addEventListener("drop", (e) => {
      const file = e.dataTransfer?.files?.[0];
      if (file) handleFile(file);
    });
    input.addEventListener("change", () => {
      if (input.files?.[0]) handleFile(input.files[0]);
    });

    async function handleFile(file) {
      dzMain.textContent = `Scoring "${file.name}"… (loads the head on first use — up to ~20s)`;
      resultBox.dataset.show = "false";

      const reach = await checkServer(1500);
      if (!reach.up) {
        dzMain.textContent = "No local server reachable — start it, then drop the file again";
        return;
      }
      if (!reach.data.scoring_available) {
        dzMain.textContent = "Server is running, but no trained head is loaded — nothing to score yet";
        return;
      }

      try {
        const form = new FormData();
        form.append("file", file);
        form.append("wait", "1");
        const model = $("#modelSelect")?.value;
        if (model) form.append("model", model);
        const res = await fetch(`${SERVER}/api/score-file`, { method: "POST", body: form });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          dzMain.textContent = `Server error: ${err.error || res.status}`;
          return;
        }
        const data = await res.json();
        renderResult(file, data);
      } catch (e) {
        dzMain.textContent = "Could not reach the server — is it still running?";
      }
    }

    function renderResult(file, data) {
      dzMain.textContent = "Drop another recording, or click to choose one";
      const mean = data?.summary?.mean_score;
      $("#resFile").textContent = file.name;
      $("#resModel").textContent = data.model || "server default";
      $("#resWindows").textContent = `${data.windows_scored ?? "0"} / ${data.expected_windows ?? "?"}`;

      const strip = $("#resStrip");
      const scores = Object.keys(data.scores || {})
        .sort((a, b) => (+a) - (+b))
        .map((k) => data.scores[k].score);

      if (mean === null || mean === undefined) {
        $("#resMean").textContent = "—";
        $("#resBand").textContent = "No windows scored (too short, or all silence)";
        $("#resBand").style.color = "var(--ink-3)";
        if (strip) strip.replaceChildren();
        setGauge(null);
      } else {
        const band = bandFor(mean);
        $("#resMean").textContent = `${(mean * 100).toFixed(1)}%`;
        $("#resBand").textContent = band.name;
        $("#resBand").style.color = band.css;
        setGauge(mean);
        if (strip) {
          strip.replaceChildren();
          for (const v of scores) {
            const bar = document.createElement("i");
            bar.className = bandFor(v).cls;
            bar.style.height = `${Math.max(6, v * 100).toFixed(0)}%`;
            bar.title = `${(v * 100).toFixed(1)}%`;
            strip.appendChild(bar);
          }
        }
      }
      resultBox.dataset.show = "true";
    }
  }

  /* ------------------------------------------------------------ embedded live capture
     Everything the old standalone /mic page did — getUserMedia, an
     AudioWorklet resampled to 16kHz, streaming PCM16 over the same
     WebSocket, showing the pairing code, and the risk timeline — ported
     onto this page so "Live Voice Detection" never navigates anywhere. The
     pairing code is now approved with a button right here (POST
     /api/approve), instead of requiring the separate Streamlit dashboard. */
  const WORKLET_SRC = `
class Cap extends AudioWorkletProcessor {
  process(inputs){
    const ch = inputs[0][0];
    if (ch) this.port.postMessage(new Float32Array(ch));
    return true;
  }
}
registerProcessor('cap', Cap);
`;

  function floatToPCM16(f32) {
    const out = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      let s = Math.max(-1, Math.min(1, f32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  function initLiveCapture() {
    const btn = $("#captureBtn");
    if (!btn) return;

    const levelBar = $("#levelBar");
    const pairingBox = $("#pairingBox");
    const pairingCode = $("#pairingCode");
    const approveBtn = $("#approveBtn");
    const capCallId = $("#capCallId");
    const capModel = $("#capModel");
    const capSr = $("#capSr");
    const capSent = $("#capSent");
    const capState = $("#capState");
    const verdictEl = $("#miniVerdict");
    const whyEl = $("#miniWhy");
    const canvas = $("#riskCanvas");
    const modelSelect = $("#captureModelSelect");
    const modelNote = $("#captureModelNote");

    populateModelSelect(modelSelect, modelNote);

    if (!navigator.mediaDevices || !window.AudioWorkletNode) {
      btn.disabled = true;
      btn.textContent = "Mic capture not supported in this browser";
      whyEl.textContent = "This browser lacks AudioWorklet support — use Chrome or Edge.";
      return;
    }

    let ws = null, ctx = null, node = null, stream = null, running = false;
    let sentSamples = 0, callId = null, scoringAvailable = false, lastScore = null;
    const scoresByIdx = new Map();

    function resetReadout() {
      lastScore = null;
      scoresByIdx.clear();
      renderScores();
      pairingBox.hidden = true;
      capCallId.textContent = "—";
      capModel.textContent = "—";
      capSr.textContent = "—";
      capSent.textContent = "0.0 s";
      capState.textContent = "—";
      levelBar.style.width = "0%";
    }

    function renderScores() {
      const vals = [...scoresByIdx.values()];
      if (vals.length) {
        const latestIdx = Math.max(...scoresByIdx.keys());
        lastScore = scoresByIdx.get(latestIdx);
      } else {
        lastScore = null;
      }
      renderBandInto(verdictEl, whyEl, lastScore, scoringAvailable);
      drawRisk();
    }

    function drawRisk() {
      if (!canvas) return;
      const dpr = window.devicePixelRatio || 1;
      const cssW = canvas.clientWidth || 460, cssH = canvas.clientHeight || 320;
      canvas.width = cssW * dpr; canvas.height = cssH * dpr;
      const g = canvas.getContext("2d");
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, cssW, cssH);

      const cs = getComputedStyle(document.documentElement);
      const good = cs.getPropertyValue("--good").trim() || "#16a34a";
      const warn = cs.getPropertyValue("--warn").trim() || "#d97706";
      const crit = cs.getPropertyValue("--crit").trim() || "#dc2626";
      const ink3 = cs.getPropertyValue("--ink-3").trim() || "#8a8f98";
      const ink = cs.getPropertyValue("--ink").trim() || "#16171a";

      const padL = 36, padR = 12, padT = 10, padB = 20;
      const w = cssW - padL - padR, h = cssH - padT - padB;
      const pts = [...scoresByIdx.entries()].sort((a, b) => a[0] - b[0]);
      const maxIdx = pts.length ? pts[pts.length - 1][0] : 0;
      const spanIdx = Math.max(20, maxIdx);
      const X = (i) => padL + (spanIdx ? (i / spanIdx) * w : 0);
      const Y = (s) => padT + (1 - s) * h;

      const bands = [[0, AMBER_AT, good], [AMBER_AT, RED_AT, warn], [RED_AT, 1, crit]];
      g.globalAlpha = 0.10;
      for (const [lo, hi, col] of bands) {
        g.fillStyle = col;
        g.fillRect(padL, Y(hi), w, Y(lo) - Y(hi));
      }
      g.globalAlpha = 1;

      g.lineWidth = 1; g.font = "10px system-ui"; g.setLineDash([4, 3]);
      for (const [y, col, txt] of [[AMBER_AT, warn, "Amber"], [RED_AT, crit, "Red"]]) {
        g.strokeStyle = col; g.beginPath(); g.moveTo(padL, Y(y)); g.lineTo(padL + w, Y(y)); g.stroke();
        g.fillStyle = col; g.fillText(txt, padL + w - 32, Y(y) - 3);
      }
      g.setLineDash([]);

      g.fillStyle = ink3;
      g.fillText("100%", 2, Y(1) + 3);
      g.fillText("50%", 6, Y(0.5) + 3);
      g.fillText("0%", 12, Y(0) + 3);
      g.fillText("time →", padL, cssH - 5);

      if (pts.length) {
        g.strokeStyle = ink; g.lineWidth = 2; g.beginPath();
        pts.forEach(([i, s], k) => { const x = X(i), y = Y(s); k ? g.lineTo(x, y) : g.moveTo(x, y); });
        g.stroke();
        g.fillStyle = ink;
        pts.forEach(([i, s]) => { g.beginPath(); g.arc(X(i), Y(s), 2.6, 0, Math.PI * 2); g.fill(); });
      }
    }

    async function start() {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false }
        });
      } catch (e) {
        whyEl.textContent = "The browser blocked microphone access. Allow it for this site — it needs a secure context (localhost is fine).";
        return;
      }

      const reach = await checkServer(1500);
      if (!reach.up) {
        whyEl.textContent = "No local server reachable at " + SERVER + " — start it, then press Start capture again.";
        stream.getTracks().forEach((t) => t.stop());
        stream = null;
        return;
      }
      scoringAvailable = !!reach.data.scoring_available;
      if (reach.data.scoring_synthetic) {
        whyEl.textContent = "UNTRAINED dev checkpoint loaded — every score will be noise (plumbing/latency test only).";
      }

      scoresByIdx.clear();
      renderScores();

      ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
      await ctx.audioWorklet.addModule(URL.createObjectURL(new Blob([WORKLET_SRC], { type: "application/javascript" })));
      capSr.textContent = ctx.sampleRate + " Hz" + (ctx.sampleRate === 16000 ? "" : " (resampled)");

      ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";

      ws.onopen = () => {
        capState.textContent = "CONNECTING";
        ws.send(JSON.stringify({
          type: "start_mic_call",
          sample_rate: ctx.sampleRate,
          caller: "website",
          model: modelSelect?.value || undefined,
        }));
      };

      ws.onmessage = (ev) => {
        if (typeof ev.data !== "string") return;
        let m; try { m = JSON.parse(ev.data); } catch { return; }

        if (m.type === "error") {
          whyEl.textContent = m.message || "The server rejected this capture request.";
          stop();
          return;
        }
        if (m.type === "mic_call_started") {
          callId = m.call_id;
          capCallId.textContent = callId;
          capModel.textContent = m.model || "server default";
          pairingCode.textContent = m.pairing_code;
          pairingBox.hidden = false;
          capState.textContent = "CONSENT_PENDING";
        }
        if (m.type === "pairing_request" && m.call_id === callId) {
          pairingCode.textContent = m.pairing_code;
        }
        if (m.type === "scores" && callId && m.data && m.data[callId]) {
          if (!scoringAvailable) return;
          const d = m.data[callId];
          const items = (Array.isArray(d.batch) && d.batch.length) ? d.batch : [{ window_idx: d.window_idx, score: d.score }];
          for (const it of items) {
            if (it && it.window_idx != null && it.score != null) {
              scoresByIdx.set(it.window_idx, Math.max(0, Math.min(1, it.score)));
            }
          }
          renderScores();
        }
        if (m.type === "call_state" && m.call_id === callId) {
          capState.textContent = m.state.toUpperCase();
          if (m.state === "listening" || m.state === "scoring") {
            pairingBox.hidden = true;
          }
        }
      };

      ws.onerror = () => { whyEl.textContent = "WebSocket error — could not reach the server."; };
      ws.onclose = () => { if (running) stop(); };

      const src = ctx.createMediaStreamSource(stream);
      node = new AudioWorkletNode(ctx, "cap");

      node.port.onmessage = (ev) => {
        const f32 = ev.data;
        let peak = 0;
        for (let i = 0; i < f32.length; i++) { const a = Math.abs(f32[i]); if (a > peak) peak = a; }
        levelBar.style.width = Math.min(100, peak * 140) + "%";

        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(floatToPCM16(f32).buffer);
          sentSamples += f32.length;
          capSent.textContent = (sentSamples / ctx.sampleRate).toFixed(1) + " s";
        }
      };

      src.connect(node);
      const mute = ctx.createGain(); mute.gain.value = 0;
      node.connect(mute).connect(ctx.destination);

      running = true;
      captureIsRunning = true;
      btn.textContent = "Stop capture";
      btn.classList.add("is-recording");
      approveBtn.disabled = false;
      if (modelSelect) modelSelect.disabled = true;
    }

    function stop() {
      running = false;
      captureIsRunning = false;
      if (ws && ws.readyState === WebSocket.OPEN) {
        if (callId) ws.send(JSON.stringify({ type: "end_call", call_id: callId }));
        ws.close();
      }
      if (node) node.disconnect();
      if (stream) stream.getTracks().forEach((t) => t.stop());
      if (ctx) ctx.close();
      ws = node = stream = ctx = null;
      callId = null;
      btn.textContent = "Start capture";
      btn.classList.remove("is-recording");
      if (modelSelect && modelSelect.options.length > 1) modelSelect.disabled = false;
      resetReadout();
      whyEl.textContent = "Press Start capture above to begin.";
    }

    btn.addEventListener("click", () => (running ? stop() : start()));

    approveBtn.addEventListener("click", async () => {
      if (!callId) return;
      approveBtn.disabled = true;
      approveBtn.textContent = "Approving…";
      try {
        const res = await fetch(`${SERVER}/api/approve`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ call_id: callId }),
        });
        if (!res.ok) throw new Error(String(res.status));
        approveBtn.textContent = "Approved";
      } catch (e) {
        approveBtn.disabled = false;
        approveBtn.textContent = "Approve on this device";
        whyEl.textContent = "Could not approve — the server may have restarted. Try Stop, then Start capture again.";
      }
    });

    window.addEventListener("resize", drawRisk);
    resetReadout();
  }

  /* ------------------------------------------------------------ boot */
  document.addEventListener("DOMContentLoaded", () => {
    initFileProtocolWarning();
    initNav();
    initThemeToggle();
    initScrollSpy();
    initToTop();
    initCounters();
    initPipelineWalkthrough();
    initCopyButtons();
    initGaugeDemo();
    initHeroCanvas();
    initLiveStatus();
    initModelPicker();
    initUpload();
    initLiveCapture();
  });
})();
