/* SONIX marketing site — vanilla JS, no build step, no framework.
   Pieces: nav (incl. accessible dropdown), the hero canvas animation, the
   server-reachability check, the model picker, and the real upload-and-score
   flow against the local API. Served by realtime/server.py at "/", so the API
   is same-origin. */

(() => {
  "use strict";

  // Same origin when served by realtime/server.py; falls back to the documented
  // port when the file is opened directly.
  const SERVER = location.protocol.startsWith("http")
    ? location.origin.replace(/\/$/, "")
    : "http://localhost:8000";
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
      const dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
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

    if (window.matchMedia) {
      window.matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", () => {
        if (reduceMotion) drawStatic();
      });
    }
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

  async function initLiveStatus() {
    const result = await checkServer();

    if (!result.up) {
      setPill("micStatus", "micStatusText", "down", "No local server detected — start it to see this live");
      setPill("uploadStatus", "uploadStatusText", "down", "No local server detected — uploads will not send anywhere");
      const verdict = $("#miniVerdict"), why = $("#miniWhy");
      if (verdict) { verdict.textContent = "—"; verdict.style.color = "var(--ink-3)"; }
      if (why) why.textContent = "Server not reachable on localhost:8000 from this page right now.";
      return;
    }

    const scoringOn = !!result.data.scoring_available;
    setPill("micStatus", "micStatusText", "up",
      scoringOn ? "Server running — scoring is live" : "Server running — no trained head loaded yet");
    setPill("uploadStatus", "uploadStatusText", "up",
      scoringOn ? "Server running — uploads will be scored for real" : "Server running — scoring is switched off (no --ckpt)");

    const verdict = $("#miniVerdict"), why = $("#miniWhy");
    if (verdict && why) {
      if (scoringOn) {
        verdict.textContent = "Ready";
        verdict.style.color = "var(--good)";
        why.textContent = "A trained head is loaded. Start capture on the mic page to see real scores.";
      } else {
        verdict.textContent = "Scoring unavailable";
        verdict.style.color = "var(--ink-3)";
        why.textContent = "Server is running without a trained head (--ckpt). Capture and consent still work live; no verdict is shown.";
      }
    }
  }

  /* ------------------------------------------------------------ model picker
     Fills the upload model dropdown from /api/models — only heads present on
     disk. Left disabled (just "Server default") when the server is unreachable
     or in mock mode. */
  async function initModelPicker() {
    const sel = $("#modelSelect"), note = $("#modelNote");
    if (!sel) return;
    let data;
    try {
      const res = await fetch(`${SERVER}/api/models`);
      if (!res.ok) return;
      data = await res.json();
    } catch (e) { return; }

    if (data.mock || !data.models || !data.models.length) {
      if (note) note.textContent = data.mock
        ? "Server is in mock mode — no real heads to choose from."
        : "";
      return;
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
      dzMain.textContent = `Scoring “${file.name}”… (loads the head on first use — up to ~20s)`;
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
      } else {
        const band = bandFor(mean);
        $("#resMean").textContent = `${(mean * 100).toFixed(1)}%`;
        $("#resBand").textContent = band.name;
        $("#resBand").style.color = band.css;
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

  /* ------------------------------------------------------------ boot */
  document.addEventListener("DOMContentLoaded", () => {
    initNav();
    initHeroCanvas();
    initLiveStatus();
    initModelPicker();
    initUpload();
  });
})();
