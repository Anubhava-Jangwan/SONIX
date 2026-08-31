/*
 * On-page overlay for meet.google.com.
 *
 * A small draggable card pinned top-right of the Meet window showing the live
 * risk band, so nobody has to watch the extension popup during a call. It is
 * created lazily on the first state message and removed when capture stops.
 */

let el = null;

function ensureOverlay() {
  if (el && document.body.contains(el)) return el;
  el = document.createElement("div");
  el.id = "sonix-overlay";
  el.innerHTML = `
    <div class="sonix-row">
      <span class="sonix-dot"></span>
      <span class="sonix-name">SONIX</span>
      <span class="sonix-band">—</span>
    </div>
    <div class="sonix-sub">Monitoring this call</div>`;
  document.body.appendChild(el);
  return el;
}

function removeOverlay() {
  if (el && el.parentNode) el.parentNode.removeChild(el);
  el = null;
}

function bandFor(score, available) {
  if (!available) return ["Monitoring", "#0b7c86", "Scoring unavailable — no trained head loaded"];
  if (score === null || score === undefined) return ["Listening", "#6e7d7d", "Waiting for the first scored window"];
  const pct = Math.round(score * 100);
  if (score >= 0.65) return [`Red · ${pct}%`, "#d03b3b", "Likely synthetic — verify by another channel"];
  if (score >= 0.35) return [`Amber · ${pct}%`, "#fab219", "Uncertain — treat with caution"];
  return [`Green · ${pct}%`, "#0ca30c", "Consistent with a real voice"];
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type !== "sonix:state") return;
  const s = msg.state;

  if (!s.capturing) {
    removeOverlay();
    return;
  }

  const node = ensureOverlay();

  if (s.callState === "consent_pending") {
    node.querySelector(".sonix-dot").style.background = "#fab219";
    node.querySelector(".sonix-band").textContent = `Code ${s.pairingCode || "------"}`;
    node.querySelector(".sonix-band").style.color = "#fab219";
    node.querySelector(".sonix-sub").textContent = "Approve in the SONIX dashboard to begin";
    return;
  }

  const [label, colour, sub] = bandFor(s.lastScore, s.scoringAvailable);
  node.querySelector(".sonix-dot").style.background = colour;
  node.querySelector(".sonix-band").textContent = label;
  node.querySelector(".sonix-band").style.color = colour;
  node.querySelector(".sonix-sub").textContent = sub;
});
