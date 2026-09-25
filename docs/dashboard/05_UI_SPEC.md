# 05 — UI specification

Invoke the **`ui-ux-pro-max`** skill before writing any component. It carries the palettes,
type pairings, chart conventions and accessibility guidelines this spec assumes; this document
says *what* to build and what it must never claim, not how to make it look good.

One app, two modes, toggled in the header and persisted to `localStorage`. Everything in this
spec is driven by real API data — there are no illustrative placeholders anywhere in the
shipped UI.

---

## Design tokens

Dark theme is the default; the room lights will be down. Light theme is a nice-to-have, and if
present it must be defined in the same token layer rather than bolted on with overrides.

```css
:root {
  --bg: #0a0d12;  --surface: #12161d;  --surface-2: #1a1f28;  --border: #262c36;
  --text: #e8ecf1;  --text-dim: #8b95a5;

  /* Bands. Semantics come from the API, never from CSS. */
  --green: #22c55e;  --amber: #f59e0b;  --red: #ef4444;
  --green-dim: #14532d;  --amber-dim: #78350f;  --red-dim: #7f1d1d;

  --accent: #3b82f6;      /* interactive, never a band colour */
  --mock: #a855f7;        /* mock-mode chrome — deliberately NOT green/amber/red */

  --radius: 10px;
  --mono: 'JetBrains Mono', ui-monospace, monospace;
}
```

**Band colour is never chosen in CSS.** `lib/bands.ts` maps a score to a band using thresholds
fetched from `/api/dashboard/summary`, and the component asks for the band. This is the fourth
place 0.35/0.65 could have been hardcoded, and `realtime/tests/test_thresholds.py` exists
because the first three were.

Colour is never the only signal — every band carries a text label and a distinct shape or icon.
A projector with the contrast turned up, and any red-green colour-blind judge, both need that.

---

## Shell — always visible

**Header:** SONIX mark · mode toggle (Demo / Operator) · server status pill · active model ·
WS connection indicator.

**Server status pill** has four states and they are not interchangeable:

| State | Shows | Means |
|---|---|---|
| `offline` | red outline, "No server at :8000" | fetch failed |
| `warming` | amber pulse, "Loading front-end…" | `warming: true` |
| `mock` | **purple**, "MOCK — not detecting" | `scoring_available: false` |
| `ready` | green, model name | warm, real head loaded |

**`MockBanner`** — when `scoring_available` is false, a full-width purple bar sits under the
header on every screen and cannot be dismissed:

> **MOCK MODE — these scores are random, not detections.** No trained head (`head*.pt`) was
> found on this machine. Scores are capped at 0.5 and never reach Red. Nothing on this screen
> is a real result.

This is the single most important piece of UI in the app. `head*.pt` is gitignored and
file-transferred (`CODE_MAP.md`), so a teammate who pulls the repo and runs the server gets
mock scoring by default — and mock output looks calm and green, which is exactly what a working
detector looks like. A demo that is accidentally mock is the worst outcome this project has
available to it.

**`OfflineState`** — server unreachable renders the launch command, copyable:

```
python -m realtime.server --ws-port 8000 --mode webrtc --ckpt outputs/models/head_v3.pt
```

Never an infinite spinner. Never placeholder data behind a loading state.

---

## Demo mode

Five minutes, a projector, and judges who have seen four dashboards already today. Ordered so
that stopping at any point still leaves a complete story.

### 1. Live band — the hero

Current band as the dominant element: band word, score as a percentage, and one plain sentence
("consistent with a real voice" / "showing synthetic characteristics"). Under it, the live
`RiskTimeline` canvas with the amber and red thresholds drawn as horizontal rules.

Start control runs the mic flow: `getUserMedia` → AudioWorklet → 16 kHz PCM → WS
`start_mic_call` → pairing code → `/api/approve` → `scores` frames. `website/script.js` already
implements this handshake end to end; read it rather than reconstructing it from the API doc.

**Before the first score exists**, show `Listening — first reading in 4s` with a real countdown
driven by samples received. Explain why in one line: SONIX needs a full 4-second window and
never pads one, because a half-silent window is what produces false positives. That sentence
turns a wait into a feature, and it is true — `docs/RESULTS_TABLE.md` has the padding numbers.

### 2. Pipeline diagram

Eight stages, SVG, lit by real telemetry as audio moves through them:

```
capture → resample → consent gate → 4s window → silence gate → embed (frozen) → head (trained) → band + audit
```

Two stages carry annotations that are the product's actual argument:

- **consent gate** — "audio is discarded, not buffered, until a human approves." Enforced in
  `Session.push_audio()`, not by convention.
- **embed / head** — "300M frozen, never trained" against "~300k trained, ~1 MB". The size
  contrast is the whole architecture in one picture.

`website/index.html` has an SVG of this pipeline with the stage copy already written. Reuse the
wording; do not invent new stage names that disagree with the existing site.

### 3. Model comparison — the centrepiece

Pick a clip (upload, or one of the bundled `sonix_real/` real/fake pairs), pick 2–4 heads from
`/api/models`, compare.

- Overlaid score timelines, one line per model, on one time axis.
- Summary table: model · label · mean · median · p95 · band · head ms.
- **The cost split, stated prominently:** "one embedding pass: 412 ms · three heads: 12 ms +
  11 ms + 13 ms". That single line is the evidence for the claim that a new detector costs a
  megabyte and no extra GPU memory.
- Model notes come from the REGISTRY (`realtime/models.py`) verbatim — including the honest
  ones. `full_indic_as5` says "NO EER HAS BEEN VERIFIED FOR THIS CHECKPOINT ON THIS MACHINE".
  Show that. A judge who finds a caveat you displayed trusts the rest; one who finds a caveat
  you hid does not.
- Heads with `exists: false` render disabled with "checkpoint not on this machine".

### 4. Results panel

From `docs/RESULTS_TABLE.md` §A only, each with its reproduce command visible:

- 1.4937% EER on ASVspoof-2019 LA eval, baseline head · `python src\metrics.py --split eval`
- eval set 71,237 clips (7,355 bonafide / 63,882 spoof)
- the padding fix before/after table

§B rows appear as **"not yet measured"** with the owner named. Not blank, not a dash, not an
estimate. Caveat 1 from §C — that the augmented head's 0.24% dev EER is *not* comparable to the
baseline's 0.16% because the test sets differ — must be adjacent to any place both appear. That
comparison is, in Yukti's words, the single easiest thing for a sharp judge to catch.

### 5. Upload & score

Drag in a file, watch windows score in, timeline plus band plus per-window table. Uses
`POST /api/score-file` — the same endpoint the demo and the extension use, which is worth saying
on screen.

---

## Operator mode

What this looks like installed at a bank. Denser, calmer, no persuasion.

### Active calls
Table: call_id · state · current band · windows scored · duration · model · source
(`voip`/`webrtc`/`upload`). **Consent state is the first column after the id**, because it is the
thing that determines whether scoring is even legal. Row click opens the call detail.

### Consent & pairing
Pending pairing codes with their 120 s expiry counting down (`PairingCodeManager.generate`),
approve and end-call actions. State the rule in the panel: audio is dropped, not buffered, in
any state other than LISTENING.

### Audit trail
Per-call events from the session audit record, timestamped, chronological, exportable as JSON.
Footer line: "No audio was retained. Scores and window statistics only." That is literally true —
the ring buffer holds one window and is overwritten — and it maps onto the problem statement's
retention requirement.

### Engine health
Throughput, batches, avg windows/batch, failed batches, queue depth, device, loaded models,
warm/warming, `last_error`. Throughput as a canvas sparkline.

### Job queue & run history
Queue with progress bars and cancel. History table, virtualized, drillable into any past run.
Failed jobs show the **full traceback** in a monospace block with a copy button — `CLAUDE.md`
rule 2, surfaced in the UI instead of buried in a log.

### Thresholds
`amber_at` / `red_at` as served, labelled **provisional, uncalibrated**, with a link to
`realtime/thresholds.py`. If you add runtime overrides, they are display-only per session and
the panel says plainly that they do not change the pinned production bands — `realtime/live_ui.py`
already has operator sliders (defaulting 0.10/0.90) and `CODE_MAP.md` is explicit that those are
a different thing that must not be "reconciled" with the production numbers.

---

## Copy rules

1. **Never invent a number.** Not in a mock-up, not in a placeholder, not in a tooltip. If data
   has not arrived, the empty state says what would fill it.
2. EER is always "X% on \<split\>". Never bare.
3. Bands are described as provisional wherever a legend appears.
4. The band is a **risk indicator, not a verdict.** Copy says "showing synthetic
   characteristics", never "this is a deepfake".
5. Model names and notes are quoted from the REGISTRY, not rewritten.
6. Say what is not built. `docs/API.md` has a "what is not built yet" table and it is one of the
   more persuasive things in the repo.

## Accessibility

Keyboard-navigable throughout; visible focus rings; charts have text alternatives and an
accessible data table behind a toggle; band state announced via `aria-live="polite"` (polite, not
assertive — it changes up to twice a second); contrast ≥ 4.5:1 on text; respects
`prefers-reduced-motion`. Works from a 1280 px projector down to a 13" laptop.
