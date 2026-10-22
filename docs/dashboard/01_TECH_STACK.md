# 01 — Tech stack

Every dependency here earns its place by removing latency, removing code, or removing a class
of bug. If a library does none of those three, it does not go in. Apply the `ponytail` skill's
test to every addition: *could the standard library or a native browser feature do this?*

---

## Backend — Python

The server already exists. You are extending it, not choosing a stack.

### Already in `requirements.txt` — use these

| Package | Version | Why it's already here |
|---|---|---|
| `aiohttp` | `>=3.9` | The HTTP + WebSocket server. `realtime/server.py` is built on it. |
| `torch`, `torchaudio` | CUDA build | Inference. Installed from the pytorch.org index, **not** from `requirements.txt`. |
| `transformers` | `>=4.35` | wav2vec2 XLS-R front-end. |
| `numpy` | `>=1.24` | Audio buffers, embedding matrices. |
| `scipy` | `>=1.10` | `resample_poly`. Without it the live path degrades to linear interpolation. |
| `soundfile` | `>=0.12` | Audio decode for uploads. |
| `pytest`, `pytest-asyncio` | `>=7.4`, `>=0.23` | The existing `realtime/tests/` suite. |

### Add exactly these — nothing else

| Package | Version | Why | Why not something bigger |
|---|---|---|---|
| `orjson` | `>=3.9` | JSON encode for WS frames and job records. 2–5× faster than `json` and it emits `bytes` directly, which is what `ws.send_bytes` wants. On the hot broadcast path this is measurable. | — |
| `aiofiles` | `>=23.2` | Non-blocking job-record writes. Job I/O on the event loop stalls every live call. | A thread executor also works; `aiofiles` is less code. Pick one, not both. |

**That is the entire backend addition.** Two packages.

### Explicitly rejected

| Rejected | Why |
|---|---|
| **FastAPI / Starlette / uvicorn** | A second ASGI server means a second process, a second port, and a second copy of the 300M front-end in memory — or an IPC hop to avoid it. Both are worse than adding six routes to the aiohttp app that already owns the engine. |
| **Celery / RQ / Dramatiq** | A broker, a worker process and a Redis dependency, to run jobs on the same box against a model that is already resident in *this* process. `asyncio.Queue` + the existing `ScoringEngine` is the whole job queue. |
| **Redis / Postgres / SQLite** | The job store is a few thousand JSON records. The repo already persists per-call JSON to `outputs/calls/`. Match that pattern: `outputs/jobs/*.json` + an in-memory index rebuilt on startup. Add a database when someone can point at the query that needs one. |
| **SQLAlchemy / Pydantic** | No database, and aiohttp handlers validate a handful of fields. A `dataclass` and an explicit `_validate()` are smaller and faster than a schema layer. |
| **Socket.IO** | `/ws` is already a working raw WebSocket with live clients (`website/script.js`, the extension, `realtime/miccapture.py`). Changing transport breaks all three for no gain. |
| **gRPC** | `docs/API.md` already lists it honestly as a future transport addition. Out of scope; do not quietly start it. |

---

## Frontend — `dashboard/`

Self-contained. Its own `package.json`. Do **not** touch the repo-root `package.json`.

### Core

| Package | Version | Role |
|---|---|---|
| `react`, `react-dom` | `^18.3` | UI. |
| `typescript` | `^5.4` | `strict: true`, `noUncheckedIndexedAccess: true`. |
| `vite` | `^5.2` | Dev server with HMR and `/api` + `/ws` proxy; production build to `dashboard/dist/`. |
| `@vitejs/plugin-react-swc` | `^3.6` | SWC over Babel — noticeably faster cold start and HMR. |
| `tailwindcss` | `^3.4` | Styling. |
| `class-variance-authority`, `clsx`, `tailwind-merge` | latest | shadcn/ui's variant plumbing. |

### shadcn/ui

Not a dependency — you copy components into `dashboard/src/components/ui/`. Install **only the
primitives you actually render**. The list you will need:

`button` · `card` · `table` · `tabs` · `badge` · `dialog` · `select` · `switch` · `tooltip` ·
`progress` · `separator` · `scroll-area` · `sonner` (toasts) · `skeleton` · `alert`

Each pulls its own `@radix-ui/react-*` peer. Do not run a bulk "add everything" — each unused
component is dead weight in the bundle and a file a reviewer has to read past.

### Charts

| Package | Version | Role |
|---|---|---|
| `recharts` | `^2.12` | The comparison chart, score histograms, engine-throughput sparklines. |

**Do not** use Recharts for the live risk timeline. See `04_PERFORMANCE.md` — that one is a
`<canvas>` you draw yourself. Recharts re-renders a React tree per data point and will drop
frames under a 2 Hz score stream with 87.5% window overlap. `website/script.js` already draws
this timeline on a canvas; read it before writing your own.

### State and data

| Package | Version | Role |
|---|---|---|
| `zustand` | `^4.5` | Client state: mode toggle, selected models, WS connection state. ~1 KB, no provider tree. |
| `@tanstack/react-query` | `^5.x` | Server state: polling `/api/dashboard/summary`, job lists, run history. Gives dedupe, background refetch, stale-while-revalidate and request cancellation for free — all of which you would otherwise write badly by hand. |

WebSocket frames bypass React Query and go straight into a zustand store, then to canvas.
Do not route a 2 Hz stream through a cache layer.

### Rejected on the frontend

| Rejected | Why |
|---|---|
| **Next.js** | SSR buys nothing for a dashboard behind a local server, and it wants to own the server. Vite builds static files that the existing aiohttp static mount serves. |
| **Redux / Redux Toolkit** | zustand covers this app's state in a fraction of the code. |
| **Axios** | `fetch` with a thin wrapper. One less dependency. |
| **Chart.js + Recharts together** | Pick one charting library. Recharts, because it is React-native and already in the ecosystem the shadcn docs assume. |
| **Framer Motion** | CSS transitions cover every animation in `05_UI_SPEC.md`. Revisit only if a spec'd interaction genuinely needs spring physics. |
| **A component library on top of shadcn** (MUI, Mantine, Ant) | shadcn *is* the component layer. Two is one too many. |
| **moment.js / date-fns** | `Intl.DateTimeFormat` and `Intl.RelativeTimeFormat` are native and format better. |
| **lodash** | `Array.prototype` methods. |

---

## Versions and lockfiles

- Commit `dashboard/package-lock.json`. A demo that builds on one laptop and not another is a
  demo that fails on demo day.
- Pin the Node version in `dashboard/.nvmrc` (Node 20 LTS).
- Do not add a dependency mid-build without noting it in `dashboard/README.md` with a one-line
  justification. If you cannot write the justification, do not add it.

## Build and run

```bash
# backend (repo root, Git Bash on Windows per CLAUDE.md)
python -m realtime.server --ws-port 8000 --mode webrtc --ckpt outputs/models/head_v3.pt

# frontend dev, separate terminal
cd dashboard && npm install && npm run dev        # :5173, proxies /api and /ws to :8000

# production build — served by the aiohttp static mount at /dashboard
cd dashboard && npm run build                     # -> dashboard/dist/
```

`dashboard/dist/` is gitignored. The server mounts it if present and returns a clear
"dashboard not built — run `npm run build` in dashboard/" page if it is not. It must never
500 or serve a blank white page because of a missing build.
