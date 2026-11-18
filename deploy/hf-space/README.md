---
title: SONIX Detection Backend
emoji: 🛡️
colorFrom: red
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

# SONIX — live voice-clone detection backend

Serves the scoring API and the tester site for
[SONIX](https://github.com/Anubhava-Jangwan/SONIX) (SIH26104).

| Route | What |
|---|---|
| `/` | the full `website/` page, same-origin (server.py:875) |
| `/ws` | live mic detection stream |
| `/api/status` | mode, scoring availability |
| `/api/models` | trained heads found on disk |
| `/api/score-file` | upload a clip, returns a `call_id` |
| `/api/telemetry` | poll scores for a `call_id` |
| `/api/approve` | operator approval action |
| `/mic` | standalone mic capture page |

## Keeping it in sync

The image clones `main` at **build time**. After pushing to GitHub, hit
*Factory rebuild* here to pick the change up.

## Variables

`SONIX_ALLOWED_ORIGINS` — comma-separated origins allowed to call the API from
a browser. Set in Settings → Variables; the Dockerfile default covers the
Vercel deployment only.

## Note on speed

Free Spaces are CPU-only. The frozen wav2vec2 XLS-R front-end is ~300M params,
so scoring runs well behind real time, and the first request pays the model
download and load. Fine for a demo, not a latency benchmark.
