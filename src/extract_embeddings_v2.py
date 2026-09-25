#!/usr/bin/env python3
"""
extract_embeddings_v2.py  --  SONIX / SIH26104

Same extractor, different pooling. This is the S4 change.

WHY
    probe_shortcut.py / probe_transfer.py (2026-09-19) measured the current
    representation -- frozen XLS-R, last_hidden_state, mean(dim=1):

        language, 12-way, within IndicSynth only ......  99.33%  (chance 8.33%)
        corpus, bonafide only, recording-split ........ 100.00%  (chance 82.54%)
        spoof probe trained EN, tested Indic .......... AUC 0.5196  (chance 0.50)
        ... after projecting out 20 language directions  AUC 0.5011

    In-domain AUC is 0.9992 and it transfers at chance. Removing language
    directions does not recover it, so language is not MASKING a transferable
    signal -- there is no transferable signal in the vector to unmask.

    Mechanism: XLS-R runs at ~50 fps, so a 4 s window is ~200 frames averaged
    into one vector. Stationary properties (language, speaker, channel, SNR)
    survive averaging intact. Vocoder artifacts are transient -- phase
    discontinuities at frame boundaries, local spectral over-smoothing,
    micro-jitter -- and average into noise. Compounding it, last_hidden_state
    is the most content-abstracted layer in the stack.

WHAT CHANGES
    * output_hidden_states=True -- every layer, not just the last
    * mean AND std pooling per layer (std is where frame-to-frame artifact
      variance lives; the old code threw it away entirely)
    * a meta.json sidecar so any single layer can be sliced back out later

WHAT DOES NOT CHANGE
    Manifest building, resume, atomic shard writes, the ffmpeg-first audio
    loader, the CLI, the shard sidecar format. This file imports all of that
    from src/extract_embeddings.py and overrides two functions. Do not fork
    that file -- it runs unattended on three machines.

OUTPUT
    <out>/<split>/shard_00000.npy         float16, (n, L * P)
    <out>/<split>/shard_00000.labels.npy  int8,    (n,)
    <out>/<split>/shard_00000.files.txt   filenames, row-aligned
    <out>/<split>/meta.json               {layers, pool, dim_per_layer, model}

    L = number of layers kept, P = 2048 for meanstd (1024 mean + 1024 std)
    or 1024 for mean. Slice layer i as  X[:, i*P:(i+1)*P].

SIZE WARNING
    All 25 layers with meanstd is 51,200 dims = ~102 KB per clip in float16.
    That is fine for a few thousand clips and NOT fine for 126k. Workflow:

      1. sweep  -- small sample, all layers, find which layer carries artifacts
      2. commit -- full extraction with --layers <the winner>

USAGE
    # 0) smoke test: confirm the shape before any long run
    python src/extract_embeddings_v2.py --split train --limit 50 --batch 4 \
        --layers all --out outputs/emb_v2_sweep

    # 1) the sweep sample (English, both labels)
    python src/extract_embeddings_v2.py --split train --limit 2000 --batch 8 \
        --layers all --out outputs/emb_v2_sweep

    # 2) once a layer wins, the real extraction
    python src/extract_embeddings_v2.py --split train --batch 8 \
        --layers 9 --pool meanstd --out outputs/embeddings_v2

    # any folder of audio (no protocol needed); labels joined later by filename
    python src/extract_embeddings_v2.py --split train --audio-dir data/indic_spoof \
        --layers all --out outputs/emb_v2_sweep_indicfake
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import extract_embeddings as ee  # noqa: E402  the battle-tested original

TARGET_SR = ee.TARGET_SR


# ===========================================================================
# Front-end -- identical load, but we ask for every hidden state
# ===========================================================================
def load_frontend_v2(model_name: str, device: str):
    import torch
    from transformers import AutoFeatureExtractor, AutoModel

    print(f"Loading front-end {model_name} on {device} (all hidden states) ...",
          flush=True)
    fe = AutoFeatureExtractor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.eval().to(device)
    use_half = device.startswith("cuda")
    if use_half:
        model.half()
    for p in model.parameters():
        p.requires_grad_(False)
    return {"model": model, "fe": fe, "device": device,
            "dtype": torch.float16 if use_half else torch.float32,
            "torch": torch}


def make_embed_batch(layers, pool, final_ln=False):
    """Build the embed_batch callable run() will use.

    layers   : list[int] | None  which hidden_states indices to keep (None = all)
    pool     : 'mean' | 'meanstd'
    final_ln : also emit last_hidden_state as a trailing block

    WHY final_ln EXISTS
        XLS-R sets do_stable_layer_norm=True, so the encoder applies one more
        layer norm AFTER the last transformer block:

            last_hidden_state == encoder.layer_norm(hidden_states[24])

        hidden_states[24] is therefore the PRE-layernorm state, and it is not
        what v1 / realtime/frontend.py serves. Measured on 50 train clips:
        |x|.mean() is 12.5 for hidden_states[24] against 0.114 for
        last_hidden_state, cosine 0.895 after mean-pooling. The difference does
        not wash out downstream either -- LayerNorm normalises each FRAME
        independently, before pooling, so standardising the pooled vector later
        is a different operation. Without this block the sweep has no baseline
        to compare a candidate layer against.
    """

    def embed_batch_v2(frontend, wavs):
        torch = frontend["torch"]
        fe, model = frontend["fe"], frontend["model"]
        device, dtype = frontend["device"], frontend["dtype"]

        inputs = fe(list(wavs), sampling_rate=TARGET_SR,
                    return_tensors="pt", padding=True)
        input_values = inputs["input_values"].to(device=device, dtype=dtype)
        kw = {}
        if "attention_mask" in inputs:
            kw["attention_mask"] = inputs["attention_mask"].to(device)

        with torch.no_grad():
            out_m = model(input_values, **kw)
            hs = out_m.hidden_states                       # tuple of (b, T, 1024)

        idx = range(len(hs)) if layers is None else layers
        blocks = [hs[i] for i in idx]
        if final_ln:
            blocks.append(out_m.last_hidden_state)         # == v1 representation

        parts = []
        for h in blocks:
            h = h.float()                                  # (b, T, 1024)
            parts.append(h.mean(dim=1))
            if pool == "meanstd":
                # unbiased=False: T is ~200, and we want the population stat,
                # not an estimator that changes with clip length
                parts.append(h.std(dim=1, unbiased=False))
        out = torch.cat(parts, dim=1)                      # (b, L*P)
        return out.cpu().numpy().astype(np.float16)

    return embed_batch_v2


# ===========================================================================
# CLI -- the original's parser plus two flags
# ===========================================================================
def build_argparser():
    ap = ee.build_argparser()
    ap.description = "Cache frozen SSL embeddings with layer choice + stats pooling."
    ap.add_argument("--layers", default="all",
                    help="'all', or comma-separated hidden_state indices, e.g. "
                         "'9' or '4,9,14,19,24'. Index 0 is the conv feature "
                         "projection; 1..24 are transformer layers for XLS-R 300M. "
                         "The extra token 'ln' adds last_hidden_state -- the "
                         "post-final-layernorm top, which is what v1 and "
                         "realtime/frontend.py use and is NOT the same tensor as "
                         "hidden_states[24]. Sweep with 'all,ln' so the table has "
                         "its own baseline row; 'ln' alone with --pool mean "
                         "reproduces v1 exactly.")
    ap.add_argument("--pool", default="meanstd", choices=["mean", "meanstd"],
                    help="mean (1024/layer, matches v1) or meanstd (2048/layer)")
    return ap


def parse_layers(spec, n_hidden):
    """'all' | comma list of ints | the token 'ln' | any mix, e.g. 'all,ln'.

    Returns (layer_indices_or_None, final_ln, n_blocks). 'ln' is not an index --
    it selects last_hidden_state, which sits after the encoder's final layer
    norm and has no hidden_states slot of its own.
    """
    toks = [t.strip().lower() for t in spec.split(",") if t.strip() != ""]
    if not toks:
        sys.exit("FATAL: --layers is empty")

    final_ln = "ln" in toks
    toks = [t for t in toks if t != "ln"]

    if "all" in toks:
        if len(toks) > 1:
            sys.exit(f"FATAL: --layers 'all' cannot be combined with explicit "
                     f"indices; got {spec!r}")
        return None, final_ln, n_hidden + int(final_ln)

    if not toks:                       # 'ln' on its own
        return [], final_ln, 1

    try:
        idx = [int(t) for t in toks]
    except ValueError:
        sys.exit(f"FATAL: --layers takes integers, 'all', and 'ln'; got {spec!r}")
    bad = [i for i in idx if not 0 <= i < n_hidden]
    if bad:
        sys.exit(f"FATAL: layer index out of range {bad}; model has "
                 f"{n_hidden} hidden states (0..{n_hidden - 1})")
    return idx, final_ln, len(idx) + int(final_ln)


def main():
    args = build_argparser().parse_args()

    # how many hidden states does this model expose? read it off the config
    # rather than assuming 25, so --model microsoft/wavlm-large also works
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model)
    n_hidden = cfg.num_hidden_layers + 1

    layers, final_ln, n_layers = parse_layers(args.layers, n_hidden)
    per_layer = 2048 if args.pool == "meanstd" else 1024
    out_dim = n_layers * per_layer

    # block labels, in the order embed_batch emits them -- ints for raw hidden
    # states, the string "ln" for the post-final-layernorm top
    block_labels = (list(range(n_hidden)) if layers is None else list(layers))
    if final_ln:
        block_labels.append("ln")

    print(f"model={args.model}  hidden_states={n_hidden}  "
          f"blocks={block_labels}  pool={args.pool}")
    print(f"output dim = {n_layers} x {per_layer} = {out_dim}  "
          f"({out_dim * 2 / 1024:.1f} KB per clip, float16)")
    if out_dim > 8192 and not args.limit:
        print("  NOTE: wide vectors and no --limit. This is a sweep configuration; "
              "for a full split pick one or two layers first.")

    # run() builds an empty (0, EMB_DIM) array if a whole shard fails, so the
    # module constant has to match or the sidecars drift out of shape
    ee.EMB_DIM = out_dim

    rc = ee.run(args,
                _load_frontend=load_frontend_v2,
                _embed_batch=make_embed_batch(layers, args.pool, final_ln))

    out_dir = Path(args.out) / args.split
    meta = {
        "model": args.model,
        "layers": block_labels,
        "pool": args.pool,
        "dim_per_layer": per_layer,
        "total_dim": out_dim,
        "window_samples": ee.TARGET_LEN,
        "pad": args.pad,
        "sample_rate": TARGET_SR,
        "note": "slice block i as X[:, i*dim_per_layer:(i+1)*dim_per_layer], "
                "where i is the POSITION in 'layers', not the layer number; for "
                "meanstd the first half of each block is mean, second is std. "
                "The block labelled \"ln\" is last_hidden_state (post final "
                "layer norm) -- the v1/realtime front-end representation, and a "
                "different tensor from hidden_states[24].",
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[{args.split}] wrote {out_dir / 'meta.json'}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
