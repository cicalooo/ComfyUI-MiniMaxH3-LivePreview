# ComfyUI-MiniMaxH3-LivePreview

A single MODEL-patch node — **MiniMax H3 Live Preview** — that shows the video building up on its
own node body while H3 samples. Drop it between the H3 model loader and the sampler; nothing else
to wire.

```
Load Diffusion Model ──▶ MiniMax H3 Live Preview ──▶ (Sol-Attn / Sigma Shift / …) ──▶ KSampler
                                  │
                                  └── preview panel updates every step
```

## Why this exists instead of KJNodes' Model Preview Override

KJNodes' generic node attaches to H3 without complaint and then emits **nothing**. H3 is unusual in
two ways that break every generic preview path:

- Its latents are a `NestedTensor(video [B,24,T,H/16,W/16], audio [B,32,2,Ta])`. `NestedTensor` has
  no `detach`/`movedim`, reports `ndim` as the max over its streams, and forwards `__getitem__` to
  *both* streams — so `Latent2RGBPreviewer`'s `x0[0, :, 0]` slices the audio tensor as if it were
  video, and the vectorised paths raise `AttributeError` into a swallowing `except`.
- Inside an `OUTER_SAMPLE` wrapper the callback receives the **flat packed** `[B, 1, N]` tensor, not
  the nested view — the nesting callback is installed further out, in `comfy/samplers.py`. The
  correct split is `comfy.utils.unpack_latents(x0, latent_shapes)`.

This node handles both, keeps the video stream and discards the audio stream.

## What you actually see

Two cheap sources, pick one with `tae_decoder`:

- **`none`** (default) — latent2rgb (`MiniMaxH3Video.latent_rgb_factors`): a single matmul, no VAE,
  no VRAM, latent resolution.
- **`taeh3.safetensors`** — the tiny VAE below: ~10 MB, full resolution, still cheap enough to run
  every step. Recommended.

### taeh3 — the tiny VAE (recommended)

[Kijai/MiniMax-H3-TAE](https://huggingface.co/Kijai/MiniMax-H3-TAE) is a quickly-trained 2D tiny VAE
decoder for H3. Download `vae_approx/taeh3.safetensors` into **`ComfyUI/models/vae_approx/`** and
select it in the node's `tae_decoder` widget.

> **Do not load it with `VAELoader`.** It fails with
> `WARNING: No VAE weights detected, VAE not initalized.` followed by
> `RuntimeError: ERROR: VAE is invalid: None`. Core ComfyUI recognises TAE checkpoints only by
> `taesd_decoder.1.weight` (2D TAESD) or `decoder.22.bias` (TAEHV); taeh3 is a *bare* decoder keyed
> by plain module indices. And even with the prefix fixed, `comfy.taesd.taesd.Decoder` hardcodes a
> 64-wide stack with three upsample stages, while taeh3 is 24 latent channels, 96 wide for its first
> half, and four upsample stages (H3's VAE is 16× spatial, not 8×). `tiny_vae.py` in this repo
> rebuilds the architecture from the checkpoint's own keys instead.

Being 2D, it decodes each latent frame independently — the 4× temporal compression is not undone, so
37 latent frames still give 37 preview frames, at 1344 × 768 each. It runs on the sampler thread
(`vae_device()`, so `--cpu-vae` is honoured) and its weights are released when the run ends.

Measured on a 3090 at 1344 × 768, fp16: **8 frames in 0.25 s, ~640 MB peak VRAM**. Frames are
decoded one at a time and moved to the CPU as they land, so that peak is per-frame — `preview_frames`
buys time, not memory. 640 MB alongside a 21 GB transformer is the one real risk: if a decode raises
(OOM being the plausible one) the node logs once and falls back to latent2rgb for the rest of that
run, rather than taking sampling down with it.

### latent2rgb resolution

A latent2rgb preview is inherently at **latent resolution**:

| generation | preview frame |
|---|---|
| 1344 × 768 | 84 × 48 |
| 672 × 480 | 42 × 30 |

`max_resolution` upscales that for display (`nearest-exact` by default, so the latent blocks stay
crisp rather than turning to mush). It is enough to read composition, motion, scene changes and
obvious failure modes — not detail. For 124 frames there are 37 latent frames, and `preview_frames`
samples evenly across all of them, so the preview is the whole clip, not just the first frame.

## Options

| input | default | what it does |
|---|---|---|
| `preview_frames` | 8 | latent frames per preview. `1` = a still JPEG; `>1` = animated |
| `preview_fps` | 8 | playback rate of the animation |
| `max_resolution` | 512 | longest side of the transmitted preview; `0` = native latent size |
| `every_n_steps` | 1 | throttle. The final step always emits |
| `upscale_method` | `nearest-exact` | `nearest-exact` / `lanczos` / `bilinear` |
| `jpeg_quality` | 80 | transport quality |
| `suppress_default_preview` | true | stop the sampler node rendering its own preview too |
| `vae` | — | optional H3 **video** VAE, for true full-resolution frames |
| `vae_decode_every_n_steps` | 0 | `0` = off. See the warning below |
| `vae_decode_frames` | 1 | full-res frames per VAE preview |
| `vae_decode_device` | `auto` | `auto` / `gpu` / `cpu` |
| `tae_decoder` | `none` | tiny VAE from `models/vae_approx` — `taeh3.safetensors` replaces latent2rgb with real full-resolution frames |

Animated previews are sent as H.264 via NVENC when PyAV exposes it, and as animated WebP otherwise.
NVENC refuses inputs below 145 × 49, so at `max_resolution=0` the WebP path is the normal one.

Attaching the node to a non-H3 model is safe: it logs one warning and passes sampling through
completely untouched.

## The `vae` option, and its real cost

**Supported full-resolution previews use `taeh3`.** The real video-VAE option below is niche: it is
mainly for very-high-VRAM cards, and most users should leave `vae_decode_every_n_steps` at `0`.

Read this before turning it on. With `taeh3` available, this option is mostly obsolete — it exists
for when you want the model's *actual* decoder rather than an approximation of it.

The H3 video VAE is **~5.2 GB**. An H3 transformer is **~21 GB**. On a 24 GB card they do not
co-fit, so:

- **`gpu`** — ComfyUI evicts the transformer to make room for the VAE and reloads it from disk
  afterwards, around *every* preview. This path deliberately runs **synchronously on the sampler
  thread**: `VAE.decode` calls `load_models_gpu`, which mutates global model-management state and
  would otherwise be free to unload the transformer while a forward pass is still running through
  it. Expect sampling to visibly stall. That stall is the design, not a bug.
- **`cpu`** — bypasses `VAE.decode` entirely, pins `first_stage_model` to CPU/fp32 once (fp16 conv3d
  on CPU is unusable), decodes off-thread, and restores the module's original device *and* dtype
  when the run ends. Sampling is not stalled, but a frame takes minutes. If a decode is still
  running when the next is due, it is dropped.
- **`auto`** — compares live free VRAM against the VAE's weights plus decode activations and logs
  the numbers behind its choice. On a 24 GB card running H3 it will pick `cpu`.

Each frame is decoded as an independent single-latent-frame slice, which hits
`MiniMaxH3VideoVAE`'s `_adaptive_decode` branch and keeps activations to a few hundred MB — the
weights are the entire problem.

Cheap previews keep flowing normally while a VAE decode runs. The node's header has `latent` / `tae`
/ `vae` tabs; each keeps its own last frame, and the panel switches to the truer stream by itself the
first time one lands, then leaves the choice to you.

## Install

1. Clone into `ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-LivePreview`
2. Restart ComfyUI
3. Download [`taeh3.safetensors`](https://huggingface.co/Kijai/MiniMax-H3-TAE) into `ComfyUI/models/vae_approx/`
4. Drop **MiniMax H3 Live Preview** between the H3 model loader and the sampler, then select `taeh3.safetensors`

## Requirements

ComfyUI with MiniMax H3 support (`comfy/ldm/minimax/`), Pillow, and PyAV. Tested against
`comfyui-frontend-package==1.51.9` on Windows / RTX 3090.

## Verification

Offline hardening checks (no live server required), from the ComfyUI root:

```bat
set PYTHONPATH=%CD%;%CD%\custom_nodes
python custom_nodes\ComfyUI-MiniMaxH3-LivePreview\scripts\offline_verify.py
```

Live TAE/latent websocket acceptance (ComfyUI must be running):

```bat
python custom_nodes\ComfyUI-MiniMaxH3-LivePreview\scripts\live_accept_preview.py
```

Accepted scope is **TAE + latent2rgb**. Full video-VAE live preview is a niche optional path and
not part of normal acceptance.

## Audit notes / remaining host limitations

Hardening details are in [`docs/MINIMAX_H3_LIVE_PREVIEW_AUDIT.md`](docs/MINIMAX_H3_LIVE_PREVIEW_AUDIT.md).
The following items are intentionally **not** changed here because they belong to ComfyUI core or
the separate H3 loader/VAE implementation:

- **Core `NestedTensor` callback contract:** the sampler still packs AV latents and reconstructs
  the nested callback view in `comfy/samplers.py`. If that contract changes upstream, this node's
  `unpack_latents` integration must be re-checked against the new sampler API.
- **Core `PromptServer` routing:** events are sent to the current `PromptServer.client_id`, as
  ComfyUI's progress system does. Multi-client/server routing changes need a corresponding core
  event-targeting update; this plugin does not add a server route or broadcast previews.
- **Full H3 video-VAE memory behavior:** the ~5.2 GiB VAE and transformer eviction/reload cost are
  properties of the connected VAE and ComfyUI model-management layer. The node can choose CPU or
  GPU and serialize shared-VAE access, but cannot make the GPU decode co-resident or interrupt an
  in-flight CPU convolution safely. This path is intentionally niche; `taeh3` is the supported
  full-resolution preview.
- **PyAV/NVENC availability:** codec support is supplied by the installed PyAV wheel and NVIDIA
  driver. The node falls back to animated WebP, but cannot install or repair those external
  dependencies.
- **Frontend lifecycle/version compatibility:** the node uses the current `window.comfyAPI`
  extension API. If an older frontend omits that API or changes qualified execution IDs, the
  frontend must be adapted to that host version.
