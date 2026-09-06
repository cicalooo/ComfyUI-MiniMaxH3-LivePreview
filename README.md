# ComfyUI-MiniMaxH3-LivePreview

A single MODEL-patch node — **MiniMax H3 Live Preview** — that shows the video building up on its own node body while H3 samples. Drop it between the H3 model loader and the sampler; nothing else to wire.

<p align="center">
  <img src="https://github.com/user-attachments/assets/204251b5-6327-4ced-a060-528aff4f40a6" alt="MiniMax H3 Live Preview on the node body" width="720">
</p>

```
Load Diffusion Model ──▶ MiniMax H3 Live Preview ──▶ (Sol-Attn / Sigma Shift / …) ──▶ KSampler
                                  │
                                  └── preview panel updates every step
```

## Install

1. Clone into `ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-LivePreview`
2. Restart ComfyUI
3. Download [`taeh3.safetensors`](https://huggingface.co/Kijai/MiniMax-H3-TAE) into `ComfyUI/models/vae_approx/`
4. Drop **MiniMax H3 Live Preview** between the H3 model loader and the sampler, then select `taeh3.safetensors`

Requires ComfyUI with MiniMax H3 support (`comfy/ldm/minimax/`), Pillow, and PyAV. Tested against `comfyui-frontend-package==1.51.9` on Windows / RTX 3090.

## What you see

Pick a source with `tae_decoder`:

- **`none`** (default) — latent2rgb (`MiniMaxH3Video.latent_rgb_factors`): a single matmul, no VAE, no VRAM, latent resolution.
- **`taeh3.safetensors`** — tiny 2D VAE (~10 MB), full resolution, cheap enough to run every step. **Recommended.**

Download `vae_approx/taeh3.safetensors` into **`ComfyUI/models/vae_approx/`**. Do **not** load it with `VAELoader` — core ComfyUI does not recognise this checkpoint.

Being 2D, it decodes each latent frame independently (the 4× temporal compression is not undone). Measured on a 3090 at 1344 × 768, fp16: **8 frames in 0.25 s, ~640 MB peak VRAM**. If a decode raises (OOM being the plausible one) the node logs once and falls back to latent2rgb for the rest of that run.

A latent2rgb preview is inherently at **latent resolution** (`1344 × 768` → `84 × 48`). `max_resolution` upscales that for display (`nearest-exact` by default). `preview_frames` samples evenly across the clip, so you see the whole video, not just the first frame.

## Options

| input | default | what it does |
|---|---|---|
| `preview_frames` | 8 | latent frames per preview. `1` = a still JPEG; `>1` = animated |
| `preview_fps` | 24 | content-timeline rate. H3 is authored at 24 fps, so `24` = realtime, `12` = half-speed |
| `max_resolution` | 512 | longest side of the transmitted preview; `0` = native latent size |
| `every_n_steps` | 1 | throttle. The final step always emits |
| `upscale_method` | `nearest-exact` | `nearest-exact` / `lanczos` / `bilinear` |
| `jpeg_quality` | 80 | transport quality |
| `suppress_default_preview` | true | stop the sampler node rendering its own preview too |
| `tae_decoder` | `none` | tiny VAE from `models/vae_approx` — `taeh3.safetensors` replaces latent2rgb |

`every_n_steps` throttles image work only; the sampler progress bar, step timing, and ETA continue to update on every callback.

Animated previews are sent as H.264 via NVENC when PyAV exposes it, and as animated WebP otherwise. Hover or focus the panel for play/pause, the scrubber, and elapsed/total time. The playhead is retained when a newer preview arrives.

Attaching the node to a non-H3 model is safe: it logs one warning and passes sampling through untouched.

<details>
<summary>Why this exists instead of KJNodes' Model Preview Override</summary>

KJNodes' generic node attaches to H3 without complaint and then emits **nothing**. H3 is unusual in two ways that break every generic preview path:

- Its latents are a `NestedTensor(video [B,24,T,H/16,W/16], audio [B,32,2,Ta])`. `NestedTensor` has no `detach`/`movedim`, reports `ndim` as the max over its streams, and forwards `__getitem__` to *both* streams — so `Latent2RGBPreviewer`'s `x0[0, :, 0]` slices the audio tensor as if it were video, and the vectorised paths raise `AttributeError` into a swallowing `except`.
- Inside an `OUTER_SAMPLE` wrapper the callback receives the **flat packed** `[B, 1, N]` tensor, not the nested view — the nesting callback is installed further out, in `comfy/samplers.py`. The correct split is `comfy.utils.unpack_latents(x0, latent_shapes)`.

This node handles both, keeps the video stream and discards the audio stream.

</details>

<details>
<summary>Why taeh3 cannot be loaded with VAELoader</summary>

It fails with `WARNING: No VAE weights detected, VAE not initalized.` followed by `RuntimeError: ERROR: VAE is invalid: None`. Core ComfyUI recognises TAE checkpoints only by `taesd_decoder.1.weight` (2D TAESD) or `decoder.22.bias` (TAEHV); taeh3 is a *bare* decoder keyed by plain module indices. And even with the prefix fixed, `comfy.taesd.taesd.Decoder` hardcodes a 64-wide stack with three upsample stages, while taeh3 is 24 latent channels, 96 wide for its first half, and four upsample stages (H3's VAE is 16× spatial, not 8×). `tiny_vae.py` in this repo rebuilds the architecture from the checkpoint's own keys instead.

</details>

<details>
<summary>Full H3 video VAE (niche, leave off)</summary>

**Supported full-resolution previews use `taeh3`.** The real video-VAE option is mainly for very-high-VRAM cards. Most users should leave `vae_decode_every_n_steps` at `0`.

| input | default | what it does |
|---|---|---|
| `vae` | — | optional H3 **video** VAE, for true full-resolution frames |
| `vae_decode_every_n_steps` | 0 | `0` = off |
| `vae_decode_frames` | 1 | full-res frames per VAE preview |
| `vae_decode_device` | `auto` | `auto` / `gpu` / `cpu` |

The H3 video VAE is **~5.2 GB**. An H3 transformer is **~21 GB**. On a 24 GB card they do not co-fit:

- **`gpu`** — ComfyUI evicts the transformer to make room for the VAE and reloads it from disk afterwards, around *every* preview. Runs **synchronously on the sampler thread**. Expect sampling to visibly stall.
- **`cpu`** — pins `first_stage_model` to CPU/fp32, decodes off-thread. Sampling is not stalled, but a frame takes minutes. If a decode is still running when the next is due, it is dropped.
- **`auto`** — compares live free VRAM against the VAE's weights plus decode activations. On a 24 GB card running H3 it will pick `cpu`.

Cheap previews keep flowing while a VAE decode runs. The node's header has `latent` / `tae` / `vae` tabs; the panel switches to the truer stream the first time one lands, then leaves the choice to you.

</details>

<details>
<summary>Verification</summary>

Offline hardening checks (no live server required), from the ComfyUI root:

```bat
set PYTHONPATH=%CD%;%CD%\custom_nodes
python custom_nodes\ComfyUI-MiniMaxH3-LivePreview\scripts\offline_verify.py
```

Use the same interpreter that launches ComfyUI; on the bundled Windows checkout this is usually `C:\comfycli\.venv\Scripts\python.exe` rather than the system `python`.

Live TAE/latent websocket acceptance (ComfyUI must be running):

```bat
python custom_nodes\ComfyUI-MiniMaxH3-LivePreview\scripts\live_accept_preview.py
```

Accepted scope is **TAE + latent2rgb**. Full video-VAE live preview is a niche optional path and not part of normal acceptance.

</details>

<details>
<summary>Audit notes / remaining host limitations</summary>

Hardening details are in [`docs/MINIMAX_H3_LIVE_PREVIEW_AUDIT.md`](docs/MINIMAX_H3_LIVE_PREVIEW_AUDIT.md). These items are intentionally **not** changed here because they belong to ComfyUI core or the separate H3 loader/VAE implementation:

- **Core `NestedTensor` callback contract:** the sampler still packs AV latents and reconstructs the nested callback view in `comfy/samplers.py`. If that contract changes upstream, this node's `unpack_latents` integration must be re-checked.
- **Core `PromptServer` routing:** events are sent to the current `PromptServer.client_id`. This plugin does not add a server route or broadcast previews.
- **Full H3 video-VAE memory behavior:** the ~5.2 GiB VAE and transformer eviction/reload cost are properties of the connected VAE and ComfyUI model-management. This path is intentionally niche; `taeh3` is the supported full-resolution preview.
- **PyAV/NVENC availability:** codec support is supplied by the installed PyAV wheel and NVIDIA driver. The node falls back to animated WebP.
- **Frontend lifecycle/version compatibility:** the node uses the current `window.comfyAPI` extension API.

</details>
