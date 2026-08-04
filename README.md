# ComfyUI-MiniMaxH3-Preview

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

Frames come from the latent2rgb approximation (`MiniMaxH3Video.latent_rgb_factors`), which is free —
a single matmul, no VAE, no VRAM. The catch is that a latent2rgb preview is inherently at **latent
resolution**:

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

Animated previews are sent as H.264 via NVENC when PyAV exposes it, and as animated WebP otherwise.
NVENC refuses inputs below 145 × 49, so at `max_resolution=0` the WebP path is the normal one.

Attaching the node to a non-H3 model is safe: it logs one warning and passes sampling through
completely untouched.

## The `vae` option, and its real cost

Read this before turning it on.

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

Latent previews keep flowing normally while a VAE decode runs. The node's header has `latent` /
`vae` tabs; each keeps its own last frame, and the panel switches to `vae` by itself the first time
a full-resolution frame lands, then leaves the choice to you.

## Requirements

ComfyUI with MiniMax H3 support (`comfy/ldm/minimax/`), Pillow, and PyAV. Tested against
`comfyui-frontend-package==1.47.12` on Windows / RTX 3090.
