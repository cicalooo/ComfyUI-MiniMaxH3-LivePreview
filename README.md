# ComfyUI-MiniMaxH3-LivePreview

One MODEL-patch node — **MiniMax H3 Live Preview** — that shows the video building on its own panel while H3 samples. Drop it between the H3 loader and the sampler; nothing else to wire.

<p align="center">
  <img src="https://github.com/user-attachments/assets/204251b5-6327-4ced-a060-528aff4f40a6" alt="MiniMax H3 Live Preview on the node body" width="720">
</p>

```
Load Diffusion Model ──▶ MiniMax H3 Live Preview ──▶ (Sol-Attn / Sigma Shift / …) ──▶ KSampler
                                   └── panel updates every step
```

## Install

1. Clone into `ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-LivePreview`
2. Restart ComfyUI
3. Download [`taeh3.safetensors`](https://huggingface.co/Kijai/MiniMax-H3-TAE) into `ComfyUI/models/vae_approx/`
4. Insert the node between loader and sampler, select `taeh3.safetensors`

Requires ComfyUI with H3 support (`comfy/ldm/minimax/`) + Pillow. PyAV is optional: with NVENC you get H.264, without it animated WebP. Tested on `comfyui-frontend-package==1.51.9`, Windows / RTX 3090.

## Sources

- `none` (default) — latent2rgb, no VRAM, latent resolution (`1344×768` → `84×48`).
- `taeh3.safetensors` — tiny 2D VAE (~10 MB), full resolution, ~0.25 s / 8 frames at 1344×768 fp16 on a 3090, ~640 MB peak. **Recommended.** Do not use `VAELoader` for it — core reports `VAE is invalid: None`.

Failed TAE decodes fall back to latent2rgb (OOM retries next step by default, see `tae_oom_retry`).

## Options

Core widgets stay visible; the rest are collapsed as advanced.

| input | default | notes |
|---|---|---|
| `preview_frames` | 8 | frames per preview; `1` = JPEG still, `>1` = animated |
| `preview_fps` | 24 | timeline rate; `24` = realtime, `12` = half-speed |
| `max_resolution` | 512 | longest side; `0` = native |
| `every_n_steps` | 1 | image throttle; final step always emits |
| `throttle_secs` | 0.0 | min seconds between decodes; progress/ETA still update every step |
| `tae_decoder` | `none` | tiny VAE from `vae_approx`; `taeh3.safetensors` = full-res |
| `tae_device` | `auto` | `auto` (honours `--cpu-vae`) / `gpu` / `cpu` |
| `tae_dtype` | `auto` | `auto` (fp16/bf16) / `float32` / `float16` / `bfloat16` |
| `tae_compile` | false | `torch.compile` the TAE; slower first preview, faster steps |
| `skip_upscale_layers` | 0 | skip N trailing 2x TAE stages (16x→8x→4x→2x→1x) |
| `tae_oom_retry` | true | OOM = latent2rgb for that step, retry next; off = disable TAE for run |
| `suppress_default_preview` | true | hide the sampler node's own preview |
| `upscale_method` | `nearest-exact` | + `lanczos` / `bilinear` |
| `jpeg_quality` | 80 | JPEG stills + WebP fallback (NVENC uses its own rate control) |
| `vae` / `vae_decode_every_n_steps` / `vae_decode_frames` / `vae_decode_device` | — / 0 / 1 / `auto` | full 5.2 GB video VAE; niche, leave off (see below) |
| `verbose` | false | detailed failure traces in log + console |

Hover/focus the panel for play/pause, scrubber, elapsed/total. Playhead survives updates; playback pauses at run end, click to replay. Non-H3 models pass through with one warning.

If ComfyUI-bleh is installed, blacklist H3 (`minimaxh3video`/`minimaxh3av` in `previews.blacklist_formats`) or bleh crashes on this file (`KeyError: 'decoder.7.conv.weight'`); this node warns on conflict.

<details>
<summary>Why not KJNodes' Model Preview Override</summary>

H3 latents are `NestedTensor(video [B,24,T,H/16,W/16], audio [B,32,2,Ta])` with no `detach`/`movedim`, and inside `OUTER_SAMPLE` the callback gets flat packed `[B,1,N]`. Generic paths slice the audio stream as video or raise into a swallowed `except`. This node splits via `comfy.utils.unpack_latents` and keeps video only.
</details>

<details>
<summary>Why VAELoader / bleh can't load taeh3</summary>

Core recognises TAEs only by `taesd_decoder.1.weight` or `decoder.22.bias`; taeh3 is a bare index-keyed decoder (24 ch, 96-wide, 4 upsample stages for 16×). `tiny_vae.py` rebuilds the arch from the keys. Bleh expects madebyollin `taeh3.pth` (`decoder.*` + `TGrow`), hence the blacklist above.
</details>

<details>
<summary>Full video VAE (niche, leave off)</summary>

`vae_decode_every_n_steps=0` = off. The ~5.2 GB VAE + ~21 GB transformer don't co-fit on 24 GB: `gpu` evicts/reloads the transformer around every preview (synchronous stall), `cpu` pins to CPU/fp32 off-thread (minutes/frame, drops when behind), `auto` picks from live free VRAM. `latent` / `tae` / `vae` tabs switch to the truer stream on first VAE frame.
</details>

<details>
<summary>Verification</summary>

From the ComfyUI root (use ComfyUI's interpreter):

```bat
set PYTHONPATH=%CD%;%CD%\custom_nodes
python custom_nodes\ComfyUI-MiniMaxH3-LivePreview\scripts\offline_verify.py
python custom_nodes\ComfyUI-MiniMaxH3-LivePreview\scripts\live_accept_preview.py
```

Live acceptance scope is TAE + latent2rgb. Details: [`docs/MINIMAX_H3_LIVE_PREVIEW_AUDIT.md`](docs/MINIMAX_H3_LIVE_PREVIEW_AUDIT.md).
</details>
