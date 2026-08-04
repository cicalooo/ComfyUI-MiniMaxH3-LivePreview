"""MiniMax H3 live inference preview.

A MODEL patch that renders an animated multi-frame preview on its own node body
while H3 samples.

H3 is the reason this is not just KJNodes' generic Model Preview Override:

  * H3 latents are ``comfy.nested_tensor.NestedTensor(video, audio)``. NestedTensor
    has no ``detach``/``movedim``, reports ``ndim`` as the max over its streams, and
    forwards ``__getitem__`` to *both* streams -- so every generic latent->RGB path
    either throws or indexes the audio tensor as if it were video.
  * Inside an OUTER_SAMPLE wrapper the callback receives the *flat packed* [B, 1, N]
    tensor, not the nested view; the nesting callback is installed outside us in
    comfy/samplers.py. ``comfy.utils.unpack_latents(x0, latent_shapes)`` is the split.
"""

import base64
import io as pyio
import logging
import queue
import threading
import time

import numpy as np
import torch
from PIL import Image

import comfy.latent_formats
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import latent_preview
from comfy_api.latest import io

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

EVENT = "minimax_h3_preview"

_RESAMPLE = {
    "nearest-exact": Image.NEAREST,
    "lanczos": Image.LANCZOS,
    "bilinear": Image.BILINEAR,
}


# --------------------------------------------------------------------------------------
# transport
# --------------------------------------------------------------------------------------

def _probe_nvenc():
    # PyPI PyAV wheels typically lack NVENC; probe once at import.
    try:
        import av  # noqa: F401
        av.Codec("h264_nvenc", "w")
        return True
    except Exception:
        return False


_NVENC_AVAILABLE = _probe_nvenc()

# NVENC H.264 rejects sub-145x49 inputs at avcodec_open2. Latent-resolution frames are
# routinely smaller than that, so the WebP fallback is the common path at max_resolution=0.
_NVENC_MIN_W = 145
_NVENC_MIN_H = 49

_nvenc_warned = False


class _Worker:
    """Single background thread with a bounded drop-on-full queue.

    The sampler must never block on preview work, so a full queue drops the job.
    """

    _STOP = object()

    def __init__(self, name, maxsize, on_close=None):
        self.q = queue.Queue(maxsize=maxsize)
        self.closed = False
        self.on_close = on_close
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, fn, block_timeout=None):
        """Queue `fn`. Returns False if it was dropped.

        Dropping is the point: a full queue means we are behind, and the sampler must
        not wait for us. `block_timeout` is for jobs worth waiting on -- the final step,
        whose frame is what the panel is left showing.
        """
        if self.closed:
            return False
        try:
            if block_timeout is None:
                self.q.put_nowait(fn)
            else:
                self.q.put(fn, timeout=block_timeout)
            return True
        except queue.Full:
            return False

    def _run(self):
        while True:
            item = self.q.get()
            if item is self._STOP:
                # Runs after any in-flight job, on this thread, so cleanup can never
                # race a decode that is still using the weights it wants to move back.
                if self.on_close is not None:
                    try:
                        self.on_close()
                    except Exception:
                        logging.exception("[MiniMaxH3Preview] worker close hook failed")
                return
            try:
                item()
            except Exception:
                logging.exception("[MiniMaxH3Preview] worker error")

    def shutdown(self, drain_timeout=5.0):
        self.closed = True
        try:
            self.q.put(self._STOP, timeout=drain_timeout)
        except queue.Full:
            pass
        # Join is best-effort: a CPU VAE decode can run for minutes and cannot be
        # interrupted. The thread is a daemon and still honours _STOP (and on_close)
        # once it gets there.
        self.thread.join(timeout=drain_timeout)


def _fit(pil, max_res, resample):
    """Scale so the longest side equals max_res, preserving aspect. 0 = leave native.

    Unlike ImageOps.contain this scales *up* too -- latent-resolution frames are ~84x48
    and would otherwise arrive unreadably small.
    """
    if not max_res or max_res <= 0:
        return pil
    w, h = pil.width, pil.height
    if max(w, h) == max_res:
        return pil
    scale = max_res / max(w, h)
    return pil.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample)


def _encode_mp4_nvenc(frames, fps):
    """Fragmented MP4 so the browser can decode mid-download.

    Returns (None, 0, 0) on any failure -- including too-small-for-NVENC -- so the
    caller falls through to WebP.
    """
    global _nvenc_warned
    if not frames:
        return None, 0, 0
    try:
        import av
    except Exception:
        return None, 0, 0

    pil_frames = [f if f.mode == "RGB" else f.convert("RGB") for f in frames]
    # yuv420p requires even dimensions.
    w0, h0 = pil_frames[0].width, pil_frames[0].height
    out_w, out_h = w0 & ~1, h0 & ~1
    if (out_w, out_h) != (w0, h0):
        pil_frames = [pf.resize((out_w, out_h), Image.NEAREST) for pf in pil_frames]
    if out_w < _NVENC_MIN_W or out_h < _NVENC_MIN_H:
        return None, 0, 0

    # Driver/GPU varies in what option combos are accepted; a bare preset always works.
    last_err = None
    for opts in ({"preset": "p1", "rc": "vbr", "cq": "23"}, {"preset": "p1"}):
        buf = pyio.BytesIO()
        try:
            container = av.open(
                buf, mode="w", format="mp4",
                options={"movflags": "frag_keyframe+empty_moov+default_base_moof"},
            )
            stream = container.add_stream("h264_nvenc", rate=int(max(1, fps)))
            stream.width = out_w
            stream.height = out_h
            stream.pix_fmt = "yuv420p"
            stream.options = opts
            for pf in pil_frames:
                for pkt in stream.encode(av.VideoFrame.from_image(pf)):
                    container.mux(pkt)
            for pkt in stream.encode():
                container.mux(pkt)
            container.close()
            return base64.b64encode(buf.getvalue()).decode("ascii"), out_w, out_h
        except Exception as e:
            last_err = e
    if not _nvenc_warned:
        _nvenc_warned = True
        logging.warning(f"[MiniMaxH3Preview] NVENC MP4 encode failed, using WebP: {last_err}")
    return None, 0, 0


def _encode_animated_webp(frames, fps, quality):
    if not frames:
        return None, 0, 0
    pil_frames = [f if f.mode == "RGB" else f.convert("RGB") for f in frames]
    buf = pyio.BytesIO()
    try:
        pil_frames[0].save(
            buf, format="WEBP", save_all=True, append_images=pil_frames[1:],
            duration=max(1, int(round(1000 / max(1, fps)))), loop=0,
            quality=quality, method=4,
        )
    except Exception as e:
        logging.warning(f"[MiniMaxH3Preview] animated WebP encode failed: {e}")
        return None, 0, 0
    return base64.b64encode(buf.getvalue()).decode("ascii"), pil_frames[0].width, pil_frames[0].height


def _encode_jpeg(pil, quality):
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buf = pyio.BytesIO()
    pil.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii"), pil.width, pil.height


def _encode_frames(frames, fps, quality):
    """-> (b64, w, h, mime). Single frame is a JPEG; multiple prefer NVENC MP4."""
    if not frames:
        return None, 0, 0, None
    if len(frames) == 1:
        b64, w, h = _encode_jpeg(frames[0], quality)
        return b64, w, h, "image/jpeg"
    if _NVENC_AVAILABLE:
        b64, w, h = _encode_mp4_nvenc(frames, fps)
        if b64:
            return b64, w, h, "video/mp4"
    b64, w, h = _encode_animated_webp(frames, fps, quality)
    return b64, w, h, "image/webp"


# --------------------------------------------------------------------------------------
# latent -> RGB
# --------------------------------------------------------------------------------------

def _pick_indices(total, count):
    if count <= 0 or count >= total:
        return list(range(total))
    return np.linspace(0, total - 1, count).round().astype(int).tolist()


def _latent_to_pil(video, latent_format, max_frames):
    """[B, 24, T, h, w] -> list of PIL frames at latent resolution (w x h).

    One bulk GPU->CPU copy for the whole stack rather than per-frame non_blocking
    copies, which tear at high resolution.
    """
    if video is None or video.ndim != 5:
        return []
    factors = getattr(latent_format, "latent_rgb_factors", None)
    if factors is None:
        return []
    try:
        reshape = getattr(latent_format, "latent_rgb_factors_reshape", None)
        if reshape is not None:
            video = reshape(video)
        x = video[0]                                    # [C, T, h, w]
        idx = _pick_indices(x.shape[1], max_frames)
        if len(idx) != x.shape[1]:
            x = x[:, idx]
        x = x.movedim(0, -1).float()                    # [T, h, w, C]
        f = torch.tensor(factors, device=x.device, dtype=x.dtype).transpose(0, 1)
        bias = getattr(latent_format, "latent_rgb_factors_bias", None)
        b = torch.tensor(bias, device=x.device, dtype=x.dtype) if bias is not None else None
        # linear() allocates, so the in-place scaling below never touches the sampler's tensor
        rgb = torch.nn.functional.linear(x, f, bias=b)
        rgb = rgb.add_(1.0).mul_(127.5).clamp_(0, 255).to(torch.uint8).cpu().numpy()
        return [Image.fromarray(rgb[i]) for i in range(rgb.shape[0])]
    except Exception as e:
        logging.warning(f"[MiniMaxH3Preview] latent2rgb decode failed: {e}")
        return []


# --------------------------------------------------------------------------------------
# optional true-VAE decode
# --------------------------------------------------------------------------------------

class _VaeDecoder:
    """Full-resolution frames from the real H3 video VAE.

    MiniMaxH3VideoVAE.decode has a single-latent-frame branch (z.shape[2] == 1 ->
    _adaptive_decode) yielding exactly one full-res frame, so activations stay around
    a few hundred MB. The 5.2 GB of *weights* is the whole problem: alongside a ~21 GB
    H3 transformer on a 24 GB card they do not co-fit, and the GPU path makes ComfyUI
    evict and reload the transformer around every preview.

    Threading: the gpu path goes through VAE.decode -> load_models_gpu, which mutates
    global model-management state and could unload the transformer while the sampler is
    mid-forward-pass. It therefore runs *synchronously on the sampler thread*. The cpu
    path bypasses VAE.decode entirely, touches no shared state, and runs off-thread.
    """

    def __init__(self, vae, mode, frames):
        self.vae = vae
        self.mode = mode
        self.frames = max(1, frames)
        self.device = None          # resolved on first decode, once the DiT is resident
        self._restore = None        # (module, device, dtype) for the CPU path

    def _weight_bytes(self):
        m = self.vae.first_stage_model
        total = 0
        for t in list(m.parameters()) + list(m.buffers()):
            total += t.numel() * t.element_size()
        return total

    def _resolve_device(self, load_device, slice_shape):
        if self.mode in ("gpu", "cpu"):
            self.device = self.mode
            logging.info(f"[MiniMaxH3Preview] VAE preview device: {self.device} (forced)")
            return
        try:
            free = comfy.model_management.get_free_memory(load_device)
            weights = self._weight_bytes()
            act = self.vae.memory_used_decode(slice_shape, self.vae.vae_dtype)
            needed = weights + act * 1.2
            self.device = "gpu" if free > needed else "cpu"
            logging.info(
                f"[MiniMaxH3Preview] VAE preview device: {self.device} (auto) -- free "
                f"{free / 2**30:.2f} GiB, weights {weights / 2**30:.2f} GiB, activations "
                f"{act / 2**30:.2f} GiB, needed {needed / 2**30:.2f} GiB"
            )
        except Exception as e:
            self.device = "cpu"
            logging.warning(f"[MiniMaxH3Preview] VAE device probe failed ({e}); using cpu")

    def _pin_cpu(self):
        # Bypass VAE.decode and drive first_stage_model directly, the same move KJNodes
        # makes to pin TAEHV. fp16 conv3d on CPU is unusable, so cast to fp32 and restore
        # both device *and* dtype afterwards -- leaving it fp32 would double its VRAM
        # footprint when ComfyUI later loads it for the real decode.
        if self._restore is not None:
            return
        m = self.vae.first_stage_model
        p = next(m.parameters(), None)
        if p is None:
            raise RuntimeError("VAE has no parameters")
        self._restore = (m, p.device, p.dtype)
        m.to(device="cpu", dtype=torch.float32)

    def resolve(self, video):
        """Pick gpu vs cpu. Call on the sampler thread -- it reads live free VRAM,
        which is only meaningful once the transformer is resident."""
        if self.device is None:
            z_shape = (1, video.shape[1], 1, video.shape[3], video.shape[4])
            self._resolve_device(video.device, z_shape)
        return self.device

    def decode(self, video):
        """[B, 24, T, h, w] -> list of full-resolution PIL frames. resolve() first."""
        t_total = video.shape[2]
        idx = _pick_indices(t_total, self.frames)
        out = []
        for i in idx:
            z = video[:1, :, i:i + 1]
            if self.device == "gpu":
                px = self.vae.decode(z)                             # [B, T, H, W, C] in 0..1
            else:
                self._pin_cpu()
                with torch.no_grad():
                    px = self.vae.first_stage_model.decode(z.to(device="cpu", dtype=torch.float32))
                # first_stage_model.decode returns [B, C, T, H, W] in -1..1; mirror the
                # default VAE.process_output plus the movedim VAE.decode applies.
                px = px.movedim(1, -1).add_(1.0).div_(2.0).clamp_(0.0, 1.0)
            if px.ndim == 5:
                px = px[0]
            if px.ndim != 4:
                continue
            u8 = (px.float() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
            out.extend(Image.fromarray(u8[j]) for j in range(u8.shape[0]))
        return out

    def restore(self):
        if self._restore is None:
            return
        m, device, dtype = self._restore
        self._restore = None
        try:
            m.to(device=device, dtype=dtype)
        except Exception as e:
            logging.warning(f"[MiniMaxH3Preview] could not restore VAE to {device}/{dtype}: {e}")


# --------------------------------------------------------------------------------------
# the wrapper
# --------------------------------------------------------------------------------------

def _suppressed_preview_image(self_, preview_format, x0):
    return None


class _H3PreviewWrapper:
    def __init__(self, node_id, preview_frames, preview_fps, max_resolution, every_n_steps,
                 upscale_method, jpeg_quality, suppress_default,
                 vae=None, vae_every_n_steps=0, vae_frames=1, vae_device="auto"):
        self.node_id = str(node_id) if node_id is not None else None
        self.preview_frames = max(1, preview_frames)
        self.preview_fps = max(1, preview_fps)
        self.max_resolution = max_resolution
        self.every_n_steps = max(1, every_n_steps)
        self.resample = _RESAMPLE.get(upscale_method, Image.NEAREST)
        self.jpeg_quality = jpeg_quality
        self.suppress_default = suppress_default
        self.vae = vae
        self.vae_every_n_steps = max(0, vae_every_n_steps)
        self.vae_frames = max(1, vae_frames)
        self.vae_device = vae_device

    # -- plumbing ----------------------------------------------------------------------

    def _send(self, payload):
        if self.node_id is None or PromptServer is None:
            return
        payload["node_id"] = self.node_id
        try:
            PromptServer.instance.send_sync(EVENT, payload, PromptServer.instance.client_id)
        except Exception as e:
            logging.warning(f"[MiniMaxH3Preview] send failed: {e}")

    def _frames_to_payload(self, frames, source):
        frames = [_fit(f, self.max_resolution, self.resample) for f in frames]
        b64, w, h, mime = _encode_frames(frames, self.preview_fps, self.jpeg_quality)
        if not b64:
            return None
        return {
            "image": b64, "mime": mime, "w": w, "h": h, "source": source,
            "fps": self.preview_fps if mime in ("video/mp4", "image/webp") else None,
        }

    def _is_h3(self, model_patcher, latent_shapes):
        lf = getattr(model_patcher.model, "latent_format", None)
        if not isinstance(lf, comfy.latent_formats.MiniMaxH3Video):
            return False
        # H3 always samples the packed (video, audio) pair
        return latent_shapes is not None and len(latent_shapes) == 2

    # -- entry point -------------------------------------------------------------------

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask,
                 callback, disable_pbar, seed, **kwargs):
        latent_shapes = kwargs.get("latent_shapes")
        model_patcher = executor.class_obj.model_patcher

        if not self._is_h3(model_patcher, latent_shapes):
            logging.warning(
                "[MiniMaxH3Preview] attached to a non-MiniMax-H3 model "
                f"({type(getattr(model_patcher.model, 'latent_format', None)).__name__}); "
                "passing through without previews."
            )
            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            callback, disable_pbar, seed, **kwargs)

        latent_format = model_patcher.model.latent_format
        original_callback = callback
        sigmas_list = sigmas.detach().cpu().tolist() if sigmas is not None else []
        total_steps_init = max(0, len(sigmas_list) - 1)

        encoder = _Worker("mmh3_preview_encode", maxsize=2)
        vae_worker = None
        vae_decoder = None
        if self.vae is not None and self.vae_every_n_steps > 0:
            vae_decoder = _VaeDecoder(self.vae, self.vae_device, self.vae_frames)
            # on_close runs after any in-flight decode, so restoring the VAE's device and
            # dtype can never race one.
            vae_worker = _Worker("mmh3_preview_vae", maxsize=1, on_close=vae_decoder.restore)

        # Boundary-0: seed the panel with the pure-noise latent and the sigma schedule.
        init = {"step": 0, "total": total_steps_init,
                "sigma": sigmas_list[0] if sigmas_list else None, "sigmas": sigmas_list}
        try:
            if sigmas is not None and len(sigmas) > 0:
                s0 = sigmas[0].to(noise.device)
                video0 = comfy.utils.unpack_latents(noise * s0, latent_shapes)[0]
                p = self._frames_to_payload(
                    _latent_to_pil(video0, latent_format, self.preview_frames), "latent")
                if p:
                    init.update(p)
        except Exception as e:
            logging.warning(f"[MiniMaxH3Preview] initial noise preview failed: {e}")
        self._send(init)

        state = {"last_time": None, "window": []}

        def new_callback(step, x0, x, total_steps):
            try:
                is_last = step >= total_steps - 1
                if step % self.every_n_steps == 0 or is_last:
                    # Never rebind x0 -- the sampler reuses that tensor downstream.
                    video = comfy.utils.unpack_latents(x0, latent_shapes)[0]

                    now = time.perf_counter()
                    step_ms = None
                    if state["last_time"] is not None:
                        step_ms = (now - state["last_time"]) * 1000.0
                        state["window"].append(step_ms)
                        if len(state["window"]) > 8:
                            state["window"].pop(0)
                    state["last_time"] = now
                    avg_ms = (sum(state["window"]) / len(state["window"])) if state["window"] else None
                    sigma_val = sigmas_list[step] if 0 <= step < len(sigmas_list) else None

                    frames = _latent_to_pil(video, latent_format, self.preview_frames)
                    if frames:
                        def _send_latent(frames=frames, step_ms=step_ms, avg_ms=avg_ms,
                                         sigma_val=sigma_val, sent=step + 1, total=total_steps):
                            p = self._frames_to_payload(frames, "latent")
                            if p:
                                p.update({"step": sent, "total": total, "sigma": sigma_val,
                                          "step_ms": step_ms, "avg_step_ms": avg_ms})
                                self._send(p)
                        # The last frame is what the panel is left displaying, so wait for
                        # a queue slot rather than dropping it. Sampling is over anyway.
                        encoder.submit(_send_latent, block_timeout=5.0 if is_last else None)

                    if vae_decoder is not None and (step % self.vae_every_n_steps == 0 or is_last):
                        device = vae_decoder.resolve(video)
                        # Private copy: the sampler's tensor is long gone by the time an
                        # off-thread decode gets to it.
                        z = video[:1].detach().clone()

                        def _send_vae(z=z, sent=step + 1, total=total_steps, sigma_val=sigma_val):
                            t0 = time.perf_counter()
                            try:
                                vframes = vae_decoder.decode(z)
                            except Exception as e:
                                logging.warning(f"[MiniMaxH3Preview] VAE preview decode failed: {e}")
                                return
                            logging.info(
                                "[MiniMaxH3Preview] VAE preview: %d frame(s) on %s in %.1fs "
                                "(step %s/%s)", len(vframes), vae_decoder.device,
                                time.perf_counter() - t0, sent, total)
                            p = self._frames_to_payload(vframes, "vae")
                            if p:
                                p.update({"step": sent, "total": total, "sigma": sigma_val})
                                self._send(p)

                        if device == "gpu":
                            # Synchronous on purpose: VAE.decode -> load_models_gpu mutates
                            # global model-management state and could evict the transformer
                            # from under a concurrent forward pass. This stalls sampling.
                            _send_vae()
                        else:
                            vae_worker.submit(_send_vae)
            except Exception as e:
                logging.warning(f"[MiniMaxH3Preview] preview callback failed: {e}")
            if original_callback is not None:
                original_callback(step, x0, x, total_steps)

        # Suppress the stock sampler-node preview. Patch every concrete
        # decode_latent_to_preview_image: subclasses (e.g. VHS's WrappedPreviewer) override
        # it and would keep emitting their own frames if only the base class were patched.
        prev_methods = []
        if self.suppress_default:
            targets = [latent_preview.LatentPreviewer]
            stack = list(latent_preview.LatentPreviewer.__subclasses__())
            while stack:
                cls = stack.pop()
                targets.append(cls)
                stack.extend(cls.__subclasses__())
            for cls in targets:
                if "decode_latent_to_preview_image" in cls.__dict__:
                    prev_methods.append((cls, cls.__dict__["decode_latent_to_preview_image"]))
                    cls.decode_latent_to_preview_image = _suppressed_preview_image

        try:
            state["last_time"] = time.perf_counter()
            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            new_callback, disable_pbar, seed, **kwargs)
        finally:
            encoder.shutdown(drain_timeout=5.0)
            if vae_worker is not None:
                # Its on_close hook does the VAE restore, whether or not the join times out.
                vae_worker.shutdown(drain_timeout=2.0)
            elif vae_decoder is not None:
                vae_decoder.restore()
            for cls, prev in prev_methods:
                cls.decode_latent_to_preview_image = prev


# --------------------------------------------------------------------------------------
# node
# --------------------------------------------------------------------------------------

class MiniMaxH3LivePreview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="MiniMaxH3LivePreview",
            display_name="MiniMax H3 Live Preview",
            category="model/patch/minimax",
            description=(
                "Drop between the H3 model loader and the sampler to watch the video build up "
                "on this node's own panel, mid-generation. Frames come from the latent2rgb "
                "approximation, so preview resolution is the latent resolution (width/16 x "
                "height/16) upscaled for display -- enough to read composition and motion. "
                "Wire the H3 video VAE and set vae_decode_every_n_steps for occasional "
                "full-resolution frames (see the tooltip for the cost)."
            ),
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 model. Non-H3 models pass through untouched."),
                io.Int.Input(
                    "preview_frames", default=8, min=1, max=256, step=1,
                    tooltip="Latent frames sampled evenly across the clip for each preview. "
                            "1 = a single still image; >1 = animated playback.",
                ),
                io.Int.Input(
                    "preview_fps", default=8, min=1, max=60, step=1,
                    tooltip="Playback rate of the animated preview. Ignored when preview_frames=1.",
                ),
                io.Int.Input(
                    "max_resolution", default=512, min=0, max=4096, step=8,
                    tooltip="Longest side of the transmitted preview, in pixels. Latent frames are "
                            "tiny (84x48 for a 1344x768 generation), so this mostly upscales. "
                            "0 = send at native latent resolution.",
                ),
                io.Int.Input(
                    "every_n_steps", default=1, min=1, max=100, step=1,
                    tooltip="Emit a preview every N sampler steps. The final step always emits.",
                ),
                io.Combo.Input(
                    "upscale_method", options=["nearest-exact", "lanczos", "bilinear"],
                    default="nearest-exact",
                    tooltip="Resampling used to scale the preview to max_resolution. "
                            "nearest-exact keeps latent blocks crisp; lanczos smooths them.",
                ),
                io.Int.Input(
                    "jpeg_quality", default=80, min=30, max=100, step=1,
                    tooltip="Quality for the JPEG/WebP preview transport.",
                ),
                io.Boolean.Input(
                    "suppress_default_preview", default=True,
                    tooltip="Suppress the standard sampler-node preview so only this node's panel "
                            "updates. The progress bar still advances normally.",
                ),
                io.Vae.Input(
                    "vae", optional=True,
                    tooltip="MiniMax H3 *video* VAE, for true full-resolution preview frames. "
                            "Only used when vae_decode_every_n_steps > 0.",
                ),
                io.Int.Input(
                    "vae_decode_every_n_steps", default=0, min=0, max=100, step=1,
                    tooltip="0 = off (recommended). Otherwise run a real VAE decode every N steps. "
                            "COSTLY: the H3 video VAE is ~5.2 GB and does not fit alongside a ~21 GB "
                            "H3 transformer on a 24 GB card, so the GPU path makes ComfyUI evict and "
                            "reload the transformer around each preview. The CPU path avoids that but "
                            "takes minutes per frame. Decoding runs off-thread either way.",
                ),
                io.Int.Input(
                    "vae_decode_frames", default=1, min=1, max=16, step=1,
                    tooltip="Full-resolution frames per VAE preview. Each is decoded as an "
                            "independent single-latent-frame slice.",
                ),
                io.Combo.Input(
                    "vae_decode_device", options=["auto", "gpu", "cpu"], default="auto",
                    tooltip="auto compares free VRAM against the VAE's weights plus decode "
                            "activations and picks cpu when they do not fit, logging the numbers.",
                ),
            ],
            outputs=[io.Model.Output(tooltip="Model with the live preview attached.")],
            hidden=[io.Hidden.unique_id],
            is_experimental=True,
        )

    @classmethod
    def execute(cls, model, preview_frames, preview_fps, max_resolution, every_n_steps,
                upscale_method, jpeg_quality, suppress_default_preview,
                vae=None, vae_decode_every_n_steps=0, vae_decode_frames=1,
                vae_decode_device="auto") -> io.NodeOutput:
        if vae_decode_every_n_steps > 0:
            if vae is None:
                logging.warning(
                    "[MiniMaxH3Preview] vae_decode_every_n_steps > 0 but no VAE is connected; "
                    "full-resolution previews are disabled."
                )
            else:
                logging.warning(
                    "[MiniMaxH3Preview] real-VAE previews enabled (every %d steps). The H3 video "
                    "VAE is ~5.2 GB and will not co-reside with the transformer on a 24 GB card: "
                    "expect transformer eviction/reload on the gpu path, or minutes per frame on "
                    "the cpu path.", vae_decode_every_n_steps,
                )

        m = model.clone()
        m.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            "minimax_h3_live_preview",
            _H3PreviewWrapper(
                cls.hidden.unique_id, preview_frames, preview_fps, max_resolution,
                every_n_steps, upscale_method, jpeg_quality, suppress_default_preview,
                vae, vae_decode_every_n_steps, vae_decode_frames, vae_decode_device,
            ),
        )
        return io.NodeOutput(m)
