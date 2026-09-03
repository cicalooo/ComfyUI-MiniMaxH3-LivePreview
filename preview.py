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
import uuid
from fractions import Fraction

import numpy as np
import torch
from PIL import Image

import comfy.latent_formats
import comfy.model_management
import comfy.patcher_extension
import comfy.utils
import latent_preview
from comfy_api.latest import io

from .tiny_vae import NONE as TAE_NONE
from .tiny_vae import list_tae_decoders, load_tae_decoder

try:
    from server import PromptServer
except ImportError:
    PromptServer = None

try:
    from comfy_execution.utils import get_executing_context
except ImportError:  # pragma: no cover - compatibility with older ComfyUI builds
    get_executing_context = None

EVENT = "minimax_h3_preview"

# Suppressing the stock preview requires a temporary class-level patch because the
# sampler creates the previewer before this OUTER_SAMPLE wrapper runs.  Keep that
# patch serialized: two simultaneous prompts must not restore each other's methods.
_PREVIEW_SUPPRESS_LOCK = threading.RLock()

_RESAMPLE = {
    "nearest-exact": Image.NEAREST,
    "lanczos": Image.LANCZOS,
    "bilinear": Image.BILINEAR,
}

# Bound the amount of image data held/encoded by one event. Schema limits protect
# normal workflows, but old workflows can still carry arbitrary values.
_MAX_PREVIEW_PIXELS = 64 * 512 * 512

# H3's EmptyMiniMaxH3LatentAV / task nodes are authored on a 24 fps pixel timeline.
# Latent tokens are not 1:1 with pixels: FRAME_PER_TOKEN cycles (1, 4, 4, 4, 4).
_H3_CONTENT_FPS = 24
_H3_FRAME_PER_TOKEN = (1, 4, 4, 4, 4)


def _bounded_int(value, default, minimum, maximum):
    """Convert an untrusted/legacy workflow value into a bounded integer."""
    try:
        value = int(value)
    except (TypeError, ValueError, OverflowError):
        value = default
    return min(maximum, max(minimum, value))


def _pixel_frames_for_latent_t(latent_t):
    """Pixel-frame span covered by ``latent_t`` H3 video latent tokens."""
    try:
        latent_t = int(latent_t)
    except (TypeError, ValueError, OverflowError):
        return 0
    if latent_t <= 0:
        return 0
    return sum(_H3_FRAME_PER_TOKEN[i % 5] for i in range(latent_t))


def _as_fraction_rate(rate):
    """Normalize an encode rate for PyAV / WebP without integer rounding drift."""
    if isinstance(rate, Fraction):
        value = rate
    else:
        try:
            value = Fraction(rate).limit_denominator(10_000)
        except (TypeError, ValueError, ZeroDivisionError, OverflowError):
            value = Fraction(_H3_CONTENT_FPS, 1)
    if value <= 0:
        value = Fraction(1, 10)
    # Keep encoder rates in a practical preview band. Sub-1 values are valid:
    # a sparse subsample of a long clip must play slower than 1 fps to stay realtime.
    lo = Fraction(1, 10)
    hi = Fraction(60, 1)
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def _frame_duration_ms(fps):
    """Per-frame duration for animated WebP, derived from an exact encode rate."""
    rate = float(_as_fraction_rate(fps))
    return max(1, int(round(1000.0 / rate)))


def _encode_fps(content_fps, preview_frame_count, pixel_frame_count):
    """Convert a content-timeline fps (24 = realtime) into an encoder fps.

    Preview payloads only carry a subsample of latent tokens. Playing those
    tokens at the content fps makes the clip appear fast-forwarded; scale so
    ``content_fps == 24`` spans the same duration as the underlying pixel clip.

    Returns a ``Fraction`` so NVENC/libx264 can keep the exact rate. Rounding
    ``8 * 24 / 124`` to ``2`` used to make a ~5.17s clip play in 4.0s (~1.29x).
    """
    content_fps = _bounded_int(content_fps, _H3_CONTENT_FPS, 1, 60)
    try:
        preview_frame_count = int(preview_frame_count)
    except (TypeError, ValueError, OverflowError):
        preview_frame_count = 1
    try:
        pixel_frame_count = int(pixel_frame_count)
    except (TypeError, ValueError, OverflowError):
        pixel_frame_count = 0
    if preview_frame_count <= 1:
        return Fraction(content_fps, 1)
    if pixel_frame_count <= 0:
        # Fallback when the latent length is unknown: treat the widget as a
        # direct encode rate (legacy behaviour).
        return Fraction(content_fps, 1)
    return _as_fraction_rate(
        Fraction(preview_frame_count * content_fps, pixel_frame_count)
    )


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

# A VAE object can be shared by nodes/prompts. CPU preview temporarily moves its
# first_stage_model, so access to a shared VAE must be serialized.
_VAE_LOCKS = {}
_VAE_LOCKS_GUARD = threading.Lock()


def _get_vae_lock(vae):
    key = id(vae)
    with _VAE_LOCKS_GUARD:
        return _VAE_LOCKS.setdefault(key, threading.RLock())


class _Worker:
    """Single background thread with a bounded drop-on-full queue.

    The sampler must never block on preview work, so a full queue drops the job.
    Shutdown is event based rather than sentinel based: a full queue can no longer
    prevent the worker from reaching its close hook, which is important for restoring
    a CPU-pinned VAE.
    """

    def __init__(self, name, maxsize, on_close=None):
        self.q = queue.Queue(maxsize=max(1, int(maxsize)))
        self.closed = False
        self._stopping = threading.Event()
        self._drain_on_stop = True
        self._state_lock = threading.Lock()
        self.on_close = on_close
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, fn, block_timeout=None):
        """Queue `fn`. Returns False if it was dropped.

        Dropping is the point: a full queue means we are behind, and the sampler must
        not wait for us. `block_timeout` is for jobs worth waiting on -- the final step,
        whose frame is what the panel is left showing.
        """
        # Keep the closed check and queue insertion atomic with respect to shutdown.
        # Otherwise shutdown can observe an empty queue, clear it, and then a racing
        # submit can enqueue work after the worker has decided to exit.
        with self._state_lock:
            if self.closed:
                return False
            try:
                if block_timeout is None:
                    self.q.put_nowait(fn)
                else:
                    self.q.put(fn, timeout=max(0.0, float(block_timeout)))
                return True
            except queue.Full:
                return False

    def _close(self):
        if self.on_close is not None:
            try:
                self.on_close()
            except Exception:
                logging.exception("[MiniMaxH3Preview] worker close hook failed")

    def _run(self):
        while True:
            try:
                item = self.q.get(timeout=0.1)
            except queue.Empty:
                if self._stopping.is_set():
                    self._close()
                    return
                continue

            if self._stopping.is_set() and not self._drain_on_stop:
                self._close()
                return
            try:
                item()
            except Exception:
                logging.exception("[MiniMaxH3Preview] worker error")

            # submit() is closed before shutdown clears the queue, so an empty queue
            # here is a stable indication that all drainable work has finished.
            if self._stopping.is_set() and (not self._drain_on_stop or self.q.empty()):
                self._close()
                return

    def shutdown(self, drain_timeout=5.0, drain=True):
        with self._state_lock:
            self.closed = True
            self._drain_on_stop = bool(drain)
            self._stopping.set()

        if not drain:
            # Do not start expensive queued VAE decodes after the sampler has ended.
            # An already-running decode cannot be interrupted safely, but it will
            # still execute _close() and restore the VAE when it completes.
            while True:
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    break

        # Join is best-effort: a CPU VAE decode can run for minutes and cannot be
        # interrupted. The thread is a daemon and will finish cleanup eventually.
        self.thread.join(timeout=max(0.0, float(drain_timeout)))


def _fit(pil, max_res, resample):
    """Scale so the longest side equals max_res, preserving aspect. 0 = leave native.

    Unlike ImageOps.contain this scales *up* too -- latent-resolution frames are ~84x48
    and would otherwise arrive unreadably small.
    """
    try:
        max_res = int(max_res)
    except (TypeError, ValueError, OverflowError):
        max_res = 0
    if max_res <= 0 or pil.width <= 0 or pil.height <= 0:
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
            stream = container.add_stream("h264_nvenc", rate=_as_fraction_rate(fps))
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
            duration=_frame_duration_ms(fps), loop=0,
            quality=max(1, min(100, int(quality))), method=4,
        )
    except Exception as e:
        logging.warning(f"[MiniMaxH3Preview] animated WebP encode failed: {e}")
        return None, 0, 0
    return base64.b64encode(buf.getvalue()).decode("ascii"), pil_frames[0].width, pil_frames[0].height


def _encode_jpeg(pil, quality):
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    buf = pyio.BytesIO()
    pil.save(buf, format="JPEG", quality=max(1, min(100, int(quality))))
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


def _tae_to_pil(video, tae, max_frames):
    """[B, 24, T, h, w] -> list of PIL frames at *full* resolution (16x the latent grid).

    Raises on failure so the caller can fall back to latent2rgb for the rest of the run.
    """
    if video is None or video.ndim != 5:
        return []
    idx = _pick_indices(video.shape[2], max_frames)
    rgb = tae.decode_video(video[:1], idx)              # [T, H, W, 3] in 0..1, on the cpu
    u8 = rgb.mul_(255.0).clamp_(0, 255).to(torch.uint8).numpy()
    return [Image.fromarray(u8[i]) for i in range(u8.shape[0])]


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
        self._lock = _get_vae_lock(vae)

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
        which is only meaningful once the transformer is resident.

        Once resolved, do not take the decode lock: a CPU decode may be in flight and
        the sampler must be able to drop the next queued decode instead of stalling.
        """
        if self.device is None:
            with self._lock:
                if self.device is None:
                    z_shape = (1, video.shape[1], 1, video.shape[3], video.shape[4])
                    self._resolve_device(video.device, z_shape)
        return self.device

    def decode(self, video):
        """[B, 24, T, h, w] -> list of full-resolution PIL frames. resolve() first."""
        with self._lock:
            if self.device not in ("gpu", "cpu"):
                raise RuntimeError("VAE preview device was not resolved")
            t_total = video.shape[2]
            idx = _pick_indices(t_total, self.frames)
            out = []
            for i in idx:
                z = video[:1, :, i:i + 1]
                if self.device == "gpu":
                    px = self.vae.decode(z)                         # [B, T, H, W, C] in 0..1
                else:
                    self._pin_cpu()
                    with torch.no_grad():
                        px = self.vae.first_stage_model.decode(z.to(device="cpu", dtype=torch.float32))
                    # Match the host VAE.decode contract. Current MiniMax H3's
                    # first_stage_model already returns [0, 1], while older builds may
                    # still expose the generic [-1, 1] process_output transform.
                    process_output = getattr(self.vae, "process_output", None)
                    if process_output is not None:
                        processed = process_output(px)
                        if processed is not None:
                            px = processed
                    px = px.movedim(1, -1).clamp_(0.0, 1.0)
                if px.ndim == 5:
                    px = px[0]
                if px.ndim != 4:
                    continue
                u8 = (px.float() * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()
                out.extend(Image.fromarray(u8[j]) for j in range(u8.shape[0]))
            return out

    def restore(self):
        with self._lock:
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
                 upscale_method, jpeg_quality, suppress_default, tae_decoder=TAE_NONE,
                 vae=None, vae_every_n_steps=0, vae_frames=1, vae_device="auto"):
        self.node_id = str(node_id) if node_id is not None else None
        self.tae_decoder = tae_decoder
        # Schema validation normally enforces these bounds, but wrappers can also
        # be invoked by old/cached workflows or directly by third-party nodes.
        self.preview_frames = _bounded_int(preview_frames, 8, 1, 256)
        self.preview_fps = _bounded_int(preview_fps, _H3_CONTENT_FPS, 1, 60)
        self.max_resolution = _bounded_int(max_resolution, 512, 0, 4096)
        self.every_n_steps = _bounded_int(every_n_steps, 1, 1, 100)
        self.resample = _RESAMPLE.get(upscale_method, Image.NEAREST)
        self.jpeg_quality = _bounded_int(jpeg_quality, 80, 30, 100)
        self.suppress_default = bool(suppress_default)
        self.vae = vae
        self.vae_every_n_steps = _bounded_int(vae_every_n_steps, 0, 0, 100)
        self.vae_frames = _bounded_int(vae_frames, 1, 1, 16)
        self.vae_device = vae_device if vae_device in ("auto", "gpu", "cpu") else "auto"

    # -- plumbing ----------------------------------------------------------------------

    def _send(self, payload, run_id=None, prompt_id=None, client_id=None):
        if self.node_id is None or PromptServer is None:
            return
        payload = dict(payload)
        payload["node_id"] = self.node_id
        if run_id is not None:
            payload["run_id"] = run_id
        if prompt_id is not None:
            payload["prompt_id"] = prompt_id
        try:
            server = getattr(PromptServer, "instance", None)
            # Never broadcast previews from a headless/CLI execution to unrelated
            # browser sessions. The normal UI path supplies the target captured at
            # the start of this sampling run.
            if server is None or client_id is None:
                return
            server.send_sync(EVENT, payload, client_id)
        except Exception as e:
            logging.warning(f"[MiniMaxH3Preview] send failed: {e}")

    def _frames_to_payload(self, frames, source, latent_t=None):
        frames = [_fit(f, self.max_resolution, self.resample) for f in frames]
        if len(frames) > 1:
            pixels = max(1, frames[0].width * frames[0].height)
            allowed = max(1, _MAX_PREVIEW_PIXELS // pixels)
            if allowed < len(frames):
                frames = [frames[i] for i in _pick_indices(len(frames), allowed)]
        pixel_frames = _pixel_frames_for_latent_t(latent_t) if latent_t is not None else 0
        encode_fps = _encode_fps(self.preview_fps, len(frames), pixel_frames)
        b64, w, h, mime = _encode_frames(frames, encode_fps, self.jpeg_quality)
        if not b64:
            return None
        return {
            "image": b64, "mime": mime, "w": w, "h": h, "source": source,
            # Report the content-timeline fps the user set (24 = realtime), not the
            # possibly much lower encoder rate used after latent subsampling.
            "fps": self.preview_fps if mime in ("video/mp4", "image/webp") else None,
            # Float keeps websocket payloads JSON-friendly while preserving the
            # fractional encode rate (e.g. 48/31 ≈ 1.548 for 8/124 @ 24fps).
            "encode_fps": (
                float(encode_fps) if mime in ("video/mp4", "image/webp") else None
            ),
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
        run_id = uuid.uuid4().hex
        context = get_executing_context() if get_executing_context is not None else None
        prompt_id = getattr(context, "prompt_id", None)
        prompt_id = str(prompt_id) if prompt_id is not None else None
        server = getattr(PromptServer, "instance", None) if PromptServer is not None else None
        target_client_id = getattr(server, "client_id", None) if server is not None else None
        sigmas_list = sigmas.detach().cpu().tolist() if sigmas is not None else []
        total_steps_init = max(0, len(sigmas_list) - 1)

        # Held locally, never on self: the wrapper outlives the run (it rides on the model
        # patcher), and a cached decoder would keep its weights on the GPU forever.
        cheap = {"tae": load_tae_decoder(self.tae_decoder)}

        def _cheap_frames(video):
            """-> (frames, source). The tiny VAE when one is loaded, latent2rgb otherwise.

            Runs on the sampler thread like latent2rgb does: taeh3 is ~10 MB and a few
            milliseconds per frame, and decoding here keeps it off the encoder thread,
            which would otherwise allocate VRAM concurrently with a forward pass.
            """
            tae = cheap["tae"]
            if tae is not None:
                try:
                    return _tae_to_pil(video, tae, self.preview_frames), "tae"
                except Exception as e:
                    cheap["tae"] = None
                    logging.warning(
                        f"[MiniMaxH3Preview] tiny-VAE decode failed ({e}); falling back to "
                        "latent2rgb for the rest of this run."
                    )
            return _latent_to_pil(video, latent_format, self.preview_frames), "latent"

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
                frames0, source0 = _cheap_frames(video0)
                p = self._frames_to_payload(frames0, source0, latent_t=video0.shape[2])
                if p:
                    init.update(p)
        except Exception as e:
            logging.warning(f"[MiniMaxH3Preview] initial noise preview failed: {e}")
        self._send(init, run_id, prompt_id, target_client_id)

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

                    frames, source = _cheap_frames(video)
                    if frames:
                        latent_t = int(video.shape[2])
                        def _send_cheap(frames=frames, source=source, step_ms=step_ms,
                                        avg_ms=avg_ms, sigma_val=sigma_val,
                                        sent=step + 1, total=total_steps,
                                        latent_t=latent_t):
                            p = self._frames_to_payload(frames, source, latent_t=latent_t)
                            if p:
                                p.update({"step": sent, "total": total, "sigma": sigma_val,
                                          "step_ms": step_ms, "avg_step_ms": avg_ms})
                                self._send(p, run_id, prompt_id, target_client_id)
                        # The last frame is what the panel is left displaying, so wait for
                        # a queue slot rather than dropping it. Sampling is over anyway.
                        encoder.submit(_send_cheap, block_timeout=5.0 if is_last else None)

                    if vae_decoder is not None and (step % self.vae_every_n_steps == 0 or is_last):
                        device = vae_decoder.resolve(video)
                        # Private copy: the sampler's tensor is long gone by the time an
                        # off-thread decode gets to it.
                        z = video[:1].detach().clone()

                        latent_t = int(video.shape[2])
                        def _send_vae(z=z, sent=step + 1, total=total_steps, sigma_val=sigma_val,
                                      latent_t=latent_t):
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
                            p = self._frames_to_payload(vframes, "vae", latent_t=latent_t)
                            if p:
                                p.update({"step": sent, "total": total, "sigma": sigma_val})
                                self._send(p, run_id, prompt_id, target_client_id)


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
                # The stock callback owns the ComfyUI progress bar as well as the
                # previewer. Suppress only its image result, and only during this
                # callback invocation; holding the class patch for the whole prompt
                # would serialize otherwise independent prompts.
                if self.suppress_default:
                    with _PREVIEW_SUPPRESS_LOCK:
                        prev_methods = []
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
                            original_callback(step, x0, x, total_steps)
                        finally:
                            for cls, prev in reversed(prev_methods):
                                cls.decode_latent_to_preview_image = prev
                else:
                    original_callback(step, x0, x, total_steps)

        try:
            state["last_time"] = time.perf_counter()
            return executor(noise, latent_image, sampler, sigmas, denoise_mask,
                            new_callback, disable_pbar, seed, **kwargs)
        finally:
            encoder.shutdown(drain_timeout=5.0)
            if vae_worker is not None:
                # Do not start queued, obsolete CPU decodes after sampling ends. An
                # already-running decode finishes and its close hook restores the VAE.
                vae_worker.shutdown(drain_timeout=2.0, drain=False)
            elif vae_decoder is not None:
                vae_decoder.restore()
            # After the encoder has drained: a queued job holds only PIL frames, but the
            # decoder's weights should not outlive the run.
            cheap["tae"] = None


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
                "on this node's own panel, mid-generation. By default frames come from the "
                "latent2rgb approximation, so preview resolution is the latent resolution "
                "(width/16 x height/16) upscaled for display -- enough to read composition and "
                "motion. Set tae_decoder to taeh3.safetensors (in models/vae_approx) for real "
                "decoded frames at full resolution, still cheap enough for every step. Wiring "
                "the H3 video VAE and setting vae_decode_every_n_steps is the expensive option "
                "(see that tooltip for the cost)."
            ),
            inputs=[
                io.Model.Input("model", tooltip="MiniMax H3 model. Non-H3 models pass through untouched."),
                io.Int.Input(
                    "preview_frames", default=8, min=1, max=256, step=1,
                    tooltip="Latent frames sampled evenly across the clip for each preview. "
                            "1 = a single still image; >1 = animated playback. Free with "
                            "latent2rgb; with tae_decoder each frame is a real decode, so lower "
                            "this if previews start costing sampler time.",
                ),
                io.Int.Input(
                    "preview_fps", default=24, min=1, max=60, step=1,
                    tooltip="Content-timeline playback rate. H3 is authored at 24 fps, so 24 = "
                            "realtime, 12 = half-speed, 48 = 2x. The encoder rate is scaled from "
                            "how densely preview_frames samples the latent clip; ignored when "
                            "preview_frames=1.",
                ),
                io.Int.Input(
                    "max_resolution", default=512, min=0, max=4096, step=8,
                    tooltip="Longest side of the transmitted preview, in pixels. Latent2rgb frames "
                            "are tiny (84x48 for a 1344x768 generation) so this mostly upscales; "
                            "tae_decoder frames are full resolution, so it downscales them for "
                            "transport. 0 = send at native resolution.",
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
                # Appended rather than grouped with the other preview widgets so existing
                # workflows keep their widget values lined up.
                io.Combo.Input(
                    "tae_decoder", options=list_tae_decoders(), default=TAE_NONE,
                    tooltip="Tiny VAE decoder from models/vae_approx, replacing the "
                            "latent-resolution latent2rgb preview with real decoded frames at "
                            "full resolution. taeh3.safetensors "
                            "(huggingface.co/Kijai/MiniMax-H3-TAE) is ~10 MB and decodes each "
                            "latent frame in 2D -- 8 frames at 1344x768 in 0.25s and ~640 MB peak "
                            "on a 3090, nothing like the 5.2 GB video VAE above. A failed decode "
                            "(OOM on a full card) falls back to latent2rgb. none = latent2rgb. "
                            "Put the file in models/vae_approx and select it here: VAELoader "
                            "cannot load it ('VAE is invalid: None').",
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
                vae_decode_device="auto", tae_decoder=TAE_NONE) -> io.NodeOutput:
        # Normalize once at the node boundary as well as in the wrapper. This keeps
        # legacy/cached workflows from failing before the wrapper can protect itself.
        preview_frames = _bounded_int(preview_frames, 8, 1, 256)
        preview_fps = _bounded_int(preview_fps, _H3_CONTENT_FPS, 1, 60)
        max_resolution = _bounded_int(max_resolution, 512, 0, 4096)
        every_n_steps = _bounded_int(every_n_steps, 1, 1, 100)
        jpeg_quality = _bounded_int(jpeg_quality, 80, 30, 100)
        vae_decode_every_n_steps = _bounded_int(vae_decode_every_n_steps, 0, 0, 100)
        vae_decode_frames = _bounded_int(vae_decode_frames, 1, 1, 16)
        vae_decode_device = vae_decode_device if vae_decode_device in ("auto", "gpu", "cpu") else "auto"
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
                tae_decoder, vae, vae_decode_every_n_steps, vae_decode_frames,
                vae_decode_device,
            ),
        )
        return io.NodeOutput(m)
