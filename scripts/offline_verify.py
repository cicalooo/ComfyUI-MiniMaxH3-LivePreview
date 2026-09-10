"""Offline verification for MiniMax H3 Live Preview hardening.

Runs without a live ComfyUI server. Intended for local acceptance of the
plugin-side fixes documented in docs/MINIMAX_H3_LIVE_PREVIEW_AUDIT.md.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import io
import json
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
COMFY = ROOT.parents[1]
CUSTOM_NODES = ROOT.parent
if str(COMFY) not in sys.path:
    sys.path.insert(0, str(COMFY))
if str(CUSTOM_NODES) not in sys.path:
    sys.path.insert(0, str(CUSTOM_NODES))


def _ok(name: str) -> None:
    print(f"PASS  {name}")


def test_imports_and_schema():
    pkg = importlib.import_module("ComfyUI-MiniMaxH3-LivePreview.preview")
    schema = pkg.MiniMaxH3LivePreview.define_schema()
    assert schema.node_id == "MiniMaxH3LivePreview"
    assert len(schema.inputs) == 14
    assert len(schema.outputs) == 1
    _ok("import + V3 schema")
    return pkg


def test_bounds(pkg):
    assert pkg._bounded_int("nope", 8, 1, 256) == 8
    assert pkg._bounded_int(-5, 8, 1, 256) == 1
    assert pkg._bounded_int(9999, 8, 1, 256) == 256
    assert pkg._bounded_int(12.7, 8, 1, 256) == 12
    wrapper = pkg._H3PreviewWrapper(
        node_id="1",
        preview_frames=10_000,
        preview_fps=0,
        max_resolution=-3,
        every_n_steps=0,
        upscale_method="nearest-exact",
        jpeg_quality=5,
        suppress_default=True,
        vae_every_n_steps=-1,
        vae_frames=99,
        vae_device="weird",
        verbose=True,
    )
    assert wrapper.preview_frames == 256
    assert wrapper.preview_fps == 1
    assert wrapper.max_resolution == 0
    assert wrapper.every_n_steps == 1
    assert wrapper.jpeg_quality == 30
    assert wrapper.vae_every_n_steps == 0
    assert wrapper.vae_frames == 16
    assert wrapper.vae_device == "auto"
    assert wrapper.verbose is True
    _ok("legacy input bounds")


def test_media_fallback_store(pkg):
    store = pkg._PreviewMediaStore(ttl_seconds=60, max_bytes=6)
    first = store.put(b"abcd")
    assert store.get(first) == b"abcd"
    assert store.snapshot() == (1, 4)

    # The bounded cache evicts the least-recently-used entry before inserting.
    second = store.put(b"efgh")
    assert store.get(first) is None
    assert store.get(second) == b"efgh"
    assert store.snapshot() == (1, 4)
    assert store.delete(second) is True
    assert store.delete(second) is False
    assert store.snapshot() == (0, 0)

    assert pkg._preview_media_range(b"012345", None) == (0, 5)
    assert pkg._preview_media_range(b"012345", "bytes=2-4") == (2, 4)
    assert pkg._preview_media_range(b"012345", "bytes=3-") == (3, 5)
    assert pkg._preview_media_range(b"012345", "bytes=-2") == (4, 5)
    assert pkg._preview_media_range(b"012345", "bytes=99-") is None
    _ok("same-origin media fallback cache + ranges")


def test_media_fallback_routes(pkg):
    if pkg.web is None:
        raise AssertionError("aiohttp is required for the media fallback routes")

    class FakeRoutes:
        def __init__(self):
            self.handlers = {}

        def _register(self, method, path):
            def decorator(handler):
                self.handlers[(method, path)] = handler
                return handler
            return decorator

        def post(self, path):
            return self._register("POST", path)

        def get(self, path):
            return self._register("GET", path)

        def delete(self, path):
            return self._register("DELETE", path)

    class FakeServer:
        def __init__(self):
            self.routes = FakeRoutes()

    class FakePromptServer:
        instance = FakeServer()

    old_prompt_server = pkg.PromptServer
    pkg.PromptServer = FakePromptServer
    try:
        pkg._register_preview_media_routes()
        handlers = FakePromptServer.instance.routes.handlers
        post = handlers[("POST", "/minimax_h3_preview/media")]
        get = handlers[("GET", "/minimax_h3_preview/media/{token}")]
        delete = handlers[("DELETE", "/minimax_h3_preview/media/{token}")]

        class Body:
            def __init__(self, data):
                self.data = data

            async def read(self, _limit):
                return self.data

        from types import SimpleNamespace

        response = asyncio.run(post(SimpleNamespace(
            headers={"Content-Type": "image/jpeg"},
            content_length=0,
            content=Body(b""),
        )))
        assert response.status == 415
        response = asyncio.run(post(SimpleNamespace(
            headers={"Content-Type": "video/mp4"},
            content_length=pkg._MAX_PREVIEW_MEDIA_BYTES + 1,
            content=Body(b""),
        )))
        assert response.status == 413

        data = b"fake-mp4"
        request = SimpleNamespace(
            headers={"Content-Type": "video/mp4"},
            content_length=len(data),
            content=Body(data),
        )
        response = asyncio.run(post(request))
        assert response.status == 200
        token = json.loads(response.text)["token"]

        response = asyncio.run(get(SimpleNamespace(
            headers={}, match_info={"token": token},
        )))
        assert response.status == 200 and response.body == data
        response = asyncio.run(get(SimpleNamespace(
            headers={"Range": "bytes=2-5"}, match_info={"token": token},
        )))
        assert response.status == 206 and response.body == data[2:6]
        assert response.headers["Content-Range"] == f"bytes 2-5/{len(data)}"

        response = asyncio.run(delete(SimpleNamespace(
            match_info={"token": token},
        )))
        assert response.status == 204
        response = asyncio.run(get(SimpleNamespace(
            headers={}, match_info={"token": token},
        )))
        assert response.status == 404
    finally:
        pkg.PromptServer = old_prompt_server
    _ok("same-origin media fallback routes")


def test_worker_shutdown_when_full(pkg):
    started = threading.Event()
    release = threading.Event()

    def blocker():
        started.set()
        release.wait(timeout=5)

    worker = pkg._Worker("mmh3_verify_worker", maxsize=1)
    assert worker.submit(blocker) is True
    assert started.wait(timeout=2)
    # Fill the single queue slot while the blocker is in-flight.
    assert worker.submit(lambda: None) is True
    assert worker.submit(lambda: None) is False
    t0 = time.perf_counter()
    # Shutdown must not hang waiting to insert a sentinel into the full queue.
    worker.shutdown(drain_timeout=0.2, drain=False)
    elapsed = time.perf_counter() - t0
    release.set()
    worker.thread.join(timeout=2)
    assert elapsed < 1.5, f"shutdown blocked too long: {elapsed:.2f}s"
    assert not worker.thread.is_alive()
    _ok("worker shutdown with full queue")


def test_worker_close_hook_after_inflight(pkg):
    closed = []
    started = threading.Event()
    release = threading.Event()

    def job():
        started.set()
        release.wait(timeout=5)

    worker = pkg._Worker(
        "mmh3_verify_close",
        maxsize=1,
        on_close=lambda: closed.append(True),
    )
    assert worker.submit(job) is True
    assert started.wait(timeout=2)
    worker.shutdown(drain_timeout=0.1, drain=False)
    release.set()
    worker.thread.join(timeout=2)
    assert closed == [True]
    _ok("worker close hook after in-flight job")


def test_worker_latest_wins(pkg):
    started = threading.Event()
    release = threading.Event()
    executed = []

    def blocker():
        started.set()
        release.wait(timeout=5)
        executed.append("active")

    worker = pkg._Worker("mmh3_verify_latest", maxsize=1)
    assert worker.submit(blocker) is True
    assert started.wait(timeout=2)
    assert worker.submit(lambda: executed.append("old")) is True
    assert worker.submit_latest(lambda: executed.append("middle")) is True
    assert worker.submit_latest(lambda: executed.append("new")) is True
    release.set()
    worker.shutdown(drain_timeout=2, drain=True)
    worker.thread.join(timeout=2)
    assert executed == ["active", "new"], executed
    _ok("latest preview job replaces stale queued work")


def test_step_timing(pkg):
    timing = pkg._StepTiming(window_size=2)
    assert timing.tick(now=1.0) == (None, None)
    step_ms, avg_ms = timing.tick(now=1.25)
    assert abs(step_ms - 250.0) < 1e-6
    assert abs(avg_ms - 250.0) < 1e-6
    step_ms, avg_ms = timing.tick(now=1.75)
    assert abs(step_ms - 500.0) < 1e-6
    assert abs(avg_ms - 375.0) < 1e-6
    _ok("sampler timing is independent of preview throttling")


def test_encode_paths(pkg):
    frames = [
        Image.new("RGB", (160, 96), color=(i * 20 % 255, 40, 80))
        for i in range(4)
    ]
    b64, w, h, mime = pkg._encode_frames(frames[:1], fps=8, quality=80)
    assert mime == "image/jpeg" and w == 160 and h == 96 and len(b64) > 32
    Image.open(io.BytesIO(base64.b64decode(b64))).verify()

    b64, w, h, mime = pkg._encode_frames(frames, fps=8, quality=80)
    assert mime in ("video/mp4", "image/webp"), mime
    assert w >= 160 and h >= 96 and len(b64) > 32
    raw = base64.b64decode(b64)
    assert len(raw) > 32
    print(f"      encode multi-frame mime={mime} bytes={len(raw)}")
    _ok("encode jpeg + animated fallback/NVENC")


def test_content_fps_scaling(pkg):
    from fractions import Fraction

    # EmptyMiniMaxH3LatentAV length=124 -> latent_t=37, pixel frames=124.
    assert pkg._pixel_frames_for_latent_t(37) == 124
    assert pkg._pixel_frames_for_latent_t(2) == 5
    assert pkg._latent_slot_offsets(2) == [0, 1]
    # Average-rate helper still reports the exact fractional rate.
    assert pkg._encode_fps(24, 8, 124) == Fraction(48, 31)
    assert pkg._encode_fps(12, 8, 124) == Fraction(24, 31)
    assert pkg._encode_fps(24, 1, 124) == Fraction(24, 1)
    assert abs(float(pkg._encode_fps(24, 8, 124)) - (8 * 24 / 124)) < 1e-12

    # Evenly spaced latent samples land on uneven pixel offsets under FRAME_PER_TOKEN.
    idx = [0, 5, 10, 15, 21, 26, 31, 36]
    offsets, durations_ms, encode_fps, pixel_frames = pkg._preview_timeline(idx, 37, 24)
    assert pixel_frames == 124
    assert offsets == [0, 17, 34, 51, 69, 86, 103, 120]
    assert durations_ms == [708, 708, 708, 750, 708, 708, 708, 167]
    assert sum(durations_ms) == 5165  # ~124/24 s in ms rounding
    assert encode_fps == Fraction(48, 31)

    wrapper = pkg._H3PreviewWrapper(
        node_id="1",
        preview_frames=8,
        preview_fps=24,
        max_resolution=160,
        every_n_steps=1,
        upscale_method="nearest-exact",
        jpeg_quality=80,
        suppress_default=True,
    )
    frames = [Image.new("RGB", (160, 96), color=(i * 10, 20, 30)) for i in range(8)]
    payload = wrapper._frames_to_payload(
        frames, "latent", latent_t=37, latent_indices=idx,
    )
    assert payload is not None
    assert payload["fps"] == 24
    assert abs(payload["encode_fps"] - (8 * 24 / 124)) < 1e-9
    assert payload["mime"] in ("video/mp4", "image/webp")
    assert payload["frame_count"] == 8
    assert payload["duration_ms"] == sum(durations_ms)
    assert payload["frame_durations_ms"] == durations_ms
    raw = base64.b64decode(payload["image"])
    if payload["mime"] == "video/mp4":
        import av

        container = av.open(io.BytesIO(raw))
        duration_s = float(container.duration) / av.time_base
        assert abs(duration_s - (124 / 24)) < 0.05, duration_s
        container.close()
    _ok("content fps uses H3 pixel timeline for subsampled latent clips")


def test_pixel_budget(pkg):
    wrapper = pkg._H3PreviewWrapper(
        node_id="1",
        preview_frames=64,
        preview_fps=24,
        max_resolution=512,
        every_n_steps=1,
        upscale_method="nearest-exact",
        jpeg_quality=80,
        suppress_default=True,
    )
    # 80 frames at 512x512 exceed the 64*512*512 budget and must be thinned.
    frames = [Image.new("RGB", (512, 512), color=(10, 20, 30)) for _ in range(80)]
    payload = wrapper._frames_to_payload(frames, "latent")
    assert payload is not None
    assert payload["source"] == "latent"
    assert payload["mime"] in ("video/mp4", "image/webp", "image/jpeg")
    assert payload["frame_count"] == 64
    _ok("preview pixel budget")


def test_send_skips_broadcast(pkg):
    sent = []

    class FakeServer:
        def send_sync(self, event, payload, sid=None):
            sent.append((event, payload, sid))

    class FakePS:
        instance = FakeServer()

    old = pkg.PromptServer
    pkg.PromptServer = FakePS
    try:
        wrapper = pkg._H3PreviewWrapper(
            node_id="7",
            preview_frames=1,
            preview_fps=24,
            max_resolution=64,
            every_n_steps=1,
            upscale_method="nearest-exact",
            jpeg_quality=80,
            suppress_default=True,
        )
        wrapper._send({"step": 1}, run_id="abc", prompt_id="p1", client_id=None)
        assert sent == []
        wrapper._send({"step": 2}, run_id="abc", prompt_id="p1", client_id="client-1")
        assert len(sent) == 1
        event, payload, sid = sent[0]
        assert event == pkg.EVENT
        assert sid == "client-1"
        assert payload["node_id"] == "7"
        assert payload["run_id"] == "abc"
        assert payload["prompt_id"] == "p1"
    finally:
        pkg.PromptServer = old
    _ok("event targeting skips broadcast")


def test_verbose_error_event(pkg):
    sent = []

    class FakeServer:
        def send_sync(self, event, payload, sid=None):
            sent.append((event, payload, sid))

    class FakePS:
        instance = FakeServer()

    old = pkg.PromptServer
    pkg.PromptServer = FakePS
    try:
        wrapper = pkg._H3PreviewWrapper(
            node_id="7",
            preview_frames=1,
            preview_fps=24,
            max_resolution=64,
            every_n_steps=1,
            upscale_method="nearest-exact",
            jpeg_quality=80,
            suppress_default=True,
            verbose=True,
        )
        wrapper._report_error(
            "encode", ValueError("codec rejected frame"), run_id="run-1",
            prompt_id="prompt-1", client_id="client-1", source="latent", step=3,
            total=8, mime="video/mp4", width=64, height=32,
        )
        assert len(sent) == 1
        event, payload, sid = sent[0]
        assert event == pkg.EVENT and sid == "client-1"
        assert payload["verbose"] is True
        assert payload["run_id"] == "run-1"
        assert payload["prompt_id"] == "prompt-1"
        assert payload["step"] == 3 and payload["total"] == 8
        assert payload["error"] == {
            "stage": "encode",
            "type": "ValueError",
            "message": "codec rejected frame",
            "source": "latent",
            "step": 3,
            "total": 8,
            "mime": "video/mp4",
            "w": 64,
            "h": 32,
        }
    finally:
        pkg.PromptServer = old
    _ok("verbose preview error event")


def test_non_h3_status_event(pkg):
    from types import SimpleNamespace

    sent = []
    wrapper = pkg._H3PreviewWrapper(
        node_id="9",
        preview_frames=1,
        preview_fps=24,
        max_resolution=64,
        every_n_steps=1,
        upscale_method="nearest-exact",
        jpeg_quality=80,
        suppress_default=True,
    )
    wrapper._send = lambda payload, *args: sent.append(payload)

    class Executor:
        class_obj = SimpleNamespace(
            model_patcher=SimpleNamespace(model=SimpleNamespace(latent_format=object()))
        )

        def __call__(self, *args, **kwargs):
            return "passthrough"

    result = wrapper(Executor(), None, None, None, None, None, None, False, 0)
    assert result == "passthrough"
    assert sent and sent[0]["disabled"] is True
    _ok("non-H3 pass-through clears stale preview state")


def test_latent2rgb_and_tae(pkg):
    import comfy.latent_formats

    tiny = importlib.import_module("ComfyUI-MiniMaxH3-LivePreview.tiny_vae")
    lf = comfy.latent_formats.MiniMaxH3Video()
    video = torch.randn(1, 24, 8, 6, 10, dtype=torch.float32)
    frames, idx = pkg._latent_to_pil(video, lf, max_frames=4)
    assert len(frames) == 4 and len(idx) == 4
    assert frames[0].size == (10, 6)

    tae = tiny.load_tae_decoder("taeh3.safetensors")
    assert tae is not None, "taeh3.safetensors not found under models/vae_approx"
    frames, idx = pkg._tae_to_pil(video, tae, max_frames=2)
    assert len(frames) == 2 and len(idx) == 2
    # 16x spatial upscale from 10x6 -> 160x96
    assert frames[0].size == (160, 96)
    try:
        pkg._latent_to_pil(video, object(), max_frames=2)
    except ValueError as err:
        assert "latent_rgb_factors" in str(err)
    else:
        raise AssertionError("latent2rgb decode failures must remain observable")
    _ok("latent2rgb + taeh3 decode")


def test_cpu_vae_process_output_contract(pkg):
    class FakeModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))

        def decode(self, z):
            # Pretend the underlying module returns NCHW in [0, 1].
            b, c, t, h, w = z.shape
            return torch.full((b, 3, t, h * 2, w * 2), 0.25, dtype=torch.float32)

    class FakeVae:
        def __init__(self):
            self.first_stage_model = FakeModule()
            self.vae_dtype = torch.float32

        def process_output(self, px):
            # Host contract for current MiniMax H3: identity on already-[0,1] output.
            return px

        def memory_used_decode(self, shape, dtype):
            return 1

        def decode(self, z):
            raise AssertionError("CPU path must not call VAE.decode")

    decoder = pkg._VaeDecoder(FakeVae(), mode="cpu", frames=1)
    video = torch.zeros(1, 24, 2, 4, 5)
    assert decoder.resolve(video) == "cpu"
    frames, idx = decoder.decode(video)
    assert len(frames) == 1 and idx == [0]
    arr = np.asarray(frames[0])
    # 0.25 * 255 ~= 63.75 -> 63 after uint8 cast
    assert arr.shape == (8, 10, 3)
    assert int(arr.mean()) in (63, 64)
    decoder.restore()
    p = next(decoder.vae.first_stage_model.parameters())
    assert p.device.type == "cpu"
    _ok("CPU VAE uses process_output contract")


def test_js_syntax():
    js = ROOT / "web" / "js" / "mmh3_preview.js"
    subprocess.run(["node", "--check", str(js)], check=True)
    source = js.read_text(encoding="utf-8")
    assert "enable verbose for details" in source
    assert "backend preview failure" in source
    assert "uploadVideoForSameOriginFallback" in source
    assert "isUrlSafetyError" in source
    assert "CONTROLS_REFRESH_MS = 250" in source
    assert "_mmh3ExecutionEnded" in source
    assert 'video.addEventListener("timeupdate", notify)' not in source
    _ok("javascript syntax")


def main():
    print(f"ComfyUI root: {COMFY}")
    print(f"Plugin root:  {ROOT}")
    try:
        pkg = test_imports_and_schema()
        test_bounds(pkg)
        test_media_fallback_store(pkg)
        test_media_fallback_routes(pkg)
        test_worker_shutdown_when_full(pkg)
        test_worker_close_hook_after_inflight(pkg)
        test_worker_latest_wins(pkg)
        test_step_timing(pkg)
        test_encode_paths(pkg)
        test_content_fps_scaling(pkg)
        test_pixel_budget(pkg)
        test_send_skips_broadcast(pkg)
        test_verbose_error_event(pkg)
        test_non_h3_status_event(pkg)
        test_cpu_vae_process_output_contract(pkg)
        test_latent2rgb_and_tae(pkg)
        test_js_syntax()
    except Exception as err:
        print(f"FAIL  {err}")
        traceback.print_exc()
        raise SystemExit(1) from err
    print("\nAll offline verification checks passed.")


if __name__ == "__main__":
    main()
