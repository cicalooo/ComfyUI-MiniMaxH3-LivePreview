"""Offline verification for MiniMax H3 Live Preview hardening.

Runs without a live ComfyUI server. Intended for local acceptance of the
plugin-side fixes documented in MINIMAX_H3_LIVE_PREVIEW_AUDIT.md.
"""

from __future__ import annotations

import base64
import importlib
import io
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parent
COMFY = ROOT.parents[1]
if str(COMFY) not in sys.path:
    sys.path.insert(0, str(COMFY))
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))


def _ok(name: str) -> None:
    print(f"PASS  {name}")


def test_imports_and_schema():
    pkg = importlib.import_module("ComfyUI-MiniMaxH3-LivePreview.preview")
    schema = pkg.MiniMaxH3LivePreview.define_schema()
    assert schema.node_id == "MiniMaxH3LivePreview"
    assert len(schema.inputs) == 13
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
    )
    assert wrapper.preview_frames == 256
    assert wrapper.preview_fps == 1
    assert wrapper.max_resolution == 0
    assert wrapper.every_n_steps == 1
    assert wrapper.jpeg_quality == 30
    assert wrapper.vae_every_n_steps == 0
    assert wrapper.vae_frames == 16
    assert wrapper.vae_device == "auto"
    _ok("legacy input bounds")


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


def test_pixel_budget(pkg):
    wrapper = pkg._H3PreviewWrapper(
        node_id="1",
        preview_frames=64,
        preview_fps=8,
        max_resolution=512,
        every_n_steps=1,
        upscale_method="nearest-exact",
        jpeg_quality=80,
        suppress_default=True,
    )
    # 64 frames at 512x512 would exceed the 64*512*512 budget and must be thinned.
    frames = [Image.new("RGB", (512, 512), color=(10, 20, 30)) for _ in range(64)]
    payload = wrapper._frames_to_payload(frames, "latent")
    assert payload is not None
    assert payload["source"] == "latent"
    assert payload["mime"] in ("video/mp4", "image/webp", "image/jpeg")
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
            preview_fps=8,
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


def test_latent2rgb_and_tae(pkg):
    import comfy.latent_formats

    tiny = importlib.import_module("ComfyUI-MiniMaxH3-LivePreview.tiny_vae")
    lf = comfy.latent_formats.MiniMaxH3Video()
    video = torch.randn(1, 24, 8, 6, 10, dtype=torch.float32)
    frames = pkg._latent_to_pil(video, lf, max_frames=4)
    assert len(frames) == 4
    assert frames[0].size == (10, 6)

    tae = tiny.load_tae_decoder("taeh3.safetensors")
    assert tae is not None, "taeh3.safetensors not found under models/vae_approx"
    frames = pkg._tae_to_pil(video, tae, max_frames=2)
    assert len(frames) == 2
    # 16x spatial upscale from 10x6 -> 160x96
    assert frames[0].size == (160, 96)
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
    frames = decoder.decode(video)
    assert len(frames) == 1
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
    _ok("javascript syntax")


def main():
    print(f"ComfyUI root: {COMFY}")
    print(f"Plugin root:  {ROOT}")
    try:
        pkg = test_imports_and_schema()
        test_bounds(pkg)
        test_worker_shutdown_when_full(pkg)
        test_worker_close_hook_after_inflight(pkg)
        test_encode_paths(pkg)
        test_pixel_budget(pkg)
        test_send_skips_broadcast(pkg)
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
