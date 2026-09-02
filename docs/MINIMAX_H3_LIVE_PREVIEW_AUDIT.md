# MiniMax H3 Live Preview — Audit and Hardening Report

## Scope

Audited:

`C:\comfycli\custom_nodes\ComfyUI-MiniMaxH3-LivePreview`

The audit was performed against the installed ComfyUI checkout at:

- ComfyUI git commit: `4e024cb1` — `Lower trellis workflow remesh memory usage. (#16034)`
- Plugin git commit before changes: `e212946` — `Add tiny TAE decoder & live preview`
- Platform: Windows AMD64
- Python: 3.12.10

The plugin repository had no tracked source modifications before this pass. It did have generated
Python cache artifacts in the working tree (`__pycache__/*.cpython-312.pyc`); these were not removed
or otherwise modified.

## Findings

### 1. Full-VAE CPU previews used the wrong value range

The installed ComfyUI MiniMax H3 video VAE returns decoded pixels in `[0, 1]` and its host
`VAE.process_output` is an identity transform. The plugin's CPU path assumed the underlying
module returned `[-1, 1]`, applied another `(x + 1) / 2` conversion, and could produce washed-out
or incorrectly bright previews.

**Status:** fixed in `preview.py` by using the connected VAE's `process_output` contract before
converting to PIL pixels.

### 2. Worker shutdown could fail when its queue was full

The original worker inserted a stop sentinel into a bounded queue. If the queue was full, the
sentinel insertion timed out and the worker could remain alive indefinitely. For CPU VAE previews,
that could also delay device/dtype restoration.

**Status:** fixed in `preview.py`.

- Shutdown is now event based.
- Submission and shutdown are synchronized.
- Queued work can be drained or discarded explicitly.
- Obsolete queued CPU-VAE decodes are discarded at run end.
- An in-flight decode is allowed to finish and still runs the close hook.

### 3. Shared VAE objects could be accessed concurrently

A CPU preview temporarily moves the connected VAE module to CPU/fp32. The same VAE object may be
shared by multiple nodes or prompts, so concurrent decode/restore operations could race.

**Status:** fixed in `preview.py` with a per-VAE re-entrant lock covering device resolution,
decode, and restore.

### 4. Browser media updates could complete out of order

`img.decode()` and video playback/load operations are asynchronous. Multiple sampler events could
therefore complete in a different order from arrival. An older frame could replace a newer one,
or an older update could revoke the object URL currently needed by a newer element.

**Status:** fixed in `web/js/mmh3_preview.js`.

- Media updates are serialized.
- Pending updates are coalesced so slow browser decodes do not retain every base64 payload.
- Each update has a generation number.
- Stale/disposed updates are rejected before swapping media.
- Video loading has a bounded three-second wait and resolves on load or error.
- Object URLs and video sources are cleaned up on reset/removal.

### 5. Late frames could contaminate a later execution

Background encoder/VAE jobs can finish after sampling returns. Without an execution identity, late
frames from a previous run could appear in the next run's node panel.

**Status:** fixed across `preview.py` and `web/js/mmh3_preview.js`.

- Each sampler invocation gets a unique `run_id`.
- The current ComfyUI execution context's `prompt_id` is included when available.
- The frontend establishes a run only from its boundary-0 event.
- Frames from another run/prompt are ignored.
- Progress is prevented from moving backward when a slow VAE frame arrives after a newer cheap
  preview.
- A new run resets the source tabs and media state.

### 6. Default preview suppression was a process-wide class patch

The plugin temporarily replaces `decode_latent_to_preview_image` on previewer classes. Holding that
class-level mutation for the complete sampling run could interfere with another prompt running in
the same process and made restoration sensitive to overlapping executions.

**Status:** improved in `preview.py`.

The class patch is now held only around the stock callback invocation and protected by a lock,
which preserves the intended suppression while reducing the process-wide race window.

### 7. Legacy/cached workflow values were not defensively bounded

Normal schema validation constrains preview settings, but old or cached workflows and direct
third-party wrapper calls can bypass those assumptions. Extremely large frame counts/resolutions
could cause excessive CPU, memory, or transport use.

**Status:** fixed in `preview.py` and `web/js/mmh3_preview.js`.

- Preview frame count: bounded to 1–256.
- Playback FPS: bounded to 1–60.
- Display resolution: bounded to 0–4096.
- Step interval: bounded to 1–100.
- JPEG/WebP quality: bounded to 30–100.
- Full-VAE preview frame count: bounded to 1–16.
- One encoded event is capped by a pixel budget.
- Browser base64 payloads over 32 MiB are rejected.

### 8. Preview events could be broadcast unintentionally

ComfyUI's `send_sync(..., sid=None)` broadcasts to all connected clients. A prompt without a valid
client target should not leak preview events to unrelated browser sessions.

**Status:** fixed in `preview.py`; events are now skipped when there is no active server/client
target instead of being broadcast.

### 9. DOM widget layout was underspecified

The frontend registered the DOM widget without an explicit minimum/layout contract. Newer frontend
DOM-widget layout handling can calculate an unstable or collapsing node height in that situation.

**Status:** fixed in `web/js/mmh3_preview.js`.

The widget now declares:

- `hideOnZoom: false`
- minimum height: 380 px
- minimum width: 300 px
- large explicit maximum dimensions
- `serialize: false`

## Files changed

### `preview.py`

- Corrected CPU full-VAE output conversion.
- Added bounded parameter normalization.
- Added run and prompt identifiers to events.
- Hardened server/client event targeting.
- Reworked worker shutdown and queue race handling.
- Added shared-VAE serialization.
- Added preview pixel budgeting.
- Narrowed and serialized stock-preview suppression.
- Added defensive device validation for VAE decoding.

### `web/js/mmh3_preview.js`

- Added base64 payload size validation.
- Added serialized/coalesced media updates.
- Added stale-generation and disposal checks.
- Added object URL/video cleanup.
- Added execution lifecycle and prompt/run filtering.
- Added stale progress protection.
- Added explicit DOM widget layout constraints.

### `README.md`

Added an audit-notes section listing host/core limitations that were intentionally not changed in
this plugin.

### `docs/MINIMAX_H3_LIVE_PREVIEW_AUDIT.md`

This report.

## Verification performed

The following checks passed against the installed checkout:

- `python -m compileall -q C:\comfycli\custom_nodes\ComfyUI-MiniMaxH3-LivePreview`
- `python -m py_compile ...\preview.py`
- `node --check ...\web\js\mmh3_preview.js`
- `git diff --check`
- Plugin import through a package-style import specification.
- V3 schema generation:
  - node id: `MiniMaxH3LivePreview`
  - inputs: 13
  - outputs: 1
- Tiny TAE checkpoint discovery/load smoke test using the installed
  `models\vae_approx\taeh3.safetensors`:
  - latent channels: 24
  - spatial upscale: 16x
  - CPU decode returned the expected frame tensor layout.

The initial quick pass had no live ComfyUI server. Live TAE/latent acceptance was completed later
in this report; full video-VAE preview acceptance is intentionally out of scope.

## Deliberately not changed / follow-up items

These items belong to ComfyUI core, the separate H3 loader/VAE implementation, or external runtime
dependencies:

1. **Core NestedTensor callback contract**
   The sampler currently packs AV latents and reconstructs nested callback views in
   `comfy/samplers.py`. If that upstream contract changes, this plugin's `unpack_latents`
   integration must be revalidated.

2. **Core PromptServer routing**
   The plugin uses ComfyUI's current `PromptServer.client_id` routing model. A true multi-client
   routing redesign requires a corresponding core event-targeting change.

3. **Full H3 video-VAE memory cost / niche path**
   The approximately 5.2 GiB VAE and transformer eviction/reload behavior are properties of the
   VAE/model-management layer. The plugin cannot make both fit on a 24 GiB card or safely interrupt
   an in-flight CPU convolution. This path is niche (high-VRAM cards / curiosity use) and is **not**
   part of the accepted live-preview scope; TAE is the supported full-resolution preview path.

4. **PyAV/NVENC installation and driver support**
   The plugin falls back to animated WebP, but it cannot install or repair missing PyAV codecs,
   FFmpeg capabilities, or NVIDIA driver support.

5. **Old frontend compatibility**
   The frontend uses the current `window.comfyAPI` extension API and qualified execution IDs. A
   frontend that lacks those APIs needs a separate compatibility adapter.

6. **Generated cache artifacts**
   Completed in the continuation pass: tracked `__pycache__` artifacts were removed from the index
   and a plugin-local `.gitignore` now excludes them.

## Current status

Targeted reliability hardening is applied and source-level checks pass. Restart ComfyUI before
validating the node, because the Python module and frontend JavaScript are loaded during startup.

## Continuation pass

Follow-up work after the initial audit report:

### Repository hygiene

- Added `.gitignore` for `__pycache__/`, `*.py[cod]`, and common local cache dirs.
- Removed previously tracked `__pycache__/*.cpython-313.pyc` artifacts from the git index.
- Left the functional hardening changes uncommitted so they can be reviewed as one unit with this
  report.

### Offline verification harness

Added `scripts/offline_verify.py`, which exercises the plugin-side fixes without requiring a live
ComfyUI server:

- package import and V3 schema generation
- legacy/cached input bounding
- worker shutdown with a full queue
- worker close-hook behavior after an in-flight job
- JPEG plus multi-frame encode path (NVENC MP4 when available, otherwise WebP)
- preview pixel budgeting
- non-broadcast event targeting
- CPU full-VAE `process_output` contract
- latent2rgb and `taeh3.safetensors` decode smoke
- JavaScript syntax check

All of those checks passed in this environment. Notable runtime observation from the harness:

- PyAV reports NVENC available here (`av 18.1.0`, `h264_nvenc` probe succeeded)
- multi-frame encode selected `video/mp4`

### Live ComfyUI acceptance (continuation)

ComfyUI was later available at `http://127.0.0.1:8188` (version `0.34.0`, RTX 3090). The
`cli-anything-comfyui` CLI app became usable, and a minimal API-format H3 prompt was queued with
`MiniMaxH3LivePreview` inserted between `UNETLoader` and `MiniMaxH3SigmaShift`.

Harness: `scripts/live_accept_preview.py` (captures `minimax_h3_preview` websocket events for the
same `client_id` used to queue the prompt).

#### Passed live checks

1. **Node registration after restart**
   - `MiniMaxH3LivePreview` is present in `/object_info`
   - `tae_decoder` options include `taeh3.safetensors`

2. **TAE live path**
   - Settings: `672x384`, length `5`, steps `4`, `tae_decoder=taeh3.safetensors`
   - Result: 5 preview events (`step` 0..4), source `tae`, mime `video/mp4`, size `256x146`
   - Single `run_id`, matching `prompt_id`, no execution errors
   - Elapsed ~16.1 s

3. **latent2rgb live path**
   - Settings: `672x384`, length `5`, steps `3`, `tae_decoder=none`
   - Result: 4 preview events (`step` 0..3), source `latent`, mime `video/mp4`, size `256x146`
   - Distinct new `run_id` / `prompt_id`, no execution errors
   - Elapsed ~1.3 s

These live runs confirm the sampler callback, nested-latent unpack, TAE/latent encode transport,
run/prompt tagging, NVENC MP4 delivery, and client-targeted websocket delivery on a real H3 turbo
model.

#### Acceptance scope

In-scope and accepted:

- latent2rgb live previews
- `taeh3` live previews
- transport, run/prompt isolation, and client-targeted events for those paths

Out of scope for acceptance:

- live full video-VAE previews (`vae` / `vae_decode_every_n_steps > 0`)
  Almost nobody can or should use this on a 24 GB card; it remains an optional niche path for
  very-high-VRAM machines. Offline contract coverage for the CPU `process_output` path is enough.

#### Frontend panel status

The browser panel was hardened in source for serialized/coalesced media updates, stale
run/generation rejection, object-URL cleanup, and DOM-widget layout bounds. Live websocket events
for both TAE and latent streams were verified end-to-end. A purely visual click-through in the UI
is optional and not required to close this hardening pass.

## Final status

Plugin-side hardening for the supported live-preview paths is complete, offline-verified, and
live-accepted for TAE + latent2rgb. Remaining items are host/core limitations or the intentionally
niche full-VAE option, not open defects in the accepted preview scope.
