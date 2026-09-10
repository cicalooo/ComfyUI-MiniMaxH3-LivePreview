// MiniMax H3 Live Preview -- on-node preview panel fed by the "minimax_h3_preview" event.
//
// Three independent streams share one stage: "latent" (cheap latent2rgb) and "tae" (taeh3
// tiny VAE, full resolution) are the two forms the every-few-steps stream can take, and
// "vae" (rare, expensive, full resolution) is the real video VAE. Each keeps its own last
// frame so the header tabs switch between them instantly.

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_ID = "MiniMaxH3LivePreview";
const EVENT = "minimax_h3_preview";
// Ordered cheapest-to-truest. A stream only ever auto-steals the stage from a lower rank,
// so a rare VAE frame surfaces itself but the latent stream never steals it back.
const SOURCES = ["latent", "tae", "vae"];
const STYLE_ID = "mmh3-preview-stylesheet";
const CSS_URL = new URL("./mmh3_preview.css", import.meta.url).href;

function ensureStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const link = document.createElement("link");
    link.id = STYLE_ID;
    link.rel = "stylesheet";
    link.href = CSS_URL;
    document.head.appendChild(link);
}

function chainCallback(object, property, callback) {
    const original = object[property];
    object[property] = function () {
        const r = original?.apply(this, arguments);
        callback.apply(this, arguments);
        return r;
    };
}

function el(tag, className, parent) {
    const e = document.createElement(tag);
    if (className) e.className = className;
    if (parent) parent.appendChild(e);
    return e;
}

// Execution ids in subgraphs look like "12:7:5" -- walk the chain to the leaf node.
// Mirrors the frontend's getNodeByExecutionId, which is not exported.
function findNodeByQualifiedId(rootGraph, qid) {
    if (!rootGraph || qid == null) return null;
    const parts = String(qid).split(":");
    let graph = rootGraph;
    for (let i = 0; i < parts.length - 1; i++) {
        const parentId = parseInt(parts[i], 10);
        if (!Number.isFinite(parentId)) return null;
        const parentNode = graph?.getNodeById?.(parentId);
        if (!parentNode?.subgraph) return null;
        graph = parentNode.subgraph;
    }
    const leafId = parseInt(parts[parts.length - 1], 10);
    if (!Number.isFinite(leafId)) return null;
    return graph?.getNodeById?.(leafId) || null;
}

const MAX_PREVIEW_B64_CHARS = 32 * 1024 * 1024;
const MAX_PREVIEW_FRAMES = 256;
const CONTROLS_REFRESH_MS = 250;
const PREVIEW_MIMES = new Set(["image/jpeg", "image/webp", "video/mp4"]);
const PREVIEW_MEDIA_PATH = "/minimax_h3_preview/media";

function b64ToBlob(b64, mime) {
    if (!PREVIEW_MIMES.has(mime)) throw new Error("unsupported preview media type");
    if (typeof b64 !== "string" || !b64 || b64.length > MAX_PREVIEW_B64_CHARS) {
        throw new Error("preview payload is missing or too large");
    }
    if (b64.length % 4 === 1) throw new Error("invalid preview payload");
    let bin;
    try {
        bin = atob(b64);
    } catch (_) {
        throw new Error("invalid preview payload");
    }
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mime });
}

function isUrlSafetyError(err) {
    return /url safety check/i.test(errorMessage(err));
}

async function uploadVideoForSameOriginFallback(blob) {
    if (!api?.fetchApi || !api?.apiURL) {
        throw new Error("same-origin preview fallback is unavailable");
    }
    const response = await api.fetchApi(PREVIEW_MEDIA_PATH, {
        method: "POST",
        headers: { "Content-Type": "video/mp4" },
        body: blob,
        cache: "no-store",
    });
    if (!response.ok) {
        throw new Error(`same-origin preview fallback upload failed (HTTP ${response.status})`);
    }
    let payload;
    try {
        payload = await response.json();
    } catch (_) {
        throw new Error("same-origin preview fallback returned invalid JSON");
    }
    const token = typeof payload?.token === "string" ? payload.token : "";
    if (!/^[0-9a-f]{32}$/i.test(token)) {
        throw new Error("same-origin preview fallback returned an invalid token");
    }
    return {
        token: token.toLowerCase(),
        url: api.apiURL(`${PREVIEW_MEDIA_PATH}/${encodeURIComponent(token)}`),
    };
}

function deleteVideoFallback(token) {
    if (!token || !api?.fetchApi) return;
    void api.fetchApi(`${PREVIEW_MEDIA_PATH}/${encodeURIComponent(token)}`, {
        method: "DELETE",
        cache: "no-store",
    }).catch(() => {});
}

function fmtDuration(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "—";
    if (seconds < 60) return `${seconds.toFixed(0)}s`;
    const m = Math.floor(seconds / 60);
    return `${m}m${String(Math.round(seconds - m * 60)).padStart(2, "0")}s`;
}

function fmtStepTime(ms) {
    if (!Number.isFinite(ms)) return "—";
    return ms >= 1000 ? `${(ms / 1000).toFixed(1)}s/it` : `${ms.toFixed(0)}ms/it`;
}

function fmtPlaybackTime(seconds) {
    if (!Number.isFinite(seconds) || seconds < 0) return "0:00";
    const whole = Math.floor(seconds);
    const minutes = Math.floor(whole / 60);
    return `${minutes}:${String(whole % 60).padStart(2, "0")}`;
}

function errorMessage(err) {
    if (err == null) return "unknown error";
    const message = typeof err === "string" ? err : err.message;
    const name = typeof err === "object" ? err.name : "";
    if (message && name && name !== "Error") return `${name}: ${message}`;
    return String(message || err);
}

function errorWithContext(message, cause) {
    const error = new Error(String(message));
    if (cause?.name && cause.name !== "Error") error.name = cause.name;
    error.cause = cause;
    return error;
}

api.addEventListener(EVENT, (e) => {
    const data = e.detail;
    if (!data || data.node_id == null) return;
    const node = findNodeByQualifiedId(app.graph, data.node_id);
    if (node?._mmh3Handler) node._mmh3Handler(data);
});

/** One source slot: transactional media swaps plus a controllable MP4/WebP player. */
function createSlot(stage) {
    const imgs = [el("img", "mmh3-media", stage), el("img", "mmh3-media", stage)];
    const videos = [el("video", "mmh3-media", stage), el("video", "mmh3-media", stage)];
    const canvas = el("canvas", "mmh3-media mmh3-canvas", stage);
    const canvasCtx = canvas.getContext?.("2d") || null;
    for (const i of imgs) {
        i.draggable = false;
        i.alt = "MiniMax H3 live preview";
    }
    for (const v of videos) {
        v.muted = true;
        v.loop = true;
        v.autoplay = false;
        v.playsInline = true;
        v.disablePictureInPicture = true;
        v.preload = "auto";
    }

    let imgIdx = 0;
    let vidIdx = 0;
    let visible = null;
    let visibleKind = null;
    let currentUrl = null;
    let currentObjectUrl = null;
    let currentMediaToken = null;
    let currentMeta = {};
    let active = false;
    let disposed = false;
    let generation = 0;
    let pendingUpdate = null;
    let processingUpdate = false;
    let decodedFrames = null;
    let frameDurations = [];
    let clipDurationMs = 0;
    let position = 0;
    let playingIntent = false;
    let playbackStart = null;
    let renderRaf = null;
    let lastDrawnFrame = null;
    let stateCallback = () => {};
    let errorCallback = () => {};
    let diagnosticCallback = () => {};

    const mediaElements = [...imgs, ...videos, canvas];

    function notify() {
        try { stateCallback(); } catch (_) { /* UI observers must not break playback */ }
    }

    function reportError(err, details = {}) {
        try {
            errorCallback(err instanceof Error ? err : new Error(String(err)), details);
        } catch (_) {}
    }

    function reportDiagnostic(details = {}) {
        try { diagnosticCallback(details); } catch (_) {}
    }

    function clamp01(value) {
        return Math.max(0, Math.min(1, Number.isFinite(value) ? value : 0));
    }

    function isAnimated() {
        if (visibleKind === "video") return true;
        if (visibleKind === "webp") return Boolean(decodedFrames?.length > 1);
        return Number(currentMeta.frame_count) > 1;
    }

    function durationMs() {
        if (visibleKind === "video" && Number.isFinite(visible?.duration) && visible.duration > 0) {
            return visible.duration * 1000;
        }
        if (visibleKind === "webp" && clipDurationMs > 0) return clipDurationMs;
        const value = Number(currentMeta.duration_ms);
        return Number.isFinite(value) && value > 0 ? value : 0;
    }

    function currentPosition() {
        if (visibleKind === "video" && visible && Number.isFinite(visible.duration) && visible.duration > 0) {
            return clamp01(visible.currentTime / visible.duration);
        }
        if (visibleKind === "webp" && playingIntent && active && clipDurationMs > 0 && playbackStart != null) {
            return clamp01(((performance.now() - playbackStart) % clipDurationMs) / clipDurationMs);
        }
        return clamp01(position);
    }

    function stopRenderLoop() {
        if (renderRaf != null) cancelAnimationFrame(renderRaf);
        renderRaf = null;
    }

    function tryPlay(video) {
        try {
            const result = video?.play?.();
            if (result?.catch) {
                result.catch(() => {
                    if (video === visible) {
                        playingIntent = false;
                        notify();
                    }
                });
            }
        } catch (_) {}
    }

    function frameAtPosition(pos) {
        if (!decodedFrames?.length) return null;
        const total = clipDurationMs || 1;
        let cursor = clamp01(pos) * total;
        for (let i = 0; i < decodedFrames.length; i++) {
            const span = frameDurations[i] || (total / decodedFrames.length);
            if (cursor < span || i === decodedFrames.length - 1) return decodedFrames[i];
            cursor -= span;
        }
        return decodedFrames[decodedFrames.length - 1];
    }

    function drawWebp() {
        if (visibleKind !== "webp" || !canvasCtx) return;
        const frame = frameAtPosition(currentPosition());
        if (!frame) return;
        if (frame === lastDrawnFrame
            && canvas.width === frame.displayWidth
            && canvas.height === frame.displayHeight) return;
        if (canvas.width !== frame.displayWidth || canvas.height !== frame.displayHeight) {
            canvas.width = frame.displayWidth;
            canvas.height = frame.displayHeight;
        }
        canvasCtx.drawImage(frame, 0, 0);
        lastDrawnFrame = frame;
        canvas.style.opacity = active ? "1" : "0";
    }

    function startRenderLoop() {
        if (renderRaf != null || !active || visibleKind !== "webp" || !playingIntent) return;
        const tick = () => {
            renderRaf = requestAnimationFrame(tick);
            drawWebp();
        };
        renderRaf = requestAnimationFrame(tick);
    }

    function closeFrames(frames) {
        for (const frame of frames || []) {
            try { frame.close?.(); } catch (_) {}
        }
    }

    async function decodeAnimatedWebp(blob, fallbackDurations) {
        if (typeof ImageDecoder === "undefined" || !canvasCtx) return null;
        let decoder = null;
        const frames = [];
        try {
            decoder = new ImageDecoder({ data: await blob.arrayBuffer(), type: "image/webp" });
            await decoder.completed;
            const track = decoder.tracks?.selectedTrack;
            if (!track || track.frameCount <= 1) return null;
            if (track.frameCount > MAX_PREVIEW_FRAMES) {
                throw new Error("animated preview has too many frames");
            }
            for (let i = 0; i < track.frameCount; i++) {
                const result = await decoder.decode({ frameIndex: i });
                if (!result?.image) throw new Error("animated WebP frame decode failed");
                frames.push(result.image);
            }
            const supplied = Array.isArray(fallbackDurations) ? fallbackDurations : [];
            const durations = frames.map((frame, i) => {
                const fromFrame = Number(frame.duration) / 1000;
                const fromPayload = Number(supplied[i]);
                return Math.max(1, Number.isFinite(fromFrame) && fromFrame > 0
                    ? fromFrame : (Number.isFinite(fromPayload) && fromPayload > 0
                        ? fromPayload : 1000 / 24));
            });
            return { frames, durations };
        } catch (err) {
            closeFrames(frames);
            return { error: err instanceof Error ? err : new Error(String(err)) };
        } finally {
            try { decoder?.close?.(); } catch (_) {}
        }
    }

    function waitForImage(image) {
        if (image.complete) {
            if (image.naturalWidth > 0) return Promise.resolve();
            return Promise.reject(new Error("preview image failed to decode"));
        }
        return new Promise((resolve, reject) => {
            let settled = false;
            const finish = (err) => {
                if (settled) return;
                settled = true;
                clearTimeout(timer);
                image.removeEventListener("load", onLoad);
                image.removeEventListener("error", onError);
                err ? reject(err) : resolve();
            };
            const onLoad = () => image.naturalWidth > 0
                ? finish() : finish(new Error("preview image has no pixels"));
            const onError = () => finish(new Error("preview image failed to decode"));
            const timer = setTimeout(() => finish(new Error("preview image decode timed out")), 3000);
            image.addEventListener("load", onLoad, { once: true });
            image.addEventListener("error", onError, { once: true });
        });
    }

    function waitForVideo(video) {
        if (video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0) {
            return Promise.resolve();
        }
        return new Promise((resolve, reject) => {
            let settled = false;
            const finish = (err) => {
                if (settled) return;
                settled = true;
                clearTimeout(timer);
                video.removeEventListener("loadeddata", onLoaded);
                video.removeEventListener("canplay", onLoaded);
                video.removeEventListener("error", onError);
                err ? reject(err) : resolve();
            };
            const onLoaded = () => video.videoWidth > 0 && video.videoHeight > 0
                ? finish() : undefined;
            const onError = () => {
                const mediaError = video.error;
                const code = Number(mediaError?.code);
                const nativeMessage = mediaError?.message ? `: ${mediaError.message}` : "";
                finish(new Error(
                    `preview video failed to decode${Number.isFinite(code) ? ` (MediaError ${code})` : ""}${nativeMessage}`,
                ));
            };
            const timer = setTimeout(() => finish(new Error("preview video decode timed out")), 3000);
            video.addEventListener("loadeddata", onLoaded, { once: true });
            video.addEventListener("canplay", onLoaded, { once: true });
            video.addEventListener("error", onError, { once: true });
        });
    }

    async function loadImage(image, decoderError = null) {
        let decodeError = decoderError;
        if (image.decode) {
            try { await image.decode(); } catch (err) { decodeError = err; }
        }
        try {
            await waitForImage(image);
        } catch (err) {
            if (!decodeError) throw err;
            throw errorWithContext(
                `${errorMessage(err)} (ImageDecoder: ${errorMessage(decodeError)})`, err,
            );
        }
    }

    function freezeImageToCanvas(image) {
        // Native animated-WebP playback cannot be paused consistently across browsers.
        // Capture its first decoded frame when possible so the compatibility fallback is
        // truly display-only and does not keep decoding while its source tab is hidden.
        if (!canvasCtx || !image?.naturalWidth || !image?.naturalHeight) return false;
        try {
            canvas.width = image.naturalWidth;
            canvas.height = image.naturalHeight;
            canvasCtx.drawImage(image, 0, 0);
            image.removeAttribute("src");
            return true;
        } catch (_) {
            return false;
        }
    }

    function show(elem) {
        for (const e of mediaElements) e.style.opacity = e === elem && active ? "1" : "0";
        visible = elem;
    }

    function promote(elem, kind, meta, nextUrl, nextObjectUrl, nextMediaToken,
                     frames, durations, preservedPos, preservedPlaying) {
        stopRenderLoop();
        if (visibleKind === "video") {
            for (const video of videos) video.pause();
        }
        closeFrames(decodedFrames);
        decodedFrames = frames || null;
        lastDrawnFrame = null;
        frameDurations = durations || [];
        clipDurationMs = frameDurations.reduce((sum, value) => sum + value, 0);
        visibleKind = kind;
        currentMeta = { ...meta };
        const previousObjectUrl = currentObjectUrl;
        const previousMediaToken = currentMediaToken;
        currentUrl = nextUrl;
        currentObjectUrl = nextObjectUrl;
        currentMediaToken = nextMediaToken;
        visible = elem;
        position = clamp01(preservedPos);
        playingIntent = Boolean(preservedPlaying && isAnimated() && (kind !== "image" || frames == null));
        if (kind === "video") {
            const duration = durationMs();
            if (duration > 0) {
                try { elem.currentTime = position * duration / 1000; } catch (_) {}
            }
        }
        show(elem);
        if (previousObjectUrl) URL.revokeObjectURL(previousObjectUrl);
        deleteVideoFallback(previousMediaToken);
        if (active && playingIntent) {
            if (kind === "video") tryPlay(elem);
            else if (kind === "webp") {
                playbackStart = performance.now() - position * clipDurationMs;
                startRenderLoop();
            }
        } else if (kind === "webp") {
            drawWebp();
        }
        notify();
    }

    async function processPending() {
        if (processingUpdate) return;
        processingUpdate = true;
        try {
            while (pendingUpdate) {
                const job = pendingUpdate;
                pendingUpdate = null;
                if (disposed || job.generation !== generation) {
                    job.resolve(false);
                    continue;
                }
                let nextUrl = null;
                let nextObjectUrl = null;
                let nextMediaToken = null;
                let fallbackAttempted = false;
                let candidateFrames = null;
                try {
                    const blob = b64ToBlob(job.b64, job.mime);
                    nextObjectUrl = URL.createObjectURL(blob);
                    nextUrl = nextObjectUrl;
                    const preservedPos = currentUrl ? currentPosition() : 0;
                    const preservedPlaying = currentUrl ? playingIntent : true;
                    let elem;
                    let kind;
                    let durations = [];
                    if (job.mime === "video/mp4") {
                        elem = videos[vidIdx ^= 1];
                        elem.pause();
                        elem.src = nextUrl;
                        elem.load();
                        try {
                            await waitForVideo(elem);
                        } catch (err) {
                            if (!isUrlSafetyError(err)) throw err;
                            fallbackAttempted = true;
                            reportDiagnostic({ status: "attempted", reason: errorMessage(err) });
                            let fallback;
                            try {
                                fallback = await uploadVideoForSameOriginFallback(blob);
                            } catch (fallbackErr) {
                                reportDiagnostic({
                                    status: "failed",
                                    reason: errorMessage(fallbackErr),
                                });
                                throw errorWithContext(
                                    `${errorMessage(err)}; same-origin fallback failed: ${errorMessage(fallbackErr)}`,
                                    fallbackErr,
                                );
                            }
                            reportDiagnostic({ status: "succeeded" });
                            nextMediaToken = fallback.token;
                            URL.revokeObjectURL(nextObjectUrl);
                            nextObjectUrl = null;
                            nextUrl = fallback.url;
                            elem.src = nextUrl;
                            elem.load();
                            await waitForVideo(elem);
                        }
                        kind = "video";
                    } else if (job.mime === "image/webp") {
                        const decoded = await decodeAnimatedWebp(blob, job.meta?.frame_durations_ms);
                        const animatedDecodeError = decoded?.error || null;
                        if (decoded?.frames) {
                            candidateFrames = decoded.frames;
                            durations = decoded.durations;
                            elem = canvas;
                            kind = "webp";
                        } else {
                            const image = imgs[imgIdx ^= 1];
                            image.src = nextUrl;
                            await loadImage(image, animatedDecodeError);
                            if (freezeImageToCanvas(image)) {
                                elem = canvas;
                                kind = "image";
                            } else {
                                elem = image;
                                kind = "image";
                            }
                        }
                    } else {
                        elem = imgs[imgIdx ^= 1];
                        elem.src = nextUrl;
                        await loadImage(elem);
                        kind = "image";
                    }
                    if (disposed || job.generation !== generation) {
                        closeFrames(candidateFrames);
                        if (nextObjectUrl) URL.revokeObjectURL(nextObjectUrl);
                        deleteVideoFallback(nextMediaToken);
                        job.resolve(false);
                        continue;
                    }
                    promote(elem, kind, job.meta || {}, nextUrl, nextObjectUrl, nextMediaToken,
                        candidateFrames, durations, preservedPos, preservedPlaying);
                    job.resolve(true);
                } catch (err) {
                    closeFrames(candidateFrames);
                    if (nextObjectUrl) URL.revokeObjectURL(nextObjectUrl);
                    deleteVideoFallback(nextMediaToken);
                    reportError(err, {
                        stage: "render",
                        mime: job.mime,
                        meta: job.meta || {},
                        fallback: fallbackAttempted,
                    });
                    job.resolve(false);
                }
            }
        } finally {
            processingUpdate = false;
            if (pendingUpdate) void processPending();
        }
    }

    function update(b64, mime, meta = {}) {
        const mine = ++generation;
        if (pendingUpdate) pendingUpdate.resolve(false);
        const promise = new Promise(resolve => {
            pendingUpdate = { b64, mime, meta, generation: mine, resolve };
        });
        void processPending();
        return promise;
    }

    function cancelPending() {
        if (pendingUpdate) pendingUpdate.resolve(false);
        pendingUpdate = null;
    }

    function play() {
        if (!isAnimated() || !controlsAvailable()) return false;
        const nextPosition = currentPosition();
        playingIntent = true;
        if (visibleKind === "video") tryPlay(visible);
        else if (visibleKind === "webp") {
            position = nextPosition;
            playbackStart = performance.now() - nextPosition * clipDurationMs;
            startRenderLoop();
        }
        notify();
        return true;
    }

    function pause() {
        if (!controlsAvailable()) return false;
        position = currentPosition();
        playingIntent = false;
        if (visibleKind === "video") visible?.pause?.();
        else if (visibleKind === "webp") {
            stopRenderLoop();
            drawWebp();
        }
        notify();
        return true;
    }

    function toggle() {
        return playingIntent ? pause() : play();
    }

    function seek(value) {
        if (!controlsAvailable()) return false;
        const next = clamp01(Number(value));
        position = next;
        const duration = durationMs();
        if (visibleKind === "video" && visible && duration > 0) {
            try { visible.currentTime = next * duration / 1000; } catch (_) {}
        } else if (visibleKind === "webp" && clipDurationMs > 0) {
            playbackStart = performance.now() - next * clipDurationMs;
            drawWebp();
        }
        notify();
        return true;
    }

    function controlsAvailable() {
        return Boolean(isAnimated() && (visibleKind === "video" || visibleKind === "webp") && durationMs() > 0);
    }

    for (const video of videos) {
        video.addEventListener("ended", () => {
            if (video === visible && !video.loop) {
                playingIntent = false;
                position = 1;
                notify();
            }
        });
    }

    return {
        setActive(value) {
            if (!value) position = currentPosition();
            active = Boolean(value);
            if (!active) {
                stopRenderLoop();
                for (const video of videos) video.pause();
            }
            for (const e of mediaElements) e.style.opacity = (active && e === visible) ? "1" : "0";
            if (active && visibleKind === "video" && playingIntent) tryPlay(visible);
            if (active && visibleKind === "webp") {
                drawWebp();
                if (playingIntent) {
                    playbackStart = performance.now() - position * clipDurationMs;
                    startRenderLoop();
                }
            }
            notify();
        },
        setStateCallback(callback) { stateCallback = callback || (() => {}); },
        setErrorCallback(callback) { errorCallback = callback || (() => {}); },
        setDiagnosticCallback(callback) { diagnosticCallback = callback || (() => {}); },
        update,
        play,
        pause,
        toggle,
        seek,
        getMeta() {
            const meta = { ...currentMeta, mime: visibleKind === "video" ? "video/mp4" : currentMeta.mime };
            if (!meta.w && visible?.videoWidth) meta.w = visible.videoWidth;
            if (!meta.h && visible?.videoHeight) meta.h = visible.videoHeight;
            return meta;
        },
        getPlaybackState() {
            const duration = durationMs();
            return {
                animated: isAnimated(),
                controls: controlsAvailable(),
                playing: Boolean(active && playingIntent && controlsAvailable()),
                position: currentPosition(),
                duration: duration / 1000,
                frameCount: Number(currentMeta.frame_count) || (decodedFrames?.length || 0),
            };
        },
        reset() {
            disposed = false;
            generation++;
            cancelPending();
            stopRenderLoop();
            for (const video of videos) video.pause();
            closeFrames(decodedFrames);
            decodedFrames = null;
            lastDrawnFrame = null;
            frameDurations = [];
            clipDurationMs = 0;
            if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
            deleteVideoFallback(currentMediaToken);
            currentUrl = null;
            currentObjectUrl = null;
            currentMediaToken = null;
            currentMeta = {};
            visible = null;
            visibleKind = null;
            position = 0;
            playingIntent = false;
            for (const e of mediaElements) {
                e.style.opacity = "0";
                e.removeAttribute("src");
                if (e.tagName === "VIDEO") e.load();
            }
            notify();
        },
        dispose() {
            disposed = true;
            generation++;
            cancelPending();
            stopRenderLoop();
            for (const video of videos) video.pause();
            closeFrames(decodedFrames);
            decodedFrames = null;
            lastDrawnFrame = null;
            if (currentObjectUrl) URL.revokeObjectURL(currentObjectUrl);
            deleteVideoFallback(currentMediaToken);
            currentUrl = null;
            currentObjectUrl = null;
            currentMediaToken = null;
            visible = null;
            visibleKind = null;
            for (const e of mediaElements) {
                e.style.opacity = "0";
                e.removeAttribute("src");
                if (e.tagName === "VIDEO") e.load();
            }
            stateCallback = () => {};
            errorCallback = () => {};
            diagnosticCallback = () => {};
        },
    };
}

let executionActive = false;
let frontendPromptId = null;
const previewNodes = new Set();
api.addEventListener("execution_start", (e) => {
    const data = e.detail || {};
    executionActive = true;
    frontendPromptId = data.prompt_id == null ? null : String(data.prompt_id);
});
for (const terminalEvent of ["execution_success", "execution_error", "execution_interrupted"]) {
    api.addEventListener(terminalEvent, (e) => {
        const promptId = e.detail?.prompt_id == null ? null : String(e.detail.prompt_id);
        executionActive = false;
        // Drop the prompt gate so a later run can claim the panel even if its
        // boundary-0 event raced ahead of execution_start, or if the browser
        // missed the next start event entirely.
        frontendPromptId = null;
        for (const node of previewNodes) node._mmh3ExecutionEnded?.(promptId);
    });
}

app.registerExtension({
    name: "MiniMaxH3.LivePreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            ensureStyles();
            const node = this;
            previewNodes.add(node);

            const root = el("div", "mmh3-root");
            root.dataset.crisp = "1";

            const header = el("div", "mmh3-header", root);
            const title = el("span", "mmh3-title", header);
            title.textContent = "MiniMax H3 Live Preview";
            const tabs = el("div", "mmh3-tabs", header);
            tabs.setAttribute("role", "tablist");
            tabs.setAttribute("aria-label", "Preview source");
            const tabEls = {};
            for (const s of SOURCES) {
                tabEls[s] = el("button", "mmh3-tab disabled", tabs);
                tabEls[s].type = "button";
                tabEls[s].setAttribute("role", "tab");
                tabEls[s].setAttribute("aria-selected", "false");
                tabEls[s].setAttribute("aria-disabled", "true");
                tabEls[s].title = `${s} preview source`;
                tabEls[s].disabled = true;
                tabEls[s].textContent = s;
            }

            const stage = el("div", "mmh3-stage", root);
            const placeholder = el("div", "mmh3-placeholder", stage);
            placeholder.textContent = "waiting for sampler…";
            const slots = {};
            for (const s of SOURCES) slots[s] = createSlot(stage);
            const progress = el("div", "mmh3-progress", stage);
            progress.setAttribute("role", "progressbar");
            progress.setAttribute("aria-label", "Sampler progress");
            progress.setAttribute("aria-valuemin", "0");
            progress.setAttribute("aria-valuemax", "1");
            const progressFill = el("div", "mmh3-progress-fill", progress);
            const status = el("div", "mmh3-status", stage);
            status.setAttribute("role", "status");
            status.setAttribute("aria-live", "polite");
            status.hidden = true;

            const controls = el("div", "mmh3-controls", stage);
            const playButton = el("button", "mmh3-play", controls);
            playButton.type = "button";
            playButton.setAttribute("aria-label", "Play preview");
            playButton.title = "Play or pause preview";
            playButton.textContent = "▶";
            const scrubber = el("input", "mmh3-scrubber", controls);
            scrubber.type = "range";
            scrubber.min = "0";
            scrubber.max = "1000";
            scrubber.step = "1";
            scrubber.value = "0";
            scrubber.setAttribute("aria-label", "Preview position");
            scrubber.title = "Scrub preview";
            const playbackTime = el("span", "mmh3-playback-time", controls);
            playbackTime.textContent = "0:00 / 0:00";

            const footer = el("div", "mmh3-footer", root);
            const stepEl = el("span", "mmh3-step", footer);
            const sigmaEl = el("span", "mmh3-sigma", footer);
            const rateEl = el("span", "mmh3-rate", footer);
            el("span", "mmh3-spacer", footer);
            const sizeEl = el("span", "mmh3-size", footer);
            const clipEl = el("span", "mmh3-clip", footer);
            const badge = el("span", "mmh3-badge", footer);
            stepEl.textContent = "idle";
            badge.textContent = "—";

            let current = null;
            const available = { latent: false, tae: false, vae: false };
            let userPinned = false;   // user clicked a tab: stop auto-switching
            let controlsTimer = null;
            let verboseEnabled = false;
            let runFinished = false;

            function stopControlsRefresh() {
                if (controlsTimer != null) clearTimeout(controlsTimer);
                controlsTimer = null;
            }

            function reportStatus(message, isError = false) {
                status.textContent = message;
                status.hidden = !message;
                status.dataset.kind = isError ? "error" : "info";
            }

            function selectedState() {
                return current && slots[current] ? slots[current].getPlaybackState() : null;
            }

            function refreshSelectedFooter() {
                const slot = current && slots[current];
                const meta = slot?.getMeta?.() || {};
                const state = slot?.getPlaybackState?.() || {};
                sizeEl.textContent = meta.w && meta.h ? `${meta.w}×${meta.h}` : "";
                const count = Number(meta.frame_count) || Number(state.frameCount) || 0;
                const duration = Number(meta.duration_ms) > 0
                    ? Number(meta.duration_ms) / 1000 : Number(state.duration) || 0;
                clipEl.textContent = count > 1
                    ? `${count}f · ${fmtPlaybackTime(duration)}` : "";
            }

            function syncPlaybackControls() {
                const state = selectedState();
                if (!state || !state.controls) {
                    stopControlsRefresh();
                    controls.hidden = true;
                    controls.dataset.playing = "0";
                    return;
                }
                controls.hidden = false;
                controls.dataset.playing = state.playing ? "1" : "0";
                playButton.textContent = state.playing ? "❚❚" : "▶";
                playButton.setAttribute("aria-label", state.playing ? "Pause preview" : "Play preview");
                scrubber.value = String(Math.round(state.position * 1000));
                playbackTime.textContent = `${fmtPlaybackTime(state.position * state.duration)} / ${fmtPlaybackTime(state.duration)}`;
                scrubber.setAttribute("aria-valuetext", playbackTime.textContent);
                refreshSelectedFooter();
                if (state.playing && controlsTimer == null) {
                    controlsTimer = setTimeout(() => {
                        controlsTimer = null;
                        syncPlaybackControls();
                    }, CONTROLS_REFRESH_MS);
                } else if (!state.playing) stopControlsRefresh();
            }

            function select(which) {
                if (!available[which]) return false;
                current = which;
                for (const s of SOURCES) {
                    slots[s].setActive(s === which);
                    tabEls[s].classList.toggle("active", s === which);
                    tabEls[s].setAttribute("aria-selected", s === which ? "true" : "false");
                }
                // Only the latent stream is block-scaled; TAE and VAE frames are true resolution.
                root.dataset.crisp = which === "latent" ? "1" : "0";
                badge.dataset.source = which;
                badge.textContent = which.toUpperCase();
                refreshSelectedFooter();
                syncPlaybackControls();
                return true;
            }

            for (const s of SOURCES) {
                tabEls[s].addEventListener("click", () => {
                    if (!available[s]) return;
                    userPinned = true;
                    select(s);
                });
            }

            controls.addEventListener("pointerdown", (event) => event.stopPropagation());
            playButton.addEventListener("click", (event) => {
                event.stopPropagation();
                if (current) slots[current].toggle();
            });
            scrubber.addEventListener("input", (event) => {
                event.stopPropagation();
                if (current) slots[current].seek(Number(scrubber.value) / 1000);
            });
            stage.addEventListener("click", (event) => {
                if (controls.contains(event.target)) return;
                const state = selectedState();
                if (state?.controls && current) {
                    event.stopPropagation();
                    slots[current].toggle();
                }
            });

            for (const s of SOURCES) {
                slots[s].setStateCallback(() => {
                    if (s === current) {
                        refreshSelectedFooter();
                        syncPlaybackControls();
                    }
                });
                slots[s].setErrorCallback((err, details = {}) => {
                    if (s === current || !current) {
                        const reason = errorMessage(err);
                        const stage = details.stage || "render";
                        const suffix = verboseEnabled
                            ? `: ${reason}`
                            : `; enable verbose for details`;
                        reportStatus(current
                            ? `preview failed during ${stage} — showing the previous frame${suffix}`
                            : `preview failed during ${stage}${suffix}`, true);
                        const diagnostic = {
                            source: s,
                            stage,
                            message: reason,
                            stack: verboseEnabled ? err?.stack : undefined,
                            mime: details.mime,
                            meta: details.meta,
                            fallback: details.fallback,
                            run_id: activeRunId,
                            prompt_id: activePromptId,
                        };
                        if (verboseEnabled) console.error("[MiniMaxH3Preview] preview failure", diagnostic, err);
                        else console.warn("[MiniMaxH3Preview] preview failure", diagnostic);
                    }
                });
                slots[s].setDiagnosticCallback((diagnostic = {}) => {
                    if (verboseEnabled) {
                        console.info("[MiniMaxH3Preview] video URL-safety fallback", {
                            ...diagnostic,
                            source: s,
                            run_id: activeRunId,
                            prompt_id: activePromptId,
                        });
                    }
                });
            }

            const onVisibilityChange = () => {
                if (current && document.hidden) slots[current].setActive(false);
                else if (current) slots[current].setActive(true);
            };
            document.addEventListener("visibilitychange", onVisibilityChange);
            const rootObserver = typeof IntersectionObserver === "function"
                ? new IntersectionObserver((entries) => {
                    const isVisible = Boolean(entries[0]?.isIntersecting);
                    if (!current) return;
                    if (isVisible && !document.hidden) slots[current].setActive(true);
                    else slots[current].setActive(false);
                }) : null;
            rootObserver?.observe(root);

            let activeRunId = null;
            let activePromptId = null;
            let lastProgressStep = -1;
            const latestMediaStep = { latent: -1, tae: -1, vae: -1 };

            function beginRun(runId, promptId, verbose = false) {
                activeRunId = runId;
                activePromptId = promptId;
                verboseEnabled = Boolean(verbose);
                runFinished = false;
                lastProgressStep = -1;
                userPinned = false;
                current = null;
                for (const s of SOURCES) available[s] = false;
                for (const s of SOURCES) latestMediaStep[s] = -1;
                for (const s of SOURCES) {
                    tabEls[s].classList.add("disabled");
                    tabEls[s].disabled = true;
                    tabEls[s].classList.remove("active");
                    tabEls[s].setAttribute("aria-selected", "false");
                    tabEls[s].setAttribute("aria-disabled", "true");
                }
                for (const s of SOURCES) slots[s].reset();
                root.dataset.crisp = "1";
                placeholder.style.display = "flex";
                placeholder.textContent = "waiting for sampler…";
                reportStatus("");
                progressFill.style.width = "0%";
                progress.setAttribute("aria-valuenow", "0");
                stepEl.textContent = "idle";
                sigmaEl.textContent = "";
                rateEl.textContent = "";
                sizeEl.textContent = "";
                clipEl.textContent = "";
                badge.textContent = "—";
                badge.removeAttribute("data-source");
                syncPlaybackControls();
            }

            node._mmh3Handler = (data) => {
                // Prompt IDs reject late frames from a prior execution. The run ID
                // additionally rejects late worker frames within the same prompt.
                const runId = data.run_id == null ? null : String(data.run_id);
                const promptId = data.prompt_id == null ? null : String(data.prompt_id);
                if (frontendPromptId && promptId && promptId !== frontendPromptId) return;

                // Prefer boundary-0, but also accept the first frame of a new run_id
                // once a fresh prompt is underway. Missing the noise preview (or a
                // page refresh mid-queue) used to leave the panel stuck on the prior
                // generation forever because non-zero steps were refused.
                const isFreshPrompt = !activePromptId || (promptId && promptId !== activePromptId);
                const isNewRun = !!(runId && runId !== activeRunId);
                const startsRun = isNewRun && (
                    data.step === 0
                    || (isFreshPrompt && Number.isFinite(data.step))
                    || (!activeRunId && Number.isFinite(data.step))
                );
                if (startsRun) {
                    if (executionActive && frontendPromptId && promptId && promptId !== frontendPromptId) return;
                    beginRun(runId, promptId, data.verbose);
                } else {
                    if (activePromptId && promptId && promptId !== activePromptId) return;
                    if (runId && activeRunId && runId !== activeRunId) return;
                    if (runId && !activeRunId && data.step !== 0) return;
                    if (runId && !activeRunId) activeRunId = runId;
                    if (promptId && !activePromptId) activePromptId = promptId;
                }
                if (typeof data.verbose === "boolean") verboseEnabled = data.verbose;
                if (data.disabled) {
                    reportStatus("not a MiniMax H3 model", true);
                    placeholder.style.display = "flex";
                    placeholder.textContent = "MiniMax H3 model required";
                    return;
                }
                const stepIsFresh = Number.isFinite(data.step) && data.step >= lastProgressStep;
                if (stepIsFresh) {
                    lastProgressStep = data.step;
                    if (Number.isFinite(data.total) && data.total > 0) {
                        const fraction = Math.max(0, Math.min(1, data.step / data.total));
                        progressFill.style.width = `${fraction * 100}%`;
                        progress.setAttribute("aria-valuenow", String(fraction));
                        progress.setAttribute("aria-valuemax", "1");
                        progress.setAttribute("aria-valuetext", `step ${data.step} of ${data.total}`);
                        stepEl.textContent = `step ${data.step}/${data.total}`;
                    }
                    if (Number.isFinite(data.sigma)) sigmaEl.textContent = `σ ${data.sigma.toFixed(3)}`;
                    if (Number.isFinite(data.avg_step_ms)) {
                        const eta = Number.isFinite(data.total) && Number.isFinite(data.step)
                            ? fmtDuration(Math.max(0, data.total - data.step) * data.avg_step_ms / 1000)
                            : "—";
                        rateEl.textContent = `${fmtStepTime(data.avg_step_ms)} · eta ${eta}`;
                    }
                }

                if (data.error) {
                    const error = data.error;
                    const stage = error.stage || "preview";
                    const reason = error.message || error.type || "unknown error";
                    const suffix = verboseEnabled
                        ? `: ${error.type ? `${error.type}: ` : ""}${reason}`
                        : `; enable verbose for details`;
                    reportStatus(`preview failed during ${stage}${suffix}`, true);
                    const diagnostic = {
                        ...error,
                        run_id: activeRunId,
                        prompt_id: activePromptId,
                        node_id: data.node_id,
                    };
                    if (verboseEnabled) console.error("[MiniMaxH3Preview] backend preview failure", diagnostic);
                    else console.warn("[MiniMaxH3Preview] backend preview failure", diagnostic);
                    placeholder.style.display = "flex";
                    placeholder.textContent = `preview failed during ${stage}`;
                    return;
                }

                if (!data.image) return;
                const source = SOURCES.includes(data.source) ? data.source : "latent";
                const mediaStep = Number.isFinite(data.step) ? data.step : null;
                if (mediaStep != null && mediaStep < latestMediaStep[source]) return;
                if (mediaStep != null) latestMediaStep[source] = mediaStep;
                const meta = {
                    w: data.w, h: data.h, mime: data.mime || "image/jpeg",
                    frame_count: data.frame_count, duration_ms: data.duration_ms,
                    frame_durations_ms: data.frame_durations_ms,
                };
                slots[source].update(data.image, meta.mime, meta).then((shown) => {
                    if (!shown) return;
                    if (!available[source]) {
                        available[source] = true;
                        tabEls[source].classList.remove("disabled");
                        tabEls[source].disabled = false;
                        tabEls[source].setAttribute("aria-disabled", "false");
                    }
                    placeholder.style.display = "none";
                    reportStatus("");
                    // A truer frame is the more interesting one -- surface it once, then leave
                    // the choice to the user. Frames for inactive slots stay hidden.
                    if (!current) {
                        select(source);
                    } else if (!userPinned && SOURCES.indexOf(source) > SOURCES.indexOf(current)) {
                        select(source);
                    } else if (source === current) {
                        refreshSelectedFooter();
                        syncPlaybackControls();
                    }
                    if (runFinished) slots[source].pause();
                });
            };

            node._mmh3ExecutionEnded = (promptId) => {
                if (promptId && activePromptId && promptId !== activePromptId) return;
                runFinished = true;
                for (const s of SOURCES) slots[s].pause();
                syncPlaybackControls();
            };

            chainCallback(node, "onRemoved", function () {
                document.removeEventListener("visibilitychange", onVisibilityChange);
                rootObserver?.disconnect?.();
                stopControlsRefresh();
                for (const s of SOURCES) slots[s].dispose();
                previewNodes.delete(node);
                delete node._mmh3Handler;
                delete node._mmh3ExecutionEnded;
            });

            const widget = node.addDOMWidget("preview", "mmh3_preview", root, {
                serialize: false,
                hideOnZoom: false,
                getMinHeight: () => 380,
            });
            widget.serialize = false;
            widget.computeLayoutSize = () => ({
                minWidth: 300,
                minHeight: 380,
                maxWidth: 1000000,
                maxHeight: 1000000,
            });
            if (node.size[1] < 380) node.setSize([Math.max(node.size[0], 300), 380]);
        });
    },
});
