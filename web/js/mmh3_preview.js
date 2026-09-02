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

const MAX_PREVIEW_B64_BYTES = 32 * 1024 * 1024;

function b64ToBlob(b64, mime) {
    if (typeof b64 !== "string" || b64.length > MAX_PREVIEW_B64_BYTES) {
        throw new Error("preview payload is missing or too large");
    }
    const bin = atob(b64);
    const arr = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
    return new Blob([arr], { type: mime });
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

api.addEventListener(EVENT, (e) => {
    const data = e.detail;
    if (!data || data.node_id == null) return;
    const node = findNodeByQualifiedId(app.graph, data.node_id);
    if (node?._mmh3Handler) node._mmh3Handler(data);
});

/** One display slot: double-buffered <img> (jpeg / animated webp) + <video> (mp4). */
function createSlot(stage) {
    const imgs = [el("img", "mmh3-media", stage), el("img", "mmh3-media", stage)];
    const videos = [el("video", "mmh3-media", stage), el("video", "mmh3-media", stage)];
    for (const i of imgs) i.draggable = false;
    for (const v of videos) {
        v.muted = true;
        v.loop = true;
        v.autoplay = true;
        v.playsInline = true;
        v.disablePictureInPicture = true;
    }

    let imgIdx = 0;
    let vidIdx = 0;
    let visible = null;
    let url = null;
    let active = false;
    let generation = 0;
    let disposed = false;
    let pendingUpdate = null;
    let processingUpdate = false;

    function show(elem) {
        if (visible && visible !== elem) visible.style.opacity = "0";
        visible = elem;
        if (active) elem.style.opacity = "1";
    }

    function waitForVideo(v) {
        if (v.readyState >= 2) return Promise.resolve();
        return new Promise(resolve => {
            let doneCalled = false;
            const done = () => {
                if (doneCalled) return;
                doneCalled = true;
                clearTimeout(timer);
                v.removeEventListener("loadeddata", done);
                v.removeEventListener("error", done);
                resolve();
            };
            const timer = setTimeout(done, 3000);
            v.addEventListener("loadeddata", done, { once: true });
            v.addEventListener("error", done, { once: true });
        });
    }

    // Serialize swaps and coalesce work that arrives while the browser is decoding.
    // This prevents an older frame from replacing a newer one or retaining every
    // in-flight base64 payload during a slow video decode.
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
                try {
                    const blob = b64ToBlob(job.b64, job.mime);
                    nextUrl = URL.createObjectURL(blob);
                    if (job.mime === "video/mp4") {
                        const v = videos[vidIdx ^= 1];
                        v.src = nextUrl;
                        await v.play().catch(() => {});
                        await waitForVideo(v);
                        if (disposed || job.generation !== generation) {
                            URL.revokeObjectURL(nextUrl);
                            job.resolve(false);
                            continue;
                        }
                        show(v);
                    } else {
                        const i = imgs[imgIdx ^= 1];
                        i.src = nextUrl;
                        if (i.decode) await i.decode().catch(() => {});
                        if (disposed || job.generation !== generation) {
                            URL.revokeObjectURL(nextUrl);
                            job.resolve(false);
                            continue;
                        }
                        show(i);
                    }
                    const prevUrl = url;
                    url = nextUrl;
                    if (prevUrl) URL.revokeObjectURL(prevUrl);
                    job.resolve(true);
                } catch (err) {
                    if (nextUrl) URL.revokeObjectURL(nextUrl);
                    job.reject(err);
                }
            }
        } finally {
            processingUpdate = false;
            if (pendingUpdate) void processPending();
        }
    }

    function update(b64, mime) {
        const mine = ++generation;
        if (pendingUpdate) pendingUpdate.resolve(false);
        const promise = new Promise((resolve, reject) => {
            pendingUpdate = { b64, mime, generation: mine, resolve, reject };
        });
        void processPending();
        return promise;
    }

    function cancelPending() {
        if (pendingUpdate) pendingUpdate.resolve(false);
        pendingUpdate = null;
    }

    return {
        get element() { return visible; },
        setActive(v) {
            active = v;
            for (const e of [...imgs, ...videos]) {
                e.style.opacity = (v && e === visible) ? "1" : "0";
            }
        },
        update,
        reset() {
            disposed = false;
            generation++;
            cancelPending();
            if (url) URL.revokeObjectURL(url);
            url = null;
            visible = null;
            for (const e of [...imgs, ...videos]) {
                e.style.opacity = "0";
                e.removeAttribute("src");
                if (e.tagName === "VIDEO") e.load();
            }
        },
        dispose() {
            disposed = true;
            generation++;
            cancelPending();
            if (url) URL.revokeObjectURL(url);
            url = null;
            visible = null;
            for (const e of [...imgs, ...videos]) {
                e.style.opacity = "0";
                if (e.tagName === "VIDEO") e.pause();
                e.removeAttribute("src");
                if (e.tagName === "VIDEO") e.load();
            }
        },
    };
}

let executionActive = false;
let frontendPromptId = null;
api.addEventListener("execution_start", (e) => {
    const data = e.detail || {};
    executionActive = true;
    frontendPromptId = data.prompt_id == null ? null : String(data.prompt_id);
});
for (const terminalEvent of ["execution_success", "execution_error", "execution_interrupted"]) {
    api.addEventListener(terminalEvent, () => {
        executionActive = false;
    });
}

app.registerExtension({
    name: "MiniMaxH3.LivePreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_ID) return;

        chainCallback(nodeType.prototype, "onNodeCreated", function () {
            ensureStyles();
            const node = this;

            const root = el("div", "mmh3-root");
            root.dataset.crisp = "1";

            const header = el("div", "mmh3-header", root);
            const title = el("span", "mmh3-title", header);
            title.textContent = "MiniMax H3 Live Preview";
            const tabs = el("div", "mmh3-tabs", header);
            const tabEls = {};
            for (const s of SOURCES) {
                // Only "latent" is available before anything has arrived; the others are the
                // opt-in streams and unlock as soon as they produce a frame.
                tabEls[s] = el("div", `mmh3-tab ${s === "latent" ? "active" : "disabled"}`, tabs);
                tabEls[s].textContent = s;
            }

            const stage = el("div", "mmh3-stage", root);
            const placeholder = el("div", "mmh3-placeholder", stage);
            placeholder.textContent = "waiting for sampler…";
            const slots = {};
            for (const s of SOURCES) slots[s] = createSlot(stage);
            const progress = el("div", "mmh3-progress", stage);
            const progressFill = el("div", "mmh3-progress-fill", progress);

            const footer = el("div", "mmh3-footer", root);
            const stepEl = el("span", null, footer);
            const sigmaEl = el("span", null, footer);
            const rateEl = el("span", null, footer);
            el("span", "mmh3-spacer", footer);
            const sizeEl = el("span", null, footer);
            const badge = el("span", "mmh3-badge", footer);
            stepEl.textContent = "idle";
            badge.textContent = "—";

            let current = "latent";
            const available = { latent: true, tae: false, vae: false };
            let userPinned = false;   // user clicked a tab: stop auto-switching

            function select(which) {
                if (!available[which]) return;
                current = which;
                for (const s of SOURCES) {
                    slots[s].setActive(s === which);
                    tabEls[s].classList.toggle("active", s === which);
                }
                // Only the latent stream is block-scaled; TAE and VAE frames are true resolution.
                root.dataset.crisp = which === "latent" ? "1" : "0";
                badge.dataset.source = which;
                badge.textContent = which.toUpperCase();
            }

            for (const s of SOURCES) {
                // Clicking a stream that has produced nothing yet is a no-op, not a pin --
                // otherwise it would silently disable auto-switching.
                tabEls[s].addEventListener("click", () => {
                    if (!available[s]) return;
                    userPinned = true;
                    select(s);
                });
            }

            let activeRunId = null;
            let activePromptId = null;
            let lastProgressStep = -1;

            node._mmh3Handler = (data) => {
                // Prompt IDs reject late frames from a prior execution. The run ID
                // additionally rejects late worker frames within the same prompt.
                const runId = data.run_id == null ? null : String(data.run_id);
                const promptId = data.prompt_id == null ? null : String(data.prompt_id);
                if (frontendPromptId && promptId && promptId !== frontendPromptId) return;
                if (activePromptId && promptId && promptId !== activePromptId) return;

                // Boundary-0 is the only event allowed to establish a new run. Do not
                // let a late non-boundary frame claim a node after a page refresh.
                const startsRun = data.step === 0 && runId && runId !== activeRunId;
                if (startsRun) {
                    if (executionActive && frontendPromptId && promptId && promptId !== frontendPromptId) return;
                    activeRunId = runId;
                    activePromptId = promptId;
                    lastProgressStep = -1;
                    userPinned = false;
                    current = "latent";
                    for (const s of SOURCES) slots[s].reset();
                    select("latent");
                    placeholder.style.display = "flex";
                } else {
                    if (runId && activeRunId && runId !== activeRunId) return;
                    if (runId && !activeRunId && data.step !== 0) return;
                    if (runId && !activeRunId) activeRunId = runId;
                }
                if (Number.isFinite(data.step) && data.step >= lastProgressStep) {
                    lastProgressStep = data.step;
                } else if (Number.isFinite(data.step)) {
                    // A slow VAE job may legitimately trail the cheap stream. Keep
                    // its media, but do not move the shared progress display backward.
                    data = { ...data, step: undefined };
                }
                if (Number.isFinite(data.total) && data.total > 0 && Number.isFinite(data.step)) {
                    progressFill.style.width = `${Math.min(100, (data.step / data.total) * 100)}%`;
                    stepEl.textContent = `step ${data.step}/${data.total}`;
                }
                if (Number.isFinite(data.sigma)) sigmaEl.textContent = `σ ${data.sigma.toFixed(3)}`;
                if (Number.isFinite(data.avg_step_ms)) {
                    const eta = Number.isFinite(data.total) && Number.isFinite(data.step)
                        ? fmtDuration(Math.max(0, data.total - data.step) * data.avg_step_ms / 1000)
                        : "—";
                    rateEl.textContent = `${fmtStepTime(data.avg_step_ms)} · eta ${eta}`;
                }
                if (data.w && data.h) sizeEl.textContent = `${data.w}×${data.h}`;

                if (!data.image) return;
                const source = SOURCES.includes(data.source) ? data.source : "latent";
                if (!available[source]) {
                    available[source] = true;
                    tabEls[source].classList.remove("disabled");
                }
                slots[source].update(data.image, data.mime || "image/jpeg").then((shown) => {
                    if (!shown) return;
                    placeholder.style.display = "none";
                    // A truer frame is the more interesting one -- surface it once, then leave
                    // the choice to the user. Frames for inactive slots stay hidden.
                    if (!userPinned && SOURCES.indexOf(source) > SOURCES.indexOf(current)) {
                        select(source);
                    }
                }).catch(() => {});
            };

            chainCallback(node, "onRemoved", function () {
                for (const s of SOURCES) slots[s].dispose();
                delete node._mmh3Handler;
            });

            select("latent");
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
