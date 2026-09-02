"""Live acceptance: queue a tiny H3 run and capture minimax_h3_preview WS events."""

from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from urllib import request

import websocket  # websocket-client


BASE = "http://127.0.0.1:8188"
WS_BASE = "ws://127.0.0.1:8188/ws"
EVENT = "minimax_h3_preview"
OUT = Path(__file__).with_name("live_accept_preview_report.json")


def build_prompt(
    *,
    unet: str,
    clip_name: str,
    clip_type: str,
    projection: str,
    video_vae: str,
    audio_vae: str,
    tae_decoder: str,
    width: int,
    height: int,
    length: int,
    steps: int,
    seed: int,
) -> dict:
    # Graph:
    # CLIPLoader -> ClipProjApply -> MiniMaxH3ReferenceToVideo
    # UNETLoader -> MiniMaxH3LivePreview -> MiniMaxH3SigmaShift -> BasicGuider/BasicScheduler
    # SamplerCustomAdvanced -> (no decode; preview is the acceptance target)
    return {
        "1": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": clip_name,
                "type": clip_type,
                "device": "default",
            },
        },
        "2": {
            "class_type": "ClipProjApply",
            "inputs": {
                "clip": ["1", 0],
                "projection": projection,
            },
        },
        "3": {
            "class_type": "UNETLoader",
            "inputs": {
                "unet_name": unet,
                "weight_dtype": "default",
            },
        },
        "4": {
            "class_type": "MiniMaxH3LivePreview",
            "inputs": {
                "model": ["3", 0],
                "preview_frames": 4,
                "preview_fps": 8,
                "max_resolution": 256,
                "every_n_steps": 1,
                "upscale_method": "nearest-exact",
                "jpeg_quality": 70,
                "suppress_default_preview": True,
                "vae_decode_every_n_steps": 0,
                "vae_decode_frames": 1,
                "vae_decode_device": "auto",
                "tae_decoder": tae_decoder,
            },
        },
        "5": {
            "class_type": "MiniMaxH3SigmaShift",
            "inputs": {
                "model": ["4", 0],
                "shift_video": 12.0,
                "shift_audio": 3.0,
            },
        },
        "6": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": video_vae},
        },
        "7": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": audio_vae},
        },
        "8": {
            "class_type": "MiniMaxH3ReferenceToVideo",
            "inputs": {
                "clip": ["2", 0],
                "vae": ["6", 0],
                "audio_vae": ["7", 0],
                "prompt": (
                    "subject_definitions:\nA quiet indoor room.\n"
                    "summary:\nEmpty room, locked-off camera, soft daylight.\n"
                    "retention_analysis:\nN/A\n"
                    "detailed_description:\nA still empty room with soft daylight and no people.\n"
                    "overall_soundscape:\nQuiet room tone.\n"
                    "non_diegetic_music:\nN/A"
                ),
                "width": width,
                "height": height,
                "length": length,
                "ref_image_size": "match",
            },
        },
        "9": {
            "class_type": "BasicGuider",
            "inputs": {
                "model": ["5", 0],
                "conditioning": ["8", 0],
            },
        },
        "10": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "res_multistep"},
        },
        "11": {
            "class_type": "BasicScheduler",
            "inputs": {
                "model": ["5", 0],
                "scheduler": "simple",
                "steps": steps,
                "denoise": 1.0,
            },
        },
        "12": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "13": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["12", 0],
                "guider": ["9", 0],
                "sampler": ["10", 0],
                "sigmas": ["11", 0],
                "latent_image": ["8", 1],
            },
        },
        # Output sink so ComfyUI accepts the prompt. We care about WS preview
        # events during sampling, not the final decode.
        "14": {
            "class_type": "PreviewAny",
            "inputs": {
                "source": ["13", 0],
            },
        },
    }


def http_json(method: str, path: str, payload: dict | None = None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = request.Request(
        BASE + path,
        data=data,
        method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        body = ""
        if hasattr(e, "read"):
            try:
                body = e.read().decode("utf-8", errors="replace")
            except Exception:
                body = ""
        if body:
            raise RuntimeError(f"{e}\n{body}") from e
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--width", type=int, default=672)
    ap.add_argument("--height", type=int, default=384)
    ap.add_argument("--length", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--tae", default="taeh3.safetensors")
    args = ap.parse_args()

    client_id = uuid.uuid4().hex
    prompt = build_prompt(
        unet="10Eros_Max_h3_TURBO-hybrid_beta4_w4a8_final_14gb.safetensors",
        clip_name="qwen3vl_4b_fp8_scaled.safetensors",
        clip_type="krea2",
        projection="mmh3-4b-ClipProj-v3.1.safetensors",
        video_vae="minimax_h3_video_vae_fp16.safetensors",
        audio_vae="minimax_h3_audio_vae_fp32.safetensors",
        tae_decoder=args.tae,
        width=args.width,
        height=args.height,
        length=args.length,
        steps=args.steps,
        seed=args.seed,
    )

    # Validate first.
    try:
        validation = http_json("POST", "/prompt", {"prompt": prompt, "client_id": client_id})
    except Exception as e:
        # /prompt both validates and queues; if it fails, surface body.
        print("queue/validate failed:", e)
        raise

    prompt_id = validation.get("prompt_id")
    print("queued", prompt_id, "client_id", client_id)

    events = []
    statuses = []
    errors = []
    done = False
    t0 = time.time()

    ws = websocket.create_connection(f"{WS_BASE}?clientId={client_id}", timeout=10)
    ws.settimeout(1.0)
    try:
        while time.time() - t0 < args.timeout:
            try:
                raw = ws.recv()
            except Exception:
                # Poll history as a backup completion signal.
                try:
                    hist = http_json("GET", f"/history/{prompt_id}")
                    if prompt_id in hist:
                        done = True
                        break
                except Exception:
                    pass
                continue

            if isinstance(raw, (bytes, bytearray)):
                continue
            msg = json.loads(raw)
            mtype = msg.get("type")
            data = msg.get("data") or {}

            if mtype == EVENT:
                # Strip huge base64 for the report summary.
                slim = {
                    k: v
                    for k, v in data.items()
                    if k != "image"
                }
                if "image" in data and isinstance(data["image"], str):
                    slim["image_b64_len"] = len(data["image"])
                events.append(slim)
                print(
                    f"preview event #{len(events)} step={slim.get('step')}/{slim.get('total')} "
                    f"source={slim.get('source')} mime={slim.get('mime')} "
                    f"size={slim.get('w')}x{slim.get('h')} run={str(slim.get('run_id'))[:8]}"
                )
            elif mtype in ("execution_error", "status", "executing", "progress", "execution_success"):
                if mtype == "execution_error":
                    errors.append(data)
                    print("execution_error", json.dumps(data)[:1000])
                    done = True
                    break
                if mtype == "status":
                    statuses.append(data)
                if mtype == "executing" and data.get("node") is None and data.get("prompt_id") == prompt_id:
                    done = True
                    break
                if mtype == "execution_success" and data.get("prompt_id") == prompt_id:
                    done = True
                    break
        else:
            print("timeout waiting for completion")
            try:
                http_json("POST", "/interrupt", {})
            except Exception:
                pass
    finally:
        ws.close()

    # Summarize.
    sources = sorted({e.get("source") for e in events if e.get("source")})
    run_ids = sorted({e.get("run_id") for e in events if e.get("run_id")})
    prompt_ids = sorted({e.get("prompt_id") for e in events if e.get("prompt_id")})
    steps = [e.get("step") for e in events if e.get("step") is not None]
    mimes = sorted({e.get("mime") for e in events if e.get("mime")})

    report = {
        "ok": done and not errors and len(events) > 0,
        "prompt_id": prompt_id,
        "client_id": client_id,
        "elapsed_s": round(time.time() - t0, 2),
        "event_count": len(events),
        "sources": sources,
        "mimes": mimes,
        "run_ids": run_ids,
        "prompt_ids_in_events": prompt_ids,
        "steps": steps,
        "errors": errors,
        "events": events,
        "settings": {
            "width": args.width,
            "height": args.height,
            "length": args.length,
            "steps": args.steps,
            "tae_decoder": args.tae,
        },
    }
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k != "events"}, indent=2))
    print("wrote", OUT)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
