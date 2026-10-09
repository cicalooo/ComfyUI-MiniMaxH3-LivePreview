"""Loader for the MiniMax H3 tiny VAE decoder (Kijai/MiniMax-H3-TAE, ``taeh3.safetensors``).

Core ComfyUI cannot build this checkpoint. ``comfy.sd.VAE`` only recognises TAE state
dicts by ``taesd_decoder.1.weight`` (2D TAESD) or ``decoder.22.bias`` (TAEHV), and taeh3
is a *bare* decoder whose keys are plain module indices -- so ``VAELoader`` reports
"No VAE weights detected" and then "VAE is invalid: None". Even with the prefix fixed,
``comfy.taesd.taesd.Decoder`` hardcodes a 64-wide stack with three upsample stages, while
taeh3 is 24 latent channels, 96 wide for its first half, and four upsample stages (H3's
VAE is 16x spatial, not 8x).

These decoders are a flat ``nn.Sequential`` keyed by positional module index, so the
architecture is recoverable from the checkpoint itself: an index carrying
``N.conv.0.weight`` is a Block, one carrying ``N.weight`` is a conv (the bias-less ones
sit right after an upsample), and the gaps are the parameterless modules.
"""

import logging

import torch
import torch.nn as nn

import comfy.model_management
import comfy.utils
import folder_paths
from comfy.taesd.taesd import Block, Clamp, conv

NONE = "none"

TAE_DEVICES = ("auto", "gpu", "cpu")
TAE_DTYPES = ("auto", "float32", "float16", "bfloat16")

_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def list_tae_decoders():
    """vae_approx filenames, with a leading "none" for the disabled state."""
    try:
        return [NONE] + folder_paths.get_filename_list("vae_approx")
    except Exception:
        return [NONE]


def _resolve_tae_device(device):
    """Map 'auto'/'gpu'/'cpu'/None/torch.device to a torch.device.

    'auto' honours --cpu-vae and friends via vae_device(). 'gpu' forces the
    torch device, 'cpu' forces CPU.
    """
    if device is None or (isinstance(device, str) and device == "auto"):
        return comfy.model_management.vae_device()
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        name = device.strip().lower()
        if name == "cpu":
            return torch.device("cpu")
        if name in ("gpu", "cuda"):
            try:
                return comfy.model_management.get_torch_device()
            except Exception:
                return torch.device("cuda")
        raise ValueError(f"unknown TAE device '{device}' (expected one of {TAE_DEVICES})")
    raise TypeError(f"bad TAE device type {type(device)}")


def _resolve_tae_dtype(dtype, device):
    """Map 'auto'/None/string/torch.dtype to a torch.dtype."""
    if dtype is None or (isinstance(dtype, str) and dtype == "auto"):
        return comfy.model_management.vae_dtype(device, [torch.float16, torch.bfloat16])
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        key = dtype.strip().lower()
        if key in _DTYPE_MAP:
            return _DTYPE_MAP[key]
        raise ValueError(f"unknown TAE dtype '{dtype}' (expected one of {TAE_DTYPES})")
    raise TypeError(f"bad TAE dtype type {type(dtype)}")


def build_tae_decoder(sd):
    """Reconstruct the decoder's nn.Sequential from a flat, index-keyed state dict."""
    by_index = {}
    for k, v in sd.items():
        head, _, rest = k.partition(".")
        if not head.isdigit():
            raise ValueError(f"not a flat TAE decoder state dict (unexpected key '{k}')")
        by_index.setdefault(int(head), {})[rest] = v

    modules = []
    for i in range(max(by_index) + 1):
        entry = by_index.get(i)
        if entry is None:
            # index 0 is the input Clamp, 2 the ReLU after the input conv, the rest upsamples
            modules.append(Clamp() if i == 0 else nn.ReLU() if i == 2 else nn.Upsample(scale_factor=2))
        elif "conv.0.weight" in entry:
            w = entry["conv.0.weight"]
            # only pass the kwarg when it is needed -- older ComfyUI has no midblock-GN variant
            if "pool.0.weight" in entry:
                modules.append(Block(w.shape[1], w.shape[0], use_midblock_gn=True))
            else:
                modules.append(Block(w.shape[1], w.shape[0]))
        elif "weight" in entry:
            w = entry["weight"]
            modules.append(conv(w.shape[1], w.shape[0], bias="bias" in entry))
        else:
            raise ValueError(f"unrecognized TAE decoder module at index {i}: {sorted(entry)}")
    return nn.Sequential(*modules)


class TinyVaeDecoder:
    """Decode-only tiny VAE. Output is [0, 1], the TAE family's convention.

    taeh3 is 2D: it decodes each latent frame independently, so T latent frames give T
    preview frames -- the 4x temporal compression is not undone. That is what makes it
    cheap enough to run inside the sampler callback (~10 MB of weights, milliseconds per
    frame), unlike the real 5.2 GB H3 video VAE.
    """

    def __init__(self, sd, device=None, dtype=None, compile=False, skip_upscale_layers=0):
        # keys may carry a "taesd_decoder."/"decoder." prefix; strip whatever is common
        first = next(iter(sd))
        if not first.split(".")[0].isdigit():
            prefix = first.split(".")[0] + "."
            sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

        resolved_device = _resolve_tae_device(device)
        resolved_dtype = _resolve_tae_dtype(dtype, resolved_device)
        try:
            skip_upscale_layers = int(skip_upscale_layers)
        except (TypeError, ValueError, OverflowError):
            skip_upscale_layers = 0
        skip_upscale_layers = max(0, skip_upscale_layers)

        self.device = resolved_device
        self.dtype = resolved_dtype
        self.compile = bool(compile)
        model = build_tae_decoder(sd)
        model.load_state_dict(sd)
        if skip_upscale_layers > 0:
            modules = list(model)
            up_idx = [i for i, m in enumerate(modules) if isinstance(m, nn.Upsample)]
            skip_upscale_layers = min(skip_upscale_layers, len(up_idx))
            if skip_upscale_layers > 0:
                drop = set(up_idx[-skip_upscale_layers:])
                model = nn.Sequential(*[m for i, m in enumerate(modules) if i not in drop])
        self.skip_upscale_layers = skip_upscale_layers
        model.eval().to(device=self.device, dtype=self.dtype)
        for p in model.parameters():
            p.requires_grad_(False)
        # Introspect before torch.compile: the compiled OptimizedModule
        # is not subscriptable/iterable, so this must run on the Sequential.
        self.latent_channels = model[1].weight.shape[1]
        self.upscale_ratio = 2 ** sum(isinstance(m, nn.Upsample) for m in model)
        if self.compile:
            try:
                model = torch.compile(model)
            except Exception as e:
                logging.warning(
                    "[MiniMaxH3Preview] torch.compile failed (%s); using eager mode.",
                    e,
                )
                self.compile = False
        self.model = model

    def decode(self, latent):
        """[B, C, h, w] -> [B, 3, h*ratio, w*ratio], float32 on the caller's device."""
        with torch.no_grad():
            out = self.model(latent.to(device=self.device, dtype=self.dtype))
        return out.to(device=latent.device, dtype=torch.float32)

    def decode_video(self, latent, frame_indices=None, output_device="cpu"):
        """[B, C, T, h, w] -> [T, h*ratio, w*ratio, 3].

        One frame at a time, each parked on `output_device` as it lands: at 16x the
        full-resolution activations are the memory peak, not the weights, and a whole
        decoded clip held on the GPU would be a second peak on top of the sampler's.
        """
        x = latent[0]
        indices = range(x.shape[1]) if frame_indices is None else frame_indices
        frames = [
            self.decode(x[:, t].unsqueeze(0))[0].movedim(0, -1).to(output_device)
            for t in indices
        ]
        return torch.stack(frames, dim=0)


def load_tae_decoder(name, device="auto", dtype="auto", compile=False, skip_upscale_layers=0):
    """Load by vae_approx filename. Returns None (and logs) if it cannot be used."""
    if not name or name == NONE:
        return None
    path = folder_paths.get_full_path("vae_approx", name)
    if path is None:
        logging.warning(
            f"[MiniMaxH3Preview] tiny VAE '{name}' not found in models/vae_approx; "
            "falling back to latent2rgb previews."
        )
        return None
    try:
        sd = comfy.utils.load_torch_file(path, safe_load=True)
        tae = TinyVaeDecoder(
            sd, device=device, dtype=dtype, compile=compile,
            skip_upscale_layers=skip_upscale_layers,
        )
    except Exception as e:
        logging.warning(
            f"[MiniMaxH3Preview] could not load tiny VAE '{name}' ({e}); "
            "falling back to latent2rgb previews."
        )
        return None
    logging.info(
        "[MiniMaxH3Preview] tiny VAE '%s' loaded: %d latent channels, %dx upscale, on %s/%s%s%s",
        name, tae.latent_channels, tae.upscale_ratio, tae.device, tae.dtype,
        " compiled" if tae.compile else "",
        f" skip_upscale={tae.skip_upscale_layers}" if tae.skip_upscale_layers else "",
    )
    return tae
