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


def list_tae_decoders():
    """vae_approx filenames, with a leading "none" for the disabled state."""
    try:
        return [NONE] + folder_paths.get_filename_list("vae_approx")
    except Exception:
        return [NONE]


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

    def __init__(self, sd, device=None, dtype=None):
        # keys may carry a "taesd_decoder."/"decoder." prefix; strip whatever is common
        first = next(iter(sd))
        if not first.split(".")[0].isdigit():
            prefix = first.split(".")[0] + "."
            sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}

        # vae_device() rather than get_torch_device() so --cpu-vae and friends are honoured.
        # On the CPU this is ~1.3 s per full-resolution frame, so that is the user's call.
        self.device = device if device is not None else comfy.model_management.vae_device()
        self.dtype = dtype if dtype is not None else comfy.model_management.vae_dtype(
            self.device, [torch.float16, torch.bfloat16])
        self.model = build_tae_decoder(sd)
        self.model.load_state_dict(sd)
        self.model.eval().to(device=self.device, dtype=self.dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.latent_channels = self.model[1].weight.shape[1]
        self.upscale_ratio = 2 ** sum(isinstance(m, nn.Upsample) for m in self.model)

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


def load_tae_decoder(name, device=None, dtype=None):
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
        tae = TinyVaeDecoder(sd, device=device, dtype=dtype)
    except Exception as e:
        logging.warning(
            f"[MiniMaxH3Preview] could not load tiny VAE '{name}' ({e}); "
            "falling back to latent2rgb previews."
        )
        return None
    logging.info(
        "[MiniMaxH3Preview] tiny VAE '%s' loaded: %d latent channels, %dx upscale, on %s/%s",
        name, tae.latent_channels, tae.upscale_ratio, tae.device, tae.dtype,
    )
    return tae
