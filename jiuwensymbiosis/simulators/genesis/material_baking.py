"""Texture operations omitted by Nyx's limited USD shader importer.

Sample caller-provided UV transforms on the unit tile. This is exact in periodicity
only when the affine matrix maps the integer UV lattice onto itself; otherwise
the repeated tile outside [0, 1] is an explicitly reported approximation.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path


def bake_texture(path: Path, directory: Path, target: str, uv, color, scale, bias) -> tuple[Path, bool, dict]:
    import numpy as np
    from PIL import Image
    from scipy.ndimage import map_coordinates

    identity = ((1.0, 1.0), 0.0, (0.0, 0.0))
    uv = uv or identity
    scale = scale or (1, 1, 1, 1)
    bias = bias or (0, 0, 0, 0)
    scalar = target in {"roughness", "metallic", "opacity"}
    normal = target == "normal"
    # EXR/HDR pixels are already linear: neither decoding them through the sRGB
    # transfer function nor declaring them sRGB downstream is correct.
    linear_source = path.suffix.lower() in {".exr", ".hdr"}
    default_values = (scale == (1, 1, 1, 1) and bias == (0, 0, 0, 0)) or (
        normal and scale == (2, 2, 2, 1) and bias == (-1, -1, -1, 0)
    )
    if uv == identity and default_values and not scalar:
        return path, color and not linear_source, {}
    digest = hashlib.sha256(path.read_bytes())
    # Version 2: linear sources are no longer pushed through the sRGB decode.
    parameters = {"version": 2, "target": target, "uv": uv, "color": color, "scale": scale, "bias": bias}
    digest.update(json.dumps(parameters, sort_keys=True).encode())
    # HDR preserves emissive intensity beyond 1.0; other channels are bounded.
    hdr = target == "emissiveColor" and max(scale[:3]) > 1
    output = directory / (digest.hexdigest() + (".hdr" if hdr else ".png"))
    angle = math.radians(uv[1])
    c, s = math.cos(angle), math.sin(angle)
    matrix = np.array([[c, -s], [s, c]]) @ np.diag(uv[0])
    detail = {
        "source": str(path),
        "baked": str(output),
        **parameters,
        "periodic_exact": bool(np.allclose(matrix, np.round(matrix), atol=1e-6)),
        "max_resolution": 2048,
    }
    if output.is_file():
        return output, bool(color and not scalar and not normal and not hdr), detail
    if linear_source:
        # Pillow cannot decode EXR textures. OpenCV's bundled OpenEXR codec needs
        # OPENCV_IO_ENABLE_OPENEXR=1 in the environment before cv2 is FIRST
        # imported; process entry points (worker, probe and batch scripts) set it.
        import cv2

        pixels = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if pixels is None:
            raise ValueError(f"Cannot decode HDR material texture: {path}")
        if pixels.ndim == 2:
            pixels = np.repeat(pixels[..., None], 3, axis=-1)
        else:
            pixels = pixels[..., :3][..., ::-1]
        pixels = pixels.astype(np.float32)
    else:
        with Image.open(path) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32) / 255
    if color and not linear_source:
        pixels = np.where(pixels <= 0.04045, pixels / 12.92, ((pixels + 0.055) / 1.055) ** 2.4)
    height, width = pixels.shape[:2]
    if uv != identity:
        # Increase sampling for repeated patterns, bounded to control GPU memory.
        output_w = min(2048, max(width, int(width * abs(uv[0][0]))))
        output_h = min(2048, max(height, int(height * abs(uv[0][1]))))
        u, v = np.meshgrid((np.arange(output_w) + 0.5) / output_w, 1 - (np.arange(output_h) + 0.5) / output_h)
        transformed_u = matrix[0, 0] * u + matrix[0, 1] * v + uv[2][0]
        transformed_v = matrix[1, 0] * u + matrix[1, 1] * v + uv[2][1]
        coordinates = [(1 - transformed_v % 1) * height - 0.5, (transformed_u % 1) * width - 0.5]
        pixels = np.stack(
            [map_coordinates(pixels[..., i], coordinates, order=1, mode="grid-wrap") for i in range(3)], axis=-1
        )
    pixels = pixels * np.asarray(scale[:3]) + np.asarray(bias[:3])
    if normal:
        pixels /= np.maximum(np.linalg.norm(pixels, axis=-1, keepdims=True), 1e-8)
        pixels = pixels * 0.5 + 0.5
    elif scalar:
        pixels = np.repeat(pixels[..., :1], 3, axis=-1)
    output_color = bool(color and not scalar and not normal and not hdr)
    if output_color:
        pixels = np.where(pixels <= 0.0031308, 12.92 * pixels, 1.055 * np.maximum(pixels, 0) ** (1 / 2.4) - 0.055)
    output.parent.mkdir(parents=True, exist_ok=True)
    # Write then rename: a half-written file must never look like a valid cache
    # entry. Keep the real extension so PIL/cv2 can infer the format.
    temporary = output.with_name(output.stem + ".tmp" + output.suffix)
    if hdr:
        import cv2

        if not cv2.imwrite(str(temporary), pixels[..., ::-1].astype(np.float32)):
            raise RuntimeError(f"Cannot write emissive texture: {output}")
    else:
        Image.fromarray(np.uint8(np.rint(np.clip(pixels, 0, 1) * 255))).save(temporary)
    os.replace(temporary, output)
    return output, output_color, detail
