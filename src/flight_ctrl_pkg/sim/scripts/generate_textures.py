#!/usr/bin/env python3
"""Generate seamless offline PBR textures for the Topic-2 UAV proving ground."""

import math
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont


OUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        '..',
        'models',
        'arena_assets',
        'materials',
        'textures',
    )
)


def _fbm_noise(size: int, seed: int, octaves=(4, 8, 16, 32, 64)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc = np.zeros((size, size), dtype=np.float32)
    weight = 1.0
    norm = 0.0
    for grid in octaves:
        base = rng.random((grid, grid), dtype=np.float32)
        up = np.array(
            Image.fromarray((base * 255).astype(np.uint8)).resize(
                (size, size), resample=Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0
        acc += weight * up
        norm += weight
        weight *= 0.55
    return acc / max(norm, 1e-6)


def make_terrain_meadow(size: int = 512) -> None:
    n1 = _fbm_noise(size, seed=101, octaves=(4, 8, 16, 32, 64))
    n2 = _fbm_noise(size, seed=202, octaves=(8, 16, 32, 64))
    # Subtle mowed turf striping (safe green/olive/earth, low red saturation)
    y_idx = np.linspace(0, 16 * math.pi, size, dtype=np.float32)[:, None]
    stripe = 0.5 + 0.5 * np.sin(y_idx)
    r = 56 + 28 * n1 + 14 * n2 + 6 * stripe
    g = 96 + 38 * n1 + 16 * n2 + 12 * stripe
    b = 44 + 18 * n1 + 10 * n2 + 4 * stripe
    rgb = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode='RGB').save(os.path.join(OUT_DIR, 'terrain_meadow.png'))


def make_terrain_hill(size: int = 256) -> None:
    n1 = _fbm_noise(size, seed=303, octaves=(4, 8, 16, 32))
    n2 = _fbm_noise(size, seed=404, octaves=(16, 32, 64))
    r = 76 + 35 * n1 + 20 * n2
    g = 92 + 32 * n1 + 18 * n2
    b = 58 + 20 * n1 + 12 * n2
    rgb = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
    Image.fromarray(rgb, mode='RGB').save(os.path.join(OUT_DIR, 'terrain_hill.png'))


def make_asphalt_runway(size: int = 512) -> None:
    n_fine = _fbm_noise(size, seed=505, octaves=(16, 32, 64, 128))
    n_patch = _fbm_noise(size, seed=606, octaves=(4, 8, 16))
    base = 48 + 24 * n_fine + 16 * n_patch
    rgb = np.stack([base, base + 2, base + 7], axis=-1).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode='RGB')
    draw = ImageDraw.Draw(img)

    # Border yellow taxi stripes (top & bottom edges in UV)
    draw.rectangle([0, 12, size, 20], fill=(215, 180, 45))
    draw.rectangle([0, size - 20, size, size - 12], fill=(215, 180, 45))

    # Crisp white dashed centerline & runway threshold bars
    cy = size // 2
    for x in range(70, size - 70, 60):
        draw.rectangle([x, cy - 4, x + 35, cy + 4], fill=(235, 238, 242))
    for k in range(-3, 4):
        if k == 0:
            continue
        yy = cy + k * 24
        draw.rectangle([21, yy - 6, 59, yy + 6], fill=(235, 238, 242))
        draw.rectangle([size - 59, yy - 6, size - 21, yy + 6], fill=(235, 238, 242))

    # Cross-grid subtle measurement lines
    for x in (128, 256, 384):
        draw.rectangle([x - 2, 22, x + 2, size - 22], fill=(95, 108, 122))
    img.save(os.path.join(OUT_DIR, 'asphalt_runway.png'))


def make_helipad(label: str, filename: str, size: int = 256) -> None:
    n = _fbm_noise(size, seed=707 + sum(ord(c) for c in label), octaves=(8, 16, 32))
    base = 64 + 18 * n
    rgb = np.stack([base, base + 4, base + 10], axis=-1).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(rgb, mode='RGB')
    draw = ImageDraw.Draw(img)

    # Outer safety ring (yellow/slate contrast, non-red)
    m = 18
    draw.ellipse([m, m, size - m, size - m], outline=(240, 195, 40), width=9)
    draw.ellipse([m + 12, m + 12, size - m - 12, size - m - 12], outline=(215, 222, 232), width=3)

    # Bold "H" emblem
    draw.rectangle([84, 65, 100, 191], fill=(242, 198, 42))
    draw.rectangle([156, 65, 172, 191], fill=(242, 198, 42))
    draw.rectangle([84, 119, 172, 137], fill=(242, 198, 42))

    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)
    except Exception:
        font = ImageFont.load_default()
    draw.text((112, 198), label, fill=(235, 240, 245), font=font)
    img.save(os.path.join(OUT_DIR, filename))


def make_architecture_textures(size: int = 256) -> None:
    # Hangar corrugated concrete/steel siding
    n = _fbm_noise(size, seed=808, octaves=(16, 32, 64))
    x_idx = np.linspace(0, 64 * math.pi, size, dtype=np.float32)[None, :]
    rib = 0.5 + 0.5 * np.sin(x_idx)
    c = 138 + 20 * n + 26 * rib
    wall = np.stack([c, c + 4, c + 10], axis=-1).clip(0, 255).astype(np.uint8)
    Image.fromarray(wall, mode='RGB').save(os.path.join(OUT_DIR, 'hangar_siding.png'))

    # Blue slate hangar roof with solar panel grid
    roof_n = _fbm_noise(size, seed=909, octaves=(8, 16, 32))
    r = 38 + 14 * roof_n
    g = 58 + 18 * roof_n
    b = 88 + 22 * roof_n
    roof_rgb = np.stack([r, g, b], axis=-1).clip(0, 255).astype(np.uint8)
    roof_img = Image.fromarray(roof_rgb, mode='RGB')
    d = ImageDraw.Draw(roof_img)
    for gx in range(32, size, 64):
        d.line([(gx, 0), (gx, size)], fill=(180, 195, 215), width=3)
    for gy in range(32, size, 64):
        d.line([(0, gy), (size, gy)], fill=(180, 195, 215), width=3)
    roof_img.save(os.path.join(OUT_DIR, 'hangar_roof.png'))

    # Blue and ochre shipping containers (high contrast non-red props)
    for fname, base_col in [
        ('container_blue.png', (36, 98, 168)),
        ('container_ochre.png', (194, 146, 38)),
    ]:
        shade = 0.85 + 0.15 * np.sin(x_idx) + 0.08 * (n - 0.5)
        arr = np.zeros((size, size, 3), dtype=np.float32)
        for ch in range(3):
            arr[:, :, ch] = base_col[ch] * shade
        Image.fromarray(arr.clip(0, 255).astype(np.uint8), mode='RGB').save(
            os.path.join(OUT_DIR, fname)
        )

    # High-visibility tactical red enemy UAV canopy livery
    tex = np.zeros((size, size, 3), dtype=np.float32)
    tex[:, :, 0] = 232
    tex[:, :, 1] = 22
    tex[:, :, 2] = 22
    t_img = Image.fromarray(tex.astype(np.uint8), mode='RGB')
    td = ImageDraw.Draw(t_img)
    td.rectangle([0, size // 2 - 22, size, size // 2 + 22], fill=(250, 48, 28))
    t_img.save(os.path.join(OUT_DIR, 'target_crimson.png'))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    make_terrain_meadow()
    make_terrain_hill()
    make_asphalt_runway()
    make_helipad('H1', 'helipad_h1.png')
    make_helipad('H2', 'helipad_h2.png')
    make_helipad('H3', 'helipad_h3.png')
    make_architecture_textures()
    print(f'Generated textures in {OUT_DIR}:', sorted(os.listdir(OUT_DIR)))


if __name__ == '__main__':
    main()
