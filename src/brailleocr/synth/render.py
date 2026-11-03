"""Photorealistic rendering of an embossed Braille page.

Embossed dots are colourless bumps, visible only through shading, so the page
is rendered from a height map rather than by pasting glyphs:

1. height map: dome-shaped recto dots (some worn flat), mirrored verso dots
   pressed in from the back ("ghost" dots of double-sided / interpoint
   Braille), and paper roughness;
2. shading: Lambertian lighting of the surface normals by a directional light
   (low raking light gives strong highlights and shadows, overhead light makes
   dots faint);
3. paper albedo (white or manila, blotchy), uneven illumination and a soft
   cast shadow;
4. camera: perspective warp onto a background, defocus and motion blur,
   sensor noise and JPEG compression.

``severity`` in [0, 1] scales every degradation. Continuous ones scale by
``severity * U(0.5, 1)`` so a high-severity page is reliably degraded;
discrete ones (cast shadow, motion blur, worn dots, ghosts) become more
likely. At 0 the page is flat, evenly and obliquely lit and sharp; at 1 it is
a poor phone photo.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .layout import PageLayout

DOT_HEIGHT_MM = 0.5


@dataclass
class RenderResult:
    image: np.ndarray            # (H, W, 3) uint8 RGB, decoded from ``jpeg``
    jpeg: bytes                  # the encoded image, exactly as saved
    dots: np.ndarray             # (N, 6): x, y, line, col, dot, amplitude (image px)
    ghost_dots: np.ndarray       # (M, 2): x, y of verso dots (image px)
    page_to_image: np.ndarray    # 3x3 homography from page mm to image px
    px_per_mm: float
    params: dict                 # sampled degradation parameters, for analysis


def render_page(
    layout: PageLayout,
    rng: np.random.Generator,
    severity: float,
    px_per_mm: float,
    ghost_layout: PageLayout | None = None,
) -> RenderResult:
    """Render ``layout`` as a photo at the given severity.

    Each degradation draws from its own stream spawned from ``rng``, and every
    variate is drawn whether or not it is used. With the same ``rng`` seed,
    renders at different severities therefore share the same paper, lighting
    direction, shadow placement and so on, and differ only in how strongly each
    effect is applied (common random numbers). Coupling is monotone: an effect
    present at one severity is present at every higher one.
    """
    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    s = severity
    g = layout.geometry
    r_res, r_dots, r_ghost, r_paper, r_light, r_albedo, r_illum, r_shadow, r_cam, r_blur, r_noise = rng.spawn(11)

    # Lower resolution (camera further away) is itself a degradation.
    ppm = px_per_mm * (1.0 - 0.35 * s * r_res.uniform(0.5, 1.0))
    w_mm, h_mm = layout.size_mm
    w, h = int(np.ceil(w_mm * ppm)), int(np.ceil(h_mm * ppm))
    params: dict = {"px_per_mm": ppm}

    # ---- 1. height map (in units of full dot height)
    height = np.zeros((h, w), np.float32)
    dots_mm = layout.dots_mm()
    n = len(dots_mm)
    dots_mm[:, :2] += (0.04 + 0.08 * s) * r_dots.standard_normal((n, 2))
    amp = r_dots.uniform(0.75, 1.0, size=n)
    worn = r_dots.uniform(size=n) < 0.12 * s
    amp = np.where(worn, r_dots.uniform(0.2, 0.45, size=n), amp)
    radius_px = 0.5 * g.dot_diameter_mm * ppm
    radii = radius_px * r_dots.uniform(0.9, 1.1, size=n)
    for (x, y), a, r in zip(dots_mm[:, :2] * ppm, amp, radii):
        _stamp(height, x, y, r, a)
    params["worn_dots"] = int(worn.sum())

    ghost_mm = np.zeros((0, 2))
    if ghost_layout is not None:
        # Verso page seen from the front: mirrored, and offset between recto lines.
        present = r_ghost.uniform() < 0.3 + 0.6 * s
        gd = ghost_layout.dots_mm()[:, :2]
        gd[:, 0] = w_mm - gd[:, 0] + r_ghost.uniform(-2.0, 2.0)
        gd[:, 1] += ghost_layout.geometry.line_pitch_mm * r_ghost.uniform(0.4, 0.6)
        strength = r_ghost.uniform(0.3, 1.0) * (0.4 + 0.6 * s)
        g_radii = radius_px * r_ghost.uniform(0.9, 1.2, size=len(gd))
        g_amp = -strength * r_ghost.uniform(0.25, 0.6, size=len(gd))
        inside = (gd[:, 0] > 0) & (gd[:, 0] < w_mm) & (gd[:, 1] > 0) & (gd[:, 1] < h_mm)
        if present:
            ghost_mm = gd[inside]
            for (x, y), a, r in zip(ghost_mm * ppm, g_amp[inside], g_radii[inside]):
                _stamp(height, x, y, r, a)
            params["ghost_strength"] = float(strength)
    params["ghost_dots"] = len(ghost_mm)

    height += _smooth_noise(r_paper, (h, w), (w_mm, h_mm), cell_mm=3.0) * (0.02 + 0.04 * s)
    height += r_paper.standard_normal((h, w)).astype(np.float32) * 0.01
    height = cv2.GaussianBlur(height, (0, 0), 0.7)

    # ---- 2. shading
    hz = height * (DOT_HEIGHT_MM * ppm)
    gy, gx = np.gradient(hz)
    norm = np.sqrt(gx * gx + gy * gy + 1.0)
    elev = np.deg2rad(r_light.uniform(25, 45) + s * r_light.uniform(17.5, 35))
    azim = r_light.uniform(0, 2 * np.pi)
    light = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    lambert = np.clip((-gx * light[0] - gy * light[1] + light[2]) / norm, 0, None) / light[2]
    ambient = r_light.uniform(0.15, 0.35)
    shade = ambient + (1 - ambient) * lambert
    params.update(light_elevation_deg=float(np.rad2deg(elev)), light_azimuth_deg=float(np.rad2deg(azim)))

    # ---- 3. albedo and illumination
    manila = r_albedo.uniform() < 0.3
    tint = (np.array([1.0, 0.95, 0.82]) if manila else np.ones(3)) * r_albedo.uniform(0.98, 1.02, size=3)
    albedo = r_albedo.uniform(0.80, 0.92) * tint
    blotch = 1.0 + 0.04 * _smooth_noise(r_albedo, (h, w), (w_mm, h_mm), cell_mm=20.0)
    white_balance = 1 + 0.06 * s * r_albedo.uniform(-1, 1, size=3)

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = xx / w - 0.5, yy / h - 0.5
    theta = r_illum.uniform(0, 2 * np.pi)
    grad_mag = r_illum.uniform(0, 0.15) + 0.35 * s * r_illum.uniform(0.5, 1.0)
    vignette = r_illum.uniform(0, 0.1) + 0.3 * s * r_illum.uniform(0.5, 1.0)
    illum = (1 + grad_mag * (xn * np.cos(theta) + yn * np.sin(theta))) * (1 - vignette * (xn**2 + yn**2))
    has_shadow = r_shadow.uniform() < 0.5 * s
    darkness = r_shadow.uniform(0.25, 0.55)
    shadow = _soft_halfplane(r_shadow, xn, yn, ppm / max(w, h))
    if has_shadow:
        illum *= 1 - darkness * shadow
        params["cast_shadow"] = True

    img = albedo[None, None, :] * (shade * blotch * illum)[..., None] * white_balance
    img = np.clip(img, 0, 1).astype(np.float32)

    # ---- 4. camera: perspective, blur, noise, JPEG
    page_px = np.diag([ppm, ppm, 1.0])
    M, out_size = _random_perspective(r_cam, w, h, s)
    bg = _background(r_cam, out_size)
    warped = cv2.warpPerspective(img, M, out_size, flags=cv2.INTER_LINEAR)
    cover = cv2.warpPerspective(np.ones((h, w), np.float32), M, out_size, flags=cv2.INTER_LINEAR)[..., None]
    img = warped * cover + bg * (1 - cover)

    blur = s * r_blur.uniform(0.9, 1.8) * ppm / 6.5
    if blur > 0.3:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    has_motion = r_blur.uniform() < 0.3 * s
    motion = _motion_kernel(3 + 10 * s * r_blur.uniform(), r_blur.uniform(0, 180))
    if has_motion:
        img = cv2.filter2D(img, -1, motion)
    noise = 0.005 + 0.035 * s * r_noise.uniform(0.5, 1.0)
    img = img + noise * r_noise.standard_normal(img.shape).astype(np.float32)
    quality = int(round(95 - 60 * s * r_noise.uniform(0.5, 1.0)))
    params.update(blur_sigma=float(blur), motion_blur=bool(has_motion), noise_std=float(noise), jpeg_quality=quality)

    rgb8 = (np.clip(img, 0, 1) * 255 + 0.5).astype(np.uint8)
    ok, enc = cv2.imencode(".jpg", cv2.cvtColor(rgb8, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    jpeg = enc.tobytes()
    image = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    page_to_image = M @ page_px
    dots = np.column_stack([_apply(page_to_image, dots_mm[:, :2]), dots_mm[:, 2:5], amp])
    return RenderResult(
        image=image,
        jpeg=jpeg,
        dots=dots,
        ghost_dots=_apply(page_to_image, ghost_mm),
        page_to_image=page_to_image,
        px_per_mm=ppm,
        params=params,
    )


def _apply(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    if len(pts) == 0:
        return np.zeros((0, 2))
    p = np.column_stack([pts, np.ones(len(pts))]) @ H.T
    return p[:, :2] / p[:, 2:3]


def _stamp(height: np.ndarray, cx: float, cy: float, r: float, amp: float) -> None:
    """Add a dome of radius ``r`` px and peak ``amp`` centred at (cx, cy)."""
    x0, x1 = max(int(cx - r) - 1, 0), min(int(cx + r) + 2, height.shape[1])
    y0, y1 = max(int(cy - r) - 1, 0), min(int(cy + r) + 2, height.shape[0])
    if x0 >= x1 or y0 >= y1:
        return
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = ((xx - cx) ** 2 + (yy - cy) ** 2) / (r * r)
    height[y0:y1, x0:x1] += amp * np.clip(1 - d2, 0, None) ** 1.5


def _smooth_noise(
    rng: np.random.Generator, shape: tuple[int, int], size_mm: tuple[float, float], cell_mm: float
) -> np.ndarray:
    """Zero-mean, unit-ish noise with features about ``cell_mm`` across.

    The random grid is defined in page millimetres, so the same stream gives the
    same pattern at any rendering resolution.
    """
    h, w = shape
    gw, gh = int(np.ceil(size_mm[0] / cell_mm)) + 2, int(np.ceil(size_mm[1] / cell_mm)) + 2
    small = rng.standard_normal((gh, gw)).astype(np.float32)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def _soft_halfplane(rng, xn: np.ndarray, yn: np.ndarray, units_per_mm: float) -> np.ndarray:
    """Soft-edged shadow (e.g. of a hand or phone) over part of the page, in [0, 1].

    ``xn``, ``yn`` are page coordinates normalised to [-0.5, 0.5].
    """
    angle = rng.uniform(0, 2 * np.pi)
    px, py = rng.uniform(-0.3, 0.3, size=2)
    dist = (xn - px) * np.cos(angle) + (yn - py) * np.sin(angle)
    softness = rng.uniform(3, 20) * units_per_mm  # 3-20 mm, in normalised units
    return 1 / (1 + np.exp(-dist / softness))


def _random_perspective(rng, w: int, h: int, s: float) -> tuple[np.ndarray, tuple[int, int]]:
    """Homography placing the page, rotated and keystoned, inside a padded frame."""
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    angle = np.deg2rad(np.clip(rng.normal(0, 1 + 4 * s), -(2 + 10 * s), 2 + 10 * s))
    c, sn = np.cos(angle), np.sin(angle)
    centred = src - [w / 2, h / 2]
    dst = centred @ np.array([[c, sn], [-sn, c]], np.float32)
    dst += rng.uniform(-1, 1, size=(4, 2)) * 0.08 * s * min(w, h)
    pad = rng.uniform(0.02, 0.08) * min(w, h)
    dst = dst - dst.min(axis=0) + pad
    size = tuple(int(np.ceil(v + pad)) for v in dst.max(axis=0))
    return cv2.getPerspectiveTransform(src, dst.astype(np.float32)).astype(np.float64), size


def _background(rng, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    base = rng.uniform(0.1, 0.6) * rng.uniform(0.7, 1.0, size=3)
    tex = 1 + 0.15 * _smooth_noise(rng, (h, w), (w, h), cell_mm=max(4.0, min(w, h) / 30))
    return np.clip(base[None, None, :] * tex[..., None], 0, 1).astype(np.float32)


def _motion_kernel(length: float, angle_deg: float) -> np.ndarray:
    length = max(int(length), 3)
    k = np.zeros((length, length), np.float32)
    k[length // 2, :] = 1
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1)
    k = cv2.warpAffine(k, rot, (length, length))
    return k / k.sum()
