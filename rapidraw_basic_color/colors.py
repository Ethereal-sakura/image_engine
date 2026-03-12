from __future__ import annotations

import math
from functools import lru_cache
from typing import Dict

import numpy as np

from .params import ColorGrading, HSL_BAND_NAMES, HslSettings


LUMA_COEFF = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
WP_D65 = np.array([0.3127, 0.3290], dtype=np.float32)
PRIMARIES_SRGB = np.array([[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], dtype=np.float32)
PRIMARIES_REC2020 = np.array([[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]], dtype=np.float32)
HSL_RANGES = {
    "reds": (358.0, 35.0),
    "oranges": (25.0, 45.0),
    "yellows": (60.0, 40.0),
    "greens": (115.0, 90.0),
    "aquas": (180.0, 60.0),
    "blues": (225.0, 60.0),
    "purples": (280.0, 55.0),
    "magentas": (330.0, 50.0),
}



def get_luma(rgb: np.ndarray) -> np.ndarray:
    return np.tensordot(rgb.astype(np.float32), LUMA_COEFF, axes=([-1], [0]))



def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    return np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4).astype(np.float32)



def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    return np.where(rgb <= 0.0031308, rgb * 12.92, 1.055 * np.power(rgb, 1.0 / 2.4) - 0.055).astype(np.float32)



def smoothstep(edge0: float | np.ndarray, edge1: float | np.ndarray, x: np.ndarray) -> np.ndarray:
    edge0_arr = np.asarray(edge0, dtype=np.float32)
    edge1_arr = np.asarray(edge1, dtype=np.float32)
    denom = np.maximum(edge1_arr - edge0_arr, 1e-8)
    t = np.clip((x - edge0_arr) / denom, 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)



def mix(a: np.ndarray, b: np.ndarray, t: np.ndarray | float) -> np.ndarray:
    return (a * (1.0 - t) + b * t).astype(np.float32)



def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    cmax = np.max(rgb, axis=-1)
    cmin = np.min(rgb, axis=-1)
    delta = cmax - cmin

    hue = np.zeros_like(cmax, dtype=np.float32)
    nz = delta > 1e-8
    rmask = nz & (cmax == r)
    gmask = nz & (cmax == g)
    bmask = nz & (cmax == b)
    hue[rmask] = (60.0 * ((g[rmask] - b[rmask]) / delta[rmask])) % 360.0
    hue[gmask] = 60.0 * (((b[gmask] - r[gmask]) / delta[gmask]) + 2.0)
    hue[bmask] = 60.0 * (((r[bmask] - g[bmask]) / delta[bmask]) + 4.0)
    sat = np.where(cmax > 1e-8, delta / np.maximum(cmax, 1e-8), 0.0).astype(np.float32)
    return np.stack([hue, sat, cmax], axis=-1).astype(np.float32)



def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    hsv = hsv.astype(np.float32)
    h = hsv[..., 0] % 360.0
    s = np.clip(hsv[..., 1], 0.0, 1.0)
    v = np.clip(hsv[..., 2], 0.0, None)
    c = v * s
    x = c * (1.0 - np.abs(((h / 60.0) % 2.0) - 1.0))
    m = v - c

    rgb = np.zeros_like(hsv, dtype=np.float32)
    conds = [
        h < 60.0,
        (h >= 60.0) & (h < 120.0),
        (h >= 120.0) & (h < 180.0),
        (h >= 180.0) & (h < 240.0),
        (h >= 240.0) & (h < 300.0),
        h >= 300.0,
    ]
    vals = [
        (c, x, 0.0),
        (x, c, 0.0),
        (0.0, c, x),
        (0.0, x, c),
        (x, 0.0, c),
        (c, 0.0, x),
    ]
    for cond, (rv, gv, bv) in zip(conds, vals):
        rgb[..., 0] = np.where(cond, rv, rgb[..., 0])
        rgb[..., 1] = np.where(cond, gv, rgb[..., 1])
        rgb[..., 2] = np.where(cond, bv, rgb[..., 2])
    return (rgb + m[..., None]).astype(np.float32)



def apply_white_balance(rgb: np.ndarray, temperature: float, tint: float) -> np.ndarray:
    temp_mult = np.array([1.0 + temperature * 0.2, 1.0 + temperature * 0.05, 1.0 - temperature * 0.2], dtype=np.float32)
    tint_mult = np.array([1.0 + tint * 0.25, 1.0 - tint * 0.25, 1.0 + tint * 0.25], dtype=np.float32)
    return (rgb * temp_mult * tint_mult).astype(np.float32)



def apply_creative_color(rgb: np.ndarray, saturation: float, vibrance: float) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    luma = get_luma(rgb)[..., None]
    out = rgb.copy()

    if abs(saturation) > 1e-8:
        out = mix(luma, out, 1.0 + saturation)
    if abs(vibrance) <= 1e-8:
        return out

    cmax = np.max(out, axis=-1)
    cmin = np.min(out, axis=-1)
    delta = cmax - cmin
    current_sat = delta / np.maximum(cmax, 1e-3)
    hsv = rgb_to_hsv(np.clip(out, 0.0, None))
    hue = hsv[..., 0]

    if vibrance > 0.0:
        sat_mask = 1.0 - smoothstep(0.4, 0.9, current_sat)
        hue_dist = np.minimum(np.abs(hue - 25.0), 360.0 - np.abs(hue - 25.0))
        skin_mask = smoothstep(35.0, 10.0, hue_dist)
        skin_dampener = 1.0 + (0.6 - 1.0) * skin_mask
        amount = vibrance * sat_mask * skin_dampener * 3.0
    else:
        desat_mask = 1.0 - smoothstep(0.2, 0.8, current_sat)
        amount = vibrance * desat_mask

    gray = np.repeat(get_luma(out)[..., None], 3, axis=-1)
    return mix(gray, out, (1.0 + amount)[..., None])




def _raw_hsl_influence(hue: np.ndarray, center: float, width: float) -> np.ndarray:
    dist = np.minimum(np.abs(hue - center), 360.0 - np.abs(hue - center))
    falloff = dist / max(width * 0.5, 1e-6)
    return np.exp(-1.5 * falloff * falloff).astype(np.float32)



def apply_hsl_mixer(rgb: np.ndarray, settings: HslSettings) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    hsv = rgb_to_hsv(np.clip(rgb, 0.0, None))
    original_luma = get_luma(rgb)
    sat_mask = smoothstep(0.05, 0.20, hsv[..., 1])
    lum_weight = smoothstep(0.0, 1.0, hsv[..., 1])
    hue = hsv[..., 0]

    infs = []
    for name in HSL_BAND_NAMES:
        center, width = HSL_RANGES[name]
        infs.append(_raw_hsl_influence(hue, center, width))
    inf = np.stack(infs, axis=-1)
    inf_sum = np.maximum(np.sum(inf, axis=-1, keepdims=True), 1e-8)
    inf = inf / inf_sum

    total_hue = np.zeros_like(original_luma)
    total_sat = np.zeros_like(original_luma)
    total_lum = np.zeros_like(original_luma)
    for i, name in enumerate(HSL_BAND_NAMES):
        band = getattr(settings, name)
        hs_inf = inf[..., i] * sat_mask
        l_inf = inf[..., i] * lum_weight
        total_hue += band.hue * 2.0 * hs_inf
        total_sat += band.saturation * hs_inf
        total_lum += band.luminance * l_inf

    new_hsv = hsv.copy()
    new_hsv[..., 0] = (new_hsv[..., 0] + total_hue + 360.0) % 360.0
    new_hsv[..., 1] = np.clip(new_hsv[..., 1] * (1.0 + total_sat), 0.0, 1.0)
    hs_rgb = hsv_to_rgb(new_hsv)
    new_luma = get_luma(hs_rgb)
    target_luma = original_luma * (1.0 + total_lum)
    scale = target_luma / np.maximum(new_luma, 1e-4)
    return (hs_rgb * scale[..., None]).astype(np.float32)



def apply_color_grading(rgb: np.ndarray, grading: ColorGrading) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    luma = get_luma(np.maximum(rgb, 0.0))
    balance = grading.balance / 200.0
    blending = grading.blending / 100.0

    base_shadow_crossover = 0.1
    base_highlight_crossover = 0.5
    balance_range = 0.5
    shadow_crossover = base_shadow_crossover + max(0.0, -balance) * balance_range
    highlight_crossover = base_highlight_crossover - max(0.0, balance) * balance_range
    feather = 0.2 * blending
    final_shadow_crossover = min(shadow_crossover, highlight_crossover - 0.01)

    shadow_mask = 1.0 - smoothstep(final_shadow_crossover - feather, final_shadow_crossover + feather, luma)
    highlight_mask = smoothstep(highlight_crossover - feather, highlight_crossover + feather, luma)
    midtone_mask = np.maximum(0.0, 1.0 - shadow_mask - highlight_mask)

    graded = rgb.copy()
    strengths = {
        "shadow_sat": 0.3,
        "shadow_lum": 0.5,
        "midtone_sat": 0.6,
        "midtone_lum": 0.8,
        "highlight_sat": 0.8,
        "highlight_lum": 1.0,
    }

    if grading.shadows.saturation > 1e-3:
        tint_rgb = hsv_to_rgb(np.dstack([
            np.full_like(luma, grading.shadows.hue),
            np.ones_like(luma),
            np.ones_like(luma),
        ]))
        graded += (tint_rgb - 0.5) * (grading.shadows.saturation / 500.0) * shadow_mask[..., None] * strengths["shadow_sat"]
    graded += (grading.shadows.luminance / 500.0) * shadow_mask[..., None] * strengths["shadow_lum"]

    if grading.midtones.saturation > 1e-3:
        tint_rgb = hsv_to_rgb(np.dstack([
            np.full_like(luma, grading.midtones.hue),
            np.ones_like(luma),
            np.ones_like(luma),
        ]))
        graded += (tint_rgb - 0.5) * (grading.midtones.saturation / 500.0) * midtone_mask[..., None] * strengths["midtone_sat"]
    graded += (grading.midtones.luminance / 500.0) * midtone_mask[..., None] * strengths["midtone_lum"]

    if grading.highlights.saturation > 1e-3:
        tint_rgb = hsv_to_rgb(np.dstack([
            np.full_like(luma, grading.highlights.hue),
            np.ones_like(luma),
            np.ones_like(luma),
        ]))
        graded += (tint_rgb - 0.5) * (grading.highlights.saturation / 500.0) * highlight_mask[..., None] * strengths["highlight_sat"]
    graded += (grading.highlights.luminance / 500.0) * highlight_mask[..., None] * strengths["highlight_lum"]

    return graded.astype(np.float32)



def _xy_to_xyz(xy: np.ndarray) -> np.ndarray:
    if xy[1] < 1e-6:
        return np.zeros(3, dtype=np.float32)
    return np.array([xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]], dtype=np.float32)



def _primaries_to_xyz_matrix(primaries: np.ndarray, white_point: np.ndarray) -> np.ndarray:
    r_xyz = _xy_to_xyz(primaries[0])
    g_xyz = _xy_to_xyz(primaries[1])
    b_xyz = _xy_to_xyz(primaries[2])
    pm = np.stack([r_xyz, g_xyz, b_xyz], axis=1)
    white_xyz = _xy_to_xyz(white_point)
    s = np.linalg.inv(pm) @ white_xyz
    return np.stack([r_xyz * s[0], g_xyz * s[1], b_xyz * s[2]], axis=1).astype(np.float32)



def _rotate_and_scale_primary(primary: np.ndarray, white_point: np.ndarray, scale: float, rotation: float) -> np.ndarray:
    p_rel = primary - white_point
    p_scaled = p_rel * scale
    sin_r = math.sin(rotation)
    cos_r = math.cos(rotation)
    return np.array([
        white_point[0] + p_scaled[0] * cos_r - p_scaled[1] * sin_r,
        white_point[1] + p_scaled[0] * sin_r + p_scaled[1] * cos_r,
    ], dtype=np.float32)


@lru_cache(maxsize=1)
def agx_matrices() -> Dict[str, np.ndarray]:
    pipe_work_profile_to_xyz = _primaries_to_xyz_matrix(PRIMARIES_SRGB, WP_D65)
    base_profile_to_xyz = _primaries_to_xyz_matrix(PRIMARIES_REC2020, WP_D65)
    xyz_to_base = np.linalg.inv(base_profile_to_xyz)
    pipe_to_base = xyz_to_base @ pipe_work_profile_to_xyz

    inset = [0.29462451, 0.25861925, 0.14641371]
    rotation = [0.03540329, -0.02108586, -0.06305724]
    outset = [0.290776401758, 0.263155400753, 0.045810721815]

    inset_rot = np.stack([
        _rotate_and_scale_primary(PRIMARIES_REC2020[i], WP_D65, 1.0 - inset[i], rotation[i])
        for i in range(3)
    ], axis=0)
    rendering_to_xyz = _primaries_to_xyz_matrix(inset_rot, WP_D65)
    base_to_rendering = xyz_to_base @ rendering_to_xyz

    outset_unrot = np.stack([
        _rotate_and_scale_primary(PRIMARIES_REC2020[i], WP_D65, 1.0 - outset[i], 0.0)
        for i in range(3)
    ], axis=0)
    outset_to_xyz = _primaries_to_xyz_matrix(outset_unrot, WP_D65)
    rendering_to_base = np.linalg.inv(xyz_to_base @ outset_to_xyz)

    return {
        "pipe_to_rendering": (base_to_rendering @ pipe_to_base).astype(np.float32),
        "rendering_to_pipe": (np.linalg.inv(pipe_to_base) @ rendering_to_base).astype(np.float32),
    }



def agx_full_transform(rgb: np.ndarray) -> np.ndarray:
    mats = agx_matrices()
    rgb = rgb.astype(np.float32)
    min_c = np.min(rgb, axis=-1, keepdims=True)
    compressed = np.where(min_c < 0.0, rgb - min_c, rgb)
    flat = compressed.reshape(-1, 3)
    agx_space = flat @ mats["pipe_to_rendering"].T

    epsilon = 1.0e-6
    min_ev = -15.2
    max_ev = 5.0
    range_ev = max_ev - min_ev
    x_relative = np.maximum(agx_space / 0.18, epsilon)
    mapped = np.clip((np.log2(x_relative) - min_ev) / range_ev, 0.0, 1.0)

    slope = 2.3843
    toe_power = 1.5
    shoulder_power = 1.5
    tx = 0.6060606
    ty = 0.43446
    intercept = -1.0112
    toe_scale = -1.0359
    shoulder_scale = 1.3475

    def sigmoid(v: np.ndarray, power: float) -> np.ndarray:
        return v / np.power(1.0 + np.power(v, power), 1.0 / power)

    def scaled_sigmoid(v: np.ndarray, scale: float, power: float) -> np.ndarray:
        return scale * sigmoid(slope * (v - tx) / scale, power) + ty

    curved = np.empty_like(mapped, dtype=np.float32)
    toe_mask = mapped < tx
    shoulder_mask = mapped > tx
    linear_mask = ~(toe_mask | shoulder_mask)
    curved[toe_mask] = scaled_sigmoid(mapped[toe_mask], toe_scale, toe_power)
    curved[linear_mask] = slope * mapped[linear_mask] + intercept
    curved[shoulder_mask] = scaled_sigmoid(mapped[shoulder_mask], shoulder_scale, shoulder_power)
    curved = np.clip(curved, 0.0, 1.0)
    curved = np.power(curved, 2.4)
    out = curved @ mats["rendering_to_pipe"].T
    return out.reshape(rgb.shape).astype(np.float32)
