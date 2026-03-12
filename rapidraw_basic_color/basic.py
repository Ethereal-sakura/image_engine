from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

from .colors import get_luma, mix, smoothstep


SCALES = {
    "exposure": 0.8,
    "brightness": 0.8,
    "contrast": 100.0,
    "highlights": 120.0,
    "shadows": 120.0,
    "whites": 30.0,
    "blacks": 70.0,
    "temperature": 25.0,
    "tint": 100.0,
    "saturation": 100.0,
    "vibrance": 100.0,
    "color_grading_saturation": 500.0,
    "color_grading_luminance": 500.0,
    "color_grading_balance": 200.0,
    "hsl_hue_multiplier": 0.3,
    "hsl_saturation": 100.0,
    "hsl_luminance": 100.0,
}



def gaussian_blur_rgb(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0:
        return image.astype(np.float32, copy=True)
    return np.stack([gaussian_filter(image[..., i], sigma=sigma, mode="nearest") for i in range(3)], axis=-1).astype(np.float32)



def apply_linear_exposure(image: np.ndarray, exposure_adj: float) -> np.ndarray:
    if abs(exposure_adj) <= 1e-8:
        return image.astype(np.float32)
    return (image * (2.0 ** exposure_adj)).astype(np.float32)



def apply_filmic_exposure(image: np.ndarray, brightness_adj: float) -> np.ndarray:
    if abs(brightness_adj) <= 1e-8:
        return image.astype(np.float32)

    luma = get_luma(image)
    mask = np.abs(luma) > 1e-5
    if not np.any(mask):
        return image.astype(np.float32)

    direct_adj = brightness_adj * (1.0 - 0.95)
    rational_adj = brightness_adj * 0.95
    scale = 2.0 ** direct_adj
    k = 2.0 ** (-rational_adj * 1.2)

    luma_abs = np.abs(luma)
    luma_floor = np.floor(luma_abs)
    luma_fract = luma_abs - luma_floor
    shaped_fract = luma_fract / np.maximum(luma_fract + (1.0 - luma_fract) * k, 1e-6)
    shaped_luma_abs = luma_floor + shaped_fract
    new_luma = np.sign(luma) * shaped_luma_abs * scale

    chroma = image - luma[..., None]
    total_scale = np.ones_like(luma, dtype=np.float32)
    total_scale[mask] = new_luma[mask] / luma[mask]
    chroma_scale = np.power(np.maximum(total_scale, 1e-6), 0.8)
    return (new_luma[..., None] + chroma * chroma_scale[..., None]).astype(np.float32)



def get_shadow_mult(luma: np.ndarray, shadows_adj: float, blacks_adj: float) -> np.ndarray:
    mult = np.ones_like(luma, dtype=np.float32)
    safe = np.maximum(luma, 1e-4)

    if abs(blacks_adj) > 1e-8:
        limit = 0.05
        x = np.clip(safe / limit, 0.0, 1.0)
        mask = (1.0 - x) ** 2
        factor = min(2.0 ** (blacks_adj * 0.75), 3.9)
        mult *= 1.0 + (factor - 1.0) * mask

    if abs(shadows_adj) > 1e-8:
        limit = 0.1
        x = np.clip(safe / limit, 0.0, 1.0)
        mask = (1.0 - x) ** 2
        factor = min(2.0 ** (shadows_adj * 1.5), 3.9)
        mult *= 1.0 + (factor - 1.0) * mask

    return mult.astype(np.float32)



def apply_tonal_adjustments(image: np.ndarray, tonal_blur: np.ndarray, contrast_adj: float, shadows_adj: float, whites_adj: float, blacks_adj: float) -> np.ndarray:
    rgb = image.astype(np.float32, copy=True)
    blur = tonal_blur.astype(np.float32, copy=True)

    if abs(whites_adj) > 1e-8:
        white_level = 1.0 - whites_adj * 0.25
        gain = 1.0 / max(white_level, 0.01)
        rgb *= gain
        blur *= gain

    pixel_luma = get_luma(np.maximum(rgb, 0.0))
    blur_luma = get_luma(np.maximum(blur, 0.0))
    safe_pixel = np.maximum(pixel_luma, 1e-4)
    safe_blur = np.maximum(blur_luma, 1e-4)

    halo_protection = smoothstep(0.05, 0.25, np.abs(np.sqrt(safe_pixel) - np.sqrt(safe_blur)))

    if abs(shadows_adj) > 1e-8 or abs(blacks_adj) > 1e-8:
        spatial = get_shadow_mult(safe_blur, shadows_adj, blacks_adj)
        pixel = get_shadow_mult(safe_pixel, shadows_adj, blacks_adj)
        mult = spatial * (1.0 - halo_protection) + pixel * halo_protection
        rgb *= mult[..., None]

    if abs(contrast_adj) > 1e-8:
        safe_rgb = np.maximum(rgb, 0.0)
        perceptual = np.power(safe_rgb, 1.0 / 2.2)
        perceptual = np.clip(perceptual, 0.0, 1.0)
        strength = 2.0 ** (contrast_adj * 1.25)
        low = 0.5 * np.power(2.0 * perceptual, strength)
        high = 1.0 - 0.5 * np.power(2.0 * (1.0 - perceptual), strength)
        curved = np.where(perceptual < 0.5, low, high)
        contrast_rgb = np.power(np.maximum(curved, 0.0), 2.2)
        mix_factor = smoothstep(1.0, 1.01, safe_rgb)
        rgb = contrast_rgb * (1.0 - mix_factor) + rgb * mix_factor

    return rgb.astype(np.float32)



def apply_highlights_adjustment(image: np.ndarray, highlights_adj: float) -> np.ndarray:
    if abs(highlights_adj) <= 1e-8:
        return image.astype(np.float32)

    luma = get_luma(np.maximum(image, 0.0))
    safe_luma = np.maximum(luma, 1e-4)
    highlight_mask = smoothstep(0.3, 0.95, np.tanh(safe_luma * 1.5))

    if highlights_adj < 0.0:
        new_luma = luma.copy()
        low_mask = luma <= 1.0
        gamma = 1.0 - highlights_adj * 1.75
        new_luma[low_mask] = np.power(np.maximum(luma[low_mask], 0.0), gamma)
        high_mask = ~low_mask
        excess = luma[high_mask] - 1.0
        compression = -highlights_adj * 6.0
        compressed_excess = excess / (1.0 + excess * compression)
        new_luma[high_mask] = 1.0 + compressed_excess
        toned = image * (new_luma / np.maximum(luma, 1e-4))[..., None]
        desat = smoothstep(1.0, 10.0, luma)[..., None]
        white_point = np.repeat(new_luma[..., None], 3, axis=-1)
        final = mix(toned, white_point, desat)
    else:
        final = image * (2.0 ** (highlights_adj * 1.75))

    return (image * (1.0 - highlight_mask[..., None]) + final * highlight_mask[..., None]).astype(np.float32)
