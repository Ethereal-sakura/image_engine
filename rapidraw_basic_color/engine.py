from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from .basic import (
    SCALES,
    apply_filmic_exposure,
    apply_highlights_adjustment,
    apply_linear_exposure,
    apply_tonal_adjustments,
    gaussian_blur_rgb,
)
from .colors import (
    agx_full_transform,
    apply_color_grading,
    apply_creative_color,
    apply_hsl_mixer,
    apply_white_balance,
    linear_to_srgb,
    srgb_to_linear,
)
from .io import load_image, save_image
from .params import BasicColorParams, ColorGrading, HslSettings


@dataclass
class StageRecorder:
    stages: dict[str, np.ndarray]

    def __init__(self) -> None:
        self.stages = {}

    def add_linear(self, name: str, image: np.ndarray) -> None:
        self.stages[name] = np.clip(linear_to_srgb(np.clip(image, 0.0, None)), 0.0, 1.0)

    def add_srgb(self, name: str, image: np.ndarray) -> None:
        self.stages[name] = np.clip(image.astype(np.float32), 0.0, 1.0)


@dataclass
class RenderOutput:
    image_srgb: np.ndarray
    recorder: Optional[StageRecorder] = None


class BasicColorRenderer:
    """RapidRAW Basic + Color 面板的非 RAW 离线版。"""

    def _normalize_basic(self, p: BasicColorParams) -> dict[str, float | str]:
        return {
            "tone_mapper": p.tone_mapper,
            "exposure": p.exposure / SCALES["exposure"],
            "brightness": p.brightness / SCALES["brightness"],
            "contrast": p.contrast / SCALES["contrast"],
            "highlights": p.highlights / SCALES["highlights"],
            "shadows": p.shadows / SCALES["shadows"],
            "whites": p.whites / SCALES["whites"],
            "blacks": p.blacks / SCALES["blacks"],
            "temperature": p.temperature / SCALES["temperature"],
            "tint": p.tint / SCALES["tint"],
            "saturation": p.saturation / SCALES["saturation"],
            "vibrance": p.vibrance / SCALES["vibrance"],
        }

    def _normalize_hsl(self, hsl: HslSettings) -> HslSettings:
        payload = {}
        for name in hsl.__dataclass_fields__.keys():
            band = getattr(hsl, name)
            payload[name] = {
                "hue": band.hue * SCALES["hsl_hue_multiplier"],
                "saturation": band.saturation / SCALES["hsl_saturation"],
                "luminance": band.luminance / SCALES["hsl_luminance"],
            }
        return HslSettings.from_dict(payload)

    def _normalize_grading(self, grading: ColorGrading) -> ColorGrading:
        data = {
            "shadows": {
                "hue": grading.shadows.hue,
                "saturation": grading.shadows.saturation / SCALES["color_grading_saturation"],
                "luminance": grading.shadows.luminance / SCALES["color_grading_luminance"],
            },
            "midtones": {
                "hue": grading.midtones.hue,
                "saturation": grading.midtones.saturation / SCALES["color_grading_saturation"],
                "luminance": grading.midtones.luminance / SCALES["color_grading_luminance"],
            },
            "highlights": {
                "hue": grading.highlights.hue,
                "saturation": grading.highlights.saturation / SCALES["color_grading_saturation"],
                "luminance": grading.highlights.luminance / SCALES["color_grading_luminance"],
            },
            "blending": grading.blending,
            "balance": grading.balance,
        }
        return ColorGrading.from_dict(data)

    def render_array(self, image_srgb_or_linear: np.ndarray, params: BasicColorParams, *, debug: bool = False) -> RenderOutput:
        recorder = StageRecorder() if debug else None
        if params.input_color_space == "linear":
            linear = image_srgb_or_linear.astype(np.float32)
        elif params.input_color_space == "srgb":
            linear = srgb_to_linear(image_srgb_or_linear.astype(np.float32))
        else:
            raise ValueError(f"Unsupported inputColorSpace: {params.input_color_space}")

        if recorder:
            recorder.add_linear("01_input", linear)

        norm = self._normalize_basic(params)
        tonal_blur = gaussian_blur_rgb(linear, sigma=3.5)

        working = apply_linear_exposure(linear, float(norm["exposure"]))
        working = apply_white_balance(working, float(norm["temperature"]), float(norm["tint"]))
        working = apply_filmic_exposure(working, float(norm["brightness"]))
        working = apply_tonal_adjustments(
            working,
            tonal_blur,
            float(norm["contrast"]),
            float(norm["shadows"]),
            float(norm["whites"]),
            float(norm["blacks"]),
        )
        working = apply_highlights_adjustment(working, float(norm["highlights"]))
        if recorder:
            recorder.add_linear("02_after_basic", working)

        working = apply_hsl_mixer(working, self._normalize_hsl(params.hsl))
        working = apply_color_grading(working, self._normalize_grading(params.color_grading))
        working = apply_creative_color(working, float(norm["saturation"]), float(norm["vibrance"]))
        if recorder:
            recorder.add_linear("03_after_color", working)

        if str(norm["tone_mapper"]) == "agx":
            out = np.clip(agx_full_transform(working), 0.0, 1.0).astype(np.float32)
        else:
            out = linear_to_srgb(np.clip(working, 0.0, None)).astype(np.float32)
        if recorder:
            recorder.add_srgb("04_output", out)
        return RenderOutput(image_srgb=out, recorder=recorder)

    def render_file(
        self,
        input_path: str | Path,
        params: BasicColorParams,
        output_path: str | Path,
        *,
        debug_dir: Optional[str | Path] = None,
        quality: int = 95,
    ) -> RenderOutput:
        image = load_image(input_path)
        result = self.render_array(image, params, debug=debug_dir is not None)
        save_image(output_path, result.image_srgb, quality=quality)

        if debug_dir and result.recorder:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            for name, img in result.recorder.stages.items():
                save_image(debug_path / f"{name}.png", img, quality=95)
        return result
