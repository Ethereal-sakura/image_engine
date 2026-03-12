from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import json


HSL_BAND_NAMES = [
    "reds",
    "oranges",
    "yellows",
    "greens",
    "aquas",
    "blues",
    "purples",
    "magentas",
]


@dataclass
class HslBand:
    hue: float = 0.0
    saturation: float = 0.0
    luminance: float = 0.0

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "HslBand":
        data = data or {}
        return HslBand(
            hue=float(data.get("hue", 0.0)),
            saturation=float(data.get("saturation", 0.0)),
            luminance=float(data.get("luminance", 0.0)),
        )


@dataclass
class HslSettings:
    reds: HslBand = field(default_factory=HslBand)
    oranges: HslBand = field(default_factory=HslBand)
    yellows: HslBand = field(default_factory=HslBand)
    greens: HslBand = field(default_factory=HslBand)
    aquas: HslBand = field(default_factory=HslBand)
    blues: HslBand = field(default_factory=HslBand)
    purples: HslBand = field(default_factory=HslBand)
    magentas: HslBand = field(default_factory=HslBand)

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "HslSettings":
        data = data or {}
        return HslSettings(**{name: HslBand.from_dict(data.get(name)) for name in HSL_BAND_NAMES})


@dataclass
class ColorGradeBand:
    hue: float = 0.0
    saturation: float = 0.0
    luminance: float = 0.0

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "ColorGradeBand":
        data = data or {}
        return ColorGradeBand(
            hue=float(data.get("hue", 0.0)),
            saturation=float(data.get("saturation", 0.0)),
            luminance=float(data.get("luminance", 0.0)),
        )


@dataclass
class ColorGrading:
    shadows: ColorGradeBand = field(default_factory=ColorGradeBand)
    midtones: ColorGradeBand = field(default_factory=ColorGradeBand)
    highlights: ColorGradeBand = field(default_factory=ColorGradeBand)
    blending: float = 50.0
    balance: float = 0.0

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "ColorGrading":
        data = data or {}
        return ColorGrading(
            shadows=ColorGradeBand.from_dict(data.get("shadows")),
            midtones=ColorGradeBand.from_dict(data.get("midtones")),
            highlights=ColorGradeBand.from_dict(data.get("highlights")),
            blending=float(data.get("blending", 50.0)),
            balance=float(data.get("balance", 0.0)),
        )


@dataclass
class ColorCalibration:
    shadows_tint: float = 0.0
    red_hue: float = 0.0
    red_saturation: float = 0.0
    green_hue: float = 0.0
    green_saturation: float = 0.0
    blue_hue: float = 0.0
    blue_saturation: float = 0.0

    @staticmethod
    def from_dict(data: Optional[Dict[str, Any]]) -> "ColorCalibration":
        data = data or {}
        return ColorCalibration(
            shadows_tint=float(data.get("shadowsTint", data.get("shadows_tint", 0.0))),
            red_hue=float(data.get("redHue", data.get("red_hue", 0.0))),
            red_saturation=float(data.get("redSaturation", data.get("red_saturation", 0.0))),
            green_hue=float(data.get("greenHue", data.get("green_hue", 0.0))),
            green_saturation=float(data.get("greenSaturation", data.get("green_saturation", 0.0))),
            blue_hue=float(data.get("blueHue", data.get("blue_hue", 0.0))),
            blue_saturation=float(data.get("blueSaturation", data.get("blue_saturation", 0.0))),
        )


@dataclass
class BasicColorParams:
    input_color_space: str = "srgb"
    tone_mapper: str = "basic"

    # Basic panel
    exposure: float = 0.0
    brightness: float = 0.0
    contrast: float = 0.0
    highlights: float = 0.0
    shadows: float = 0.0
    whites: float = 0.0
    blacks: float = 0.0

    # Color panel
    temperature: float = 0.0
    tint: float = 0.0
    saturation: float = 0.0
    vibrance: float = 0.0
    hsl: HslSettings = field(default_factory=HslSettings)
    color_grading: ColorGrading = field(default_factory=ColorGrading)
    color_calibration: ColorCalibration = field(default_factory=ColorCalibration)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BasicColorParams":
        data = dict(data or {})
        return cls(
            input_color_space=str(data.get("inputColorSpace", data.get("input_color_space", "srgb"))).lower(),
            tone_mapper=str(data.get("toneMapper", data.get("tone_mapper", "basic"))).lower(),
            exposure=float(data.get("exposure", 0.0)),
            brightness=float(data.get("brightness", 0.0)),
            contrast=float(data.get("contrast", 0.0)),
            highlights=float(data.get("highlights", 0.0)),
            shadows=float(data.get("shadows", 0.0)),
            whites=float(data.get("whites", 0.0)),
            blacks=float(data.get("blacks", 0.0)),
            temperature=float(data.get("temperature", 0.0)),
            tint=float(data.get("tint", 0.0)),
            saturation=float(data.get("saturation", 0.0)),
            vibrance=float(data.get("vibrance", 0.0)),
            hsl=HslSettings.from_dict(data.get("hsl")),
            color_grading=ColorGrading.from_dict(data.get("colorGrading", data.get("color_grading"))),
            color_calibration=ColorCalibration.from_dict(data.get("colorCalibration", data.get("color_calibration"))),
        )

    @classmethod
    def from_json_file(cls, path: str | Path) -> "BasicColorParams":
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        return cls.from_dict(payload)
