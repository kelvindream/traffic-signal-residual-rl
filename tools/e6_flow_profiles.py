import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


_FLOW_GROUPS = {"horizontal", "vertical", "turn"}


def validate_e6_profile(intensity, direction_profile=None, flow_groups=None):
    try:
        intensity_value = float(intensity)
    except (TypeError, ValueError) as exc:
        raise ValueError("intensity must be a finite positive number") from exc
    if not math.isfinite(intensity_value) or intensity_value <= 0:
        raise ValueError("intensity must be a finite positive number")

    if direction_profile is not None:
        for multiplier in direction_profile.values():
            try:
                multiplier_value = float(multiplier)
            except (TypeError, ValueError) as exc:
                raise ValueError("direction profile multipliers must be finite and nonnegative") from exc
            if not math.isfinite(multiplier_value) or multiplier_value < 0:
                raise ValueError("direction profile multipliers must be finite and nonnegative")

    if flow_groups is not None:
        for group in flow_groups.values():
            if group not in _FLOW_GROUPS:
                raise ValueError(f"invalid flow group: {group}")


def scale_route_xml(xml_text, intensity=1.0, direction_profile=None, flow_groups=None) -> str:
    validate_e6_profile(intensity, direction_profile, flow_groups)

    root = ET.fromstring(xml_text)
    base_multiplier = float(intensity)
    profile = direction_profile or {}
    groups = flow_groups or {}
    default_multiplier = float(profile.get("default", 1.0))

    for flow in root.iter("flow"):
        multiplier = base_multiplier
        if direction_profile is not None:
            group = groups.get(flow.get("id"))
            if group in _FLOW_GROUPS:
                multiplier *= float(profile.get(group, default_multiplier))
            else:
                multiplier *= default_multiplier

        if multiplier != 1.0:
            scaled_rate = float(flow.attrib["vehsPerHour"]) * multiplier
            flow.set("vehsPerHour", format(scaled_rate, ".12g"))

    return ET.tostring(root, encoding="unicode")


def make_scaled_route_file(
    source_path,
    intensity=1.0,
    direction_profile=None,
    flow_groups=None,
    temp_dir=None,
) -> Path:
    source = Path(source_path)
    scaled_xml = scale_route_xml(
        source.read_text(encoding="utf-8"),
        intensity=intensity,
        direction_profile=direction_profile,
        flow_groups=flow_groups,
    )

    directory = None if temp_dir is None else str(temp_dir)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".rou.xml",
        prefix="e6_scaled_",
        dir=directory,
        delete=False,
    ) as temporary_file:
        temporary_file.write(scaled_xml)
        return Path(temporary_file.name)


def cleanup_scaled_route_file(path):
    Path(path).unlink(missing_ok=True)
