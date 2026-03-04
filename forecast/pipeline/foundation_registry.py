from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FoundationSpec:
    key: str
    model_name: str
    default_model_id: str
    backend_status: str
    backend_note: str


FOUNDATION_REGISTRY: dict[str, FoundationSpec] = {
    "chronos2": FoundationSpec(
        key="chronos2",
        model_name="chronos2_zero_shot",
        default_model_id="amazon/chronos-2",
        backend_status="native",
        backend_note="Chronos-2 adapter using chronos package",
    ),
    "timesfm": FoundationSpec(
        key="timesfm",
        model_name="timesfm_zero_shot",
        default_model_id="google/timesfm-2.0-500m-pytorch",
        backend_status="native",
        backend_note="TimesFM adapter using timesfm package",
    ),
    "moirai": FoundationSpec(
        key="moirai",
        model_name="moirai_zero_shot",
        default_model_id="Salesforce/moirai-1.1-R-small",
        backend_status="native",
        backend_note="Moirai adapter using uni2ts package",
    ),
}


def get_foundation_spec(model_key: str) -> FoundationSpec:
    key = str(model_key).strip().lower()
    if key not in FOUNDATION_REGISTRY:
        raise ValueError(f"Unknown foundation model key: {model_key}")
    return FOUNDATION_REGISTRY[key]
