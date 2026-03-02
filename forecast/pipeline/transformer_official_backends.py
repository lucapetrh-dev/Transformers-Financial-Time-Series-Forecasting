from __future__ import annotations

from dataclasses import dataclass

from forecast.pipeline.transformers import ITransformerLikeRegressor, PatchTSTLikeRegressor


CANONICAL_MODEL_FAMILIES = ("itransformer", "patchtst")


@dataclass(frozen=True)
class CanonicalBackendConfig:
    model_family: str
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    dropout: float = 0.1
    lr: float = 1e-3
    patch_len: int | None = None
    stride: int | None = None

    def validated(self) -> "CanonicalBackendConfig":
        family = canonical_model_name(self.model_family)
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.patch_len is not None and self.patch_len <= 0:
            raise ValueError("patch_len must be positive when provided")
        if self.stride is not None and self.stride <= 0:
            raise ValueError("stride must be positive when provided")
        return CanonicalBackendConfig(
            model_family=family,
            d_model=int(self.d_model),
            n_layers=int(self.n_layers),
            n_heads=int(self.n_heads),
            dropout=float(self.dropout),
            lr=float(self.lr),
            patch_len=int(self.patch_len) if self.patch_len is not None else None,
            stride=int(self.stride) if self.stride is not None else None,
        )


def canonical_model_name(model_family: str) -> str:
    key = str(model_family).strip().lower()
    if key in {"itransformer", "itransformer_like"}:
        return "itransformer"
    if key in {"patchtst", "patchtst_like"}:
        return "patchtst"
    raise ValueError(f"Unknown canonical model family: {model_family}")


def parse_canonical_models(value: str) -> list[str]:
    models = [canonical_model_name(m.strip()) for m in value.split(",") if m.strip()]
    if not models:
        raise ValueError("No canonical models requested")
    return models


def build_canonical_model(
    cfg: CanonicalBackendConfig,
    *,
    lookback: int,
    n_features: int,
    output_dim: int = 1,
):
    family = canonical_model_name(cfg.model_family)
    if family == "itransformer":
        return ITransformerLikeRegressor(
            lookback=lookback,
            n_features=n_features,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
            output_dim=output_dim,
        )

    patch_len = cfg.patch_len if cfg.patch_len is not None else max(4, lookback // 10)
    stride = cfg.stride if cfg.stride is not None else max(2, lookback // 20)
    return PatchTSTLikeRegressor(
        lookback=lookback,
        n_features=n_features,
        patch_len=patch_len,
        stride=stride,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        dropout=cfg.dropout,
        output_dim=output_dim,
    )


def config_provenance(cfg: CanonicalBackendConfig) -> dict[str, object]:
    return {
        "canonical_backend": canonical_model_name(cfg.model_family),
        "d_model": int(cfg.d_model),
        "n_layers": int(cfg.n_layers),
        "n_heads": int(cfg.n_heads),
        "dropout": float(cfg.dropout),
        "lr": float(cfg.lr),
        "patch_len": int(cfg.patch_len) if cfg.patch_len is not None else None,
        "stride": int(cfg.stride) if cfg.stride is not None else None,
    }
