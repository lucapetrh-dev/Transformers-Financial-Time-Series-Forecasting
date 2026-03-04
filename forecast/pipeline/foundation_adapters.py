from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from forecast.pipeline.foundation_registry import FoundationSpec, get_foundation_spec


@dataclass
class FoundationPrediction:
    point: float
    q10: float
    q50: float
    q90: float


class FoundationAdapter(Protocol):
    model_name: str
    backend_status: str
    backend_note: str

    def predict_batch(self, contexts: list[np.ndarray], *, horizon: int) -> list[FoundationPrediction]:
        ...


def _safe_quantiles_from_samples(samples: np.ndarray) -> tuple[float, float, float]:
    q10 = float(np.quantile(samples, 0.1))
    q50 = float(np.quantile(samples, 0.5))
    q90 = float(np.quantile(samples, 0.9))
    return q10, q50, q90


class Chronos2NativeAdapter:
    model_name = "chronos2_zero_shot"
    backend_status = "native"

    def __init__(self, *, model_id: str, batch_size: int, context_length: int) -> None:
        self.model_id = model_id
        self.batch_size = int(batch_size)
        self.context_length = int(context_length)
        self.backend_note = "Chronos-2 adapter using chronos package"
        self._pipeline = None

    def _load(self):
        if self._pipeline is not None:
            return self._pipeline
        try:
            from chronos import Chronos2Pipeline
        except Exception as exc:  # pragma: no cover - import is environment-dependent
            raise ImportError("chronos package is required for native Chronos-2 runs") from exc

        self._pipeline = Chronos2Pipeline.from_pretrained(self.model_id, device_map="cpu")
        return self._pipeline

    def predict_batch(self, contexts: list[np.ndarray], *, horizon: int) -> list[FoundationPrediction]:
        if not contexts:
            return []
        pipeline = self._load()
        inputs = [ctx.reshape(1, -1).astype(np.float32) for ctx in contexts]
        q_list, mean_list = pipeline.predict_quantiles(
            inputs,
            prediction_length=horizon,
            quantile_levels=[0.1, 0.5, 0.9],
            batch_size=self.batch_size,
            context_length=self.context_length,
        )

        out: list[FoundationPrediction] = []
        for series_input, q_fcst, mean_fcst in zip(inputs, q_list, mean_list):
            last_obs = float(series_input[0, -1])
            q_last = q_fcst[0, horizon - 1, :].detach().cpu().numpy()
            mean_last = float(mean_fcst[0, horizon - 1].detach().cpu().item())
            out.append(
                FoundationPrediction(
                    point=float(mean_last - last_obs),
                    q10=float(q_last[0] - last_obs),
                    q50=float(q_last[1] - last_obs),
                    q90=float(q_last[2] - last_obs),
                )
            )
        return out


class TimesFMNativeAdapter:
    model_name = "timesfm_zero_shot"
    backend_status = "native"

    def __init__(self, *, model_id: str, batch_size: int, context_length: int) -> None:
        self.model_id = model_id
        self.batch_size = int(batch_size)
        self.context_length = int(context_length)
        self.backend_note = "TimesFM adapter using timesfm package"
        self._model = None
        self._quantiles = [0.1 * k for k in range(1, 10)]

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            import timesfm
        except Exception as exc:  # pragma: no cover
            raise ImportError("timesfm package is required for TimesFM runs") from exc

        hparams = timesfm.TimesFmHparams(
            backend="cpu",
            per_core_batch_size=max(1, self.batch_size),
            context_len=max(128, self.context_length),
            horizon_len=128,
            input_patch_len=32,
            output_patch_len=128,
            num_layers=50,
            model_dims=1280,
            use_positional_embedding=False,
            quantiles=self._quantiles,
        )
        checkpoint = timesfm.TimesFmCheckpoint(huggingface_repo_id=self.model_id)
        self._model = timesfm.TimesFm(hparams=hparams, checkpoint=checkpoint)
        return self._model

    def predict_batch(self, contexts: list[np.ndarray], *, horizon: int) -> list[FoundationPrediction]:
        if not contexts:
            return []
        model = self._load()
        in_series = [np.asarray(ctx, dtype=np.float32) for ctx in contexts]
        freq = [0] * len(in_series)
        point_fcst, quant_fcst = model.forecast(in_series, freq=freq)

        q10_i = self._quantiles.index(0.1)
        q50_i = self._quantiles.index(0.5)
        q90_i = self._quantiles.index(0.9)
        out: list[FoundationPrediction] = []
        for i, ctx in enumerate(in_series):
            last_obs = float(ctx[-1])
            point_abs = float(point_fcst[i, horizon - 1])
            if quant_fcst is not None and np.ndim(quant_fcst) >= 3:
                qv = quant_fcst[i, horizon - 1, :]
                q10_abs = float(qv[q10_i])
                q50_abs = float(qv[q50_i])
                q90_abs = float(qv[q90_i])
            elif quant_fcst is not None and np.ndim(quant_fcst) == 2:
                q10_abs = float(quant_fcst[i, q10_i])
                q50_abs = float(quant_fcst[i, q50_i])
                q90_abs = float(quant_fcst[i, q90_i])
            else:
                q10_abs = q50_abs = q90_abs = point_abs

            out.append(
                FoundationPrediction(
                    point=point_abs - last_obs,
                    q10=q10_abs - last_obs,
                    q50=q50_abs - last_obs,
                    q90=q90_abs - last_obs,
                )
            )
        return out


class MoiraiNativeAdapter:
    model_name = "moirai_zero_shot"
    backend_status = "native"

    def __init__(self, *, model_id: str, batch_size: int, context_length: int, num_samples: int = 100) -> None:
        self.model_id = model_id
        self.batch_size = int(batch_size)
        self.context_length = int(context_length)
        self.num_samples = int(num_samples)
        self.backend_note = "Moirai adapter using uni2ts package"
        self._module = None
        self._predictors: dict[int, object] = {}

    def _load_module(self):
        if self._module is not None:
            return self._module
        try:
            from uni2ts.model.moirai import MoiraiModule
        except Exception as exc:  # pragma: no cover
            raise ImportError("uni2ts package is required for Moirai runs") from exc
        self._module = MoiraiModule.from_pretrained(self.model_id)
        return self._module

    def _get_predictor(self, horizon: int):
        if horizon in self._predictors:
            return self._predictors[horizon]
        from uni2ts.model.moirai import MoiraiForecast

        module = self._load_module()
        forecast = MoiraiForecast(
            prediction_length=int(horizon),
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=max(16, self.context_length),
            patch_size="auto",
            num_samples=self.num_samples,
            module=module,
        )
        predictor = forecast.create_predictor(batch_size=max(1, self.batch_size), device="cpu")
        self._predictors[horizon] = predictor
        return predictor

    def predict_batch(self, contexts: list[np.ndarray], *, horizon: int) -> list[FoundationPrediction]:
        if not contexts:
            return []
        from gluonts.dataset.common import ListDataset
        import pandas as pd

        predictor = self._get_predictor(horizon)
        ds_rows = []
        for ctx in contexts:
            ds_rows.append(
                {
                    "start": pd.Period("2000-01-01", freq="D"),
                    "target": np.asarray(ctx, dtype=float),
                }
            )
        ds = ListDataset(ds_rows, freq="D")
        forecasts = list(predictor.predict(ds))
        out: list[FoundationPrediction] = []
        for ctx, fcst in zip(contexts, forecasts):
            last_obs = float(ctx[-1])
            samples = np.asarray(fcst.samples)
            if samples.ndim == 3:
                samples_1 = samples[:, horizon - 1, 0]
            elif samples.ndim == 2:
                samples_1 = samples[:, horizon - 1]
            else:
                samples_1 = samples.reshape(-1)
            q10_abs, q50_abs, q90_abs = _safe_quantiles_from_samples(samples_1)
            out.append(
                FoundationPrediction(
                    point=q50_abs - last_obs,
                    q10=q10_abs - last_obs,
                    q50=q50_abs - last_obs,
                    q90=q90_abs - last_obs,
                )
            )
        return out


class PersistenceFallbackAdapter:
    def __init__(self, spec: FoundationSpec, *, model_id: str) -> None:
        self.spec = spec
        self.model_id = model_id
        self.model_name = spec.model_name
        self.backend_status = "fallback_persistence"
        self.backend_note = f"{spec.backend_note}; model_id={model_id}"

    def predict_batch(self, contexts: list[np.ndarray], *, horizon: int) -> list[FoundationPrediction]:
        out: list[FoundationPrediction] = []
        for ctx in contexts:
            if len(ctx) >= 2:
                drift = float((ctx[-1] - ctx[0]) / max(1, len(ctx) - 1))
            else:
                drift = 0.0
            point = float(horizon * drift)
            out.append(FoundationPrediction(point=point, q10=point, q50=point, q90=point))
        return out


def build_foundation_adapter(
    *,
    model_key: str,
    model_id: str | None = None,
    batch_size: int = 128,
    context_length: int = 120,
) -> FoundationAdapter:
    spec = get_foundation_spec(model_key)
    resolved_model_id = model_id or spec.default_model_id
    if spec.key == "chronos2":
        return Chronos2NativeAdapter(model_id=resolved_model_id, batch_size=batch_size, context_length=context_length)
    if spec.key == "timesfm":
        return TimesFMNativeAdapter(model_id=resolved_model_id, batch_size=batch_size, context_length=context_length)
    if spec.key == "moirai":
        return MoiraiNativeAdapter(model_id=resolved_model_id, batch_size=batch_size, context_length=context_length)
    return PersistenceFallbackAdapter(spec, model_id=resolved_model_id)
