from __future__ import annotations

import copy
import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainConfig:
    epochs: int = 8
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 3
    loss: str = "huber"
    seed: int = 42
    quantiles: tuple[float, ...] | None = None
    track_history: bool = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_sliding_windows(
    frame,
    feature_cols: list[str],
    target_col: str,
    lookback: int,
):
    if lookback < 2:
        raise ValueError("lookback must be >= 2")

    X, y = [], []
    for end in range(lookback - 1, len(frame)):
        start = end - lookback + 1
        X.append(frame.iloc[start : end + 1][feature_cols].to_numpy(dtype=np.float32))
        y.append(float(frame.iloc[end][target_col]))

    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


class SequenceStandardizer:
    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.std_: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> "SequenceStandardizer":
        flat = X.reshape(-1, X.shape[-1])
        self.mean_ = flat.mean(axis=0)
        self.std_ = flat.std(axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Scaler not fitted")
        return ((X - self.mean_) / self.std_).astype(np.float32)


class PatchTSTLikeRegressor(nn.Module):
    def __init__(
        self,
        lookback: int,
        n_features: int,
        patch_len: int = 6,
        stride: int = 3,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        if lookback < patch_len:
            raise ValueError("lookback must be >= patch_len")

        self.patch_len = patch_len
        self.stride = stride
        self.n_features = n_features
        self.output_dim = output_dim
        self.n_patches = 1 + (lookback - patch_len) // stride

        if self.n_patches < 1:
            raise ValueError("No patches generated. Increase lookback or lower patch_len/stride")

        self.patch_proj = nn.Linear(patch_len * n_features, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.n_patches, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        # [B, Np, F, patch_len] -> [B, Np, patch_len, F]
        patches = patches.permute(0, 1, 3, 2)
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)

        z = self.patch_proj(patches)
        z = z + self.pos_emb[:, : z.shape[1], :]
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        out = self.head(pooled)
        return out.squeeze(-1) if self.output_dim == 1 else out


class ITransformerLikeRegressor(nn.Module):
    def __init__(
        self,
        lookback: int,
        n_features: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.var_proj = nn.Linear(lookback, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, n_features, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F] -> [B, F, L]
        tokens = x.transpose(1, 2)
        z = self.var_proj(tokens)
        z = z + self.pos_emb
        z = self.encoder(z)
        pooled = z.mean(dim=1)
        out = self.head(pooled)
        return out.squeeze(-1) if self.output_dim == 1 else out


class DLinearLikeRegressor(nn.Module):
    """
    Lightweight DLinear-style forecaster:
    decompose each variable into trend/seasonal and apply linear heads.
    """

    def __init__(
        self,
        lookback: int,
        n_features: int,
        moving_avg: int = 5,
        output_dim: int = 1,
    ):
        super().__init__()
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.lookback = lookback
        self.n_features = n_features
        self.output_dim = output_dim
        self.moving_avg = max(2, min(moving_avg, lookback))

        self.linear_seasonal = nn.Linear(lookback, output_dim)
        self.linear_trend = nn.Linear(lookback, output_dim)

    def _moving_average(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, F, L] -> trailing causal moving average
        pad = self.moving_avg - 1
        x_pad = F.pad(x, (pad, 0), mode="replicate")
        return F.avg_pool1d(x_pad, kernel_size=self.moving_avg, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        xt = x.transpose(1, 2)
        trend = self._moving_average(xt)
        seasonal = xt - trend

        b, f, l = seasonal.shape
        seasonal_out = self.linear_seasonal(seasonal.reshape(b * f, l))
        trend_out = self.linear_trend(trend.reshape(b * f, l))
        out = (seasonal_out + trend_out).reshape(b, f, self.output_dim).mean(dim=1)
        return out.squeeze(-1) if self.output_dim == 1 else out


class _CausalConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=self.pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, dilation=dilation, padding=self.pad)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.BatchNorm1d(out_ch)
        self.norm2 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()
        self.residual = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def _trim(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.pad] if self.pad > 0 else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.residual(x)
        z = self._trim(self.conv1(x))
        z = self.norm1(z)
        z = self.act(z)
        z = self.dropout(z)

        z = self._trim(self.conv2(z))
        z = self.norm2(z)
        z = self.act(z)
        z = self.dropout(z)
        return self.act(z + res)


class TCNLikeRegressor(nn.Module):
    def __init__(
        self,
        lookback: int,
        n_features: int,
        channels: tuple[int, ...] = (64, 64, 64),
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_dim: int = 1,
    ):
        super().__init__()
        self.output_dim = output_dim
        layers: list[nn.Module] = []
        in_ch = n_features
        for i, ch in enumerate(channels):
            dilation = 2**i
            layers.append(_CausalConvBlock(in_ch, ch, kernel_size=kernel_size, dilation=dilation, dropout=dropout))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.head = nn.Linear(in_ch, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F] -> [B, F, L]
        z = x.transpose(1, 2)
        z = self.tcn(z)
        pooled = z.mean(dim=-1)
        out = self.head(pooled)
        return out.squeeze(-1) if self.output_dim == 1 else out


def _select_loss(name: str):
    if name == "huber":
        return nn.SmoothL1Loss()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    raise ValueError("loss must be 'huber', 'mse', or 'mae'")


def _pinball_loss(pred: torch.Tensor, y: torch.Tensor, quantiles: tuple[float, ...]) -> torch.Tensor:
    if pred.ndim != 2:
        raise ValueError("Quantile predictions must be 2D: [batch, n_quantiles]")
    if pred.shape[1] != len(quantiles):
        raise ValueError("Prediction second dim must match quantile count")
    y_exp = y.unsqueeze(1)
    q = torch.tensor(quantiles, dtype=pred.dtype, device=pred.device).unsqueeze(0)
    err = y_exp - pred
    loss = torch.maximum(q * err, (q - 1.0) * err)
    return loss.mean()


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
) -> nn.Module:
    set_seed(config.seed)
    device = torch.device("cpu")
    model = model.to(device)

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = _select_loss(config.loss)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad_epochs = 0
    history_rows: list[dict[str, float | int]] = []

    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            if config.quantiles is None:
                loss = loss_fn(pred, yb)
            else:
                loss = _pinball_loss(pred, yb, config.quantiles)
            loss.backward()
            opt.step()
            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                if config.quantiles is None:
                    val_losses.append(float(loss_fn(pred, yb).item()))
                else:
                    val_losses.append(float(_pinball_loss(pred, yb, config.quantiles).item()))

        train_loss = float(np.mean(train_losses)) if train_losses else float("nan")
        val_loss = float(np.mean(val_losses))
        history_rows.append({"epoch": epoch + 1, "train_loss": train_loss, "val_loss": val_loss})
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    if config.track_history:
        setattr(model, "_training_history", history_rows)
    return model


def predict_model(model: nn.Module, X: np.ndarray) -> np.ndarray:
    device = torch.device("cpu")
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(X).to(device)
        pred = model(xb).cpu().numpy()
    return pred.astype(np.float32)
