"""Trains STFNN on the masked-station interpolation task and reports test metrics."""

import logging
from pathlib import Path

import hydra
import numpy as np
import torch as th
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from tqdm import tqdm

from data import build_loaders
from model import STFNN

log = logging.getLogger(__name__)


def forward(model, batch, device):
    """Returns (pred (B,) float32, label (B,) float32, curl ()) in normalized units."""
    query = batch["query"].to(device, non_blocking=True)
    ngb = batch["ngb"].to(device, non_blocking=True)
    dist = batch["dist"].to(device, non_blocking=True)
    with th.autocast(device.type, enabled=device.type == "cuda"):
        pred, curl = model(query[:, :3], ngb, dist)
    return pred.float(), query[:, -1], curl.float()


class Metrics:
    """MAE, MAPE and RMSE over all accumulated values, not averaged per batch."""

    def __init__(self):
        self.abs = self.ape = self.sq = 0.0
        self.n = 0

    def update(self, pred: np.ndarray, label: np.ndarray):
        err = pred - label
        self.abs += np.abs(err).sum()
        self.ape += (np.abs(err) / np.abs(label)).sum()
        self.sq += np.square(err).sum()
        self.n += err.size

    def result(self) -> dict:
        return {"mae": self.abs / self.n, "mape": self.ape / self.n, "rmse": np.sqrt(self.sq / self.n)}


def train_epoch(model, loader, optimizer, grad_scaler, mask_weight, acc_steps, device) -> dict:
    """One pass over `loader`; returns per-batch-averaged losses."""
    model.train()
    totals = {"loss": 0.0, "mask": 0.0, "non_mask": 0.0, "curl": 0.0}
    optimizer.zero_grad()
    bar = tqdm(loader, desc="train", leave=False)
    for i, batch in enumerate(bar):
        pred, label, curl = forward(model, batch, device)
        masked = batch["masked"].to(device)
        loss_mask = F.mse_loss(pred[masked], label[masked])
        loss_non_mask = F.mse_loss(pred[~masked], label[~masked])
        loss = mask_weight * loss_mask + (1 - mask_weight) * loss_non_mask
        grad_scaler.scale(loss / acc_steps).backward()
        if (i + 1) % acc_steps == 0 or i + 1 == len(loader):
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()
        for k, v in zip(totals, (loss, loss_mask, loss_non_mask, curl)):
            totals[k] += v.item() / len(loader)
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return totals


@th.no_grad()
def evaluate(model, loader, label_scaler, device) -> dict:
    """De-normalized metrics on the masked queries."""
    model.eval()
    metrics = Metrics()
    for batch in tqdm(loader, desc="eval", leave=False):
        pred, label, _ = forward(model, batch, device)
        masked, station = batch["masked"].numpy(), batch["station"].numpy()
        pred = label_scaler.denorm(pred.cpu().numpy(), station)
        label = label_scaler.denorm(label.cpu().numpy(), station)
        metrics.update(pred[masked], label[masked])
    return metrics.result()


class BestCheckpoint:
    """Saves the model whenever the validation MAE improves and remembers that epoch."""

    def __init__(self, path: Path):
        self.path = path
        self.mae = np.inf
        self.epoch = -1

    def update(self, mae: float, epoch: int, model) -> bool:
        """Returns True and saves `model` if `mae` is a new best."""
        if not mae < self.mae:
            return False
        self.mae, self.epoch = mae, epoch
        # write then rename, so an interrupted save never corrupts the previous best
        tmp = self.path.with_suffix(".tmp")
        th.save(model.state_dict(), tmp)
        tmp.replace(self.path)
        return True


def fmt(values: dict) -> str:
    return " ".join(f"{k}={v:.4f}" for k, v in values.items())


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg) -> float:
    log.info("config:\n%s", OmegaConf.to_yaml(cfg))
    np.random.seed(cfg.seed)
    th.manual_seed(cfg.seed)
    device = th.device(cfg.device if th.cuda.is_available() else "cpu")
    loaders, label_scaler = build_loaders(cfg.data, cfg.seed)

    model = STFNN(cfg.model).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.w_decay)
    scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.train.lr_step, gamma=cfg.train.lr_gamma)
    grad_scaler = th.amp.GradScaler(device.type, enabled=device.type == "cuda")
    mask_weight = max(cfg.data.mask_ratio, 1 - cfg.data.mask_ratio)

    best = BestCheckpoint(Path(HydraConfig.get().runtime.output_dir) / "model.pth")
    for epoch in range(cfg.train.epochs):
        losses = train_epoch(
            model, loaders["train"], optimizer, grad_scaler, mask_weight, cfg.train.acc_steps, device
        )
        val = evaluate(model, loaders["val"], label_scaler, device)
        if not np.isfinite(val["mae"]):
            raise FloatingPointError(f"validation MAE is {val['mae']} at epoch {epoch}")
        # test metrics are only reported; checkpointing and early stopping use validation alone
        test = evaluate(model, loaders["test"], label_scaler, device)
        msg = f"epoch {epoch} train {fmt(losses)} | val {fmt(val)} | test {fmt(test)}"
        if best.update(val["mae"], epoch, model):
            msg += " (saved)"
        log.info(msg)
        scheduler.step()
        if epoch - best.epoch >= cfg.train.patience:
            log.info("early stop: no val improvement for %d epochs", cfg.train.patience)
            break

    log.info("best val mae=%.4f at epoch %d", best.mae, best.epoch)
    model.load_state_dict(th.load(best.path, map_location=device))
    test = evaluate(model, loaders["test"], label_scaler, device)
    log.info("test %s", fmt(test))
    return test["mae"]


if __name__ == "__main__":
    main()
