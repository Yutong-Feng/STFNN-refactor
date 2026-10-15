"""Air-quality data loading, normalization and neighbor sampling."""

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import DataLoader, Dataset

LABEL_COLUMNS = {
    "pm25": 0, "pm10": 1, "no2": 2, "co": 3, "o3": 4, "so2": 5,
    "rain": 6, "temp": 7, "press": 8,
}
# raw feature layout: air quality 0-5, continuous meteorology 6-10,
# discrete: wind direction 11, weather 12, hour 13, weekday 14, holiday 15
CONTINUOUS = range(11)
DISCRETE = range(11, 16)


class StationScaler:
    """Per-station z-score: (x - mean) / std, with statistics over the time axis.

    Args:
        x: (T, N, ...) float, NaN for missing; only the rows used for fitting.
        pool_stations: share one statistic, taken over time and stations, by all stations.
    """

    def __init__(self, x: np.ndarray, pool_stations: bool = False):
        axis = (0, 1) if pool_stations else 0
        with warnings.catch_warnings():
            # all-NaN series get NaN statistics and stay missing after `norm`
            warnings.simplefilter("ignore", RuntimeWarning)
            self.mean = np.broadcast_to(np.nanmean(x, axis=axis), x.shape[1:])
            std = np.broadcast_to(np.nanstd(x, axis=axis, ddof=1), x.shape[1:])
        # constant or single-observation series: keep them centered instead of dividing by ~0
        self.std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)

    def norm(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    def denorm(self, x: np.ndarray, station: np.ndarray) -> np.ndarray:
        """Inverts `norm` for values `x` (B,) belonging to stations `station` (B,) int."""
        return x * self.std[station] + self.mean[station]


def select_stations(loc: pd.DataFrame, drop: list) -> np.ndarray:
    """Boolean mask (N_raw,) of stations to keep.

    Co-located stations make the query-neighbor direction zero (NaN in the model),
    so only the first station at each coordinate is kept.
    """
    keep = ~loc.duplicated(["longitude", "latitude"]).to_numpy()
    keep[list(drop)] = False
    return keep


@dataclass
class Split:
    """Rows and stations of one split.

    Attributes:
        windows: (S, W) int rows of `data`; column 0 is the query time,
            column i is i * skip_step rows earlier.
        query_stations: (N,) bool, stations that may be queries.
        ngb_stations: (N,) bool, stations that may be neighbors.
        mask_ratio: share of the queries that are masked (targets, never neighbors).
    """

    windows: np.ndarray
    query_stations: np.ndarray
    ngb_stations: np.ndarray
    mask_ratio: float


def split_stations(n: int, cfg, seed: int) -> dict:
    """Random disjoint station groups: split -> (N,) bool."""
    order = np.random.default_rng(seed).permutation(n)
    bounds = np.cumsum([0, int(cfg.station_train_ratio * n), int(cfg.station_val_ratio * n)])
    groups = {"train": order[: bounds[1]], "val": order[bounds[1] : bounds[2]], "test": order[bounds[2] :]}
    return {k: np.isin(np.arange(n), v) for k, v in groups.items()}


def load_arrays(cfg):
    """Reads the raw tensor and station coordinates.

    Returns:
        raw: (T_raw, N, 16) float64, NaN for missing.
        xy: (N, 2) float64 longitude / latitude.
    """
    raw = np.load(Path(cfg.data_path).expanduser())
    loc = pd.read_csv(Path(cfg.loc_path).expanduser())
    assert len(loc) == raw.shape[1], f"{len(loc)} locations vs {raw.shape[1]} stations"
    keep = select_stations(loc, cfg.drop_stations)
    return raw[:, keep], loc.loc[keep, ["longitude", "latitude"]].to_numpy()


def prepare(raw: np.ndarray, xy: np.ndarray, cfg, seed: int):
    """Normalizes features with train-period statistics and builds the splits.

    `cfg.split` is "transductive" (splits differ in time only) or "inductive"
    (splits also differ in stations, see `split_stations`).

    Args:
        raw: (T_raw, N, 16) float64.
        xy: (N, 2) float64.

    Returns:
        data: (T, N, F) float32; continuous features, discrete codes, normalized label (last).
        hours: (T,) int, hours since the first raw row.
        coords: (N, 2) float64, z-scored lon / lat.
        splits: dict "train" / "val" / "test" -> Split.
        label_scaler: StationScaler of the label.
    """
    if cfg.split not in ("transductive", "inductive"):
        raise ValueError(f"unknown split {cfg.split!r}")
    inductive = cfg.split == "inductive"
    label_col = LABEL_COLUMNS[cfg.label]
    hours = np.arange(len(raw))
    keep_rows = np.isnan(raw[:, :, label_col]).mean(axis=1) <= cfg.drop_limit
    raw, hours = raw[keep_rows], hours[keep_rows]

    n_rows = len(raw)
    n_train = int(cfg.train_ratio * n_rows)
    n_val = int((cfg.train_ratio + cfg.val_ratio) * n_rows)
    span = cfg.window * cfg.skip_step
    assert span < n_train, "window does not fit into the training period"

    label = raw[:, :, label_col]
    n_stations = raw.shape[1]
    if inductive:
        stations = split_stations(n_stations, cfg, seed)
        # held-out stations have no history to normalize with: one scale from training stations
        label_scaler = StationScaler(np.where(stations["train"], label[:n_train], np.nan), pool_stations=True)
    else:
        stations = dict.fromkeys(("train", "val", "test"), np.ones(n_stations, dtype=bool))
        label_scaler = StationScaler(label[:n_train])
    cont = raw[:, :, [c for c in CONTINUOUS if c != label_col]]
    cont = np.nan_to_num(StationScaler(cont[:n_train]).norm(cont), nan=0.0)
    disc = np.nan_to_num(raw[:, :, DISCRETE] + 1, nan=0.0)  # code 0 marks missing
    data = np.concatenate([cont, disc, label_scaler.norm(label)[..., None]], axis=2)

    # every split looks backwards, so no query row belongs to two splits
    query_rows = {
        "train": np.arange(span, n_train),
        "val": np.arange(n_train, n_val),
        "test": np.arange(n_val, n_rows),
    }
    offsets = np.arange(cfg.window + 1) * cfg.skip_step
    splits = {
        k: Split(
            windows=q[:, None] - offsets,
            query_stations=stations[k],
            ngb_stations=stations["train"],
            # inductive evaluation predicts every held-out station from training stations
            mask_ratio=1.0 if inductive and k != "train" else cfg.mask_ratio,
        )
        for k, q in query_rows.items()
    }
    coords = (xy - xy.mean(axis=0)) / xy.std(axis=0)
    return data.astype(np.float32), hours, coords, splits, label_scaler


class AQIDataset(Dataset):
    """One sample per time window; every query-pool station observed at the query time is a query.

    A `split.mask_ratio` share of the queries is masked: masked stations are never used as
    neighbors. Each query takes its `k_top` nearest observed, unmasked neighbor-pool stations
    at each of the W window steps as neighbors.

    Args:
        data: (T, N, F) float32 from `prepare`.
        hours: (T,) int.
        coords: (N, 2) float.
        split: Split.
        seed: fixes the mask of each sample (evaluation); None draws a fresh mask
            from the global numpy RNG, which the DataLoader seeds per worker (training).
    """

    def __init__(self, data, hours, coords, split: Split, k_top, seed=None):
        super().__init__()
        self.data = data
        self.windows = split.windows
        self.hours = hours[split.windows]  # (S, W)
        self.coords = coords
        self.dist = np.sum((coords[:, None] - coords[None]) ** 2, axis=-1)  # (N, N) squared
        np.fill_diagonal(self.dist, np.inf)
        self.query_stations = split.query_stations
        self.ngb_stations = split.ngb_stations
        self.mask_ratio = split.mask_ratio
        self.k_top = k_top
        self.seed = seed

    def __len__(self):
        return len(self.windows)

    def _rng(self, index):
        return np.random if self.seed is None else np.random.default_rng((self.seed, index))

    def __getitem__(self, index):
        x = self.data[self.windows[index]]  # (W, N, F)
        t = self.hours[index]  # (W,)
        n_steps = len(x)
        missing = np.isnan(x[:, :, -1])  # (W, N)
        query = np.flatnonzero(~missing[0] & self.query_stations)  # (Q,)
        masked = self._rng(index).choice(query, int(self.mask_ratio * len(query)), replace=False)

        blocked = missing | ~self.ngb_stations  # (W, N)
        blocked[:, masked] = True
        # (W, 1, N) | (1, Q, N) -> (W, Q, N): distance from each query to each candidate
        dist = np.where(blocked[:, None, :], np.inf, self.dist[query][None])
        ngb = np.argsort(dist, axis=2)[:, :, : self.k_top]  # (W, Q, K)
        ngb_dist = np.take_along_axis(dist, ngb, axis=2)  # (W, Q, K)
        ngb_x = x[np.arange(n_steps)[:, None, None], ngb]  # (W, Q, K, F)
        ngb_t = np.broadcast_to(t[:, None, None, None], ngb.shape + (1,)).astype(np.float64)
        ngb_info = np.concatenate([self.coords[ngb], ngb_t, ngb_x], axis=3)  # (W, Q, K, 3 + F)
        # (W, Q, K, 3 + F) -> (Q, W * K, 3 + F), step-major
        ngb_info = ngb_info.transpose(1, 0, 2, 3).reshape(len(query), -1, ngb_info.shape[-1])

        query_t = np.full((len(query), 1), t[0], dtype=np.float64)
        query_info = np.concatenate([self.coords[query], query_t, x[0, query]], axis=1)  # (Q, 3 + F)
        return (
            query_info.astype(np.float32),
            ngb_info.astype(np.float32),
            ngb_dist.transpose(1, 0, 2).astype(np.float32),  # (Q, W, K)
            np.isin(query, masked),  # (Q,)
            query,  # (Q,) station index
        )


def collate(batch):
    """Concatenates samples along the query axis; B is the total number of queries."""
    keys = ("query", "ngb", "dist", "masked", "station")
    return {k: th.from_numpy(np.concatenate(v)) for k, v in zip(keys, zip(*batch))}


def build_loaders(cfg, seed: int):
    """Returns ({split: DataLoader}, label StationScaler) for `cfg` = the `data` config."""
    data, hours, coords, splits, label_scaler = prepare(*load_arrays(cfg), cfg, seed)

    def loader(split):
        train = split == "train"
        dataset = AQIDataset(data, hours, coords, splits[split], cfg.k_top, seed=None if train else seed)
        return DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=train,
            collate_fn=collate,
            num_workers=cfg.num_workers,
            prefetch_factor=cfg.prefetch_factor if cfg.num_workers > 0 else None,
            pin_memory=th.cuda.is_available(),
            generator=th.Generator().manual_seed(seed),
        )

    return {split: loader(split) for split in splits}, label_scaler
