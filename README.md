# STFNN

English | [中文](README.zh-CN.md)

Spatio-temporal field neural network (STFNN) for station-level air-quality interpolation. The value at a target station is estimated from its nearest observed neighbors in space and time: the model learns the gradient of a spatio-temporal field, integrates it along the path from each neighbor to the target, and combines the neighbor estimates with learned weights.

Paper: [Spatio-Temporal Field Neural Networks for Air Quality Inference](https://doi.org/10.24963/ijcai.2024/803), IJCAI 2024 (AI for Good track).

> [!NOTE]
> **This is the official refactor of STFNN by the authors of the paper**, carried out with AI assistance. It was checked against the original implementation:
>
> - **Model**: with the original weights loaded, both implementations give the same predictions (largest difference about 1e-5, floating-point rounding) in the configuration of all original experiments (`st_embed=simple`, `dym_graph=true`, GELU).
> - **Neighbor sampling**: for the same random mask, the samples are identical element by element.
>
> The data pipeline and evaluation differ on purpose, so results are not directly comparable with those of the original code:
>
> - **Fixes**: neighbor timestamps match their data rows; training and validation windows no longer overlap; normalization uses training rows only; metrics are pooled over all queries instead of averaged per batch (RMSE was biased low); evaluation masks are seeded and fixed; IDW weights use inverse distance; boolean options can be switched off on the command line; co-located stations are dropped.
> - **Removed**: `poly` / `linear` space-time embeddings, SELU and the 36-station dataset, none of which the original experiments used; `mm_size` micro-batching, replaced by `data.batch_size` with `train.acc_steps`.
> - **Added**: Hydra configuration, the `inductive` split, early stopping, per-epoch validation and test metrics, `evaluate.py`.

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.10+ and a CUDA GPU for training at the default batch size (about 11 GiB of GPU memory).

### Data

| File | Content |
|---|---|
| `air_2018.npy` | (Time, Station, 16) float array, NaN for missing values |
| `air_2018_locations.csv` | `longitude`, `latitude` of each station, one row per station in the order of the station axis |

The 16 features of `air_2018.npy`:

| Index | Group | Features |
|---|---|---|
| 0-5 | air quality | PM2.5, PM10, NO2, CO, O3, SO2 |
| 6-12 | meteorology | rainfall, temperature, pressure, humidity, wind speed, wind direction, weather |
| 13-15 | time context | hour of day, day of week, holiday flag |

The dataset is not part of this repository. By default the files are read from `~/dataset/air_2018.npy` and `~/dataset/air_2018_locations.csv`; change `data.data_path` and `data.loc_path` in `configs/config.yaml` or on the command line to use another location.

## Run

Train with the default configuration:

```bash
python train.py
```

All hyperparameters live in `configs/config.yaml` and can be overridden with Hydra syntax:

```bash
python train.py data.batch_size=2 train.acc_steps=8   # less GPU memory, same effective batch size
python train.py data.split=inductive                  # hold out stations as well as time
python train.py -m model.max_step=2,4,8,16            # sweep, one run per value
```

Frequently used options:

| Option | Default | Meaning |
|---|---|---|
| `data.label` | `pm25` | target variable: `pm25`, `pm10`, `no2`, `co`, `o3`, `so2`, `rain`, `temp`, `press` |
| `data.split` | `transductive` | `transductive`: splits differ in time only, every station is seen in training; `inductive`: validation and test stations are also unseen in training |
| `data.mask_ratio` | `0.25` | share of stations masked as prediction targets in each sample |
| `train.epochs` | `200` | maximum number of epochs |
| `train.patience` | `20` | stop after this many epochs without a better validation MAE |
| `seed` | `42` | seeds data splits, masks, shuffling and initialization |

Data are split chronologically into 60% / 20% / 20% train / validation / test time rows. Each epoch logs MAE, MAPE and RMSE on the validation and the test split; only the validation MAE selects the checkpoint and triggers early stopping. After training, the best checkpoint is evaluated on the test split.

Each run writes its configuration (`.hydra/`), `train.log` and the best checkpoint `model.pth` to `outputs/<date>/<time>/` (`multirun/` for sweeps).

Re-evaluate a saved checkpoint with the configuration of its run:

```bash
python evaluate.py run_dir=outputs/<date>/<time>
```

## Layout

```
configs/config.yaml   hyperparameters
data.py               loading, normalization, splits, neighbor sampling
model.py              STFNN model
train.py              training entry point
evaluate.py           evaluation of a saved checkpoint
requirements.txt      dependencies
```

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{ijcai2024p803,
  title     = {Spatio-Temporal Field Neural Networks for Air Quality Inference},
  author    = {Feng, Yutong and Wang, Qiongyan and Xia, Yutong and Huang, Junlin and Zhong, Siru and Liang, Yuxuan},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {7260--7268},
  year      = {2024},
  month     = {8},
  note      = {AI for Good},
  doi       = {10.24963/ijcai.2024/803},
  url       = {https://doi.org/10.24963/ijcai.2024/803},
}
```
