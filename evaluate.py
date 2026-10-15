"""Evaluates a saved checkpoint on the validation and test splits with the config of its run.

Usage: python evaluate.py run_dir=outputs/<run> [data.batch_size=4 device=cpu ...]
"""

import sys
from pathlib import Path

import torch as th
from omegaconf import OmegaConf

from data import build_loaders
from model import STFNN
from train import evaluate, fmt


def main():
    cli = OmegaConf.from_cli()
    if "run_dir" not in cli:
        sys.exit(__doc__)
    run_dir = Path(cli.pop("run_dir"))
    # current defaults fill keys that runs of older code do not have
    cfg = OmegaConf.merge(
        OmegaConf.load(Path(__file__).parent / "configs" / "config.yaml"),
        OmegaConf.load(run_dir / ".hydra" / "config.yaml"),
    )
    OmegaConf.set_struct(cfg, True)  # reject overrides of unknown keys
    cfg = OmegaConf.merge(cfg, cli)

    device = th.device(cfg.device if th.cuda.is_available() else "cpu")
    loaders, label_scaler = build_loaders(cfg.data, cfg.seed)
    model = STFNN(cfg.model).to(device)
    model.load_state_dict(th.load(run_dir / "model.pth", map_location=device))
    for split in ("val", "test"):
        print(f"{split} {fmt(evaluate(model, loaders[split], label_scaler, device))}", flush=True)


if __name__ == "__main__":
    main()
