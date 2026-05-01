from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml


def load_yaml(config_path: str | Path) -> dict[str, Any]:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data or {}


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)

    for key, value in override.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def parse_common_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STNet runner")

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to yaml config file",
    )

    parser.add_argument(
        "--mode",
        choices=["all", "eval", "eval_detailed", "train", "tuning"],
        default="train",
    )

    # optional overrides
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)

    parser.add_argument("--num_genes", type=int, default=None)
    parser.add_argument("--genes_criteria", type=str, default=None)
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--optimizer_name", type=str, default=None)

    # distributed setitngs
    parser.add_argument("--distributed", type=lambda x: x.lower() == "true", default=None)

    return parser.parse_args()


def str_to_bool(value: str | None) -> bool | None:
    if value is None:
        return None

    lowered = value.lower()
    if lowered in {"true", "1", "yes", "y"}:
        return True
    if lowered in {"false", "0", "no", "n"}:
        return False

    raise ValueError(f"Cannot convert to bool: {value}")


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    if args.seed is not None:
        cfg["seed"] = args.seed

    cfg.setdefault("paths", {})
    cfg.setdefault("model", {})
    cfg.setdefault("train", {})
    cfg.setdefault("runtime", {})

    if args.device is not None:
        cfg["runtime"]["device"] = args.device

    if args.batch_size is not None:
        cfg["train"]["batch_size"] = args.batch_size

    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs

    if args.lr is not None:
        cfg["train"]["lr"] = args.lr

    if args.num_workers is not None:
        cfg["train"]["num_workers"] = args.num_workers

    if args.num_genes is not None:
        cfg["model"]["num_genes"] = args.num_genes

    if args.genes_criteria is not None:
        cfg["model"]["genes_criteria"] = args.genes_criteria

    if args.pretrained is not None:
        cfg["model"]["pretrained"] = str_to_bool(args.pretrained)

    if args.optimizer_name is not None:
        cfg["train"]["optimizer_name"] = args.optimizer_name

    return cfg