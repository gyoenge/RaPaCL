from __future__ import annotations
"""
CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 \
  -m src.rapacl.pretrain_transtab \
  --config configs/pretrain_transtab/idc_allxenium.yaml \
  --mode train
"""
"""
python -m src.rapacl.pretrain_transtab \
  --config configs/pretrain_transtab/idc_allxenium.yaml \
  --distributed false \
  --mode eval
"""
"""
python -m src.rapacl.pretrain_transtab \
  --config configs/pretrain_transtab/idc_allxenium.yaml \
  --distributed false \
  --mode eval_detailed
"""

import os

import torch
import torch.distributed as dist

from src.common.config import load_yaml, parse_common_args, apply_cli_overrides
from src.common.logger import setup_logger
from src.common.utils import ensure_dir, save_yaml, seed_everything

from src.rapacl.transtab_custom import (
    load_data, 
    build_contrastive_learner, 
    build_classifier,
    train,
    predict,
    # utils 
    unwrap_dataset, 
    save_column_info, 
)
from src.rapacl.eval_representation import run_eval_detailed 

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report

# ignore warning logs 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
import logging
logging.getLogger().setLevel(logging.ERROR)


def evaluate_classifier(model, testset, logger):
    x_test, y_test = unwrap_dataset(testset)

    y_pred = predict(model, x_test)
    y_pred = np.asarray(y_pred)

    logger.info("Prediction shape: %s", y_pred.shape)

    # binary classification
    if y_pred.ndim == 1 or (y_pred.ndim == 2 and y_pred.shape[1] == 1):
        y_score = y_pred.reshape(-1)
        y_label = (y_score >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_label)
        f1 = f1_score(y_test, y_label)
        try:
            auc = roc_auc_score(y_test, y_score)
        except Exception as e:
            auc = None
            logger.warning("Failed to compute ROC-AUC: %s", e)

        logger.info("=== Test Classification Metrics ===")
        logger.info("Test Accuracy: %.6f", acc)
        logger.info("Test F1      : %.6f", f1)
        if auc is not None:
            logger.info("Test AUROC   : %.6f", auc)

        logger.info("\n%s", classification_report(y_test, y_label, digits=4))

    # multiclass classification
    else:
        y_label = np.argmax(y_pred, axis=1)

        acc = accuracy_score(y_test, y_label)
        f1_macro = f1_score(y_test, y_label, average="macro")

        logger.info("=== Test Classification Metrics ===")
        logger.info("Test Accuracy : %.6f", acc)
        logger.info("Test F1-macro : %.6f", f1_macro)

        try:
            auc_macro_ovr = roc_auc_score(y_test, y_pred, multi_class="ovr", average="macro")
            logger.info("Test AUROC(ovr, macro): %.6f", auc_macro_ovr)
        except Exception as e:
            logger.warning("Failed to compute multiclass ROC-AUC: %s", e)

        logger.info("\n%s", classification_report(y_test, y_label, digits=4))


def setup_distributed(distributed: bool):
    if not distributed:
        return 0, 0, 1, "cuda:0" if torch.cuda.is_available() else "cpu"

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    dist.init_process_group(backend="nccl", init_method="env://")
    torch.cuda.set_device(local_rank)

    return rank, local_rank, world_size, f"cuda:{local_rank}"


def cleanup_distributed(distributed: bool):
    if distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def main() -> None:
    args = parse_common_args()

    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)
    cfg["mode"] = args.mode

    distributed = (
        args.distributed
        if args.distributed is not None
        else cfg["runtime"].get("distributed", False)
    )
    rank, local_rank, world_size, device = setup_distributed(distributed)

    seed = cfg.get("seed", 42)
    seed_everything(seed + rank)

    output_root = ensure_dir(cfg["paths"]["output_root"])
    log_dir = ensure_dir(cfg["paths"].get("log_dir", output_root / "logs"))
    timestamp, logger = setup_logger(log_dir, name=f"pretrain_transtab_rank{rank}")

    if is_main_process(rank):
        logger.info("Loaded config from: %s", args.config)
        logger.info("Execution mode: %s", args.mode)
        logger.info("Distributed: %s", distributed)
        logger.info("Rank: %s", rank)
        logger.info("Local rank: %s", local_rank)
        logger.info("World size: %s", world_size)
        logger.info("Device: %s", device)
        logger.info("Batch size: %s", cfg["train"].get("batch_size"))
        logger.info("Learning rate: %s", cfg["train"].get("lr"))

    run_dir = ensure_dir(output_root / f"run_{timestamp}")
    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])

    if is_main_process(rank) and cfg["experiment"].get("save_config", True):
        save_yaml(cfg, run_dir / "config_final.yaml")
        logger.info("Final config saved to: %s", run_dir / "config_final.yaml")

    if is_main_process(rank):
        logger.info("Preparing custom TransTab dataset from: %s", cfg["paths"]["data_root"])

    allset, trainset, valset, testset, cat_cols, num_cols, bin_cols = load_data(
        [f'{cfg["paths"]["data_root"]}']
    )

    if is_main_process(rank):
        logger.info("Detected column types:")
        logger.info("  categorical: %d", len(cat_cols))
        logger.info("  numerical  : %d", len(num_cols))
        logger.info("  binary     : %d", len(bin_cols))

        save_column_info(
            run_dir=run_dir,
            categorical_columns=cat_cols,
            numerical_columns=num_cols,
            binary_columns=bin_cols,
        )

    # num of patches per label
    if is_main_process(rank):
        try:
            if isinstance(allset, (list, tuple)) and len(allset) == 1:
                _, y_all = allset[0]
            else:
                _, y_all = allset

            logger.info("allset type: %s", type(allset))
            logger.info("allset len: %s", len(allset))

            label_counts = y_all.value_counts().sort_index()

            logger.info("=== Allset label distribution ===")
            for label, count in label_counts.items():
                logger.info("Label %s: %d samples", label, count)

            logger.info("Total samples: %d", len(y_all))

        except Exception as e:
            logger.warning("Failed to log allset label distribution: %s", e)
            logger.warning("allset repr: %s", repr(allset)[:500])


    model_cfg = cfg["model"]
    train_cfg = cfg["train"]

    # 1) contrastive pretraining model
    model, collate_fn = build_contrastive_learner(
        cat_cols=cat_cols,
        num_cols=num_cols,
        bin_cols=bin_cols,
        supervised=model_cfg["supervised"],
        num_partition=model_cfg["num_partition"],
        overlap_ratio=model_cfg["overlap_ratio"],
        device=device,
    )

    # 2) pretraining
    if args.mode in {"all", "train"}:
        if is_main_process(rank):
            logger.info("Start training...")

        train(
            model=model,
            trainset=trainset,
            valset=valset,
            num_epoch=train_cfg["max_epochs"],
            batch_size=train_cfg["batch_size"],
            eval_batch_size=train_cfg["eval_batch_size"],
            lr=train_cfg["lr"],
            weight_decay=train_cfg["weight_decay"],
            patience=train_cfg["patience"],
            warmup_ratio=train_cfg["warmup_ratio"],
            warmup_steps=train_cfg["warmup_steps"],
            eval_metric=train_cfg["eval_metric"],
            output_dir=str(checkpoint_dir),
            collate_fn=collate_fn,
            num_workers=train_cfg["num_workers"],
            ignore_duplicate_cols=model_cfg["ignore_duplicate_cols"],
            eval_less_is_better=train_cfg["eval_less_is_better"],
            distributed=distributed,
            local_rank=local_rank,
            rank=rank,
            world_size=world_size,
            device=device,
        )

        if is_main_process(rank):
            logger.info("Training finished.")

    # 3) finetune classifier + evaluate on testset
    if args.mode in {"all", "eval"}:
        if is_main_process(rank):
            logger.info("Build classifier from pretrained checkpoint: %s", checkpoint_dir)

            _, y_train = unwrap_dataset(trainset)
            num_class = len(np.unique(y_train))
            logger.info("Building classifier with num_class=%d", num_class)
            clf = build_classifier(
                checkpoint=str(checkpoint_dir),
                num_class=num_class,
                cat_cols=cat_cols,
                num_cols=num_cols,
                bin_cols=bin_cols,
            )
            logger.info("Start classifier finetuning...")

            train(
                clf,
                trainset,
                valset,
                num_epoch=train_cfg.get("ft_max_epochs", 20),
                batch_size=train_cfg["batch_size"],
                eval_batch_size=train_cfg["eval_batch_size"],
                lr=train_cfg.get("ft_lr", 1e-4),
                weight_decay=train_cfg["weight_decay"],
                patience=train_cfg["patience"],
                eval_metric=train_cfg.get("ft_eval_metric", "auc"),
                eval_less_is_better=False,
                output_dir=str(run_dir / "classifier_ckpt"),
                num_workers=train_cfg["num_workers"],
            )

            logger.info("Classifier finetuning finished.")

            # best ckpt reload
            clf.load(str(run_dir / "classifier_ckpt"))

            # 
            logger.info("Start evaluating...")
            evaluate_classifier(clf, testset, logger)

        # 4) detailed evaluations
        if args.mode == "eval_detailed":
            if is_main_process(rank):
                logger.info("Build contrastive learner for detailed evaluation: %s", checkpoint_dir)

                cl_model, _ = build_contrastive_learner(
                    cat_cols=cat_cols,
                    num_cols=num_cols,
                    bin_cols=bin_cols,
                    supervised=model_cfg["supervised"],
                    num_partition=model_cfg["num_partition"],
                    overlap_ratio=model_cfg["overlap_ratio"],
                    device=device,
                    checkpoint=str(checkpoint_dir),
                )

                run_eval_detailed(
                    model=cl_model,
                    allset=allset,
                    trainset=trainset,
                    valset=valset,
                    testset=testset,
                    run_dir=run_dir,
                    logger=logger,
                    device=device,
                    cfg=cfg,
                )
        
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
