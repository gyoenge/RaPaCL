from __future__ import annotations

import numpy as np

from radtranstab.utils.config import load_yaml, parse_common_args, apply_cli_overrides
from radtranstab.utils.logging import setup_logger, setup_warnings
from radtranstab.utils.misc import ensure_dir, save_yaml, seed_everything, unwrap_dataset
from radtranstab.data.dataset import load_data
from radtranstab.models.build import build_contrastive_learner, build_classifier, train
from radtranstab.engine.evaluator import evaluate_classifier
from .eval_representation import run_eval_detailed 
from radtranstab.utils.distributed import setup_distributed, is_main_process, cleanup_distributed


def build_runtime_context() -> None:
    """
    Manages: 
        - parse CLI 
        - load config  
        - initialize distributed  
        - prepare logger 
        - prepare dataset 
        - prepare arguments 
    """

    # parse CLI 
    args = parse_common_args()

    # load configs
    cfg = load_yaml(args.config)
    cfg = apply_cli_overrides(cfg, args)
    cfg["mode"] = args.mode

    # initialize distributed 
    distributed = (
        args.distributed
        if args.distributed is not None
        else cfg["runtime"].get("distributed", False)
    )
    rank, local_rank, world_size, device = setup_distributed(distributed)

    # initialize seed 
    seed = cfg.get("seed", 42)
    seed_everything(seed + rank)

    # prepare logger 
    output_root = ensure_dir(cfg["paths"]["output_root"])
    log_dir = ensure_dir(cfg["paths"].get("log_dir", output_root / "logs"))
    timestamp, logger = setup_logger(
        log_dir=log_dir,
        name=f"pretrain_transtab_rank{rank}",
        rank=rank,
    )

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

    # set run/checkpoint saving directory 
    run_dir = ensure_dir(output_root / f"run_{timestamp}")
    checkpoint_dir = ensure_dir(cfg["paths"]["checkpoint_dir"])

    # save 
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

        # save_column_info(
        #     run_dir=run_dir,
        #     categorical_columns=cat_cols,
        #     numerical_columns=num_cols,
        #     binary_columns=bin_cols,
        # )

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

    # build_contrastive_learner args 
    args_build_contrastive_learner = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "bin_cols": bin_cols,
        "supervised": model_cfg["supervised"],
        "num_partition": model_cfg["num_partition"],
        "overlap_ratio": model_cfg["overlap_ratio"],
        "device": device,
    }

    # build_classifier args 
    _, y_train = unwrap_dataset(trainset)
    num_class = len(np.unique(y_train))
    args_build_classifier = {
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "bin_cols": bin_cols,
        "num_class": num_class, 
        "checkpoint": str(checkpoint_dir),
    }
    
    # train(pretrain) args 
    args_pretrain = {
        "trainset": trainset,
        "valset": valset,
        "num_epoch": train_cfg["max_epochs"],
        "batch_size": train_cfg["batch_size"],
        "eval_batch_size": train_cfg["eval_batch_size"],
        "lr": train_cfg["lr"],
        "weight_decay": train_cfg["weight_decay"],
        "patience": train_cfg["patience"],
        "warmup_ratio": train_cfg["warmup_ratio"],
        "warmup_steps": train_cfg["warmup_steps"],
        "eval_metric": train_cfg["eval_metric"],
        "output_dir": str(checkpoint_dir),
        "num_workers": train_cfg["num_workers"],
        "ignore_duplicate_cols": model_cfg["ignore_duplicate_cols"],
        "eval_less_is_better": train_cfg["eval_less_is_better"],
        "distributed": distributed,
        "local_rank": local_rank,
        "rank": rank,
        "world_size": world_size,
        "device": device,
    }

    # train(finetune) args
    args_cls_train = {
        "trainset": trainset,
        "valset": valset,
        "num_epoch": train_cfg.get("ft_max_epochs", 20),
        "batch_size": train_cfg["batch_size"],
        "eval_batch_size": train_cfg["eval_batch_size"],
        "lr": train_cfg.get("ft_lr", 1e-4),
        "weight_decay": train_cfg["weight_decay"],
        "patience": train_cfg["patience"],
        "eval_metric": train_cfg.get("ft_eval_metric", "auc"),
        "eval_less_is_better": False,
        "output_dir": str(run_dir / "classifier_ckpt"),
        "num_workers": train_cfg["num_workers"],
    }

    # eval args 
    args_eval = {
        "testset": testset,
    }

    # eval_detailed args
    args_eval_detailed = {
        "allset": allset,
        "trainset": trainset,
        "valset": valset,
        "testset": testset,
        "run_dir": run_dir,
        "logger": logger,
        "device": device,
        "cfg": cfg,
    }

    return (
        distributed, 
        {
            "mode": args.mode, 
            "logger": logger, 
            "rank": rank, 
            "checkpoint_dir": checkpoint_dir, 
            "args_build_contrastive_learner": args_build_contrastive_learner, 
            "args_build_classifier": args_build_classifier, 
            "args_pretrain": args_pretrain, 
            "args_cls_train": args_cls_train, 
            "args_eval": args_eval, 
            "args_eval_detailed": args_eval_detailed, 
        }
    ) 


def run() -> None:
    # setting
    setup_warnings("ignore")
    distributed, settings = build_runtime_context()

    logger = settings["logger"]
    rank = settings["rank"]

    # pretraining
    if settings["mode"] in {"all", "train"}:
        logger.info("Start training...")
        model_pretrain, collate_fn = build_contrastive_learner(
            **settings["args_build_contrastive_learner"], 
        )
        train(
            model=model_pretrain,
            collate_fn=collate_fn,
            **settings["args_pretrain"], 
        )

        if is_main_process(rank):
            logger.info("Training finished.")

    # finetune classifier + evaluate on testset
    if settings["mode"] in {"all", "eval"}:
        logger.info(
            "Build classifier from pretrained checkpoint: %s",
            settings["args_build_classifier"]["checkpoint"],
        )
        logger.info("Building classifier with num_class=%d", settings["args_build_classifier"]["num_class"])
        classifier = build_classifier(
            **settings["args_build_classifier"]
        )
        logger.info("Start classifier finetuning...")
        train(
            classifier,
            **settings["args_cls_train"]
        )
        logger.info("Classifier finetuning finished.")
        classifier.load(str(settings["args_cls_train"]["output_dir"])) # best ckpt reload
        logger.info("Start evaluating...")
        evaluate_classifier(classifier, settings["args_eval"]["testset"], logger)

    # detailed evaluations
    if settings["mode"] == "eval_detailed":
        logger.info("Build contrastive learner for detailed evaluation: %s", settings["checkpoint_dir"])
        model_finetune, _ = build_contrastive_learner(
            **settings["args_build_contrastive_learner"], 
            checkpoint=settings["checkpoint_dir"],
        )
        run_eval_detailed(
            model=model_finetune,
            **settings["args_eval_detailed"]
        )

    cleanup_distributed(distributed)


if __name__ == "__main__":
    run()

