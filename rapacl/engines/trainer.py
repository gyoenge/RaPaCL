import os
import json
import torch
from torch.utils.data import DataLoader

from rapacl.data.dataset import HestRadiomicsDataset, radiomics_collate_fn
from rapacl.data.constants_radfeatcols import RADIOMICS_FEATURES_NAMES
from rapacl.model.radtranstab.build import build_radiomics_learner
from rapacl.engines.trainer_utils import (
    set_seed,
    load_model_radiomics_from_full_checkpoint,
    train_one_epoch,
    evaluate,
    save_checkpoint,
)
import rapacl.engines.constants as constants


def build_model_radiomics(device):
    model_radiomics = build_radiomics_learner(
        checkpoint=None,
        numerical_columns=RADIOMICS_FEATURES_NAMES,
        num_class=constants.NUM_CLASS,
        # hidden_dim=constants.HIDDEN_DIM,
        # num_layer=constants.NUM_LAYER,
        hidden_dropout_prob=constants.DROPOUT,
        projection_dim=constants.PROJECTION_DIM,
        activation=constants.ACTIVATION,
        # num_sub_cols=constants.NUM_SUB_COLS,
        ape_drop_rate=constants.APE_DROP_RATE,
        device=device,
    )

    # APE 없이 학습하므로 아래 로직은 사용하지 않는다.
    # model_radiomics.input_encoder.feature_extractor.update(num=ape_cols)

    return model_radiomics.to(device)


def build_dataloaders():
    train_dataset = HestRadiomicsDataset(
        radiomics_file=constants.TRAIN_RADIOMCIS_FILE,
        root_dir=constants.ROOT_DIR,
        label_col=constants.LABEL_COL,
        id_col=constants.ID_COL,
    )

    val_dataset = HestRadiomicsDataset(
        radiomics_file=constants.VAL_RADIOMCIS_FILE,
        root_dir=constants.ROOT_DIR,
        label_col=constants.LABEL_COL,
        id_col=constants.ID_COL,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=constants.BATCH_SIZE,
        shuffle=True,
        num_workers=constants.NUM_WORKERS,
        pin_memory=True,
        collate_fn=radiomics_collate_fn,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=constants.BATCH_SIZE,
        shuffle=False,
        num_workers=constants.NUM_WORKERS,
        pin_memory=True,
        collate_fn=radiomics_collate_fn,
        drop_last=False,
    )

    return train_loader, val_loader


def main():
    set_seed(constants.SEED)

    device = torch.device(constants.DEVICE)
    os.makedirs(constants.OUTPUT_DIR, exist_ok=True)

    model_radiomics = build_model_radiomics(device)

    if constants.CHECKPOINT_PATH is not None:
        load_model_radiomics_from_full_checkpoint(
            model_radiomics=model_radiomics,
            checkpoint_path=constants.CHECKPOINT_PATH,
            device=device,
            strict=False,
        )

    train_loader, val_loader = build_dataloaders()

    optimizer = torch.optim.AdamW(
        model_radiomics.parameters(),
        lr=constants.LR,
        weight_decay=constants.WEIGHT_DECAY,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
    scaler = torch.amp.GradScaler() if constants.USE_AMP else None

    best_val_loss = float("inf")
    history = {
        "train": {},
        "val": {},
    }

    for epoch in range(constants.EPOCHS):
        train_metrics = train_one_epoch(
            model_radiomics=model_radiomics,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epoch=epoch,
            scaler=scaler,
        )

        val_metrics = evaluate(
            model_radiomics=model_radiomics,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        history["train"][epoch] = train_metrics
        history["val"][epoch] = val_metrics

        print(
            f"[Epoch {epoch}] "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['acc']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['acc']:.4f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]

            save_path = save_checkpoint(
                output_dir=constants.OUTPUT_DIR,
                model_radiomics=model_radiomics,
                optimizer=optimizer,
                epoch=epoch,
                metrics=val_metrics,
                args=vars(constants),
                name="best_model_radiomics",
            )

            print(f"Saved best checkpoint: {save_path}")

        metrics_path = os.path.join(constants.OUTPUT_DIR, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=4)


if __name__ == "__main__":
    main()

