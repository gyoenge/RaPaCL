# check_load_checkpoint.py
import torch

from rapacl.data.constants_radfeatcols import RADIOMICS_FEATURES_NAMES
from rapacl.model.radtranstab.build import build_radiomics_learner
from rapacl.engines.trainer_utils import (
    set_seed,
    load_model_radiomics_from_full_checkpoint,
)
import rapacl.engines.constants as constants


def build_model_radiomics(device):
    model_radiomics = build_radiomics_learner(
        checkpoint=None,
        numerical_columns=RADIOMICS_FEATURES_NAMES,
        num_class=constants.NUM_CLASS,
        hidden_dropout_prob=constants.DROPOUT,
        projection_dim=constants.PROJECTION_DIM,
        activation=constants.ACTIVATION,
        ape_drop_rate=constants.APE_DROP_RATE,
        device=device,
    )

    return model_radiomics.to(device)


def main():
    set_seed(constants.SEED)

    device = torch.device(constants.DEVICE)

    print(f"[INFO] device: {device}")
    print(f"[INFO] checkpoint path: {constants.CHECKPOINT_PATH}")

    if constants.CHECKPOINT_PATH is None:
        raise ValueError("constants.CHECKPOINT_PATH is None")

    model_radiomics = build_model_radiomics(device)

    print("[INFO] model built successfully")

    load_model_radiomics_from_full_checkpoint(
        model_radiomics=model_radiomics,
        checkpoint_path=constants.CHECKPOINT_PATH,
        device=device,
        strict=False,
    )

    print("[INFO] checkpoint loaded successfully")

    total_params = sum(p.numel() for p in model_radiomics.parameters())
    trainable_params = sum(p.numel() for p in model_radiomics.parameters() if p.requires_grad)

    print(f"[INFO] total params: {total_params:,}")
    print(f"[INFO] trainable params: {trainable_params:,}")


if __name__ == "__main__":
    main()
