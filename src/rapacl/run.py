import os
import random
import argparse
from typing import Optional

import numpy as np
import torch
import pandas as pd

from src.rapacl.models.rapacl import RaPaCL
from src.rapacl.losses.rapacl_criterion import RaPaCLCriterion


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Run RaPaCL training / debugging entrypoint.")

    # -------------------------------------------------
    # basic runtime
    # -------------------------------------------------
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output_dir", type=str, default="./outputs/rapacl")

    # -------------------------------------------------
    # patch encoder
    # -------------------------------------------------
    parser.add_argument("--patch_pretrained", action="store_true")
    parser.add_argument("--patch_feat_dim", type=int, default=1024)

    # -------------------------------------------------
    # radiomics encoder
    # -------------------------------------------------
    parser.add_argument("--radiomics_hidden_dim", type=int, default=128)
    parser.add_argument("--num_sub_cols", type=int, nargs="+", default=[93, 62, 31, 15, 7, 3, 1])
    parser.add_argument("--separate_contrast_token", action="store_true")

    # -------------------------------------------------
    # projection / contrastive
    # -------------------------------------------------
    parser.add_argument("--proj_dim", type=int, default=128)
    parser.add_argument("--projection_use_mlp", action="store_true")
    parser.add_argument("--projection_hidden_dim", type=int, default=None)
    parser.add_argument("--projection_dropout", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.07)

    # -------------------------------------------------
    # classification
    # -------------------------------------------------
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--classifier_dropout", type=float, default=0.0)

    # -------------------------------------------------
    # batch correction
    # -------------------------------------------------
    parser.add_argument("--use_batch_correction", action="store_true")
    parser.add_argument("--num_batch_labels", type=int, default=None)
    parser.add_argument("--batch_disc_hidden_dim", type=int, default=None)
    parser.add_argument("--batch_disc_nlayers", type=int, default=2)
    parser.add_argument("--batch_disc_dropout", type=float, default=0.1)
    parser.add_argument("--batch_disc_grl_lambda", type=float, default=1.0)

    # -------------------------------------------------
    # loss weights
    # -------------------------------------------------
    parser.add_argument("--lambda_contrastive", type=float, default=1.0)
    parser.add_argument("--lambda_patch_cls", type=float, default=1.0)
    parser.add_argument("--lambda_rad_cls", type=float, default=1.0)
    parser.add_argument("--lambda_batch", type=float, default=1.0)

    # -------------------------------------------------
    # debug
    # -------------------------------------------------
    parser.add_argument("--debug_forward", action="store_true")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_radiomics_features", type=int, default=93)

    return parser.parse_args()


def build_model(args) -> RaPaCL:
    numerical_columns = [f"rad_feat_{i}" for i in range(args.num_radiomics_features)]

    model = RaPaCL(
        patch_pretrained=args.patch_pretrained,
        patch_feat_dim=args.patch_feat_dim,
        radiomics_hidden_dim=args.radiomics_hidden_dim,
        proj_dim=args.proj_dim,
        projection_use_mlp=args.projection_use_mlp,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_dropout=args.projection_dropout,
        num_classes=args.num_classes,
        classifier_dropout=args.classifier_dropout,
        use_batch_correction=args.use_batch_correction,
        num_batch_labels=args.num_batch_labels,
        batch_disc_hidden_dim=args.batch_disc_hidden_dim,
        batch_disc_nlayers=args.batch_disc_nlayers,
        batch_disc_dropout=args.batch_disc_dropout,
        batch_disc_grl_lambda=args.batch_disc_grl_lambda,
        separate_contrast_token=args.separate_contrast_token,
        numerical_columns=numerical_columns,
        num_sub_cols=args.num_sub_cols,
        hidden_dim=args.radiomics_hidden_dim,
        device=args.device,
    )
    return model


def build_criterion(args) -> RaPaCLCriterion:
    criterion = RaPaCLCriterion(
        lambda_contrastive=args.lambda_contrastive,
        lambda_patch_cls=args.lambda_patch_cls,
        lambda_rad_cls=args.lambda_rad_cls,
        lambda_batch=args.lambda_batch,
        temperature=args.temperature,
        use_batch_correction=args.use_batch_correction,
    )
    return criterion


def make_dummy_batch(args, device: torch.device):
    patch_x = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=device,
    )

    radiomics_columns = [f"rad_feat_{i}" for i in range(args.num_radiomics_features)]
    radiomics_array = np.random.randn(args.batch_size, args.num_radiomics_features)
    radiomics_x = pd.DataFrame(radiomics_array, columns=radiomics_columns)

    batch = {
        "patch_x": patch_x,
        "radiomics_x": radiomics_x,
        "labels": torch.randint(0, args.num_classes, (args.batch_size,), device=device),
        "sample_ids": torch.arange(args.batch_size, device=device),
    }

    if args.use_batch_correction:
        if args.num_batch_labels is None:
            raise ValueError("`num_batch_labels` must be set when `--use_batch_correction` is enabled.")
        batch["batch_labels"] = torch.randint(
            0,
            args.num_batch_labels,
            (args.batch_size,),
            device=device,
        )

    return batch


def debug_forward_pass(model: RaPaCL, criterion: RaPaCLCriterion, args, device: torch.device):
    model.eval()

    batch = make_dummy_batch(args, device)

    with torch.no_grad():
        outputs = model(
            patch_x=batch["patch_x"],
            radiomics_x=batch["radiomics_x"],
        )

    print("\n[Model Outputs]")
    for k, v in outputs.items():
        if isinstance(v, torch.Tensor):
            print(f"{k}: shape={tuple(v.shape)}")
        else:
            print(f"{k}: type={type(v)}")

    # criterion이 아직 완성되지 않았을 수도 있으니 안전하게 처리
    try:
        loss_dict = criterion(outputs, batch)
        print("\n[Loss Dict]")
        for k, v in loss_dict.items():
            if isinstance(v, torch.Tensor):
                print(f"{k}: {v.item():.6f}")
            else:
                print(f"{k}: {v}")
    except Exception as e:
        print("\n[Criterion check skipped / failed]")
        print(f"{type(e).__name__}: {e}")


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = build_model(args).to(device)
    criterion = build_criterion(args)

    print("[INFO] Model and criterion successfully built.")
    print(model.__class__.__name__)
    print(criterion.__class__.__name__)

    if args.debug_forward:
        debug_forward_pass(model, criterion, args, device)
        return

    print("\n[INFO] Trainer is not connected yet.")
    print("[INFO] Next step: add dataloader / optimizer / trainer loop here.")


if __name__ == "__main__":
    main()
