from __future__ import annotations

import os
import random
from contextlib import nullcontext
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from rapacl.data._dataset import HestRadiomicsDataset
from rapacl.data.constants_radfeatcols import RADIOMICS_FEATURES_NAMES
from rapacl.model.radtranstab.build import build_radiomics_learner
from rapacl.engines.trainer_utils import set_seed
import rapacl.configs.default.train as CONSTANTS


# ============================================================
# Constants
# ============================================================
NUM_CELLTYPE_CLASS = 5
NUM_SUB_COLS = [72, 54, 36, 18, 9, 3, 1]
PROJECTION_DIM = 384
NUM_RADIOMICS = len(RADIOMICS_FEATURES_NAMES)
NUM_GENES = 250

CKPT_PATH = "/root/workspace/RaPaCL/rapacl/checkpoints/radiomics_retrieval/transtab/pytorch_model.bin"
SAVE_DIR = "/root/workspace/RaPaCL/rapacl/checkpoints/ratranstab_from-ckpt_transfer_ablation/verticalcl_then_train"

# ----------------------------
# Stage 1: Vertical CL only
# ----------------------------
STAGE1_NUM_EPOCHS = 50  # 50 ~ 100
STAGE1_LR = 1e-4
STAGE1_WEIGHT_DECAY = 1e-4
STAGE1_TEMPERATURE = 0.7
STAGE1_VIEW_SIZES = [72, 54, 36, 18, 9, 3, 1] # [72, 36, 9]  # 계산량 줄이려면 3개 정도 추천
STAGE1_SAVE_NAME = "stage1_vertical_cl.pt"

# ----------------------------
# Stage 2: Current training
# no freeze + gene + recon + cls
# ----------------------------
STAGE2_NUM_EPOCHS = 50
STAGE2_LR = 1e-4
STAGE2_WEIGHT_DECAY = 1e-4
GRAD_ACCUM_STEPS = 1

USE_AMP = True
AMP_DTYPE = torch.float16
CLIP_GRAD_NORM = 1.0

# Loss weights for Stage 2
W_RECON = 1.0
W_CLS = 1.0
W_GENE = 1.0

# DataLoader tuning
DEFAULT_NUM_WORKERS = 0
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True


# ============================================================
# Distributed utils
# ============================================================
def setup_distributed() -> tuple[bool, int, int, int, torch.device]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return True, rank, local_rank, world_size, device

    rank = 0
    local_rank = 0
    world_size = 1
    device = torch.device(CONSTANTS.DEVICE if torch.cuda.is_available() else "cpu")
    return False, rank, local_rank, world_size, device


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(rank: int) -> bool:
    return rank == 0


def rank0_print(rank: int, *args, **kwargs) -> None:
    if is_main_process(rank):
        print(*args, **kwargs)


def seed_everything(seed: int, rank: int = 0) -> None:
    seed = seed + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_seed(seed)


def reduce_mean(value: float | torch.Tensor, device: torch.device, world_size: int) -> float:
    if not torch.is_tensor(value):
        value = torch.tensor(value, dtype=torch.float32, device=device)
    else:
        value = value.detach().float().to(device)

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value /= world_size
    return value.item()


def gather_tensor_variable_length(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return tensor

    tensor = tensor.to(device)
    local_size = torch.tensor([tensor.size(0)], dtype=torch.long, device=device)
    sizes = [torch.zeros_like(local_size) for _ in range(dist.get_world_size())]
    dist.all_gather(sizes, local_size)
    sizes = [int(s.item()) for s in sizes]
    max_size = max(sizes)

    if tensor.size(0) < max_size:
        pad_shape = (max_size - tensor.size(0),) + tuple(tensor.shape[1:])
        pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=device)
        tensor = torch.cat([tensor, pad], dim=0)

    gathered = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)

    gathered = [g[:s] for g, s in zip(gathered, sizes)]
    return torch.cat(gathered, dim=0)


# ============================================================
# Model
# ============================================================
def build_radiomics_model_from_ckpt(device: torch.device) -> nn.Module:
    model = build_radiomics_learner(
        checkpoint=None,
        numerical_columns=RADIOMICS_FEATURES_NAMES,
        num_class=NUM_CELLTYPE_CLASS,
        projection_dim=PROJECTION_DIM,
        activation="leakyrelu",
        ape_drop_rate=0.0,
        num_sub_cols=NUM_SUB_COLS,
        device=device,
    ).to(device)

    print(f"[INFO] Load RadTransTab checkpoint: {CKPT_PATH}")
    state_dict = torch.load(CKPT_PATH, map_location="cpu")

    filtered_prefix = "clf."
    filtered = {k: v for k, v in state_dict.items() if not k.startswith(filtered_prefix)}
    missing, unexpected = model.load_state_dict(filtered, strict=False)

    print("[INFO] Loaded RadTransTab")
    print(f"[INFO] Skipped keys: {filtered_prefix}*")
    print(f"[INFO] Missing keys: {missing}")
    print(f"[INFO] Unexpected keys: {unexpected}")
    return model


class MLPHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RadTransTabGenePredModel(nn.Module):
    def __init__(self, device: torch.device, use_pandas_fallback: bool = True):
        super().__init__()
        self.device = device
        self.use_pandas_fallback = use_pandas_fallback

        self.radiomics_model = build_radiomics_model_from_ckpt(device=device).to(device)

        self.recon_head = MLPHead(
            in_dim=128,
            out_dim=NUM_RADIOMICS,
            hidden_dim=512,
            dropout=0.1,
        )
        self.cls_head = MLPHead(
            in_dim=128,
            out_dim=NUM_CELLTYPE_CLASS,
            hidden_dim=256,
            dropout=0.1,
        )
        self.gene_head = MLPHead(
            in_dim=PROJECTION_DIM,
            out_dim=NUM_GENES,
            hidden_dim=512,
            dropout=0.1,
        )

    def _make_input_for_transtab(self, radiomics: torch.Tensor | pd.DataFrame):
        if isinstance(radiomics, pd.DataFrame):
            return radiomics

        if not self.use_pandas_fallback:
            return radiomics

        return pd.DataFrame(
            radiomics.detach().float().cpu().numpy(),
            columns=RADIOMICS_FEATURES_NAMES,
        )

    def _encode_radiomics(self, radiomics: torch.Tensor | pd.DataFrame) -> dict[str, torch.Tensor]:
        x = self._make_input_for_transtab(radiomics)

        feat = self.radiomics_model.input_encoder(x)
        feat = self.radiomics_model.contrastive_token(**feat)
        feat = self.radiomics_model.cls_token(**feat)
        enc = self.radiomics_model.encoder(**feat)

        # 현재 코드 기준 유지
        rad_cls_h = enc[:, 0, :]
        rad_contrast_h = enc[:, -1, :]
        rad_contrast_z = self.radiomics_model.projection_head(rad_contrast_h)

        return {
            "rad_cls_h": rad_cls_h,
            "rad_contrast_h": rad_contrast_h,
            "rad_contrast_z": rad_contrast_z,
        }

    def forward_cl(self, radiomics: torch.Tensor | pd.DataFrame) -> torch.Tensor:
        rad = self._encode_radiomics(radiomics)
        return rad["rad_contrast_z"]

    def forward(self, radiomics: torch.Tensor | pd.DataFrame) -> dict[str, torch.Tensor]:
        rad = self._encode_radiomics(radiomics)

        pred_radiomics = self.recon_head(rad["rad_contrast_h"])
        pred_class_logits = self.cls_head(rad["rad_cls_h"])
        pred_gene = self.gene_head(rad["rad_contrast_z"])

        return {
            **rad,
            "pred_radiomics": pred_radiomics,
            "pred_class_logits": pred_class_logits,
            "pred_gene": pred_gene,
        }


def print_trainable_parameters(model: nn.Module, rank: int = 0):
    if not is_main_process(rank):
        return

    trainable, total = 0, 0
    print("\n[INFO] Trainable parameters:")
    for name, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()
            print(f"  [T] {name}")
        else:
            print(f"  [F] {name}")

    print(
        f"\n[INFO] trainable params: {trainable:,} / {total:,} "
        f"({100 * trainable / total:.2f}%)"
    )


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


# ============================================================
# Loss / Metric
# ============================================================
def symmetric_info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.7) -> torch.Tensor:
    a = F.normalize(a, dim=1)
    b = F.normalize(b, dim=1)
    logits = a @ b.t() / temperature
    labels = torch.arange(a.size(0), device=a.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def make_vertical_masked_view(
    radiomics: torch.Tensor,
    keep_cols: int,
) -> torch.Tensor:
    """
    TransTab input_encoder가 전체 column 이름을 기준으로 동작하므로,
    feature dimension을 줄이지 않고 mask-out 방식으로 vertical view를 만든다.

    keep_cols개 feature만 유지하고 나머지는 0으로 masking한다.
    """
    batch_size, num_features = radiomics.shape
    keep_cols = min(keep_cols, num_features)

    idx = torch.randperm(num_features, device=radiomics.device)[:keep_cols]
    mask = torch.zeros(num_features, device=radiomics.device, dtype=radiomics.dtype)
    mask[idx] = 1.0

    return radiomics * mask.unsqueeze(0)


def compute_vertical_cl_loss(
    model: nn.Module,
    radiomics: torch.Tensor,
    view_sizes: list[int],
    temperature: float,
) -> torch.Tensor:
    raw_model = unwrap_model(model)

    z_list = []
    for keep_cols in view_sizes:
        view = make_vertical_masked_view(radiomics, keep_cols=keep_cols)
        z = raw_model.forward_cl(view)
        z_list.append(z)

    loss = torch.zeros((), device=radiomics.device)
    count = 0

    for i in range(len(z_list)):
        for j in range(i + 1, len(z_list)):
            loss = loss + symmetric_info_nce(z_list[i], z_list[j], temperature=temperature)
            count += 1

    return loss / max(count, 1)


def compute_genewise_pcc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    pred = pred.detach().float().cpu()
    target = target.detach().float().cpu()

    pred_c = pred - pred.mean(dim=0, keepdim=True)
    target_c = target - target.mean(dim=0, keepdim=True)

    denom = torch.sqrt((pred_c ** 2).sum(dim=0) * (target_c ** 2).sum(dim=0)) + eps
    pcc_per_gene = (pred_c * target_c).sum(dim=0) / denom
    return pcc_per_gene.mean().item(), pcc_per_gene.numpy()


def compute_genewise_pcc_pearsonr(pred: torch.Tensor, target: torch.Tensor):
    pred_np = pred.detach().float().cpu().numpy()
    target_np = target.detach().float().cpu().numpy()

    pcc_list = []
    for i in range(pred_np.shape[1]):
        r, _ = pearsonr(pred_np[:, i], target_np[:, i])
        pcc_list.append(r)

    pcc_array = np.array(pcc_list)
    return float(np.nanmean(pcc_array)), pcc_array


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return (logits.argmax(dim=1) == target).float().mean().item()


def get_batch_tensor(batch: dict[str, Any], names: tuple[str, ...], device: torch.device) -> torch.Tensor:
    for name in names:
        if name in batch:
            return batch[name].to(device, non_blocking=True).float()
    raise KeyError(f"None of keys {names} found in batch. Available keys: {list(batch.keys())}")


def compute_stage2_losses(
    out: dict[str, torch.Tensor],
    radiomics: torch.Tensor,
    target_label: torch.Tensor,
    gene: torch.Tensor,
):
    recon_loss = F.mse_loss(out["pred_radiomics"], radiomics)
    cls_loss = F.cross_entropy(out["pred_class_logits"], target_label)
    gene_loss = F.mse_loss(out["pred_gene"], gene)
    loss = W_RECON * recon_loss + W_CLS * cls_loss + W_GENE * gene_loss
    return loss, recon_loss, cls_loss, gene_loss


# ============================================================
# Stage 1: Vertical CL train / eval
# ============================================================
def train_epoch_stage1_vertical_cl(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    rank: int,
    world_size: int,
    epoch: int,
):
    model.train()

    total_samples = 0
    loss_sum = 0.0

    optimizer.zero_grad(set_to_none=True)

    iterator = tqdm(loader, desc=f"Stage1-VerticalCL E{epoch}", leave=False, disable=not is_main_process(rank))
    amp_context = torch.autocast(device_type="cuda", dtype=AMP_DTYPE) if USE_AMP and device.type == "cuda" else nullcontext()

    for step, batch in enumerate(iterator, start=1):
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)

        with amp_context:
            loss = compute_vertical_cl_loss(
                model=model,
                radiomics=radiomics,
                view_sizes=STAGE1_VIEW_SIZES,
                temperature=STAGE1_TEMPERATURE,
            )
            loss_for_backward = loss / GRAD_ACCUM_STEPS

        if USE_AMP and device.type == "cuda":
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = (step % GRAD_ACCUM_STEPS == 0) or (step == len(loader))
        if should_step:
            if USE_AMP and device.type == "cuda":
                scaler.unscale_(optimizer)
                if CLIP_GRAD_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                if CLIP_GRAD_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        bs = radiomics.size(0)
        total_samples += bs
        loss_sum += loss.detach().item() * bs

    local_loss = loss_sum / max(total_samples, 1)
    global_loss = reduce_mean(local_loss, device=device, world_size=world_size)
    return {"vertical_cl_loss": global_loss}


@torch.no_grad()
def eval_epoch_stage1_vertical_cl(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    world_size: int,
    epoch: int,
):
    model.eval()

    total_samples = 0
    loss_sum = 0.0

    iterator = tqdm(loader, desc=f"Stage1-VerticalCL-Val E{epoch}", leave=False, disable=not is_main_process(rank))
    amp_context = torch.autocast(device_type="cuda", dtype=AMP_DTYPE) if USE_AMP and device.type == "cuda" else nullcontext()

    for batch in iterator:
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)

        with amp_context:
            loss = compute_vertical_cl_loss(
                model=model,
                radiomics=radiomics,
                view_sizes=STAGE1_VIEW_SIZES,
                temperature=STAGE1_TEMPERATURE,
            )

        bs = radiomics.size(0)
        total_samples += bs
        loss_sum += loss.detach().item() * bs

    local_loss = loss_sum / max(total_samples, 1)
    global_loss = reduce_mean(local_loss, device=device, world_size=world_size)
    return {"vertical_cl_loss": global_loss}


# ============================================================
# Stage 2: Gene + Recon + Cls train / eval
# ============================================================
def train_epoch_stage2(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    rank: int,
    world_size: int,
    epoch: int,
):
    model.train()

    total_samples = 0
    metric_sums = {
        "loss": 0.0,
        "loss-recon": 0.0,
        "loss-cls": 0.0,
        "loss-gene": 0.0,
        "cls-acc": 0.0,
    }

    optimizer.zero_grad(set_to_none=True)

    iterator = tqdm(loader, desc=f"Stage2 E{epoch}", leave=False, disable=not is_main_process(rank))
    amp_context = torch.autocast(device_type="cuda", dtype=AMP_DTYPE) if USE_AMP and device.type == "cuda" else nullcontext()

    for step, batch in enumerate(iterator, start=1):
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        target_label = get_batch_tensor(batch, ("target_label", "label", "celltype_label"), device).long()
        gene = get_batch_tensor(batch, ("gene", "expression", "expr"), device)

        with amp_context:
            out = model(radiomics=radiomics)
            loss, recon_loss, cls_loss, gene_loss = compute_stage2_losses(out, radiomics, target_label, gene)
            loss_for_backward = loss / GRAD_ACCUM_STEPS

        if USE_AMP and device.type == "cuda":
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = (step % GRAD_ACCUM_STEPS == 0) or (step == len(loader))
        if should_step:
            if USE_AMP and device.type == "cuda":
                scaler.unscale_(optimizer)
                if CLIP_GRAD_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
            else:
                if CLIP_GRAD_NORM is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD_NORM)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        bs = radiomics.size(0)
        total_samples += bs
        metric_sums["loss"] += loss.detach().item() * bs
        metric_sums["loss-recon"] += recon_loss.detach().item() * bs
        metric_sums["loss-cls"] += cls_loss.detach().item() * bs
        metric_sums["loss-gene"] += gene_loss.detach().item() * bs
        metric_sums["cls-acc"] += accuracy(out["pred_class_logits"].detach(), target_label) * bs

    local_metrics = {k: v / max(total_samples, 1) for k, v in metric_sums.items()}
    global_metrics = {
        k: reduce_mean(v, device=device, world_size=world_size)
        for k, v in local_metrics.items()
    }
    return global_metrics


@torch.no_grad()
def eval_epoch_stage2(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    rank: int,
    world_size: int,
    epoch: int,
):
    model.eval()

    total_samples = 0
    metric_sums = {
        "loss": 0.0,
        "loss-recon": 0.0,
        "loss-cls": 0.0,
        "loss-gene": 0.0,
        "cls-acc": 0.0,
    }

    preds = []
    targets = []

    iterator = tqdm(loader, desc=f"Stage2-Val E{epoch}", leave=False, disable=not is_main_process(rank))
    amp_context = torch.autocast(device_type="cuda", dtype=AMP_DTYPE) if USE_AMP and device.type == "cuda" else nullcontext()

    for batch in iterator:
        radiomics = get_batch_tensor(batch, ("radiomics", "radiomics_features"), device)
        target_label = get_batch_tensor(batch, ("target_label", "label", "celltype_label"), device).long()
        gene = get_batch_tensor(batch, ("gene", "expression", "expr"), device)

        with amp_context:
            out = model(radiomics=radiomics)
            loss, recon_loss, cls_loss, gene_loss = compute_stage2_losses(out, radiomics, target_label, gene)

        bs = radiomics.size(0)
        total_samples += bs
        metric_sums["loss"] += loss.detach().item() * bs
        metric_sums["loss-recon"] += recon_loss.detach().item() * bs
        metric_sums["loss-cls"] += cls_loss.detach().item() * bs
        metric_sums["loss-gene"] += gene_loss.detach().item() * bs
        metric_sums["cls-acc"] += accuracy(out["pred_class_logits"].detach(), target_label) * bs

        preds.append(out["pred_gene"].detach().float())
        targets.append(gene.detach().float())

    local_metrics = {k: v / max(total_samples, 1) for k, v in metric_sums.items()}
    global_metrics = {
        k: reduce_mean(v, device=device, world_size=world_size)
        for k, v in local_metrics.items()
    }

    pred_local = torch.cat(preds, dim=0) if preds else torch.empty((0, NUM_GENES), device=device)
    target_local = torch.cat(targets, dim=0) if targets else torch.empty((0, NUM_GENES), device=device)

    pred_all = gather_tensor_variable_length(pred_local, device=device)
    target_all = gather_tensor_variable_length(target_local, device=device)

    mean_pcc, pcc_per_gene = compute_genewise_pcc(pred_all, target_all)
    global_metrics["mean_pcc"] = mean_pcc
    global_metrics["pcc_per_gene"] = pcc_per_gene

    return global_metrics


# ============================================================
# DataLoader builder
# ============================================================
def build_dataloaders(is_distributed: bool, rank: int, world_size: int):
    trainset = HestRadiomicsDataset(
        bench_data_root=CONSTANTS.ROOT_DIR,
        split_csv_path=CONSTANTS.TRAIN_SPLIT_CSV,
        gene_list_path=CONSTANTS.GENE_LIST_PATH,
        feature_list_path=CONSTANTS.FEATURE_LIST_PATH,
        radiomics_dir=getattr(CONSTANTS, "RADIOMICS_DIR", "radiomics_features"),
    )
    evalset = HestRadiomicsDataset(
        bench_data_root=CONSTANTS.ROOT_DIR,
        split_csv_path=CONSTANTS.VAL_SPLIT_CSV,
        gene_list_path=CONSTANTS.GENE_LIST_PATH,
        feature_list_path=CONSTANTS.FEATURE_LIST_PATH,
        radiomics_dir=getattr(CONSTANTS, "RADIOMICS_DIR", "radiomics_features"),
    )

    train_sampler = DistributedSampler(
        trainset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=False,
    ) if is_distributed else None

    eval_sampler = DistributedSampler(
        evalset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        drop_last=False,
    ) if is_distributed else None

    num_workers = getattr(CONSTANTS, "NUM_WORKERS", DEFAULT_NUM_WORKERS)
    if num_workers is None or num_workers <= 0:
        num_workers = DEFAULT_NUM_WORKERS

    train_loader = DataLoader(
        trainset,
        batch_size=CONSTANTS.BATCH_SIZE,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=PERSISTENT_WORKERS if num_workers > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
    )
    eval_loader = DataLoader(
        evalset,
        batch_size=CONSTANTS.BATCH_SIZE,
        shuffle=False,
        sampler=eval_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=PERSISTENT_WORKERS if num_workers > 0 else False,
        prefetch_factor=PREFETCH_FACTOR if num_workers > 0 else None,
    )

    return trainset, evalset, train_loader, eval_loader, train_sampler, eval_sampler, num_workers


# ============================================================
# Main
# ============================================================
def main():
    is_distributed, rank, local_rank, world_size, device = setup_distributed()
    seed_everything(CONSTANTS.SEED, rank=rank)

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    rank0_print(rank, f"[INFO] distributed: {is_distributed}")
    rank0_print(rank, f"[INFO] world_size: {world_size}")
    rank0_print(rank, f"[INFO] device: {device}")

    trainset, evalset, train_loader, eval_loader, train_sampler, eval_sampler, num_workers = build_dataloaders(
        is_distributed=is_distributed,
        rank=rank,
        world_size=world_size,
    )

    rank0_print(rank, f"[INFO] train samples: {len(trainset)}")
    rank0_print(rank, f"[INFO] val samples: {len(evalset)}")
    rank0_print(rank, f"[INFO] num_genes: {len(trainset.genes)}")
    rank0_print(rank, f"[INFO] num_radiomics_features: {len(RADIOMICS_FEATURES_NAMES)}")
    rank0_print(rank, f"[INFO] batch_size per GPU: {CONSTANTS.BATCH_SIZE}")
    rank0_print(rank, f"[INFO] effective batch_size: {CONSTANTS.BATCH_SIZE * world_size * GRAD_ACCUM_STEPS}")
    rank0_print(rank, f"[INFO] num_workers per process: {num_workers}")

    if is_main_process(rank):
        os.makedirs(SAVE_DIR, exist_ok=True)
        rank0_print(rank, f"[INFO] save directory: {SAVE_DIR}")

    model = RadTransTabGenePredModel(device=device, use_pandas_fallback=True).to(device)

    # No freeze: all parameters trainable
    for param in model.parameters():
        param.requires_grad = True

    print_trainable_parameters(model, rank=rank)

    if is_distributed:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )

    # ========================================================
    # Stage 1: Vertical CL only
    # ========================================================
    rank0_print(rank, "\n==============================")
    rank0_print(rank, "[STAGE 1] Vertical CL only")
    rank0_print(rank, "==============================")

    stage1_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=STAGE1_LR,
        weight_decay=STAGE1_WEIGHT_DECAY,
    )
    stage1_scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    stage1_best_path = os.path.join(SAVE_DIR, "stage1_best_vertical_cl.pt")
    stage1_last_path = os.path.join(SAVE_DIR, STAGE1_SAVE_NAME)
    best_stage1_val = float("inf")

    for epoch in range(1, STAGE1_NUM_EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        train_m = train_epoch_stage1_vertical_cl(
            model=model,
            loader=train_loader,
            optimizer=stage1_optimizer,
            scaler=stage1_scaler,
            device=device,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
        )
        eval_m = eval_epoch_stage1_vertical_cl(
            model=model,
            loader=eval_loader,
            device=device,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
        )

        if is_main_process(rank):
            print(
                f"[INFO][Stage1][Epoch {epoch}] "
                f"train_vertical_cl={train_m['vertical_cl_loss']:.4f} | "
                f"val_vertical_cl={eval_m['vertical_cl_loss']:.4f}"
            )

            raw_model = unwrap_model(model)
            ckpt = {
                "stage": "stage1_vertical_cl",
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": stage1_optimizer.state_dict(),
                "scaler_state_dict": stage1_scaler.state_dict(),
                "train_metrics": train_m,
                "val_metrics": eval_m,
                "config": {
                    "stage1_num_epochs": STAGE1_NUM_EPOCHS,
                    "stage1_lr": STAGE1_LR,
                    "stage1_weight_decay": STAGE1_WEIGHT_DECAY,
                    "stage1_temperature": STAGE1_TEMPERATURE,
                    "stage1_view_sizes": STAGE1_VIEW_SIZES,
                    "batch_size_per_gpu": CONSTANTS.BATCH_SIZE,
                    "world_size": world_size,
                    "grad_accum_steps": GRAD_ACCUM_STEPS,
                    "effective_batch_size": CONSTANTS.BATCH_SIZE * world_size * GRAD_ACCUM_STEPS,
                    "use_amp": USE_AMP,
                    "amp_dtype": str(AMP_DTYPE),
                },
            }
            torch.save(ckpt, stage1_last_path)

            if eval_m["vertical_cl_loss"] < best_stage1_val:
                best_stage1_val = eval_m["vertical_cl_loss"]
                torch.save(ckpt, stage1_best_path)
                print(f"[INFO] Saved Stage1 best checkpoint: {stage1_best_path} | best_val={best_stage1_val:.4f}")

    if is_distributed:
        dist.barrier()

    # ========================================================
    # Stage 2: Gene + Recon + Cls, no freeze
    # ========================================================
    rank0_print(rank, "\n==============================")
    rank0_print(rank, "[STAGE 2] Gene + Recon + Cls, no freeze")
    rank0_print(rank, "==============================")

    # Stage 2는 Stage 1에서 이어진 모델 weight 그대로 사용
    # optimizer는 새로 시작하는 것이 안정적
    stage2_optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=STAGE2_LR,
        weight_decay=STAGE2_WEIGHT_DECAY,
    )
    stage2_scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device.type == "cuda"))

    stage2_best_path = os.path.join(SAVE_DIR, "stage2_best.pt")
    stage2_last_path = os.path.join(SAVE_DIR, "stage2_last.pt")
    best_stage2_val = float("inf")

    for epoch in range(1, STAGE2_NUM_EPOCHS + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch + STAGE1_NUM_EPOCHS)

        train_m = train_epoch_stage2(
            model=model,
            loader=train_loader,
            optimizer=stage2_optimizer,
            scaler=stage2_scaler,
            device=device,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
        )

        eval_m = eval_epoch_stage2(
            model=model,
            loader=eval_loader,
            device=device,
            rank=rank,
            world_size=world_size,
            epoch=epoch,
        )

        if is_main_process(rank):
            print(
                f"[INFO][Stage2][Epoch {epoch}] "
                f"train_loss={train_m['loss']:.4f} "
                f"train_recon={train_m['loss-recon']:.4f} "
                f"train_cls={train_m['loss-cls']:.4f} "
                f"train_acc={train_m['cls-acc']:.4f} "
                f"train_gene={train_m['loss-gene']:.4f} | "
                f"val_loss={eval_m['loss']:.4f} "
                f"val_recon={eval_m['loss-recon']:.4f} "
                f"val_cls={eval_m['loss-cls']:.4f} "
                f"val_acc={eval_m['cls-acc']:.4f} "
                f"val_gene={eval_m['loss-gene']:.4f} "
                f"mean_pcc={eval_m['mean_pcc']:.4f}"
            )

            raw_model = unwrap_model(model)
            ckpt = {
                "stage": "stage2_gene_recon_cls_no_freeze",
                "epoch": epoch,
                "model_state_dict": raw_model.state_dict(),
                "optimizer_state_dict": stage2_optimizer.state_dict(),
                "scaler_state_dict": stage2_scaler.state_dict(),
                "train_metrics": train_m,
                "val_metrics": {k: v for k, v in eval_m.items() if k != "pcc_per_gene"},
                "pcc_per_gene": eval_m["pcc_per_gene"],
                "config": {
                    "stage1_num_epochs": STAGE1_NUM_EPOCHS,
                    "stage2_num_epochs": STAGE2_NUM_EPOCHS,
                    "stage2_lr": STAGE2_LR,
                    "stage2_weight_decay": STAGE2_WEIGHT_DECAY,
                    "batch_size_per_gpu": CONSTANTS.BATCH_SIZE,
                    "world_size": world_size,
                    "grad_accum_steps": GRAD_ACCUM_STEPS,
                    "effective_batch_size": CONSTANTS.BATCH_SIZE * world_size * GRAD_ACCUM_STEPS,
                    "use_amp": USE_AMP,
                    "amp_dtype": str(AMP_DTYPE),
                    "w_recon": W_RECON,
                    "w_cls": W_CLS,
                    "w_gene": W_GENE,
                    "freeze": False,
                },
            }

            torch.save(ckpt, stage2_last_path)

            # gene prediction 목적이면 val_loss보다 mean_pcc 기준 저장도 고려 가능
            # 현재는 기존 코드와 동일하게 val_loss 기준
            if eval_m["loss"] < best_stage2_val:
                best_stage2_val = eval_m["loss"]
                torch.save(ckpt, stage2_best_path)
                print(f"[INFO] Saved Stage2 best checkpoint: {stage2_best_path} | best_val={best_stage2_val:.4f}")

    cleanup_distributed(is_distributed)


if __name__ == "__main__":
    main()

