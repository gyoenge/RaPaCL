# 기존 Trainer를 최대한 유지하면서 DDP 옵션을 추가한 버전
"""
distributed, local_rank, rank, world_size 추가
DataLoader에 DistributedSampler 사용
DDP wrapping
rank 0만 저장/출력
validation metric은 all-reduce 해서 평균내기
"""

import os
import pdb
import math
import time
import json

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import pandas as pd
from tqdm.autonotebook import trange

from transtab import constants
from transtab.evaluator import predict, get_eval_metric_fn, EarlyStopping
from transtab.modeling_transtab import TransTabFeatureExtractor
from transtab.trainer_utils import SupervisedTrainCollator, TrainDataset
from transtab.trainer_utils import get_parameter_names
from transtab.trainer_utils import get_scheduler

import logging
logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self,
        model,
        train_set_list,
        test_set_list=None,
        collate_fn=None,
        output_dir='./ckpt',
        num_epoch=10,
        batch_size=64,
        lr=1e-4,
        weight_decay=0,
        patience=5,
        eval_batch_size=256,
        warmup_ratio=None,
        warmup_steps=None,
        balance_sample=False,
        load_best_at_last=True,
        ignore_duplicate_cols=False,
        eval_metric='auc',
        eval_less_is_better=False,
        num_workers=0,
        distributed=False,
        local_rank=0,
        rank=0,
        world_size=1,
        device=None,
        **kwargs,
        ):
        self.model = model
        if isinstance(train_set_list, tuple):
            train_set_list = [train_set_list]
        if isinstance(test_set_list, tuple):
            test_set_list = [test_set_list]

        self.train_set_list = train_set_list
        self.test_set_list = test_set_list
        self.collate_fn = collate_fn
        self.distributed = distributed
        self.local_rank = local_rank
        self.rank = rank
        self.world_size = world_size
        self.device = device if device is not None else f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'

        if collate_fn is None:
            self.collate_fn = SupervisedTrainCollator(
                categorical_columns=model.categorical_columns,
                numerical_columns=model.numerical_columns,
                binary_columns=model.binary_columns,
                ignore_duplicate_cols=ignore_duplicate_cols,
            )

        self.trainloader_list = []
        self.train_sampler_list = []

        for trainset in train_set_list:
            loader, sampler = self._build_dataloader(
                trainset,
                batch_size,
                collator=self.collate_fn,
                num_workers=num_workers,
                shuffle=True,
                distributed=self.distributed,
            )
            self.trainloader_list.append(loader)
            self.train_sampler_list.append(sampler)

        if test_set_list is not None:
            self.testloader_list = []
            self.test_sampler_list = []
            for testset in test_set_list:
                loader, sampler = self._build_dataloader(
                    testset,
                    eval_batch_size,
                    collator=self.collate_fn,
                    num_workers=num_workers,
                    shuffle=False,
                    distributed=self.distributed,
                )
                self.testloader_list.append(loader)
                self.test_sampler_list.append(sampler)
        else:
            self.testloader_list = None
            self.test_sampler_list = None

        self.output_dir = output_dir
        self.early_stopping = EarlyStopping(
            output_dir=output_dir,
            patience=patience,
            verbose=False,
            less_is_better=eval_less_is_better
        )

        self.args = {
            'lr': lr,
            'weight_decay': weight_decay,
            'batch_size': batch_size,
            'num_epoch': num_epoch,
            'eval_batch_size': eval_batch_size,
            'warmup_ratio': warmup_ratio,
            'warmup_steps': warmup_steps,
            'num_training_steps': self.get_num_train_steps(train_set_list, num_epoch, batch_size),
            'eval_metric': get_eval_metric_fn(eval_metric),
            'eval_metric_name': eval_metric,
            'eval_less_is_better': eval_less_is_better,
        }
        self.args['steps_per_epoch'] = int(
            self.args['num_training_steps'] / (num_epoch * len(self.train_set_list))
        )

        if self.is_main_process() and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # model to device
        self.model.to(self.device)

        # wrap DDP
        if self.distributed:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            )

        self.optimizer = None
        self.lr_scheduler = None
        self.balance_sample = balance_sample
        self.load_best_at_last = load_best_at_last

    def is_main_process(self):
        return (not self.distributed) or self.rank == 0

    def unwrap_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def train(self):
        args = self.args
        self.create_optimizer()
        if args['warmup_ratio'] is not None or args['warmup_steps'] is not None:
            num_train_steps = args['num_training_steps']
            if self.is_main_process():
                logger.info(f'set warmup training in initial {num_train_steps} steps')
            self.create_scheduler(num_train_steps, self.optimizer)

        start_time = time.time()

        epoch_iter = trange(args['num_epoch'], desc='Epoch', disable=not self.is_main_process())
        for epoch in epoch_iter:
            ite = 0
            train_loss_all = 0.0
            self.model.train()

            if self.distributed:
                for sampler in self.train_sampler_list:
                    if sampler is not None:
                        sampler.set_epoch(epoch)

            for dataindex in range(len(self.trainloader_list)):
                for data in self.trainloader_list[dataindex]:
                    self.optimizer.zero_grad()
                    logits, loss = self.model(data[0], data[1])
                    loss.backward()
                    self.optimizer.step()

                    train_loss_all += loss.item()
                    ite += 1

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

            # average train loss across ranks
            if self.distributed:
                loss_tensor = torch.tensor([train_loss_all, ite], device=self.device, dtype=torch.float64)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                train_loss_all = (loss_tensor[0] / loss_tensor[1]).item()
            else:
                train_loss_all = train_loss_all / max(ite, 1)

            if self.test_set_list is not None:
                eval_res_list = self.evaluate()
                eval_res = np.mean(eval_res_list)

                if self.is_main_process():
                    print(f'epoch: {epoch}, test {self.args["eval_metric_name"]}: {eval_res:.6f}')
                    self.early_stopping(-eval_res if not self.args['eval_less_is_better'] else eval_res, self.unwrap_model())
                    if self.early_stopping.early_stop:
                        print('early stopped')

                if self.distributed:
                    stop_flag = torch.tensor(
                        [1 if (self.is_main_process() and self.early_stopping.early_stop) else 0],
                        device=self.device
                    )
                    dist.broadcast(stop_flag, src=0)
                    if stop_flag.item() == 1:
                        break
                else:
                    if self.early_stopping.early_stop:
                        break

            if self.is_main_process():
                print(
                    'epoch: {}, train loss: {:.4f}, lr: {:.6f}, spent: {:.1f} secs'.format(
                        epoch,
                        train_loss_all,
                        self.optimizer.param_groups[0]['lr'],
                        time.time() - start_time
                    )
                )

        if os.path.exists(self.output_dir):
            if self.test_set_list is not None and self.is_main_process():
                logger.info(f'load best at last from {self.output_dir}')
                state_dict = torch.load(
                    os.path.join(self.output_dir, constants.WEIGHTS_NAME),
                    map_location='cpu'
                )
                self.unwrap_model().load_state_dict(state_dict)

            if self.is_main_process():
                self.save_model(self.output_dir)

        if self.is_main_process():
            logger.info('training complete, cost {:.1f} secs.'.format(time.time()-start_time))

    def evaluate(self):
        self.model.eval()
        eval_res_list = []

        for dataindex in range(len(self.testloader_list)):
            y_test, pred_list, loss_list = [], [], []

            for data in self.testloader_list[dataindex]:
                if data[1] is not None:
                    label = data[1]
                    if isinstance(label, pd.Series):
                        label = label.values
                    y_test.append(label)

                with torch.no_grad():
                    logits, loss = self.model(data[0], data[1])

                if loss is not None:
                    loss_list.append(loss.item())

                if logits is not None:
                    if logits.shape[-1] == 1:
                        pred_list.append(logits.sigmoid().detach().cpu().numpy())
                    else:
                        pred_list.append(torch.softmax(logits, -1).detach().cpu().numpy())

            # 기존 metric 계산 방식 유지
            if len(pred_list) > 0:
                pred_all = np.concatenate(pred_list, 0)
                if logits.shape[-1] == 1:
                    pred_all = pred_all.flatten()

            if self.args['eval_metric_name'] == 'val_loss':
                local_eval_res = np.mean(loss_list)
                if self.distributed:
                    metric_tensor = torch.tensor([local_eval_res], device=self.device, dtype=torch.float64)
                    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                    eval_res = (metric_tensor / self.world_size).item()
                else:
                    eval_res = local_eval_res
            else:
                # metric like auc/acc는 rank별 prediction 전체를 gather해야 정확함
                # 간단 버전: 각 rank metric 평균
                y_test = np.concatenate(y_test, 0)
                local_eval_res = self.args['eval_metric'](y_test, pred_all)

                if self.distributed:
                    metric_tensor = torch.tensor([local_eval_res], device=self.device, dtype=torch.float64)
                    dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
                    eval_res = (metric_tensor / self.world_size).item()
                else:
                    eval_res = local_eval_res

            eval_res_list.append(eval_res)

        return eval_res_list

    def save_model(self, output_dir=None):
        if output_dir is None:
            output_dir = self.output_dir

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        logger.info(f'saving model checkpoint to {output_dir}')
        self.unwrap_model().save(output_dir)
        self.collate_fn.save(output_dir)

        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), os.path.join(output_dir, constants.OPTIMIZER_NAME))
        if self.lr_scheduler is not None:
            torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, constants.SCHEDULER_NAME))
        if self.args is not None:
            train_args = {}
            for k, v in self.args.items():
                if isinstance(v, (int, str, float, bool)):
                    train_args[k] = v
            with open(os.path.join(output_dir, constants.TRAINING_ARGS_NAME), 'w', encoding='utf-8') as f:
                f.write(json.dumps(train_args))

    def create_optimizer(self):
        if self.optimizer is None:
            decay_parameters = get_parameter_names(self.unwrap_model(), [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]

            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.unwrap_model().named_parameters() if n in decay_parameters],
                    "weight_decay": self.args['weight_decay'],
                },
                {
                    "params": [p for n, p in self.unwrap_model().named_parameters() if n not in decay_parameters],
                    "weight_decay": 0.0,
                },
            ]
            self.optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=self.args['lr'])

    def create_scheduler(self, num_training_steps, optimizer):
        self.lr_scheduler = get_scheduler(
            'cosine',
            optimizer=optimizer,
            num_warmup_steps=self.get_warmup_steps(num_training_steps),
            num_training_steps=num_training_steps,
        )
        return self.lr_scheduler

    def get_num_train_steps(self, train_set_list, num_epoch, batch_size):
        total_step = 0
        for trainset in train_set_list:
            x_train, _ = trainset
            total_step += np.ceil(len(x_train) / batch_size)

        if self.distributed:
            total_step = np.ceil(total_step / self.world_size)

        total_step *= num_epoch
        return total_step

    def get_warmup_steps(self, num_training_steps):
        warmup_steps = (
            self.args['warmup_steps']
            if self.args['warmup_steps'] is not None
            else math.ceil(num_training_steps * self.args['warmup_ratio'])
        )
        return warmup_steps

    def _build_dataloader(self, trainset, batch_size, collator, num_workers=8, shuffle=True, distributed=False):
        dataset = TrainDataset(trainset)
        sampler = None

        if distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=shuffle,
                drop_last=False,
            )

        trainloader = DataLoader(
            dataset,
            collate_fn=collator,
            batch_size=batch_size,
            shuffle=(sampler is None and shuffle),
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )
        return trainloader, sampler