

# reference: Radiomics Retrieval 

import os 
import json
import math 
import collections 
from typing import List, Dict, Mapping, Optional, Tuple, Any, Union
import pandas as pd 
import numpy as np 

import torch 
import torch.nn as nn 
from torch import Tensor 
from torch.autograd import Function
import torch.nn.init as nn_init 
import torch.nn.functional as F 

from src.pretrain_transtab.transtab_custom.models.base import (
    TransTabFeatureExtractor, 
    TransTabFeatureProcessor, 
    TransTabInputEncoder, 
    TransTabEncoder, 
    # TransTabModel, # only sinlge CLS token 
    TransTabProjectionHead, 
    TransTabLinearClassifier, 
)
import src.pretrain_transtab.transtab_custom.constants as constants 

import logging
logger = logging.getLogger(__name__)


### TransTabForRadiomicsCL

class TransTabModelCustom(nn.Module):
    """added CLS(+Contrastive) selectively."""
    def __init__(self, 
        # special token setting 
        separate_contrast_token: bool=True, # False이면 [CLS], True이면 [CLS]+[Contrastive]
        # basics 
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation="relu",
        device="cuda:0",
        **kwargs,             
    ) -> None: 
        super().__init__()

        # TransTabFeatureExtractor 
        self.categorical_columns=categorical_columns
        self.numerical_columns=numerical_columns
        self.binary_columns=binary_columns
        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))
        if feature_extractor is None: 
            feature_extractor = TransTabFeatureExtractor(
                categorical_columns=self.categorical_columns,
                numerical_columns=self.numerical_columns, 
                binary_columns=self.binary_columns,
                **kwargs, 
            )

        # TransTabFeatureProcessor
        feature_processor = TransTabFeatureProcessor(
            vocab_size=feature_extractor.vocab_size, 
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim, 
            hidden_dropout_prob=hidden_dropout_prob,
            device=device, 
        )

        # TransTabInputEncoder 
        self.input_encoder = TransTabInputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
        )

        # TransTabEncoder
        self.encoder = TransTabEncoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation, 
        )

        # Token 
        # [CLS, CONTRAST, feature...]
        self.cls_token = AdditionalToken(
            hidden_dim=hidden_dim, 
        )
        self.contrastive_token = AdditionalToken(
            hidden_dim=hidden_dim, 
        ) if separate_contrast_token else None 

        # device setting 
        self.device = device 
        self.to(device)

    def forward(self, 
        x, # input table features 
        y=None, # input table labels (optional for supervision)
    ): 
        # extract the embeddings based on input tables 
        embedded = self.input_encoder(x)
        if self.contrastive_token is not None: 
            embedded = self.contrastive_token(**embedded)
        embedded = self.cls_token(**embedded)

        # pass through transformer layers to obtain final token embedding(s)
        encoder_output = self.encoder(**embedded)

        # get CLS/Contrastive token
        final_cls_embedding = encoder_output[:, 0, :]
        final_contrastive_embedding = encoder_output[:, 1, :] if self.contrastive_token is not None else None 

        return final_cls_embedding, final_contrastive_embedding 
    
    def load(self, 
        ckpt_dir, 
    ): 
        """load model state_dict and feature_extractor configuration from the `ckpt_dir`"""
        # load model weights, state dict 
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location="cpu")
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f"[load model] missing keys: {missing_keys}")
        logger.info(f"[load model] unexpected keys: {unexpected_keys}")
        logger.info(f"[load model] load model from: {ckpt_dir}")

        # load feature extractor 
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

    def save(self,
        ckpt_dir, 
    ): 
        """save model state_dict and feature_extractor configuration to the `ckpt_dir`"""
        # save model weights, state dict 
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict() 
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        # save feature extractor 
        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)
        # save input encoder separtely 
        state_dict_input_encoder = self.input_encoder.state_dict() 
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))

    def update(self, 
        config,            
    ): 
        """update the configuration of column map for cat, num, and bin cols
        or update the number of classes for the output classifier layer. (classifier will be implemented at child `TransTabForRadiomics`)"""
        # update column map 
        col_map = {}
        for k,v in config.items():
            if k in ['cat','num','bin']: col_map[k] = v
        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns 
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns
        # update number of classes 
        if 'num_class' in config: 
            num_class = config['num_class'] 
            self._adapt_to_new_num_class(num_class) 
        return None 
    
    def _check_column_overlap(self, 
        cat_cols=None, 
        num_cols=None, 
        bin_cols=None,
    ): 
        """check overlapping columns"""
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        unq_length = len(list(set(all_cols)))
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1]
        return org_length == unq_length, duplicate_cols

    def _solve_duplicate_cols(self, 
        duplicate_cols, 
    ):
        """solve the issue of duplicate cols: separate with different prefixes"""
        for col in duplicate_cols:
            logger.warning(f'Find duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_columns: 
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')

    def _adapt_to_new_num_class(self, 
        num_class, 
    ):
        """adapting new num_class, to deal with task change. (classifier will be implemented at child `TransTabForRadiomics`)"""
        if num_class != self.num_class:
            self.num_class = num_class 
            self.clf = TransTabLinearClassifier(num_class, hidden_dim=self.cls_token.hidden_dim)
            self.clf.to(self.device)
            if self.num_class > 2:
                self.loss_fn = nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            logger.info(f'Build a new classifier with num {num_class} classes outputs, need further finetune to work.') 


class AdditionalToken(nn.Module): 
    """add a learnable [CLS/CONTRAST] token embedding at the **first** of each sequence.
        
        call order matters: contrastive_token first, then cls_token
        => final token order becomes [CLS, CONTRAST, feature...]
    """
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.weight = nn.Parameter(Tensor(hidden_dim))
        nn_init.uniform_(self.weight, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))
        self.hidden_dim = hidden_dim

    def expand(self, *leading_dimensions):
        new_dims = (1,) * (len(leading_dimensions)-1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)
    
    def forward(self, embedding, attention_mask=None, **kwargs) -> Tensor:
        embedding = torch.cat([self.expand(len(embedding), 1), embedding], dim=1) # add position: first
        outputs = {'embedding': embedding}
        if attention_mask is not None: 
            attention_mask = torch.cat(
                [torch.ones(attention_mask.shape[0], 1, device=attention_mask.device), attention_mask],
                dim=1
            )
        outputs['attention_mask'] = attention_mask
        return outputs 


class TransTabForRadiomicsCL(TransTabModelCustom):
    """Radiomics specific TransTab Model. 
        - num_sub_cols 으로 partition 관리 
        - separate_contrast_token 으로 CLS/CONTRASTIVE 분리 가능 
        - multi-task objectives 으로 SUPERVISION/BATCHCORRECTION 가능 
    """ 
    def __init__(self, 
        # feature 
        categorical_columns=None,
        numerical_columns=None, 
        binary_columns=None, 
        feature_extractor=None, 
        # architecture 
        hidden_dim=128, 
        num_layer=2, 
        num_attention_head=8,
        hidden_dropout_prob=0.1, # RadRet에서는 0
        ffn_dim=256, 
        projection_dim=128, # ==hidden_dim 
        # sub column partitioning 
        num_sub_cols: List[int]=[93,62,31,15,7,3,1], # radiomics feature subset views inside the positive.
        # special token setting 
        separate_contrast_token: bool=True, # False이면 [CLS], True이면 [CLS]+[Contrastive]
        # contrastive setting 
        temperature=10,
        base_temperature=10,
        # supervision setting 
        supervision_type: str="categorical", 
        num_class: int=6, # ex) ["Tumor-dominant", "Immune-dominant", "Stroma-dominant", "Necrotic", "Normal epithelial", "Mixed / ambiguous"]
        # batch correction setting 
        batch_correction: bool=True, # True이면 gradient reverse discriminator 적용
        num_batch_labels: int=None,  
        # others 
        activation="relu",
        device="cuda:0",
        **kwargs,
    ) -> None: 
        super().__init__(
            # special token setting 
            separate_contrast_token=separate_contrast_token, 
            # basics 
            categorical_columns=categorical_columns,
            numerical_columns=numerical_columns,
            binary_columns=binary_columns,
            feature_extractor=feature_extractor,
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            device=device,
            **kwargs,
        )

        # sub modules 
        # (i) projection head & contrative setting 
        self.projection_dim = projection_dim
        self.projection_head = TransTabProjectionHead(
            hidden_dim=hidden_dim,
            projection_dim=projection_dim, 
        )
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.base_temperature = base_temperature
        # (ii) classifier for supervision 
        self.num_class = num_class
        if supervision_type == "categorical": 
            self.classifier = TransTabLinearClassifier(
                num_class=num_class, 
                hidden_dim=hidden_dim, 
            )
        else: 
            raise NotImplementedError(f"Not Implemented: supervision type of {supervision_type}")
        # (iii)
        self.grad_reverse_discriminator = AdversarialDiscriminator(
            hidden_dim, 
            n_cls=num_batch_labels, 
            reverse_grad=True, 
        ) if batch_correction else None 

        # partitioning setting 
        self.num_sub_cols = num_sub_cols

        # others (state save)
        self.activation = activation
        self.device = device
        self.to(device)

    def forward(self, 
        x, # a batch of raw tabular samples. (pd.DataFrame)
    ): 
        """generates **random** views, for training
        
        Returns: 
            feat_x_multiview: the embeddings of the input tabular samples. (torch.Tensor)
            logits: the classification logits. (torch.Tensor)
        """
        # Perform positive sampling with multiple radiomics subsets 
        feat_x_list = []
        feat_x_for_classification = None 
        if isinstance(x, pd.DataFrame):
            sub_x_list = self._build_sub_x_list_random(x, self.num_sub_cols)
            for i, sub_x in enumerate(sub_x_list):
                # encode each subset(view) feature sample into embedding
                feat_x = self.input_encoder(sub_x) 
                if self.contrastive_token is not None:
                    feat_x = self.contrastive_token(**feat_x) 
                feat_x = self.cls_token(**feat_x) 
                feat_x = self.encoder(**feat_x)
                # [CLS, CONTRAST, feature...] 
                if i == 0: # full feature column
                    feat_x_for_classification = feat_x # classifier 내부적으로 CLS 임베딩을 예측에 사용. 
                # use contrastive token (idx=1), project the embedding, for contrastive learning 
                feat_x_proj = feat_x[:,1,:] 
                feat_x_proj = self.projection_head(feat_x_proj)  
                feat_x_list.append(feat_x_proj)
        else:
            raise ValueError(f'expect input x to be pd.DataFrame, get {type(x)} instead')

        # prepare for loss calculation 
        logits = self.classifier(feat_x_for_classification)
        feat_x_multiview = torch.stack(feat_x_list, axis=1)  # (bs, num_partition, projection_dim) 

        # multi-task losses 
        loss = _

        return loss 

    def _build_sub_x_list_random(self, 
        x, # DataFrame with `num_sub_cols`` radiomics feature columns 
        num_sub_cols: List[int], 
    ): 
        """returns a list of sub-DataFrames, each containing a random subset of columns"""
        cols = x.columns.tolist()
        total_cols = len(cols)
        if total_cols != num_sub_cols[0]:
            raise ValueError(f"expect {num_sub_cols[0]} columns, get {total_cols} instead")
        sub_x_list = [] 
        for count in num_sub_cols: 
            if count == total_cols: 
                selected_cols = cols 
            else: 
                # select count columns randomly 
                indices = np.random.choice(total_cols, count, replace=False)
                selected_cols = [cols[i] for i in indices]
            sub_x = x.copy()[selected_cols]
            sub_x_list.append(sub_x)
        return sub_x_list 
    
    def _build_selfsupervised_contrastive_loss(
        self,
        feat_x_multiview,  # (batch_size, num_partition, projection_dim)
    ):
        """
        Self-supervised contrastive loss.
        Positive pairs are different views of the same sample.
        """
        device = feat_x_multiview.device
        batch_size = feat_x_multiview.shape[0]
        contrast_count = feat_x_multiview.shape[1]

        if contrast_count < 2:
            raise ValueError("Self-supervised contrastive loss requires at least 2 views per sample.")

        # each sample is only positive with itself across different views
        mask = torch.eye(batch_size, dtype=torch.float32, device=device)

        # (batch_size * num_partition, projection_dim)
        contrast_feature = torch.cat(torch.unbind(feat_x_multiview, dim=1), dim=0)

        # use all views as anchors
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # similarity logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature
        )

        # numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # expand mask to match flattened view structure
        # (batch_size, batch_size) -> (batch_size * anchor_count, batch_size * contrast_count)
        mask = mask.repeat(anchor_count, contrast_count)

        # remove self-contrast: diagonal entries should not be counted
        logits_mask = torch.ones_like(mask)
        logits_mask.scatter_(
            1,
            torch.arange(batch_size * anchor_count, device=device).view(-1, 1),
            0
        )

        mask = mask * logits_mask

        # log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        # mean over positives
        mask_sum = mask.sum(dim=1)
        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / (mask_sum + 1e-12)

        # final loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss 
    
    def forward_withSubX(self, 
        sub_x_list: List[pd.DataFrame], 
    ): 
        """utilizes **fixed(no-shuffle)** view, for inference&analysis"""
        feat_x_list = []
        feat_x_for_cl = None
        for i, sub_x in enumerate(sub_x_list):
            # encode two subset feature samples 
            feat_x = self.input_encoder(sub_x)
            if self.contrastive_token is not None:
                feat_x = self.contrastive_token(**feat_x)
            feat_x = self.cls_token(**feat_x)
            feat_x = self.encoder(**feat_x) # 
            if i == 0:
                feat_x_for_cl = feat_x
            feat_x_proj = feat_x[:,1,:]  # take the contrastive token embedding 
            feat_x_proj = self.projection_head(feat_x_proj) # bs, projection_dim 
            feat_x_list.append(feat_x_proj)
        logits = self.clf(feat_x_for_cl)
        feat_x_multiview = torch.stack(feat_x_list, axis=1) # bs, num_partition, projection_dim 
        return feat_x_multiview, logits

    ######################################
    # use parent's load method 
    # def load(self, ckpt_dir): pass 

    # override parent's save method 
    def save(self, 
        ckpt_dir,
    ): 
        """save the model state_dict and feature_extractor configuration to the `ckpt_dir`."""
        # save model weight state dict 
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict() 
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None: 
            self.input_encoder.feature_extractor.save(ckpt_dir)
        # save model parameters 
        model_params = {
            'categorical_columns': self.input_encoder.feature_extractor.categorical_columns,
            'numerical_columns': self.input_encoder.feature_extractor.numerical_columns, 
            'binary_columns': self.input_encoder.feature_extractor.binary_columns, 
            'num_class': self.num_class, 
            'hidden_dim': self.encoder.hidden_dim, 
            'num_layer': self.encoder.num_layer, 
            'num_attention_head': self.encoder.num_attention_head, 
            'hidden_dropout_prob': self.encoder.hidden_dropout_prob, 
            'ffn_dim': self.encoder.ffn_dim, 
            'projection_dim': self.projection_dim, 
            'num_sub_cols': self.num_sub_cols, 
            'activation': self.activation, 
        }
        with open(os.path.join(ckpt_dir, constants.TRANSTAB_PARAMS_NAME), 'w') as f:
            json.dump(model_params, f, indent=4)
        # save the input encoder separately 
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None 


### About Batch Correction ### 

class GradReverse(Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    return GradReverse.apply(x, lambd)


class AdversarialDiscriminator(nn.Module):
    """
    Discriminator for the adversarial training for batch correction.
    """

    def __init__(
        self,
        d_model: int,
        n_cls: int,
        nlayers: int = 3,
        activation: callable = nn.LeakyReLU,
        reverse_grad: bool = False,
    ):
        super().__init__()
        # module list
        self._decoder = nn.ModuleList()
        for i in range(nlayers - 1):
            self._decoder.append(nn.Linear(d_model, d_model))
            self._decoder.append(activation())
            self._decoder.append(nn.LayerNorm(d_model))
        self.out_layer = nn.Linear(d_model, n_cls)
        self.reverse_grad = reverse_grad

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, embsize]
        """
        if self.reverse_grad:
            x = grad_reverse(x, lambd=1.0)
        for layer in self._decoder:
            x = layer(x)
        return self.out_layer(x)


# The code is modified from https://github.com/wgchang/DSBN/blob/master/model/dsbn.py
class _DomainSpecificBatchNorm(nn.Module):
    _version = 2

    def __init__(
        self,
        num_features: int,
        num_domains: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ):
        super(_DomainSpecificBatchNorm, self).__init__()
        self._cur_domain = None
        self.num_domains = num_domains
        self.bns = nn.ModuleList(
            [
                self.bn_handle(num_features, eps, momentum, affine, track_running_stats)
                for _ in range(num_domains)
            ]
        )

    @property
    def bn_handle(self) -> nn.Module:
        raise NotImplementedError

    @property
    def cur_domain(self) -> Optional[int]:
        return self._cur_domain

    @cur_domain.setter
    def cur_domain(self, domain_label: int):
        self._cur_domain = domain_label

    def reset_running_stats(self):
        for bn in self.bns:
            bn.reset_running_stats()

    def reset_parameters(self):
        for bn in self.bns:
            bn.reset_parameters()

    def _check_input_dim(self, input: torch.Tensor):
        raise NotImplementedError

    def forward(self, x: torch.Tensor, domain_label: int) -> torch.Tensor:
        self._check_input_dim(x)
        if domain_label >= self.num_domains:
            raise ValueError(
                f"Domain label {domain_label} exceeds the number of domains {self.num_domains}"
            )
        bn = self.bns[domain_label]
        self.cur_domain = domain_label
        return bn(x)


class DomainSpecificBatchNorm1d(_DomainSpecificBatchNorm):
    @property
    def bn_handle(self) -> nn.Module:
        return nn.BatchNorm1d

    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() > 3:
            raise ValueError(
                "expected at most 3D input (got {}D input)".format(input.dim())
            )


class DomainSpecificBatchNorm2d(_DomainSpecificBatchNorm):
    @property
    def bn_handle(self) -> nn.Module:
        return nn.BatchNorm2d

    def _check_input_dim(self, input: torch.Tensor):
        if input.dim() != 4:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))

