import os
import math
import collections
import json
from typing import Dict
from loguru import logger
from transformers import BertTokenizer, BertTokenizerFast
import torch
from torch import nn
from torch import Tensor
import torch.nn.init as nn_init
import numpy as np
import pandas as pd

from rapacl.engines.constants import CHECKPOINT_PATH
import rapacl.model.radtranstab.constants as constants


class TransTabWordEmbedding(nn.Module):
    r'''
    Encode tokens drawn from column names, categorical and binary features.
    '''
    def __init__(self,
        vocab_size,
        hidden_dim,
        padding_idx=0,
        hidden_dropout_prob=0,
        layer_norm_eps=1e-5,
        ) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_dim, padding_idx)
        nn_init.kaiming_normal_(self.word_embeddings.weight)
        self.norm = nn.LayerNorm(hidden_dim, eps=layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(self, input_ids) -> Tensor:
        embeddings = self.word_embeddings(input_ids)
        embeddings = self.norm(embeddings)
        embeddings =  self.dropout(embeddings)
        return embeddings

class TransTabNumEmbedding(nn.Module):
    r'''
    Encode tokens drawn from column names and the corresponding numerical features.
    '''
    def __init__(self, hidden_dim) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.num_bias = nn.Parameter(Tensor(1, 1, hidden_dim)) # add bias
        nn_init.uniform_(self.num_bias, a=-1/math.sqrt(hidden_dim), b=1/math.sqrt(hidden_dim))

    def forward(self, num_col_emb, x_num_ts, num_mask=None) -> Tensor:
        '''args:
        num_col_emb: numerical column embedding, (# numerical columns, emb_dim)
        x_num_ts: numerical features, (bs, emb_dim)
        num_mask: the mask for NaN numerical features, (bs, # numerical columns)
        '''
        num_col_emb = num_col_emb.unsqueeze(0).expand((x_num_ts.shape[0],-1,-1))
        num_feat_emb = num_col_emb * x_num_ts.unsqueeze(-1).float() + self.num_bias
        return num_feat_emb

class TransTabFeatureExtractor:
    r'''
    Process input dataframe to input indices towards transtab encoder,
    usually used to build dataloader for paralleling loading.
    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        disable_tokenizer_parallel=False,
        ignore_duplicate_cols=False,
        **kwargs,
        ) -> None:
        '''args:
        categorical_columns: a list of categories feature names
        numerical_columns: a list of numerical feature names
        binary_columns: a list of yes or no feature names, accept binary indicators like
            (yes,no); (true,false); (0,1).
        disable_tokenizer_parallel: true if use extractor for collator function in torch.DataLoader
        ignore_duplicate_cols: check if exists one col belongs to both cat/num or cat/bin or num/bin,
            if set `true`, the duplicate cols will be deleted, else throws errors.
        '''
        if os.path.exists('./transtab/tokenizer'):
            self.tokenizer = BertTokenizerFast.from_pretrained('./transtab/tokenizer')
        else:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
            self.tokenizer.save_pretrained(os.path.join(CHECKPOINT_PATH, 'tokenizer'))
        self.tokenizer.__dict__['model_max_length'] = 512
        if disable_tokenizer_parallel: # disable tokenizer parallel
            os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.vocab_size = self.tokenizer.vocab_size
        self.pad_token_id = self.tokenizer.pad_token_id

        self.categorical_columns = categorical_columns
        self.numerical_columns = numerical_columns
        self.binary_columns = binary_columns
        self.ignore_duplicate_cols = ignore_duplicate_cols

        if categorical_columns is not None:
            self.categorical_columns = list(set(categorical_columns))
        if numerical_columns is not None:
            self.numerical_columns = list(set(numerical_columns))
        if binary_columns is not None:
            self.binary_columns = list(set(binary_columns))

        # check if column exists overlap
        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns, self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(f'Find duplicate cols named `{col}`, please process the raw data or set `ignore_duplicate_cols` to True!')
            assert col_no_overlap, 'The assigned categorical_columns, numerical_columns, binary_columns should not have overlap! Please check your input.'
        else:
            self._solve_duplicate_cols(duplicate_cols)

    def __call__(self, x, shuffle=False) -> Dict:
        '''
        Parameters
        ----------
        x: pd.DataFrame 
            with column names and features.

        shuffle: bool
            if shuffle column order during the training.

        Returns
        -------
        encoded_inputs: a dict with {
                'x_num': tensor contains numerical features,
                'num_col_input_ids': tensor contains numerical column tokenized ids,
                'x_cat_input_ids': tensor contains categorical column + feature ids,
                'x_bin_input_ids': tesnor contains binary column + feature ids,
            }
        '''
        encoded_inputs = {
            'x_num':None,
            'num_col_input_ids':None,
            'x_cat_input_ids':None,
            'x_bin_input_ids':None,
        }
        col_names = x.columns.tolist()
        cat_cols = [c for c in col_names if c in self.categorical_columns] if self.categorical_columns is not None else []
        num_cols = [c for c in col_names if c in self.numerical_columns] if self.numerical_columns is not None else []
        bin_cols = [c for c in col_names if c in self.binary_columns] if self.binary_columns is not None else []

        if len(cat_cols+num_cols+bin_cols) == 0:
            # take all columns as categorical columns!
            cat_cols = col_names

        if shuffle:
            np.random.shuffle(cat_cols)
            np.random.shuffle(num_cols)
            np.random.shuffle(bin_cols)

        # TODO:
        # mask out NaN values like done in binary columns
        if len(num_cols) > 0:
            x_num = x[num_cols]
            x_num = x_num.fillna(0) # fill Nan with zero
            x_num_ts = torch.tensor(x_num.values, dtype=float)
            num_col_ts = self.tokenizer(num_cols, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            encoded_inputs['x_num'] = x_num_ts
            encoded_inputs['num_col_input_ids'] = num_col_ts['input_ids']
            encoded_inputs['num_att_mask'] = num_col_ts['attention_mask'] # mask out attention

        if len(cat_cols) > 0:
            x_cat = x[cat_cols].astype(str)
            x_mask = (~pd.isna(x_cat)).astype(int)
            x_cat = x_cat.fillna('')
            x_cat = x_cat.apply(lambda x: x.name + ' '+ x) * x_mask # mask out nan features
            x_cat_str = x_cat.agg(' '.join, axis=1).values.tolist()
            x_cat_ts = self.tokenizer(x_cat_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')

            encoded_inputs['x_cat_input_ids'] = x_cat_ts['input_ids']
            encoded_inputs['cat_att_mask'] = x_cat_ts['attention_mask']

        if len(bin_cols) > 0:
            x_bin = x[bin_cols] # x_bin should already be integral (binary values in 0 & 1)
            x_bin_str = x_bin.apply(lambda x: x.name + ' ') * x_bin
            x_bin_str = x_bin_str.agg(' '.join, axis=1).values.tolist()
            x_bin_ts = self.tokenizer(x_bin_str, padding=True, truncation=True, add_special_tokens=False, return_tensors='pt')
            if x_bin_ts['input_ids'].shape[1] > 0: # not all false
                encoded_inputs['x_bin_input_ids'] = x_bin_ts['input_ids']
                encoded_inputs['bin_att_mask'] = x_bin_ts['attention_mask']

        return encoded_inputs

    def save(self, path):
        '''save the feature extractor configuration to local dir.
        '''
        # save_path = os.path.join(path, constants.EXTRACTOR_STATE_DIR)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        # save tokenizer
        # tokenizer_path = os.path.join(save_path, constants.TOKENIZER_DIR)
        tokenizer_path = os.path.join(CHECKPOINT_PATH, constants.TOKENIZER_DIR)
        self.tokenizer.save_pretrained(tokenizer_path)

        # save other configurations
        # coltype_path = os.path.join(save_path, constants.EXTRACTOR_STATE_NAME)
        coltype_path = os.path.join(CHECKPOINT_PATH, constants.EXTRACTOR_STATE_NAME)
        col_type_dict = {
            'categorical': self.categorical_columns,
            'binary': self.binary_columns,
            'numerical': self.numerical_columns,
        }
        with open(coltype_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(col_type_dict))

    def load(self, path):
        '''load the feature extractor configuration from local dir.
        '''
        tokenizer_path = os.path.join(path, constants.TOKENIZER_DIR)
        coltype_path = os.path.join(path, constants.EXTRACTOR_STATE_NAME)

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
        with open(coltype_path, 'r', encoding='utf-8') as f:
            col_type_dict = json.loads(f.read())

        self.categorical_columns = col_type_dict['categorical']
        self.numerical_columns = col_type_dict['numerical']
        self.binary_columns = col_type_dict['binary']
        logger.info(f'load feature extractor from {coltype_path}')

    def update(self, cat=None, num=None, bin=None):
        '''update cat/num/bin column maps.
        '''
        if cat is not None:
            self.categorical_columns.extend(cat)
            self.categorical_columns = list(set(self.categorical_columns))

        if num is not None:
            self.numerical_columns.extend(num)
            self.numerical_columns = list(set(self.numerical_columns))

        if bin is not None:
            self.binary_columns.extend(bin)
            self.binary_columns = list(set(self.binary_columns))

        col_no_overlap, duplicate_cols = self._check_column_overlap(self.categorical_columns, self.numerical_columns, self.binary_columns)
        if not self.ignore_duplicate_cols:
            for col in duplicate_cols:
                logger.error(f'Find duplicate cols named `{col}`, please process the raw data or set `ignore_duplicate_cols` to True!')
            assert col_no_overlap, 'The assigned categorical_columns, numerical_columns, binary_columns should not have overlap! Please check your input.'
        else:
            self._solve_duplicate_cols(duplicate_cols)

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
        if org_length == 0:
            logger.warning('No cat/num/bin cols specified, will take ALL columns as categorical! Ignore this warning if you specify the `checkpoint` to load the model.')
            return True, []
        unq_length = len(list(set(all_cols)))
        duplicate_cols = [item for item, count in collections.Counter(all_cols).items() if count > 1]
        return org_length == unq_length, duplicate_cols

    def _solve_duplicate_cols(self, duplicate_cols):
        for col in duplicate_cols:
            logger.warning('Find duplicate cols named `{col}`, will ignore it during training!')
            if col in self.categorical_columns:
                self.categorical_columns.remove(col)
                self.categorical_columns.append(f'[cat]{col}')
            if col in self.numerical_columns:
                self.numerical_columns.remove(col)
                self.numerical_columns.append(f'[num]{col}')
            if col in self.binary_columns:
                self.binary_columns.remove(col)
                self.binary_columns.append(f'[bin]{col}')

class TransTabFeatureProcessor(nn.Module):
    r'''
    Process inputs from feature extractor to map them to embeddings.
    '''
    def __init__(self,
        vocab_size=None,
        hidden_dim=128,
        hidden_dropout_prob=0,
        pad_token_id=0,
        device='cuda:0',
        ) -> None:
        '''args:
        categorical_columns: a list of categories feature names
        numerical_columns: a list of numerical feature names
        binary_columns: a list of yes or no feature names, accept binary indicators like
            (yes,no); (true,false); (0,1).
        '''
        super().__init__()
        self.word_embedding = TransTabWordEmbedding(
            vocab_size=vocab_size,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            padding_idx=pad_token_id
            )
        self.num_embedding = TransTabNumEmbedding(hidden_dim)
        self.align_layer = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.device = device

    def _avg_embedding_by_mask(self, embs, att_mask=None):
        if att_mask is None:
            return embs.mean(1)
        else:
            embs[att_mask==0] = 0
            embs = embs.sum(1) / att_mask.sum(1,keepdim=True).to(embs.device)
            return embs

    def forward(self,
        x_num=None,
        num_col_input_ids=None,
        num_att_mask=None,
        x_cat_input_ids=None,
        cat_att_mask=None,
        x_bin_input_ids=None,
        bin_att_mask=None,
        **kwargs,
        ) -> Tensor:
        '''args:
        x: pd.DataFrame with column names and features.
        shuffle: if shuffle column order during the training.
        num_mask: indicate the NaN place of numerical features, 0: NaN 1: normal.
        '''
        num_feat_embedding = None
        cat_feat_embedding = None
        bin_feat_embedding = None

        if x_num is not None and num_col_input_ids is not None:
            num_col_emb = self.word_embedding(num_col_input_ids.to(self.device)) # number of cat col, num of tokens, embdding size
            x_num = x_num.to(self.device)
            num_col_emb = self._avg_embedding_by_mask(num_col_emb, num_att_mask)
            num_feat_embedding = self.num_embedding(num_col_emb, x_num)
            num_feat_embedding = self.align_layer(num_feat_embedding)

        if x_cat_input_ids is not None:
            cat_feat_embedding = self.word_embedding(x_cat_input_ids.to(self.device))
            cat_feat_embedding = self.align_layer(cat_feat_embedding)

        if x_bin_input_ids is not None:
            if x_bin_input_ids.shape[1] == 0: # all false, pad zero
                x_bin_input_ids = torch.zeros(x_bin_input_ids.shape[0],dtype=int)[:,None]
            bin_feat_embedding = self.word_embedding(x_bin_input_ids.to(self.device))
            bin_feat_embedding = self.align_layer(bin_feat_embedding)

        # Concatenate all available feature embeddings (numerical, categorical, binary)
        emb_list = []
        att_mask_list = []
        if num_feat_embedding is not None:
            emb_list += [num_feat_embedding]
            att_mask_list += [torch.ones(num_feat_embedding.shape[0], num_feat_embedding.shape[1])]
        if cat_feat_embedding is not None:
            emb_list += [cat_feat_embedding]
            att_mask_list += [cat_att_mask]
        if bin_feat_embedding is not None:
            emb_list += [bin_feat_embedding]
            att_mask_list += [bin_att_mask]
        if len(emb_list) == 0: raise Exception('no feature found belonging into numerical, categorical, or binary, check your data!')
        all_feat_embedding = torch.cat(emb_list, 1).float()
        attention_mask = torch.cat(att_mask_list, 1).to(all_feat_embedding.device)
        return {'embedding': all_feat_embedding, 'attention_mask': attention_mask}
