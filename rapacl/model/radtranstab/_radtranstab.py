import os
import json
from loguru import logger
import torch
import numpy as np
import pandas as pd

from rapacl.model.radtranstab._token import ContrastiveToken
from rapacl.model.radtranstab._head import TransTabProjectionHead, TransTabLinearClassifier
from rapacl.model.radtranstab._transtab import TransTabModel
import rapacl.model.radtranstab.constants as constants 


class TransTabForRadiomics(TransTabModel):
    '''The contrasstive learning and clssification model subclass from :class:`transtab.modeling_transtab.TransTabModel`.

    Parameters
    ----------
    categorical_columns: list
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    feature_extractor: TransTabFeatureExtractor
        a feature extractor to tokenize the input tables. if not passed the model will build itself.

    num_class: int
        number of output classes to be predicted.

    hidden_dim: int
        the dimension of hidden embeddings.

    num_layer: int
        the number of transformer layers used in the encoder.

    num_attention_head: int
        the numebr of heads of multihead self-attention layer in the transformers.

    hidden_dropout_prob: float
        the dropout ratio in the transformer encoder.

    ffn_dim: int
        the dimension of feed-forward layer in the transformer layer.

    projection_dim: int
        the dimension of projection head on the top of encoder.

    num_sub_cols: list
        the number of columns for each subset of the input tabular data.

    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.

    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    A TransTabForRadiomics model.

    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        num_class=2,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0,
        ffn_dim=256,
        projection_dim=128,
        num_sub_cols=[72, 54, 36, 18, 9, 3, 1],
        gpe_drop_rate=0.1,
        activation='relu',
        device='cuda:0',
        **kwargs,
        ) -> None:
        super().__init__(
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
        self.projection_head = TransTabProjectionHead(hidden_dim, projection_dim)
        self.num_sub_cols = num_sub_cols
        self.num_class = num_class
        self.clf = TransTabLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)
        self.contrastive_token = ContrastiveToken(hidden_dim=hidden_dim)
        self.gpe_drop_rate = gpe_drop_rate
        self.projection_dim = projection_dim
        self.activation = activation
        self.device = device
        self.to(device)

    def forward(self, x, gpe=None):
        '''Make forward pass given the input feature ``x`` and the global positional embeddings ``gpe`` (optional).

        Parameters
        ----------
        x: pd.DataFrame
            a batch of raw tabular samples.

        gpe: pd.DataFrame
            a batch of global positional embeddings of the same size as x.

        Returns
        -------
        feat_x_multiview: torch.Tensor
            the embeddings of the input tabular samples.

        logits: torch.Tensor
            the classification logits.

        '''
        # Perform positive sampling with multiple radiomics subsets
        feat_x_list = []
        feat_x_for_cl = None
        if isinstance(x, pd.DataFrame):
            sub_x_list = self._build_sub_x_list_random(x, self.num_sub_cols)
            
            # concatenate with the gpes with a certain drop rate
            if gpe is not None and np.random.rand() > self.gpe_drop_rate:
                for i in range(len(sub_x_list)):
                    sub_x_list[i] = pd.concat([sub_x_list[i], gpe], axis=1)
            
            if gpe is not None:
                sub_x_list.append(gpe)

            for i, sub_x in enumerate(sub_x_list):
                # encode two subset feature samples
                feat_x = self.input_encoder(sub_x)
                feat_x = self.contrastive_token(**feat_x)
                feat_x = self.cls_token(**feat_x)
                feat_x = self.encoder(**feat_x)
                if i == 0:
                    feat_x_for_cl = feat_x
                feat_x_proj = feat_x[:,1,:] # Extract the contrastive token embedding (at index 1)
                feat_x_proj = self.projection_head(feat_x_proj) # bs, projection_dim
                feat_x_list.append(feat_x_proj)
        else:
            raise ValueError(f'expect input x to be pd.DataFrame, get {type(x)} instead')

        logits = self.clf(feat_x_for_cl)

        feat_x_multiview = torch.stack(feat_x_list, axis=1) # bs, num_partition, projection_dim

        return feat_x_multiview, logits
    
    def _build_sub_x_list_random(self, x, num_sub_cols):
        """
        x: DataFrame with 72 radiomics feature columns
        Returns: A list of sub-DataFrames, each containing a random subset of columns 
                with lengths [72, 54, 36, 18, 9, 3, 1] respectively.
        """
        cols = x.columns.tolist()
        total_cols = len(cols)
        
        if total_cols != 72:
            raise ValueError(f'expect 72 columns, get {total_cols} instead')
        
        sub_x_list = []
        for count in num_sub_cols:
            # select count columns randomly
            if count == total_cols:
                selected_cols = cols
            else:
                indices = np.random.choice(total_cols, count, replace=False)
                selected_cols = [cols[i] for i in indices]
            sub_x = x.copy()[selected_cols]
            sub_x_list.append(sub_x)
        
        return sub_x_list

    def forward_withSubX(self, sub_x_list, gpe=None):
        '''Make forward pass given the input feature ``x`` and the global positional embeddings ``gpe`` (optional).

        Parameters
        ----------
        x: pd.DataFrame
            a batch of raw tabular samples.

        gpe: pd.DataFrame
            a batch of global positional embeddings of the same size as x.

        Returns
        -------
        feat_x_multiview: torch.Tensor
            the embeddings of the input tabular samples.

        logits: torch.Tensor
            the classification logits.

        '''
        # do positive sampling
        feat_x_list = []
        feat_x_for_cl = None
        # sub_x_list = self._build_sub_x_list_random(x, self.num_sub_cols)
            
        # concatenate with the gpes with a certain drop rate
        if gpe is not None and np.random.rand() > self.gpe_drop_rate:
            for i in range(len(sub_x_list)):
                sub_x_list[i] = pd.concat([sub_x_list[i], gpe], axis=1)

        if gpe is not None:
            sub_x_list.append(gpe)

        for i, sub_x in enumerate(sub_x_list):
            # encode two subset feature samples
            feat_x = self.input_encoder(sub_x)
            feat_x = self.contrastive_token(**feat_x)
            feat_x = self.cls_token(**feat_x)
            feat_x = self.encoder(**feat_x)
            if i == 0:
                feat_x_for_cl = feat_x
            feat_x_proj = feat_x[:,1,:] # take the contrastive token embedding
            feat_x_proj = self.projection_head(feat_x_proj) # bs, projection_dim
            feat_x_list.append(feat_x_proj)

        logits = self.clf(feat_x_for_cl)

        feat_x_multiview = torch.stack(feat_x_list, axis=1) # bs, num_partition, projection_dim

        return feat_x_multiview, logits
    
    def load(self, ckpt_dir):
        '''Load the model state_dict and feature_extractor configuration
        from the ``ckpt_dir``.

        Parameters
        ----------
        ckpt_dir: str
            the directory path to load.

        Returns
        -------
        None

        '''
        # Load model weights (state_dict)
        model_name = os.path.join(ckpt_dir, constants.WEIGHTS_NAME)
        state_dict = torch.load(model_name, map_location='cpu')
        missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=False)
        logger.info(f'missing keys: {missing_keys}')
        logger.info(f'unexpected keys: {unexpected_keys}')
        logger.info(f'load model from {ckpt_dir}')

        # load feature extractor
        self.input_encoder.feature_extractor.load(os.path.join(ckpt_dir, constants.EXTRACTOR_STATE_DIR))
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

    def save(self, ckpt_dir):
        '''Save the model state_dict and feature_extractor configuration
        to the ``ckpt_dir``.

        Parameters
        ----------
        ckpt_dir: str
            the directory path to save.

        Returns
        -------
        None

        '''
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
            'gpe_drop_rate': self.gpe_drop_rate,
            'activation': self.activation,
        }
        with open(os.path.join(ckpt_dir, constants.TRANSTAB_PARAMS_NAME), 'w') as f:
            json.dump(model_params, f, indent=4)

        # save the input encoder separately
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None
