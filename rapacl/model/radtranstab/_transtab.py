import os
import collections
from loguru import logger
import torch
from torch import nn
import pandas as pd

from rapacl.model.radtranstab._embed import TransTabFeatureExtractor, TransTabFeatureProcessor
from rapacl.model.radtranstab._encoder import TransTabEncoder, TransTabInputEncoder
from rapacl.model.radtranstab._token import TransTabCLSToken
from rapacl.model.radtranstab._head import TransTabLinearClassifier
import rapacl.model.radtranstab.constants as constants


class TransTabModel(nn.Module):
    '''The base transtab model for downstream tasks like contrastive learning, binary classification, etc.
    All models subclass this basemodel and usually rewrite the ``forward`` function. Refer to the source code of
    :class:`transtab.modeling_transtab.TransTabClassifier` or :class:`transtab.modeling_transtab.TransTabForCL` for the implementation details.

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

    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.

    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    A TransTabModel model.

    '''
    def __init__(self,
        categorical_columns=None,
        numerical_columns=None,
        binary_columns=None,
        feature_extractor=None,
        hidden_dim=128,
        num_layer=2,
        num_attention_head=8,
        hidden_dropout_prob=0.1,
        ffn_dim=256,
        activation='relu',
        device='cuda:0',
        **kwargs,
        ) -> None:

        super().__init__()
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

        feature_processor = TransTabFeatureProcessor(
            vocab_size=feature_extractor.vocab_size,
            pad_token_id=feature_extractor.pad_token_id,
            hidden_dim=hidden_dim,
            hidden_dropout_prob=hidden_dropout_prob,
            device=device,
            )
        
        self.input_encoder = TransTabInputEncoder(
            feature_extractor=feature_extractor,
            feature_processor=feature_processor,
            device=device,
            )

        self.encoder = TransTabEncoder(
            hidden_dim=hidden_dim,
            num_layer=num_layer,
            num_attention_head=num_attention_head,
            hidden_dropout_prob=hidden_dropout_prob,
            ffn_dim=ffn_dim,
            activation=activation,
            )

        self.cls_token = TransTabCLSToken(hidden_dim=hidden_dim)
        self.device = device
        self.to(device)

    def forward(self, x, y=None):
        '''Extract the embeddings based on input tables.

        Parameters
        ----------
        x: pd.DataFrame
            a batch of samples stored in pd.DataFrame.

        y: pd.Series
            the corresponding labels for each sample in ``x``. ignored for the basemodel.

        Returns
        -------
        final_cls_embedding: torch.Tensor
            the [CLS] embedding at the end of transformer encoder.

        '''
        embeded = self.input_encoder(x)
        embeded = self.cls_token(**embeded)

        # go through transformers, get final cls embedding
        # Pass through transformer layers to obtain final CLS token embedding
        encoder_output = self.encoder(**embeded)

        # get cls token
        final_cls_embedding = encoder_output[:,0,:]
        return final_cls_embedding

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
        # load model weight state dict
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
        # Save model weights (state_dict)
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir, exist_ok=True)
        state_dict = self.state_dict()
        torch.save(state_dict, os.path.join(ckpt_dir, constants.WEIGHTS_NAME))
        if self.input_encoder.feature_extractor is not None:
            self.input_encoder.feature_extractor.save(ckpt_dir)

        # save the input encoder separately
        state_dict_input_encoder = self.input_encoder.state_dict()
        torch.save(state_dict_input_encoder, os.path.join(ckpt_dir, constants.INPUT_ENCODER_NAME))
        return None

    def update(self, config):
        '''Update the configuration of feature extractor's column map for cat, num, and bin cols.
        Or update the number of classes for the output classifier layer.

        Parameters
        ----------
        config: dict
            a dict of configurations: keys cat:list, num:list, bin:list are to specify the new column names;
            key num_class:int is to specify the number of classes for finetuning on a new dataset.

        Returns
        -------
        None

        '''

        col_map = {}
        for k,v in config.items():
            if k in ['cat','num','bin']: col_map[k] = v

        self.input_encoder.feature_extractor.update(**col_map)
        self.binary_columns = self.input_encoder.feature_extractor.binary_columns
        self.categorical_columns = self.input_encoder.feature_extractor.categorical_columns
        self.numerical_columns = self.input_encoder.feature_extractor.numerical_columns

        if 'num_class' in config:
            num_class = config['num_class']
            self._adapt_to_new_num_class(num_class)

        return None

    def _check_column_overlap(self, cat_cols=None, num_cols=None, bin_cols=None):
        all_cols = []
        if cat_cols is not None: all_cols.extend(cat_cols)
        if num_cols is not None: all_cols.extend(num_cols)
        if bin_cols is not None: all_cols.extend(bin_cols)
        org_length = len(all_cols)
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

    def _adapt_to_new_num_class(self, num_class):
        if num_class != self.num_class:
            self.num_class = num_class
            self.clf = TransTabLinearClassifier(num_class, hidden_dim=self.cls_token.hidden_dim)
            self.clf.to(self.device)
            if self.num_class > 2:
                self.loss_fn = nn.CrossEntropyLoss(reduction='none')
            else:
                self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
            logger.info(f'Build a new classifier with num {num_class} classes outputs, need further finetune to work.')


class TransTabClassifier(TransTabModel):
    '''The classifier model subclass from :class:`transtab.modeling_transtab.TransTabModel`.

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

    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.

    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    Returns
    -------
    A TransTabClassifier model.

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
        self.num_class = num_class
        self.clf = TransTabLinearClassifier(num_class=num_class, hidden_dim=hidden_dim)
        if self.num_class > 2:
            self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(reduction='none')
        self.to(device)

    def forward(self, x, y=None):
        '''Make forward pass given the input feature ``x`` and label ``y`` (optional).

        Parameters
        ----------
        x: pd.DataFrame or dict
            pd.DataFrame: a batch of raw tabular samples; dict: the output of TransTabFeatureExtractor.

        y: pd.Series
            the corresponding labels for each sample in ``x``. if label is given, the model will return
            the classification loss by ``self.loss_fn``.

        Returns
        -------
        logits: torch.Tensor
            the [CLS] embedding at the end of transformer encoder.

        loss: torch.Tensor or None
            the classification loss.

        '''
        if isinstance(x, dict):
            # input is the pre-tokenized encoded inputs
            inputs = x
        elif isinstance(x, pd.DataFrame):
            # Input is a raw DataFrame of tabular features
            inputs = self.input_encoder.feature_extractor(x)
        else:
            raise ValueError(f'TransTabClassifier takes inputs with dict or pd.DataFrame, find {type(x)}.')

        outputs = self.input_encoder.feature_processor(**inputs)
        outputs = self.cls_token(**outputs)

        # go through transformers, get the first cls embedding
        encoder_output = self.encoder(**outputs) # bs, seqlen+1, hidden_dim

        # classifier
        logits = self.clf(encoder_output)

        if y is not None:
            # compute classification loss
            if self.num_class == 2:
                y_ts = torch.tensor(y.values).to(self.device).float()
                loss = self.loss_fn(logits.flatten(), y_ts)
            else:
                y_ts = torch.tensor(y.values).to(self.device).long()
                loss = self.loss_fn(logits, y_ts)
            loss = loss.mean()
        else:
            loss = None

        return logits, loss

