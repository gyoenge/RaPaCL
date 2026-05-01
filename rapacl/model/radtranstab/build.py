#################################################################################
# TransTabModel for Radiomics Specific Task 
#################################################################################
"""
Source: 
    https://github.com/nainye/RadiomicsRetrieval

Adapted version of the TransTab model for tabular data processing and embedding.

Original implementation:
    https://github.com/RyanWangZf/transtab

This version includes modifications for radiomics-based retrieval tasks,
including integration with anatomical positional embeddings (APE).
"""

import os

from rapacl.model.radtranstab._embed import TransTabFeatureExtractor
from rapacl.model.radtranstab._transtab import TransTabClassifier
from rapacl.model.radtranstab._radtranstab import TransTabForRadiomics
import rapacl.model.radtranstab.constants as constants 


def build_extractor(
    categorical_columns=constants.DEFAULT_CATEGORICAL_COLUMNS,
    numerical_columns=constants.DEFAULT_NUMERICAL_COLUMNS,
    binary_columns=constants.DEFAULT_BINARY_COLUMNS,
    ignore_duplicate_cols=constants.IGNORE_DUPLICATE_COLS,
    disable_tokenizer_parallel=constants.DISABLE_TOKENIZER_PARALLEL,
    checkpoint=None,
    **kwargs,) -> TransTabFeatureExtractor:
    '''Build a feature extractor for TransTab model.

    Parameters
    ----------
    categorical_columns: list 
        a list of categorical feature names.

    numerical_columns: list
        a list of numerical feature names.

    binary_columns: list
        a list of binary feature names, accept binary indicators like (yes,no); (true,false); (0,1).

    ignore_duplicate_cols: bool
        if there is one column assigned to more than one type, e.g., the feature age is both nominated
        as categorical and binary columns, the model will raise errors. set True to avoid this error as 
        the model will ignore this duplicate feature.

    disable_tokenizer_parallel: bool
        if the returned feature extractor is leveraged by the collate function for a dataloader,
        try to set this False in case the dataloader raises errors because the dataloader builds 
        multiple workers and the tokenizer builds multiple workers at the same time.

    checkpoint: str
        the directory of the predefined TransTabFeatureExtractor.

    Returns
    -------
    A TransTabFeatureExtractor module.

    '''
    feature_extractor = TransTabFeatureExtractor(
        categorical_columns=categorical_columns,
        numerical_columns=numerical_columns,
        binary_columns=binary_columns,
        disable_tokenizer_parallel=disable_tokenizer_parallel,
        ignore_duplicate_cols=ignore_duplicate_cols,
    )
    if checkpoint is not None:
        extractor_path = os.path.join(checkpoint, constants.EXTRACTOR_STATE_DIR)
        if os.path.exists(extractor_path):
            feature_extractor.load(extractor_path)
        else:
            feature_extractor.load(checkpoint)
    return feature_extractor

def build_classifier(
    categorical_columns=constants.DEFAULT_CATEGORICAL_COLUMNS,
    numerical_columns=constants.DEFAULT_NUMERICAL_COLUMNS,
    binary_columns=constants.DEFAULT_BINARY_COLUMNS,
    feature_extractor=constants.CLASSIFIER_FEATURE_EXTRACTOR,
    num_class=constants.CLASSIFIER_NUM_CLASS,
    hidden_dim=constants.CLASSIFIER_HIDDEN_DIM,
    num_layer=constants.CLASSIFIER_NUM_LAYER,
    num_attention_head=constants.CLASSIFIER_NUM_ATTENTION_HEAD,
    hidden_dropout_prob=constants.CLASSIFIER_HIDDEN_DROPOUT_PROB,
    ffn_dim=constants.CLASSIFIER_FFN_DIM,
    activation=constants.CLASSIFIER_ACTIVATION,
    device=constants.CLASSIFIER_DEVICE,
    checkpoint=None,
    **kwargs) -> TransTabClassifier:
    '''Build a :class:`transtab.modeling_transtab.TransTabClassifier`.

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
    
    checkpoint: str
        the directory to load the pretrained TransTab model.

    Returns
    -------
    A TransTabClassifier model.

    '''
    model = TransTabClassifier(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor = feature_extractor,
        num_class=num_class,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        activation=activation,
        device=device,
        **kwargs,
        )
    
    if checkpoint is not None:
        model.load(checkpoint)

    return model

def build_radiomics_learner(
    categorical_columns=constants.DEFAULT_CATEGORICAL_COLUMNS,
    numerical_columns=constants.DEFAULT_NUMERICAL_COLUMNS,
    binary_columns=constants.DEFAULT_BINARY_COLUMNS,
    feature_extractor=constants.RADTRANSTAB_FEATURE_EXTRACTOR,
    num_class=constants.RADTRANSTAB_NUM_CLASS,
    hidden_dim=constants.RADTRANSTAB_HIDDEN_DIM,
    num_layer=constants.RADTRANSTAB_NUM_LAYER,
    num_attention_head=constants.RADTRANSTAB_NUM_ATTENTION_HEAD,
    hidden_dropout_prob=constants.RADTRANSTAB_HIDDEN_DROPOUT_PROB,
    ffn_dim=constants.RADTRANSTAB_FFN_DIM,
    projection_dim=constants.RADTRANSTAB_PROJECTION_DIM,
    num_sub_cols=constants.RADTRANSTAB_NUM_SUB_COLS,
    gpe_drop_rate=constants.RADTRANSTAB_GPE_DROP_RATE,
    activation=constants.RADTRANSTAB_ACTIVATION,
    device=constants.RADTRANSTAB_DEVICE,
    checkpoint=None,
    ignore_duplicate_cols=True,
    **kwargs,
    ): 
    '''Build a contrastive learning and classification model for radiomics feature extraction.

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
    
    overlap_ratio: float
        the overlap ratio of columns of different partitions when doing subsetting.
    
    num_partition: int
        the number of partitions made for vertical-partition contrastive learning.

    supervised: bool
        whether or not to take supervised VPCL, otherwise take self-supervised VPCL.
    
    temperature: float
        temperature used to compute logits for contrastive learning.

    base_temperature: float
        base temperature used to normalize the temperature.
    
    activation: str
        the name of used activation functions, support ``"relu"``, ``"gelu"``, ``"selu"``, ``"leakyrelu"``.
    
    device: str
        the device, ``"cpu"`` or ``"cuda:0"``.

    checkpoint: str
        the directory of the pretrained transtab model.
    
    ignore_duplicate_cols: bool
        if there is one column assigned to more than one type, e.g., the feature age is both nominated
        as categorical and binary columns, the model will raise errors. set True to avoid this error as 
        the model will ignore this duplicate feature.
    
    Returns
    -------
    A TransTabForRadiomics model.

    '''

    model = TransTabForRadiomics(
        categorical_columns = categorical_columns,
        numerical_columns = numerical_columns,
        binary_columns = binary_columns,
        feature_extractor=feature_extractor,
        num_class=num_class,
        hidden_dim=hidden_dim,
        num_layer=num_layer,
        num_attention_head=num_attention_head,
        hidden_dropout_prob=hidden_dropout_prob,
        ffn_dim=ffn_dim,
        projection_dim=projection_dim,
        num_sub_cols=num_sub_cols,
        gpe_drop_rate=gpe_drop_rate,
        activation=activation,
        device=device,
    )
    if checkpoint is not None:
        model.load(checkpoint)

    return model



