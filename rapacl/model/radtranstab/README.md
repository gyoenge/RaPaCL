### Description 

Current version follows [Radiomics Retrieval](https://github.com/nainye/RadiomicsRetrieval) style TransTab adaptation. 

- `build.py`: Factory style modeling builder 

    - `_embed.py`: `TransTabWordEmbedding`, `TransTabNumEmbedding`, `TransTabFeatureExtractor`, `TransTabFeatureProcessor` 
    - `_encoder.py`: `_get_activate_f`, `TransTabTransformerLayer`, `TransTabEncoder` 
    - `_token.py`: `TransTabCLSToken`, `ContrastiveToken` 
    - `_head.py`: `TransTabLinearClassifier`, `TransTabProjectionHead` 
    - `_transtab.py`: `TransTabModel`, `TransTabClassifier`
    - `_radtranstab.py`: `TransTabForRadiomics` 

- `constants.py`

