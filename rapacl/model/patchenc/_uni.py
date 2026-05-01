# import timm
# import torch.nn as nn


# def build_uni(checkpoint_path=None):
#     model = timm.create_model(
#         "vit_large_patch16_224",
#         img_size=224,
#         num_classes=0,
#     )

#     if checkpoint_path is not None:
#         state = torch.load(checkpoint_path, map_location="cpu")
#         model.load_state_dict(state, strict=False)

#     feature_dim = model.num_features

#     return model, feature_dim
