# %% ##################################### 
 # Descripttion: 
 # version: 
 # Author: Yuanjie Gu @ Fudan
 # Date: 2024-06-05
 # LastEditors: Yuanjie Gu
 # LastEditTime: 2024-09-06
# %% #####################################
# --------------------------------------------------------
# SimMIM
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# Modified by Zhenda Xie
# --------------------------------------------------------

from .swin_transformer import build_swin
from .masked_autoencoder import build_masked_autoencoder


def build_model(config, is_pretrain=True):
    if is_pretrain:
        model = build_masked_autoencoder(config)
    else:
        model_type = config.MODEL.TYPE
        if model_type == 'swin':
            model = build_swin(config)
        else:
            raise NotImplementedError(f"Unknown fine-tune model: {model_type}")

    return model
