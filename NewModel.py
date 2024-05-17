import torch
from transformers import GPT2Tokenizer, GPT2Model, GPT2Config
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import torchvision
# from compGCN import CompGraphConv
import torch.nn as nn
import torch
# import clip
import collections
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import open_clip
from open_clip import *
from typing import Any, Dict, Optional, Tuple, Union
import torch
import copy
import copy
import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from functools import partial
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification
import torch
from torch import nn
from transformers import ViTModel, ViTConfig, ViTFeatureExtractor

''' 
Load state_dict in pre_model to model
Solve the problem that model and pre_model have some different keys
'''
device = "cuda:0" if torch.cuda.is_available() else "cpu" # If using GPU then use mixed precision training.

"""
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_model_name)
        self.tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        self.tokenizer.pad_token = tokenizer.eos_token
        self.transformer.resize_token_embeddings(len(tokenizer))
        # fix model padding token id
        self.transformer.config.pad_token_id = self.transformer.config.eos_token_id
"""      

class Pair_CLIP_SI(nn.Module):
    output_dict: torch.jit.Final[bool]

    def __init__(
            self,
            init_logit_scale: float = np.log(1 / 0.07),
            init_logit_bias: Optional[float] = None,
            output_dict: bool = False,
    ):
        super().__init__()
        self.output_dict = output_dict

        # self.visual = open_clip.model._build_vision_tower(embed_dim, vision_cfg, quick_gelu, cast_dtype)
        # self.state_dict = torch.load('checkpoints/RemoteCLIP-ViT-B-32.pt')
        # self.state_dict = torch.load('vit/vit-base-patch16-384')
        # 配置文件
        self.config = ViTConfig.from_pretrained('google/vit-base-patch32-384')#,local_files_only=True) #/vit-base-patch32-384
# 加载模型
        self.visual = ViTModel.from_pretrained('google/vit-base-patch32-384')#,local_files_only=True)
 
        # self.transformer = GPT2Model(config)
        self.gpt_model_name = "gpt2"  # 预训练模型的名称
        self.transformer = GPT2Model.from_pretrained(self.gpt_model_name)#,local_files_only=True)
        self.gpt_model_name = "gpt2"
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.gpt_model_name)#,local_files_only=True)
        self.tokenizer.padding_side = "left"
        # Define PAD Token = EOS Token = 50256
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.transformer.resize_token_embeddings(len(self.tokenizer))
        # fix model padding token id
        self.transformer.config.pad_token_id = self.transformer.config.eos_token_id

        self.logit_scale = nn.Parameter(torch.ones([]) * init_logit_scale)


    def encode_image(self, image, normalize: bool = False):
        outputs = self.visual(image)
        features= outputs.last_hidden_state[:,0]
        # print('visual_embedding: ',features.shape)
        return F.normalize(features, dim=-1) if normalize else features

    def encode_text(self,input_ids, normalize: bool = False):
        transformer_outputs = self.transformer(**input_ids)
        # print('transformer_outputs[0]: ',transformer_outputs[0].shape)
        text_embedding = transformer_outputs[0].squeeze(1)[:,-1,:]
        # print('text_embedding: ',text_embedding.shape)
        # if input_ids is not None:
        #     batch_size, _ = input_ids.shape[:2]
        # text_embedding = transformer_outputs[0][range(batch_size),-1]

        return F.normalize(text_embedding, dim=-1) if normalize else text_embedding

    def get_logits(self, image, text):
        image_features = self.encode_image(image, normalize=True)
        text_features = self.encode_text(text, normalize=True)
        image_logits = self.logit_scale.exp() * image_features @ text_features.T
        text_logits = image_logits.T
        return image_logits, text_logits

    def forward(
            self,
            image: Optional[torch.Tensor] = None,
            text: Optional[torch.Tensor] = None,
    ):
        image_features = self.encode_image(image, normalize=True) if image is not None else None
        text_features = self.encode_text(text, normalize=True) if text is not None else None

        if self.output_dict:
            out_dict = {
                "image_features": image_features,
                "text_features": text_features,
                "logit_scale": self.logit_scale.exp()
            }
            if self.logit_bias is not None:
                out_dict['logit_bias'] = self.logit_bias
            return out_dict

        if self.logit_bias is not None:
            return image_features, text_features, self.logit_scale.exp(), self.logit_bias
        return image_features, text_features, self.logit_scale.exp()

    def text2text(self, input_ids):
        # 这里编写你自己的函数逻辑
        # text_features = self.encode_text(text, normalize=True)
        transformer_outputs = self.transformer(**input_ids)
        # print('transformer_outputs[0]: ',transformer_outputs[0].shape)
        text_features = transformer_outputs[0].squeeze(1)[:,-1,:]
        text_features = F.normalize(text_features, dim=-1)
        text_logits = self.logit_scale.exp() * text_features @ text_features.T
        del text_features
        # if self.logit_bias is not None:
        #     text_logits += self.logit_bias
        return text_logits