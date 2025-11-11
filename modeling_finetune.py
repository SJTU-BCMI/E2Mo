# --------------------------------------------------------
# BEiT v2: Masked EEGe Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on BEiT, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-EEGe-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops import rearrange
from modeling_modules import TemporalConv_EYE,MultiWay_Block,DropPath,Mlp,Attention,Block,PatchEmbed,TemporalConv,RelativePositionBias,VisionTransformer,VisionTransformer_EYE
from modeling_pretrain import MultiWayTransformerForMaskedEegEyeModeling,MultiWayTransformerForMaskedEegEyeModeling_drop
from collections import OrderedDict
# from flash_attn import flash_attn_func


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': (0.5, 0.5, 0.5), 'std': (0.5, 0.5, 0.5),
        **kwargs
    }

# 由两个EEMo_classifier组成，一个专门处理EEG数据，一个专门处理眼动数据
class concat_EEMo_classifier(nn.Module):
    def __init__(self,modality="EEG", img_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200, depth=12,
                num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                use_mean_pooling=True, init_scale=0.001,multiffn_start_layer_index=10, **kwargs):
        super().__init__()
        self.EEG_classifier = EEMo_classifier(modality="EEG", img_size=img_size, patch_size=patch_size, in_chans=in_chans, out_chans=out_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values,
                use_abs_pos_emb=use_abs_pos_emb, use_rel_pos_bias=use_rel_pos_bias, use_shared_rel_pos_bias=use_shared_rel_pos_bias,
                use_mean_pooling=use_mean_pooling, init_scale=init_scale,multiffn_start_layer_index=multiffn_start_layer_index)
        self.EYE_classifier = EEMo_classifier(modality="EYE", img_size=img_size, patch_size=patch_size, in_chans=in_chans, out_chans=out_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth,
                num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values,
                use_abs_pos_emb=use_abs_pos_emb, use_rel_pos_bias=use_rel_pos_bias, use_shared_rel_pos_bias=use_shared_rel_pos_bias,
                use_mean_pooling=use_mean_pooling, init_scale=init_scale,multiffn_start_layer_index=multiffn_start_layer_index)

        self.fc_norm = norm_layer(embed_dim*2)
        self.classifier_head = nn.Linear(embed_dim*2, num_classes) if num_classes > 0 else nn.Identity()
        self.load_fromcheckpoint()
    
    def load_fromcheckpoint(self):
        EEG_model_path = "/data/jiangweibang/result_yhl/labram/checkpoint/pretrain_EEMo_EEG_base7/checkpoint-49.pth"
        EYE_model_path = "/data/jiangweibang/result_yhl/labram/checkpoint/pretrain_EEMo_EYE_base11_unfree/checkpoint-49.pth"
        EEG_model = torch.load(EEG_model_path,map_location="cpu")
        EYE_model = torch.load(EYE_model_path,map_location="cpu")
        EEG_model = EEG_model["model"]
        EYE_model = EYE_model["model"]

        all_keys_EEG = list(EEG_model.keys())
        new_dict_EEG = OrderedDict()
            # else:
        for key in all_keys_EEG:
            if key.startswith('student.'):
                new_dict_EEG[key[8:]] = EEG_model[key]
            else:
                pass
        
        all_keys_EYE = list(EYE_model.keys())
        new_dict_EYE = OrderedDict()
            # else:
        for key in all_keys_EYE:
            if key.startswith('student.'):
                new_dict_EYE[key[8:]] = EYE_model[key]
            else:
                pass
        self.EEG_classifier.load_state_dict(new_dict_EEG,strict=False)
        self.EYE_classifier.load_state_dict(new_dict_EYE,strict=False)


    def forward(self,x,input_chans):
        EEG_CLS = self.EEG_classifier.get_cls_token(x[0],input_chans)
        EYE_CLS = self.EYE_classifier.get_cls_token(x[1],input_chans)
        x = torch.cat([EEG_CLS, EYE_CLS], dim=1)
        x = self.fc_norm(x)
        x = self.classifier_head(x)
        return x

class EEMo_classifier(MultiWayTransformerForMaskedEegEyeModeling):
    def __init__(self,modality="EEG", img_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200, depth=12,
                num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                use_mean_pooling=True, init_scale=0.001,multiffn_start_layer_index=10, **kwargs):
        attn_head_dim = None
        init_std = 0.02
        vocab_size = 8192
        super().__init__(img_size, patch_size, in_chans, out_chans, vocab_size, embed_dim, depth,
                num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, 
                attn_head_dim, 
                use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias, init_std,multiffn_start_layer_index)
        self.modality = modality
        self.num_classes = num_classes
        
        #分类头
        if self.modality == "multi" or self.modality == "concat":
            self.fc_norm = norm_layer(embed_dim*2)
            self.classifier_head = nn.Linear(embed_dim*2, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.fc_norm = norm_layer(embed_dim)
            self.classifier_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if isinstance(self.classifier_head, nn.Linear):
            trunc_normal_(self.classifier_head.weight, std=.02)
        if isinstance(self.classifier_head, nn.Linear):
            self.classifier_head.weight.data.mul_(init_scale)
            self.classifier_head.bias.data.mul_(init_scale)

    def forward(self, x, input_chans=None, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False, return_all_patch_tokens=False):
        if bool_masked_pos is None:
            if self.modality == "multi" or self.modality == "concat":
                bool_masked_pos = (torch.zeros((x[0].shape[0], x[0].shape[1] * x[0].shape[2]), dtype=torch.bool).to(x[0].device),
                                      torch.zeros((x[1].shape[0], x[1].shape[1] * x[1].shape[2]), dtype=torch.bool).to(x[1].device))
            else:
                bool_masked_pos =torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        if self.modality == "concat":
            x_EEG = x[0]
            x_EYE = x[1]
            x_EEG = self.forward_features(x_EEG, input_chans=input_chans,modality_type="EEG",bool_masked_pos=bool_masked_pos[0])
            x_EYE = self.forward_features(x_EYE, input_chans=input_chans,modality_type="EYE",bool_masked_pos=bool_masked_pos[1])
            x_EEG = x_EEG[:,0]
            x_EYE = x_EYE[:,0]
            x = torch.cat([x_EEG, x_EYE], dim=1)
            x = self.fc_norm(x)
            x = self.classifier_head(x)
            return x
        else:
            x = self.forward_features(x, input_chans=input_chans,modality_type=self.modality, bool_masked_pos=bool_masked_pos)
            if self.modality == "multi":
                x_EEG = x[0]
                x_EYE = x[1]
                x_EEG = x_EEG[:, 0]
                x_EYE = x_EYE[:, 0]
                x = torch.cat([x_EEG, x_EYE], dim=1)
                x = self.fc_norm(x)
                x = self.classifier_head(x)
            else:
                x = x[:]
                x = x[:,0]
                x = self.fc_norm(x)
                x = self.classifier_head(x)
            return x
        # 获取每一层的输出，输入必须是多模态
    def collect_layer_output(self, x, input_chans=None,modality_type=None, bool_masked_pos=None, return_all_tokens=False):
        layer_output = []
        if modality_type == "multi":
            x_EEG = x[0]
            x_EYE = x[1]
            x_EEG = self.EEG_embedding(x_EEG,input_chans)
            x_EEG = x_EEG + self.token_type_embeddings(
                torch.ones((x_EEG.shape[0],x_EEG.shape[1])).long().to(x_EEG.device)
            )
            x_EYE = self.EYE_embedding(x_EYE)
            x_EYE = x_EYE + self.token_type_embeddings(
                torch.zeros((x_EYE.shape[0],x_EYE.shape[1])).long().to(x_EYE.device)
            )
            x = torch.cat([x_EYE,x_EEG],dim=1)
            cls_output = []
            for blk in self.blocks:
                x = blk(x,modality_type="multi",eye_length=x_EYE.shape[1])
                layer_output.append(x)
            x = self.norm(x)
            x_EYE = x[:, :x_EYE.shape[1]]
            x_EEG = x[:, x_EYE.shape[1]:]
            x_EEG = x_EEG[:, 0]
            x_EYE = x_EYE[:, 0]
            cls_output.append(x_EEG)
            cls_output.append(x_EYE)
            x = torch.cat([x_EEG, x_EYE], dim=1)
            layer_output.append(x)
            x = self.fc_norm(x)
            x = self.classifier_head(x)
            layer_output.append(x)
            cls_output.append(x)

        return cls_output # layer_output

    def get_cls_token(self,x,input_chans):
        bool_masked_pos =torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x = self.forward_features(x, input_chans=input_chans,modality_type=self.modality, bool_masked_pos=bool_masked_pos)
        return x[:,0]


class EEMo_classifier_drop(MultiWayTransformerForMaskedEegEyeModeling_drop):
    def __init__(self,modality="EEG", img_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200, depth=12,
                num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
                use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
                use_mean_pooling=True, init_scale=0.001,multiffn_start_layer_index=10, **kwargs):
        attn_head_dim = None
        init_std = 0.02
        vocab_size = 8192
        super().__init__(img_size, patch_size, in_chans, out_chans, vocab_size, embed_dim, depth,
                num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, 
                attn_head_dim, 
                use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias, init_std,multiffn_start_layer_index)
        self.modality = modality
        self.num_classes = num_classes
        
        #分类头
        if self.modality == "multi" or self.modality == "concat":
            self.fc_norm = norm_layer(embed_dim*2)
            self.classifier_head = nn.Linear(embed_dim*2, num_classes) if num_classes > 0 else nn.Identity()
        else:
            self.fc_norm = norm_layer(embed_dim)
            self.classifier_head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if isinstance(self.classifier_head, nn.Linear):
            trunc_normal_(self.classifier_head.weight, std=.02)
        if isinstance(self.classifier_head, nn.Linear):
            self.classifier_head.weight.data.mul_(init_scale)
            self.classifier_head.bias.data.mul_(init_scale)

    def forward(self, x, input_chans=None, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False, return_all_patch_tokens=False):
        if bool_masked_pos is None:
            if self.modality == "multi" or self.modality == "concat":
                bool_masked_pos = (torch.zeros((x[0].shape[0], x[0].shape[1] * x[0].shape[2]), dtype=torch.bool).to(x[0].device),
                                      torch.zeros((x[1].shape[0], x[1].shape[1] * x[1].shape[2]), dtype=torch.bool).to(x[1].device))
            else:
                bool_masked_pos =torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        if self.modality == "concat":
            x_EEG = x[0]
            x_EYE = x[1]
            x_EEG = self.forward_features(x_EEG, input_chans=input_chans,modality_type="EEG",bool_masked_pos=bool_masked_pos[0])
            x_EYE = self.forward_features(x_EYE, input_chans=input_chans,modality_type="EYE",bool_masked_pos=bool_masked_pos[1])
            x_EEG = x_EEG[:,0]
            x_EYE = x_EYE[:,0]
            x = torch.cat([x_EEG, x_EYE], dim=1)
            x = self.fc_norm(x)
            x = self.classifier_head(x)
            return x
        else:
            x = self.forward_features(x, input_chans=input_chans,modality_type=self.modality, bool_masked_pos=bool_masked_pos)
            if self.modality == "multi":
                x_EEG = x[0]
                x_EYE = x[1]
                x_EEG = x_EEG[:, 0]
                x_EYE = x_EYE[:, 0]
                x = torch.cat([x_EEG, x_EYE], dim=1)
                x = self.fc_norm(x)
                x = self.classifier_head(x)
            else:
                x = x[:]
                x = x[:,0]
                x = self.fc_norm(x)
                x = self.classifier_head(x)
            return x
        # 获取每一层的输出，输入必须是多模态
    def collect_layer_output(self, x, input_chans=None,modality_type=None, bool_masked_pos=None, return_all_tokens=False):
        layer_output = []
        if modality_type == "multi":
            x_EEG = x[0]
            x_EYE = x[1]
            x_EEG = self.EEG_embedding(x_EEG,input_chans)
            x_EEG = x_EEG + self.token_type_embeddings(
                torch.ones((x_EEG.shape[0],x_EEG.shape[1])).long().to(x_EEG.device)
            )
            x_EYE = self.EYE_embedding(x_EYE)
            x_EYE = x_EYE + self.token_type_embeddings(
                torch.zeros((x_EYE.shape[0],x_EYE.shape[1])).long().to(x_EYE.device)
            )
            x = torch.cat([x_EYE,x_EEG],dim=1)
            cls_output = []
            for blk in self.blocks:
                x = blk(x,modality_type="multi",eye_length=x_EYE.shape[1])
                layer_output.append(x)
            x = self.norm(x)
            x_EYE = x[:, :x_EYE.shape[1]]
            x_EEG = x[:, x_EYE.shape[1]:]
            x_EEG = x_EEG[:, 0]
            x_EYE = x_EYE[:, 0]
            cls_output.append(x_EEG)
            cls_output.append(x_EYE)
            x = torch.cat([x_EEG, x_EYE], dim=1)
            layer_output.append(x)
            x = self.fc_norm(x)
            x = self.classifier_head(x)
            layer_output.append(x)
            cls_output.append(x)

        return cls_output # layer_output


# 跨模态分类模型，即仅使用眼动信号做下游任务分类
class EEMo_cross_classifier(MultiWayTransformerForMaskedEegEyeModeling):
    def __init__(self,modality="EYE", img_size=1600, patch_size=200, in_chans=1, out_chans=8, num_classes=1000, embed_dim=200, depth=12,
            num_heads=10, mlp_ratio=4., qkv_bias=False, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
            drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None,
            use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False,
            use_mean_pooling=True, init_scale=0.001,multiffn_start_layer_index=10, **kwargs):
        attn_head_dim = None
        init_std = 0.02
        vocab_size = 8192
        super().__init__(img_size, patch_size, in_chans, out_chans, vocab_size, embed_dim, depth,
                num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, 
                attn_head_dim, 
                use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias, init_std,multiffn_start_layer_index)
        self.modality = modality
        self.num_classes = num_classes
        # 用于补充EEG信号的token
        self.cross_EEG_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.EEG_length = 18 # 补充的EEG的长度
        trunc_normal_(self.cls_token, std=self.init_std)
        
        #分类头
        
        self.fc_norm = norm_layer(embed_dim*2)
        self.classifier_head = nn.Linear(embed_dim*2, num_classes) if num_classes > 0 else nn.Identity()
        
        if isinstance(self.classifier_head, nn.Linear):
            trunc_normal_(self.classifier_head.weight, std=.02)
        if isinstance(self.classifier_head, nn.Linear):
            self.classifier_head.weight.data.mul_(init_scale)
            self.classifier_head.bias.data.mul_(init_scale)

    def forward(self, x, input_chans=None, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False, return_all_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos =torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x_EYE = x
        cls_tokens = self.cls_token.expand(x_EYE.shape[0], -1, -1)
        x_EEG = self.cross_EEG_token.expand(x_EYE.shape[0], self.EEG_length*4, -1)
        x_EEG = torch.cat((cls_tokens,x_EEG),dim=1)
        x_EEG = x_EEG + self.token_type_embeddings(
            torch.ones((x_EEG.shape[0],x_EEG.shape[1])).long().to(x_EEG.device)
        )
        x_EYE = self.EYE_embedding(x_EYE,bool_masked_pos)
        x_EYE = x_EYE + self.token_type_embeddings(
            torch.zeros((x_EYE.shape[0],x_EYE.shape[1])).long().to(x_EYE.device)
        )
        x = torch.cat([x_EYE,x_EEG],dim=1)
        layer_output = [] # 收集模型每一层的输出
        cls_output = []
        for blk in self.blocks:
            x = blk(x,modality_type="multi",eye_length=x_EYE.shape[1])
            layer_output.append(x)
        x = self.norm(x)
        x_EYE = x[:, :x_EYE.shape[1]]
        x_EEG = x[:, x_EYE.shape[1]:]

        x_EEG = x_EEG[:, 0]
        x_EYE = x_EYE[:, 0]
        cls_output.append(x_EEG)
        cls_output.append(x_EYE)
        x = torch.cat([x_EEG, x_EYE], dim=1)
        layer_output.append(x)
        x = self.fc_norm(x)
        x = self.classifier_head(x)
        layer_output.append(x)
        cls_output.append(x)
        return cls_output# layer_output

class AttentionFusionModel(nn.Module):
    def __init__(self, input_dim=200, output_dim=200):
        super(AttentionFusionModel, self).__init__()
        
        # 定义注意力层的参数
        self.query_fc = nn.Linear(input_dim, output_dim)
        self.key_fc = nn.Linear(input_dim, output_dim)
        self.value_fc = nn.Linear(input_dim, output_dim)
        
        # 输出层，保持输出维度为(B, 200)
        self.output_fc = nn.Linear(output_dim, output_dim)

    def forward(self, modal1_input, modal2_input):
        # 通过全连接层转换为query, key, value
        query1 = self.query_fc(modal1_input)  # 第一个模态的query
        key1 = self.key_fc(modal1_input)      # 第一个模态的key
        value1 = self.value_fc(modal1_input)  # 第一个模态的value
        
        query2 = self.query_fc(modal2_input)  # 第二个模态的query
        key2 = self.key_fc(modal2_input)      # 第二个模态的key
        value2 = self.value_fc(modal2_input)  # 第二个模态的value
        
        # 计算自注意力
        attention_scores1 = torch.matmul(query1, key1.transpose(-1, -2)) / query1.size(-1) ** 0.5  # 第一个模态的注意力分数
        attention_scores2 = torch.matmul(query2, key2.transpose(-1, -2)) / query2.size(-1) ** 0.5  # 第二个模态的注意力分数
        
        # 使用softmax归一化得到注意力权重
        attention_weights1 = F.softmax(attention_scores1, dim=-1)
        attention_weights2 = F.softmax(attention_scores2, dim=-1)
        
        # 通过注意力权重加权得到加权的value
        attended_values1 = torch.matmul(attention_weights1, value1)  # 第一个模态的加权值
        attended_values2 = torch.matmul(attention_weights2, value2)  # 第二个模态的加权值
        
        # 将加权后的特征融合（拼接）
        fused_features = attended_values1 + attended_values2  # 融合两个模态
        
        # 通过输出层保持输出维度为 (B, 200)
        output = self.output_fc(fused_features)
        
        return output
    
@register_model
def beit_base_patch200_200(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def EEMo_EEG_base_finetune(pretrained=False, **kwargs):
    model = EEMo_classifier(
        modality="EEG",patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),multiffn_start_layer_index=10, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def EEMo_EEG_huge_finetune(pretrained=False, **kwargs):
    model = EEMo_classifier(
        modality="EEG",patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6),out_chans=32, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),multiffn_start_layer_index=40, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def EEMo_EYE_base_finetune(pretrained=False, **kwargs):
    model = EEMo_classifier(
        modality="EYE",patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),multiffn_start_layer_index=10, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def  EEMo_EYE_base_finetune_drop(pretrained=False, **kwargs):
    model = EEMo_classifier_drop(
        modality="EYE",patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),multiffn_start_layer_index=10, **kwargs)
    model.default_cfg = _cfg()
    return model
   

@register_model
def EEMo_EYE_large_finetune(pretrained=False, **kwargs):
    model = EEMo_classifier(
        modality="EYE",patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6),out_chans=16, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),multiffn_start_layer_index=20, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def EEMo_EYE_huge_finetune(pretrained=False, **kwargs):
    model = EEMo_classifier(
        modality="EYE",patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6),out_chans=32, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),multiffn_start_layer_index=40, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def EEMo_cross_base_finetune(pretrained=False, **kwargs):
    model = EEMo_cross_classifier(
        modality="EYE",patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),multiffn_start_layer_index=10, **kwargs)
    model.default_cfg = _cfg()
    return model
# multi
@register_model
def EEMo_multi_base_finetune(pretrained=False, **kwargs):
    model = EEMo_classifier(
        modality="multi",patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),multiffn_start_layer_index=10, **kwargs)
    model.default_cfg = _cfg()
    return model



def EEMo_sconcat_base_finetune(pretrained=False):
    kwargs = {'num_classes': 1, 'in_chans': 3, 'drop_rate': 0.0, 'drop_path_rate': 0.1, 'attn_drop_rate': 0.0, 'use_mean_pooling': True, 'init_scale': 0.001, 'use_rel_pos_bias': False, 'use_abs_pos_emb': True, 'init_values': 0.1, 'qkv_bias': False}
    model = concat_EEMo_classifier(
        modality="multi",patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),multiffn_start_layer_index=10, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def EEMo_multi_large_finetune(pretrained=False, **kwargs):
    model = EEMo_classifier(
        modality="multi",patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6),out_chans=16, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),multiffn_start_layer_index=20, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def EEMo_multi_huge_finetune(pretrained=False, **kwargs):
    model = EEMo_classifier(
        modality="multi",patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6),out_chans=32, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),multiffn_start_layer_index=40, **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def EEMo_concat_base_finetune(pretrained=False, **kwargs):
    model = EEMo_classifier(
        modality="concat",patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),multiffn_start_layer_index=10, **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def beit_small_patch200_200(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=200, embed_dim=200, depth=6, num_heads=8, mlp_ratio=4, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def beit_large_patch200_200(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, out_chans=16, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def beit_huge_patch200_200(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, out_chans=32, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def beit_1B_patch200_200(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=200, embed_dim=1600, depth=48, num_heads=16, mlp_ratio=4, out_chans=64, qk_norm=partial(nn.LayerNorm, eps=1e-6), # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model




@register_model
def beit_base_patch16_256(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=256, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def beit_base_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, #qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def beit_24x544_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=16, embed_dim=544, depth=24, num_heads=16, mlp_ratio=4, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def beit_large_patch16_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, #qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def beit_large_patch16_384(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, #qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model


@register_model
def beit_large_patch16_512(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=512, patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, #qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def beit_huge_patch14_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=14, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model

@register_model
def beit_giant_patch14_224(pretrained=False, **kwargs):
    model = VisionTransformer(
        img_size=224, patch_size=14, embed_dim=1408, depth=40, num_heads=16, mlp_ratio=6144 / 1408, # qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    model.default_cfg = _cfg()
    return model