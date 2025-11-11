# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on BEiT, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import math
import torch
import torch.nn as nn
from functools import partial

from modeling_modules import Block, _cfg, PatchEmbed, RelativePositionBias,MultiWay_Block,TemporalConv_EYE,MultiWay_Block_checkpoint,TemporalConv,TemporalConv_freq,TemporalConv_EYE_fp,TemporalConv_freq_EYE
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from einops import rearrange
import numpy as np
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


    def __init__(self, img_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, multiffn_start_layer_index = 10):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.depth = depth # 模型深度
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed = TemporalConv(out_chans=out_chans)
        self.patch_embed_EYE = TemporalConv_EYE_drop(out_chans=out_chans)
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_EYE = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token_EYE = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim))
            self.pos_embed_EYE = nn.Parameter(torch.zeros(1, 33 + 1, embed_dim))
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=(62, img_size // patch_size), num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.multiffn_start_layer_index = multiffn_start_layer_index
        if embed_dim == -800:#huge
            self.blocks = nn.ModuleList([
            MultiWay_Block_checkpoint(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                layer_scale_init_values=init_values, 
                act_layer=nn.GELU, with_vlffn=(i >= self.multiffn_start_layer_index),max_EYE_len = 33,)
            # Block( 
            #     dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
            #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
            #     init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
            #     attn_head_dim=attn_head_dim,
            # )
            for i in range(depth)])
        else:


            self.blocks = nn.ModuleList([
                MultiWay_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                    layer_scale_init_values=init_values, 
                    act_layer=nn.GELU, with_vlffn=(i >= self.multiffn_start_layer_index),max_EYE_len = 33,)
                # Block( 
                #     dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                #     drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                #     init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                #     attn_head_dim=attn_head_dim,
                # )
                for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
            trunc_normal_(self.pos_embed_EYE, std=self.init_std)
        trunc_normal_(self.time_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.cls_token_EYE, std=self.init_std)
        trunc_normal_(self.mask_token_EYE, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)

        self.token_type_embeddings = nn.Embedding(2, embed_dim)
        trunc_normal_(self.token_type_embeddings.weight, std=self.init_std)
        
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp_EEG.fc2.weight.data, layer_id + 1)
            rescale(layer.mlp_EYE.fc2.weight.data, layer_id + 1)
            if layer_id >= self.multiffn_start_layer_index:
                rescale(layer.mlp_multi.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, input_chans,modality_type, bool_masked_pos,):
        if modality_type == "EEG":
            x = self.EEG_embedding(x,input_chans,bool_masked_pos)
            x = x + self.token_type_embeddings(
                torch.ones((x.shape[0],x.shape[1])).long().to(x.device)
            )
            for blk in self.blocks:
                x = blk(x,modality_type="EEG")
            return self.norm(x)
        
        elif modality_type == "EYE":
            x = self.EYE_embedding(x,bool_masked_pos)
            x = x + self.token_type_embeddings(
                torch.zeros((x.shape[0],x.shape[1])).long().to(x.device)
            )
            for blk in self.blocks:
                x = blk(x,modality_type="EYE")
            return self.norm(x)
        elif modality_type == "multi":
            x_EEG = x[0]
            x_EYE = x[1]
            x_EEG = self.EEG_embedding(x_EEG,input_chans,bool_masked_pos[0])
            x_EEG = x_EEG + self.token_type_embeddings(
                torch.ones((x_EEG.shape[0],x_EEG.shape[1])).long().to(x_EEG.device)
            )
            x_EYE = self.EYE_embedding(x_EYE,bool_masked_pos[1])
            x_EYE = x_EYE + self.token_type_embeddings(
                torch.zeros((x_EYE.shape[0],x_EYE.shape[1])).long().to(x_EYE.device)
            )
            x = torch.cat([x_EYE,x_EEG],dim=1)
            for blk in self.blocks:
                x = blk(x,modality_type="multi",eye_length=x_EYE.shape[1])
            x = self.norm(x)
            x_EYE = x[:, :x_EYE.shape[1]]
            x_EEG = x[:, x_EYE.shape[1]:]
            return (x_EEG,x_EYE)
        
    def forward(self, x, input_chans=None,modality_type="None", bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False, return_all_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x = self.forward_features(x, input_chans=input_chans,modality_type=modality_type, bool_masked_pos=bool_masked_pos)
        if modality_type == "multi":
            return x
        else:
            if return_all_patch_tokens:
                return x
            x = x[:, 1:]
            if return_patch_tokens:
                return x
            if return_all_tokens:
                return self.lm_head(x)
            else:
                # return the masked tokens
                return self.lm_head(x[bool_masked_pos])
    
    
    
    def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                # with torch.cuda.amp.autocast(enabled=False):
                x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)

        if split_out_as_qkv:
            x = self.norm(x)
            x = self.lm_head(x) # [b, n+1, 3*c]
            q, k, v = x.chunk(3, dim=-1) # [b, n+1, c]
            b, n, c =q.shape
            q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            k = k.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            return x, q, k, v
        else:
            x = self.norm(x)
            x = x[:, 1:]
            x = self.lm_head(x[bool_masked_pos])

            q, k, v = qkv[0], qkv[1], qkv[2]

        return x, q, k, v


    def forward_intermediate(self, x, bool_masked_pos=None, layer_id=12):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(x.device)
        batch_size, _, time_window, _ = x.size()
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + self.pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if l in layer_id:
                    output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")

    def get_last_selfattention(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                # return attention of the last block
                return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)

    def EEG_embedding(self,x,input_chans=None,bool_masked_pos=None):
        batch_size, c, time_window, _ = x.size()
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        if bool_masked_pos is not None:
            # replace the masked visual tokens by mask_token
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2) # pos_embed[:,1:,:] for visual tokens
            pos_embed = torch.cat((pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1) # pos_embed[:,0:1,:] for cls token
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed[:, 0:time_window, :].unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        return x

    def EYE_embedding(self,x,bool_masked_pos=None):
        batch_size, c, time_window, _ = x.size()
        x = self.patch_embed_EYE(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token_EYE.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token_EYE.expand(batch_size, seq_len, -1)

        if bool_masked_pos is not None:
            # replace the masked visual tokens by mask_token
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed_used = self.pos_embed_EYE[:,:c+1]
        if self.pos_embed_EYE is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2) # pos_embed[:,1:,:] for visual tokens
            pos_embed = torch.cat((pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1) # pos_embed[:,0:1,:] for cls token
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed[:, 0:time_window, :].unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        return x
class MultiWayTransformerForMaskedEegEyeModeling(nn.Module):
    def __init__(self, img_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02, multiffn_start_layer_index = 10):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.depth = depth # 模型深度
        self.patch_embed = TemporalConv(out_chans=out_chans)
        self.patch_embed_freq = TemporalConv_freq(out_chans=out_chans)
        # self.patch_embed = TemporalConv_test(out_chans=out_chans)
        self.patch_embed_EYE_pd = TemporalConv_EYE(out_chans=out_chans)
        self.patch_embed_EYE_fp = TemporalConv_EYE_fp(out_chans=out_chans)
        self.patch_embed_freq_EYE = TemporalConv_freq_EYE(out_chans=out_chans)


        self.num_heads = num_heads
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.cls_token_EYE = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token_EYE = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim))
            self.pos_embed_EYE = nn.Parameter(torch.zeros(1, 33 + 1, embed_dim))
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=(62, img_size // patch_size), num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.multiffn_start_layer_index = multiffn_start_layer_index

        self.blocks = nn.ModuleList([
            MultiWay_Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                layer_scale_init_values=init_values, 
                act_layer=nn.GELU, with_vlffn=(i >= self.multiffn_start_layer_index),max_EYE_len = 33,)
                for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
            trunc_normal_(self.pos_embed_EYE, std=self.init_std)
        trunc_normal_(self.time_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.cls_token_EYE, std=self.init_std)
        trunc_normal_(self.mask_token_EYE, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)

        self.token_type_embeddings = nn.Embedding(2, embed_dim)
        trunc_normal_(self.token_type_embeddings.weight, std=self.init_std)


        
        
        self.apply(self._init_weights)
        self.fix_init_weight()
        # self.gate = nn.Sequential(nn.Linear(embed_dim*2, embed_dim), nn.Sigmoid())
        # self.fft_gate = nn.Sequential(nn.Linear(embed_dim*2, embed_dim), nn.Sigmoid())

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            kdim=embed_dim,
            vdim=embed_dim,
            num_heads=4,
            batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.layer_norm = nn.LayerNorm(embed_dim)
        # self.gate_first_d = nn.Sequential(nn.Linear(embed_dim*2, embed_dim), nn.Sigmoid())

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp_EEG.fc2.weight.data, layer_id + 1)
            rescale(layer.mlp_EYE.fc2.weight.data, layer_id + 1)
            if layer_id >= self.multiffn_start_layer_index:
                rescale(layer.mlp_multi.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, input_chans,modality_type, bool_masked_pos,):
        if modality_type == "EEG":
            x = self.EEG_embedding(x,input_chans,bool_masked_pos)
            x = x + self.token_type_embeddings(
                torch.ones((x.shape[0],x.shape[1])).long().to(x.device)
            )
            for blk in self.blocks:
                x = blk(x,modality_type="EEG")
            return self.norm(x)
        
        elif modality_type == "EYE":
            x = self.EYE_embedding(x,bool_masked_pos)
            x = x + self.token_type_embeddings(
                torch.zeros((x.shape[0],x.shape[1])).long().to(x.device)
            )
            for blk in self.blocks:
                x = blk(x,modality_type="EYE")
            return self.norm(x)
        elif modality_type == "multi":
            x_EEG = x[0]
            x_EYE = x[1]
            x_EEG = self.EEG_embedding(x_EEG,input_chans,bool_masked_pos[0])
            x_EEG = x_EEG + self.token_type_embeddings(
                torch.ones((x_EEG.shape[0],x_EEG.shape[1])).long().to(x_EEG.device)
            )
            x_EYE = self.EYE_embedding(x_EYE,bool_masked_pos[1])
            x_EYE = x_EYE + self.token_type_embeddings(
                torch.zeros((x_EYE.shape[0],x_EYE.shape[1])).long().to(x_EYE.device)
            )
            x = torch.cat([x_EYE,x_EEG],dim=1)
            for blk in self.blocks:
                x = blk(x,modality_type="multi",eye_length=x_EYE.shape[1])
            x = self.norm(x)
            x_EYE = x[:, :x_EYE.shape[1]]
            x_EEG = x[:, x_EYE.shape[1]:]
            return (x_EEG,x_EYE)
        
    def forward(self, x, input_chans=None,modality_type="None", bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False, return_all_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x = self.forward_features(x, input_chans=input_chans,modality_type=modality_type, bool_masked_pos=bool_masked_pos)
        if modality_type == "multi":
            return x
        else:
            if return_all_patch_tokens:
                return x
            x = x[:, 1:]
            if return_patch_tokens:
                return x
            if return_all_tokens:
                return self.lm_head(x)
            else:
                # return the masked tokens
                return self.lm_head(x[bool_masked_pos])
    
    
    
    def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                # with torch.cuda.amp.autocast(enabled=False):
                x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)

        if split_out_as_qkv:
            x = self.norm(x)
            x = self.lm_head(x) # [b, n+1, 3*c]
            q, k, v = x.chunk(3, dim=-1) # [b, n+1, c]
            b, n, c =q.shape
            q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            k = k.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            return x, q, k, v
        else:
            x = self.norm(x)
            x = x[:, 1:]
            x = self.lm_head(x[bool_masked_pos])

            q, k, v = qkv[0], qkv[1], qkv[2]

        return x, q, k, v


    def forward_intermediate(self, x, bool_masked_pos=None, layer_id=12):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(x.device)
        batch_size, _, time_window, _ = x.size()
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + self.pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if l in layer_id:
                    output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")

    def get_last_selfattention(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                # return attention of the last block
                return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)

    def _get_average_psd_torch(self,energy_graph, freq_bands, sample_rate, stft_n=256):
        start_index = torch.floor(freq_bands[0] / sample_rate * stft_n).long()
        end_index = torch.floor(freq_bands[1] / sample_rate * stft_n).long()
        ave_psd = torch.mean(energy_graph[:, start_index:end_index] ** 2, dim=1)
        return ave_psd

    def get_psd_feature_torch(
        self,
        eeg: torch.Tensor, 
        window_size: int, 
        stride_size: int, 
        sample_rate =200,
        stft_n=256,
        freq_bands=[[1, 4], [4, 8], [8, 14], [14, 31], [31, 49]]
    ) -> torch.Tensor:
        """PyTorch版PSD特征提取 (支持GPU加速)
        
        Args:
            eeg: 输入EEG信号 (batch, n_channels, n_samples)
            sample_rate: 采样率(Hz)
            window_size: 分析窗口长度(样本数)
            stride_size: 滑动步长(样本数)
            stft_n: FFT点数
            freq_bands: 频带划分列表
            
        Returns:
            PSD特征张量 (batch, n_channels, n_bands)
        """
        batch, n_channels, n_samples = eeg.shape
        
        # 转换为适合FFT的格式 (合并batch和channel维度)
        window_data = rearrange(eeg, 'b c t -> (b c) t')
        
        # 执行FFT (PyTorch>=1.8的规范写法)
        # fft_data = torch.fft.fft(window_data, n=stft_n, dim=-1)
        # energy_graph = torch.abs(fft_data[..., :stft_n // 2])
        
        fft_data = torch.fft.fft(window_data, n=stft_n, dim=-1) / stft_n
        energy_graph = torch.abs(fft_data[..., :stft_n // 2])

        # 初始化输出张量
        psd = torch.zeros((len(freq_bands), batch * n_channels), 
                        device=eeg.device, dtype=eeg.dtype)
        
        # 计算各频带PSD
        freq_bands = torch.tensor(freq_bands, device=eeg.device)
        for band_idx in range(len(freq_bands)):
            band_ave_psd = self._get_average_psd_torch(
                energy_graph, 
                freq_bands[band_idx], 
                sample_rate, 
                stft_n
            )
            psd[band_idx] = band_ave_psd
        
        # 恢复原始维度
        return rearrange(psd, 'f (b c) -> b c f', c=n_channels)

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / std
        return x





    def EEG_embedding(self,x,input_chans=None,bool_masked_pos=None):
        batch_size, c, time_window, _ = x.size()


        ## fft加注意力融合
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        # amplitude[:,:,:,99] = amplitude[:,:,:,100]
        amplitude = amplitude[:,:,:,:amplitude.shape[3]//2]
        
        amplitude = self.std_norm(amplitude)
        x_fft = self.patch_embed_freq(amplitude)
        x = self.patch_embed(x)
        x = x+x_fft


        # x = self.patch_embed(x)


        # ## binear_fusion融合
        # x_fft = torch.fft.fft(x, dim=-1)
        # amplitude = torch.abs(x_fft)
        # amplitude = self.std_norm(amplitude)
        # x_fft = self.patch_embed_freq(amplitude)
        # x = self.patch_embed(x)
        # # x = rearrange(x, "B CT D -> (B CT) D")
        # # x_fft = rearrange(x_fft, "B CT D -> (B CT) D")
        
        # binear_fused = self.binear_fusion(x,x_fft)


        # # 门控融合
        # gate_value = self.gate(torch.cat([x, x_fft], dim=-1))

        # x = x * gate_value + binear_fused * (1 - gate_value)

        # # x = rearrange(x, "(B CT) D -> B CT D", B=batch_size, CT=c*time_window)






        # # psd 门控注意力融合
        # psd = self.get_psd_feature_torch(rearrange(x,"B C T D -> B (C T) D"), window_size=200, stride_size=200)
        # psd = torch.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0)
        # psd = self.psd_layer_norm(psd)
        # psd = self.psd_projection(psd)
        # x = self.patch_embed(x)
        # # 门控融合
        # gate_value = self.gate(torch.cat([x, psd], dim=-1))
        # # x = x * gate_value + psd * (1 - gate_value)
        # attended, _ = self.cross_attn(
        #     query=x,
        #     key=psd,
        #     value=psd
        # )
        # # x = self.norm(x + attended)
        # x = x * gate_value + attended * (1 - gate_value)



        # 注意力融合
        
        # x = x+psd
        # x = rearrange(x,"B C T D -> B (C T) D")
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        if bool_masked_pos is not None:
            # replace the masked visual tokens by mask_token
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2) # pos_embed[:,1:,:] for visual tokens
            pos_embed = torch.cat((pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1) # pos_embed[:,0:1,:] for cls token
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed[:, 0:time_window, :].unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        return x

    def EYE_embedding(self,x,bool_masked_pos=None):
        batch_size, c, time_window, _ = x.size()


        x_pd = x[:,:2]
        x_fp = x[:,2:]
        x_fp = self.patch_embed_EYE_fp(x_fp)

        # 瞳孔直径的处理
        x_pd_fft = torch.fft.fft(x_pd, dim=-1)
        amplitude = torch.abs(x_pd_fft)
        amplitude = amplitude[:,:,:,:amplitude.shape[3]//2+1]
        amplitude = self.std_norm(amplitude)
        x_pd_fft = self.patch_embed_freq_EYE(amplitude)
        x_pd = self.patch_embed_EYE_pd(x_pd)
        x_pd = x_pd+x_pd_fft

        x = torch.cat((x_pd, x_fp), dim=1)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token_EYE.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token_EYE.expand(batch_size, seq_len, -1)

        if bool_masked_pos is not None:
            # replace the masked visual tokens by mask_token
            # bool_masked_pos = bool_masked_pos[:,:12]
            w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
            x = x * (1 - w) + mask_token * w
        # c = c-1
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed_used = self.pos_embed_EYE[:,:c+1]
        if self.pos_embed_EYE is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2) # pos_embed[:,1:,:] for visual tokens
            pos_embed = torch.cat((pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1) # pos_embed[:,0:1,:] for cls token
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed[:, 0:time_window, :].unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)
        return x





# EEMo
class EEMo(nn.Module):
    def __init__(self, img_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02,multiffn_start_layer_index=10,pretrained_cfg=None,pretrained_cfg_overlay=None):
        super().__init__()
        self.init_std = init_std
        self.patch_size = patch_size
        self.student = MultiWayTransformerForMaskedEegEyeModeling(img_size, patch_size, in_chans, out_chans, vocab_size, embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, attn_head_dim,
                 use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias, init_std,multiffn_start_layer_index)
        
        # MLM
        self.embed_dim = -800
        self.checkpoint_layer = 0
        self.lm_head = nn.Linear(embed_dim, vocab_size) # project embedding to vocab
        self.lm_head_EYE = nn.Linear(embed_dim, vocab_size) 
        trunc_normal_(self.lm_head.weight, std=init_std)
        trunc_normal_(self.lm_head_EYE.weight, std=init_std)

        self.pooler = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )

        # ITC
        self.itc_eeg_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.itc_eye_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.itc_multi_eeg_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.itc_multi_eye_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        trunc_normal_(self.itc_eeg_proj.weight, std=self.init_std)
        trunc_normal_(self.itc_eye_proj.weight, std=self.init_std)
        trunc_normal_(self.itc_multi_eeg_proj.weight, std=self.init_std)
        trunc_normal_(self.itc_multi_eye_proj.weight, std=self.init_std)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_multi_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        # ITM
        self.itm_score = nn.Linear(embed_dim, 2)
        trunc_normal_(self.itm_score.weight, std=self.init_std)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'logit_scale'}

    def loss_selector(self, loss_type, pred, target):
        if loss_type == 'mse':
            return F.mse_loss(pred, target, reduction="mean")
        elif loss_type == 'kld':
            return F.kl_div(F.log_softmax(pred, dim=-1), F.softmax(target, dim=-1), reduction='batchmean')
        elif loss_type == 'smoothl1':
            return F.smooth_l1_loss(pred, target)
    
    def _init_teacher(self):  
        # init the weights of teacher with those of backbone
        for param_encoder, param_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_encoder.data)
            param_teacher.requires_grad = False

    def momentum_update(self, base_momentum=0):
        """Momentum update of the teacher network."""
        for param_encoder, param_teacher in zip(self.student.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + \
                param_encoder.data * (1. - base_momentum)
    
    def forward(self, x, input_chans=None,modality=None, bool_masked_pos=None):
        batch_size = x.size(0)

        x_masked = self.student(x, input_chans,modality, bool_masked_pos, return_all_patch_tokens=True)
        x_masked_no_cls = x_masked[:, 1:]
        if modality == "EEG":
            x_rec = self.lm_head(x_masked_no_cls[bool_masked_pos])
        elif modality == "EYE":
            x_rec = self.lm_head_EYE(x_masked_no_cls[bool_masked_pos])
        # x_rec = self.lm_head(x_masked_no_cls[bool_masked_pos]) # rec shape (batch_size*patch_num/2, vocab_size)

        #symetric
        x_masked_sym = self.student(x, input_chans,modality, ~bool_masked_pos, return_all_patch_tokens=True)
        x_masked_no_cls_sym = x_masked_sym[:, 1:]
        if modality == "EEG":
            x_rec_sym = self.lm_head(x_masked_no_cls_sym[~bool_masked_pos])
        elif modality == "EYE":
            x_rec_sym = self.lm_head_EYE(x_masked_no_cls_sym[~bool_masked_pos])
        return x_rec, x_rec_sym#, loss_align + loss_align_sym


        # 输入脑电单模态数据,输出单模态路径和多模态路径的cls token
    def infer_EEG(self,x,input_chans=None,bool_masked_pos=None):
        
        x = self.student.EEG_embedding(x,input_chans,bool_masked_pos)
        x = x + self.student.token_type_embeddings(
            torch.ones((x.shape[0],x.shape[1])).long().to(x.device)
        )
        all_hidden_states = []
        for i, blk in enumerate(self.student.blocks):
            if self.embed_dim == 800:
                if i<self.checkpoint_layer:
                    x = blk(x,modality_type="EEG")
                else:
                    x = checkpoint(lambda x: blk(x,modality_type="EEG"), x)
            else:
                x = blk(x,modality_type="EEG")
            all_hidden_states.append(x)
        multiffn_hiddens = all_hidden_states[self.student.multiffn_start_layer_index-1]
        for multiffn_index in range(self.student.multiffn_start_layer_index, self.student.depth):
            if self.embed_dim == 800:
                multiffn_hiddens = checkpoint(lambda multiffn_hiddens: blk(multiffn_hiddens, modality_type="multi"), multiffn_hiddens)
            else:
                multiffn_hiddens = blk(multiffn_hiddens, modality_type="multi")
            
        
        eegffn_hiddens = all_hidden_states[-1] 
        eegffn_hiddens = self.student.norm(eegffn_hiddens) #EEG单模态输出

        eeg_feats, eye_feats = (
            eegffn_hiddens,
            None
        )
        cls_feats = self.itc_eeg_proj(eegffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        multiffn_hiddens = self.student.norm(multiffn_hiddens)
        cls_multiffn_feats = self.itc_multi_eeg_proj(multiffn_hiddens[:, 0])
        cls_multiffn_feats = cls_multiffn_feats / cls_multiffn_feats.norm(dim=-1, keepdim=True)

        ret = {
            "eeg_feats": eeg_feats,
            "eye_feats": eye_feats,
            "cls_feats": cls_feats,
            "cls_multiffn_feats": cls_multiffn_feats,
            "raw_cls_feats": x[:, 0],
            "eeg_masks": bool_masked_pos,
            "eye_masks": None,
        }
        return ret
    
    def infer_EYE(self,x,bool_masked_pos=None):
        
        x = self.student.EYE_embedding(x,bool_masked_pos)
        x = x + self.student.token_type_embeddings(
            torch.zeros((x.shape[0],x.shape[1])).long().to(x.device)
        )
        all_hidden_states = []
        for i, blk in enumerate(self.student.blocks):
            if self.embed_dim == 800:
                if i <self.checkpoint_layer:
                    x = blk(x,modality_type="EYE")
                else:
                    x = checkpoint(lambda x: blk(x,modality_type="EYE"), x)
            else:
                x = blk(x,modality_type="EYE")
            all_hidden_states.append(x)
        multiffn_hiddens = all_hidden_states[self.student.multiffn_start_layer_index-1]
        for multiffn_index in range(self.student.multiffn_start_layer_index, self.student.depth):
            if self.embed_dim == 800:
                multiffn_hiddens = checkpoint(lambda multiffn_hiddens: blk(multiffn_hiddens, modality_type="multi"), multiffn_hiddens)
            else:
                multiffn_hiddens = blk(multiffn_hiddens, modality_type="multi")

        
        eyeffn_hiddens = all_hidden_states[-1] 
        eyeffn_hiddens = self.student.norm(eyeffn_hiddens) # EEG单模态输出

        eeg_feats, eye_feats = (
            None,
            eyeffn_hiddens
            
        )
        cls_feats = self.itc_eye_proj(eyeffn_hiddens[:, 0])
        cls_feats = cls_feats / cls_feats.norm(dim=-1, keepdim=True)

        multiffn_hiddens = self.student.norm(multiffn_hiddens)
        cls_multiffn_feats = self.itc_multi_eye_proj(multiffn_hiddens[:, 0])
        cls_multiffn_feats = cls_multiffn_feats / cls_multiffn_feats.norm(dim=-1, keepdim=True)


        ret = {
            "eeg_feats": eeg_feats,
            "eye_feats": eye_feats,
            "cls_feats": cls_feats,
            "cls_multiffn_feats": cls_multiffn_feats,
            "raw_cls_feats": x[:, 0],
            "eeg_masks": None,
            "eye_masks": bool_masked_pos,
        }
        return ret

    def infer(
        self,
        x_EEG,
        x_EYE,
        input_chans=None,
        bool_masked_pos_EEG=None,
        bool_masked_pos_EYE=None,
    ):

        EEG_embeds = self.student.EEG_embedding(x_EEG, input_chans, bool_masked_pos_EEG)
        EYE_embeds = self.student.EYE_embedding(x_EYE, bool_masked_pos_EYE)

        EEG_embeds = EEG_embeds + self.student.token_type_embeddings(
            torch.ones((EEG_embeds.shape[0], EEG_embeds.shape[1])).long().to(EEG_embeds.device))
        EYE_embeds = EYE_embeds + self.student.token_type_embeddings(
            torch.zeros((EYE_embeds.shape[0], EYE_embeds.shape[1])).long().to(EYE_embeds.device))

        # 眼动在前，脑电在后
        co_embeds = torch.cat([EYE_embeds,EEG_embeds], dim=1)
        # if bool_masked_pos_EEG is not None and bool_masked_pos_EYE is not None:
        #     co_masks = torch.cat([bool_masked_pos_EYE,bool_masked_pos_EEG], dim=1) # 这里的mask均没有考虑cls token
        # else:
        #     co_masks = None

        x = co_embeds
        for i, blk in enumerate(self.student.blocks):
            if self.embed_dim == 800:
                if i<self.checkpoint_layer:
                    x = blk(x, modality_type="multi",eye_length=EYE_embeds.shape[1])
                else:
                    x = checkpoint(lambda x: blk(x,modality_type="multi",eye_length=EYE_embeds.shape[1]), x)
            else:
                x = blk(x, modality_type="multi",eye_length=EYE_embeds.shape[1])

        x = self.student.norm(x)

        EYE_feats, EEG_feats = (
            x[:, : EYE_embeds.shape[1]],
            x[:, EYE_embeds.shape[1] :],
        )
        cls_feats = self.pooler(x[:,0])

        ret = {
            "eeg_feats": EEG_feats,
            "eye_feats": EYE_feats,
            "cls_feats": cls_feats,
            "raw_cls_feats": x[:, 0],
            "eeg_data":x_EEG,
            "eye_data":x_EYE,
        }
        return ret
    
    
    def multiMLM(self,x_EEG,x_EYE,input_chans=None,bool_masked_pos_EEG=None,bool_masked_pos_EYE=None):

        ret = self.infer(x_EEG,x_EYE,input_chans,bool_masked_pos_EEG,bool_masked_pos_EYE)
        EEG_embeds = ret["eeg_feats"]
        EYE_embeds = ret["eye_feats"]
        EEG_masked_no_cls = EEG_embeds[:, 1:]
        EYE_masked_no_cls = EYE_embeds[:, 1:]
        EEG_rec = self.lm_head(EEG_masked_no_cls[bool_masked_pos_EEG])
        EYE_rec = self.lm_head_EYE(EYE_masked_no_cls[bool_masked_pos_EYE])

        ret_sym = self.infer(x_EEG,x_EYE,input_chans,~bool_masked_pos_EEG,~bool_masked_pos_EYE)
        EEG_embeds_sym = ret_sym["eeg_feats"]
        EYE_embeds_sym = ret_sym["eye_feats"]
        EEG_masked_no_cls_sym = EEG_embeds_sym[:, 1:]
        EYE_masked_no_cls_sym = EYE_embeds_sym[:, 1:]
        EEG_rec_sym = self.lm_head(EEG_masked_no_cls_sym[~bool_masked_pos_EEG])
        EYE_rec_sym = self.lm_head_EYE(EYE_masked_no_cls_sym[~bool_masked_pos_EYE])
        return EEG_rec,EYE_rec,EEG_rec_sym,EYE_rec_sym

# labram
class VisionTransformerForMaskedImageModeling(nn.Module):
    def __init__(self, img_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed = TemporalConv(out_chans=out_chans)
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if use_abs_pos_emb:
            self.pos_embed = nn.Parameter(torch.zeros(1, 128 + 1, embed_dim))
        else:
            self.pos_embed = None
        self.time_embed = nn.Parameter(torch.zeros(1, 16, embed_dim), requires_grad=True)
        self.pos_drop = nn.Dropout(p=drop_rate)

        if use_shared_rel_pos_bias:
            self.rel_pos_bias = RelativePositionBias(window_size=(62, img_size // patch_size), num_heads=num_heads)
        else:
            self.rel_pos_bias = None

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block( 
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_norm=qk_norm, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.init_std = init_std
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=self.init_std)
        trunc_normal_(self.time_embed, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)
        trunc_normal_(self.mask_token, std=self.init_std)
        trunc_normal_(self.lm_head.weight, std=self.init_std)
        self.apply(self._init_weights)
        self.fix_init_weight()

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'time_embed'}

    def get_num_layers(self):
        return len(self.blocks)

    def forward_features(self, x, input_chans, bool_masked_pos):
        batch_size, c, time_window, _ = x.size()
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2) # pos_embed[:,1:,:] for visual tokens
            pos_embed = torch.cat((pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1) # pos_embed[:,0:1,:] for cls token
            x = x + pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed[:, 0:time_window, :].unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for blk in self.blocks:
            x = blk(x, rel_pos_bias=rel_pos_bias)

        return self.norm(x)

    def forward(self, x, input_chans=None, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False, return_all_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x = self.forward_features(x, input_chans=input_chans, bool_masked_pos=bool_masked_pos)
        if return_all_patch_tokens:
            return x
        x = x[:, 1:]
        if return_patch_tokens:
            return x
        if return_all_tokens:
            return self.lm_head(x)
        else:
            # return the masked tokens
            return self.lm_head(x[bool_masked_pos])
    
    def forward_return_qkv(self, x, bool_masked_pos=None, split_out_as_qkv=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x = self.patch_embed(x, bool_masked_pos=bool_masked_pos)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                # with torch.cuda.amp.autocast(enabled=False):
                x, qkv = blk(x, rel_pos_bias=rel_pos_bias, return_qkv=True)

        if split_out_as_qkv:
            x = self.norm(x)
            x = self.lm_head(x) # [b, n+1, 3*c]
            q, k, v = x.chunk(3, dim=-1) # [b, n+1, c]
            b, n, c =q.shape
            q = q.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            k = k.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            v = v.reshape(b, n, self.num_heads, -1).permute(0, 2, 1, 3)
            return x, q, k, v
        else:
            x = self.norm(x)
            x = x[:, 1:]
            x = self.lm_head(x[bool_masked_pos])

            q, k, v = qkv[0], qkv[1], qkv[2]

        return x, q, k, v


    def forward_intermediate(self, x, bool_masked_pos=None, layer_id=12):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], self.patch_embed.num_patches), dtype=torch.bool).to(x.device)
        batch_size, _, time_window, _ = x.size()
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            pos_embed = self.pos_embed[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + self.pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, 62, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        if isinstance(layer_id, list):
            output_list = []
            for l, blk in enumerate(self.blocks):
                x = blk(x, rel_pos_bias=rel_pos_bias)
                if l in layer_id:
                    output_list.append(x[:, 1:])
            return output_list
        elif isinstance(layer_id, int):
            for l, blk in enumerate(self.blocks):
                if l < layer_id:
                    x = blk(x, rel_pos_bias=rel_pos_bias)
                elif l == layer_id:
                    x = blk.norm1(x)
                else:
                    break
            return x[:, 1:]
        else:
            raise NotImplementedError(f"Not support for layer id is {layer_id} now!")

    def get_last_selfattention(self, x):
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.pos_embed is not None:
            x = x + self.pos_embed
        x = self.pos_drop(x)
        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None

        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x, rel_pos_bias=rel_pos_bias)
            else:
                # return attention of the last block
                return blk(x, rel_pos_bias=rel_pos_bias, return_attention=True)
# labram                
class VisionTransformerForMaskedImageModelingCLS(VisionTransformerForMaskedImageModeling):
    def __init__(self, img_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02,
                 early_layers=6, head_layers=2, shared_lm_head=True):
        super().__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, vocab_size=vocab_size, embed_dim=embed_dim, depth=depth,
                 num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, init_values=init_values, attn_head_dim=attn_head_dim,
                 use_abs_pos_emb=use_abs_pos_emb, use_rel_pos_bias=use_rel_pos_bias, use_shared_rel_pos_bias=use_shared_rel_pos_bias, init_std=init_std)

        self.early_layers = early_layers
        print(f'early layer {early_layers}, late layer {depth - early_layers}, condenser head layers {head_layers}, shared_lm_head {shared_lm_head}')

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, max(depth, early_layers + head_layers))]  # stochastic depth decay rule
        self.cls_pt_layers = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values, window_size=self.patch_embed.patch_shape if use_rel_pos_bias else None,
                attn_head_dim=attn_head_dim,
            )
            for i in range(early_layers, early_layers + head_layers)])
        self.fix_init_cls_pt_weight()

        self.shared_lm_head = shared_lm_head
        if not shared_lm_head:
            self.cls_pt_norm = norm_layer(embed_dim)
            self.cls_pt_lm_head = nn.Linear(embed_dim, vocab_size)

            self.cls_pt_norm.apply(self._init_weights)
            self.cls_pt_lm_head.apply(self._init_weights)

    def fix_init_cls_pt_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.cls_pt_layers):
            rescale(layer.attn.proj.weight.data, self.early_layers + layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, self.early_layers + layer_id + 1)

    def forward_features(self, x, input_chans, bool_masked_pos):
        batch_size, c, time_window, _ = x.size()
        x = self.patch_embed(x)
        batch_size, seq_len, _ = x.size()

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        mask_token = self.mask_token.expand(batch_size, seq_len, -1)

        # replace the masked visual tokens by mask_token
        w = bool_masked_pos.unsqueeze(-1).type_as(mask_token)
        x = x * (1 - w) + mask_token * w

        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed_used = self.pos_embed[:, input_chans] if input_chans is not None else self.pos_embed
        if self.pos_embed is not None:
            pos_embed = pos_embed_used[:, 1:, :].unsqueeze(2).expand(batch_size, -1, time_window, -1).flatten(1, 2)
            pos_embed = torch.cat((self.pos_embed[:,0:1,:].expand(batch_size, -1, -1), pos_embed), dim=1)
            x = x + self.pos_embed
        if self.time_embed is not None:
            time_embed = self.time_embed.unsqueeze(1).expand(batch_size, c, -1, -1).flatten(1, 2)
            x[:, 1:, :] += time_embed
        x = self.pos_drop(x)

        rel_pos_bias = self.rel_pos_bias() if self.rel_pos_bias is not None else None
        for i, blk in enumerate(self.blocks):
            x = blk(x, rel_pos_bias=rel_pos_bias)
            if i + 1 == self.early_layers:
                early_states = x[:, 1:]

        x_cls_pt = torch.cat([x[:, [0]], early_states], dim=1)
        for blk in self.cls_pt_layers:
            x_cls_pt = blk(x_cls_pt, rel_pos_bias=rel_pos_bias)

        return self.norm(x), self.norm(x_cls_pt) if self.shared_lm_head else self.cls_pt_norm(x_cls_pt)

    def forward(self, x, input_chans=None, bool_masked_pos=None, return_all_tokens=False, return_patch_tokens=False):
        if bool_masked_pos is None:
            bool_masked_pos = torch.zeros((x.shape[0], x.shape[1] * x.shape[2]), dtype=torch.bool).to(x.device)
        x, x_cls_pt = self.forward_features(x, input_chans=input_chans, bool_masked_pos=bool_masked_pos)
        x = x[:, 1:]
        x_cls_pt = x_cls_pt[:, 1:]
        if return_patch_tokens:
            return [x, x_cls_pt]
        if return_all_tokens:
            return [self.lm_head(x), self.lm_head(x_cls_pt) if self.shared_lm_head else self.cls_pt_lm_head(x_cls_pt)]
        else:
            # return the masked tokens
            return [self.lm_head(x[bool_masked_pos]), self.lm_head(x_cls_pt[bool_masked_pos]) if self.shared_lm_head else self.cls_pt_lm_head(x_cls_pt[bool_masked_pos])]
# labram
class VisionTransformerForMIMAlign(nn.Module):
    def __init__(self, img_size=1600, patch_size=200, in_chans=1, out_chans=8, vocab_size=8192, embed_dim=200, depth=12,
                 num_heads=10, mlp_ratio=4., qkv_bias=True, qk_norm=None, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=None, init_values=None, attn_head_dim=None,
                 use_abs_pos_emb=True, use_rel_pos_bias=False, use_shared_rel_pos_bias=False, init_std=0.02):
        super().__init__()
        self.patch_size = patch_size
        self.student = VisionTransformerForMaskedImageModeling(img_size, patch_size, in_chans, out_chans, vocab_size, embed_dim, depth,
                 num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, attn_head_dim,
                 use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias, init_std)
        # self.teacher = VisionTransformerForMaskedImageModeling(img_size, patch_size, in_chans, out_chans, vocab_size, embed_dim, depth,
        #          num_heads, mlp_ratio, qkv_bias, qk_norm, qk_scale, drop_rate, attn_drop_rate, drop_path_rate, norm_layer, init_values, attn_head_dim,
        #          use_abs_pos_emb, use_rel_pos_bias, use_shared_rel_pos_bias, init_std)
        
        self.lm_head = nn.Linear(embed_dim, vocab_size) # project embedding to vocab
        self.projection_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU()
        )
        #self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        #self.loss_contra = nn.CrossEntropyLoss()
        #self.loss_align = nn.MSELoss()
        
        #self._init_teacher()
        trunc_normal_(self.lm_head.weight, std=init_std)
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'logit_scale'}

    def loss_selector(self, loss_type, pred, target):
        if loss_type == 'mse':
            return F.mse_loss(pred, target, reduction="mean")
        elif loss_type == 'kld':
            return F.kl_div(F.log_softmax(pred, dim=-1), F.softmax(target, dim=-1), reduction='batchmean')
        elif loss_type == 'smoothl1':
            return F.smooth_l1_loss(pred, target)
    
    def _init_teacher(self):  
        # init the weights of teacher with those of backbone
        for param_encoder, param_teacher in zip(self.student.parameters(), self.teacher.parameters()):
            param_teacher.detach()
            param_teacher.data.copy_(param_encoder.data)
            param_teacher.requires_grad = False

    def momentum_update(self, base_momentum=0):
        """Momentum update of the teacher network."""
        for param_encoder, param_teacher in zip(self.student.parameters(),
                                                self.teacher.parameters()):
            param_teacher.data = param_teacher.data * base_momentum + \
                param_encoder.data * (1. - base_momentum)
    
    def forward(self, x, input_chans=None, bool_masked_pos=None):
        batch_size = x.size(0)

        x_masked = self.student(x, input_chans, bool_masked_pos, return_all_patch_tokens=True)
        x_masked_no_cls = x_masked[:, 1:]
        x_rec = self.lm_head(x_masked_no_cls[bool_masked_pos]) # rec shape (batch_size*patch_num/2, vocab_size)
        #x_masked_contra = self.projection_head(x_masked[:, 1:].mean(1))

        #symetric
        x_masked_sym = self.student(x, input_chans, ~bool_masked_pos, return_all_patch_tokens=True)
        x_masked_no_cls_sym = x_masked_sym[:, 1:]
        x_rec_sym = self.lm_head(x_masked_no_cls_sym[~bool_masked_pos])
        #x_masked_contra_sym = self.projection_head(x_masked_sym[:, 1:].mean(1))
        
        # with torch.no_grad():
        #     x_unmasked = self.teacher(x, input_chans, return_all_patch_tokens=True)
        #     #x_unmasked_contra = self.projection_head(x_unmasked[:, 0])
        #     self.momentum_update(0.996)

        # x_unmasked_no_cls = x_unmasked[:, 1:]
        # loss_align = self.loss_selector('smoothl1', x_masked_no_cls[bool_masked_pos], x_unmasked_no_cls[bool_masked_pos])
        # loss_align_sym = self.loss_selector('smoothl1', x_masked_no_cls_sym[~bool_masked_pos], x_unmasked_no_cls[~bool_masked_pos])
        
        # logit_scale = self.logit_scale.exp()
        # x_masked_contra = x_masked_contra / x_masked_contra.norm(dim=-1, keepdim=True)
        # x_masked_contra_sym = x_masked_contra_sym / x_masked_contra_sym.norm(dim=-1, keepdim=True)
        # logits_contra_1 = logit_scale * x_masked_contra @ x_masked_contra_sym.t()
        # logits_contra_2 = logits_contra_1.t()

        # contra_labels = torch.eye(batch_size).float().to(x.device)
        # loss_contra = (self.loss_contra(logits_contra_1, contra_labels) + self.loss_contra(logits_contra_2, contra_labels)) / 2

        return x_rec, x_rec_sym#, loss_align + loss_align_sym



@register_model
def beit_base_patch200_1600_8k_vocab_cls_pt(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMIMAlign(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qkv_bias=True, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_base_patch200_1600_8k_vocab_align(pretrained=False, **kwargs): #5M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMIMAlign(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model


@register_model
def EEMo_base(pretrained=False, **kwargs): #5M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = EEMo(
        patch_size=200, embed_dim=200, depth=12, num_heads=10, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size,multiffn_start_layer_index=10, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def EEMo_large(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = EEMo(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),out_chans=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size,multiffn_start_layer_index=20, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def EEMo_huge(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = EEMo(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),out_chans=32,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size,multiffn_start_layer_index=40, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_small_patch200_1600_8k_vocab_align(pretrained=False, **kwargs):
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMIMAlign(
        patch_size=200, embed_dim=100, depth=6, num_heads=10, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6),  out_chans=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_large_patch200_1600_8k_vocab_align(pretrained=False, **kwargs): #50M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMIMAlign(
        patch_size=200, embed_dim=400, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_huge_patch200_1600_8k_vocab_align(pretrained=False, **kwargs): #380M
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMIMAlign(
        patch_size=200, embed_dim=800, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=32,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def beit_1B_patch200_1600_8k_vocab_align(pretrained=False, **kwargs): # 1.47B
    if "num_classes" in kwargs:
        _ = kwargs.pop("num_classes")
    if 'vocab_size' in kwargs:
        vocab_size = kwargs['vocab_size']
        _ = kwargs.pop("vocab_size")
    else:
        vocab_size = 8192
    model = VisionTransformerForMIMAlign(
        patch_size=200, embed_dim=1600, depth=48, num_heads=16, mlp_ratio=4, qkv_bias=False, qk_norm=partial(nn.LayerNorm, eps=1e-6), out_chans=64,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), vocab_size=vocab_size, **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
