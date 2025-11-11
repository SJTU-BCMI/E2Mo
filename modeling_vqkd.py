# --------------------------------------------------------
# BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers (https://arxiv.org/abs/2208.06366)
# Github source: https://github.com/microsoft/unilm/tree/master/beitv2
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Zhiliang Peng
# Based on VQGAN code bases
# https://github.com/CompVis/taming-transformers
# --------------------------------------------------------'

import torch
import numpy as np
from torch import nn, einsum
import torch.nn.functional as F
import math
from collections import OrderedDict
from functools import partial, reduce
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model

from modeling_modules import VisionTransformer, VisionTransformer_EYE,VisionTransformer_EYE_attn_trace,VisionTransformer_freq,TemporalConv,TemporalConv_freq
from norm_ema_quantizer import NormEMAVectorQuantizer

import utils

# from vqkd_teacher import clip, get_dino_vit_base


class VQKD(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_embed=8192, 
                 embed_dim=32,
                 decay=0.99,
                 process_type='default',
                 quantize_kmeans_init=True,
                 teacher_model_type='clip',
                 decoder_out_dim=200,
                 rec_loss_type='cosine',
                 smooth_l1_loss = False,
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        if decoder_config['in_chans'] != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config['in_chans']} to {embed_dim}")
            decoder_config['in_chans'] = embed_dim

        # encoder & decode params
        print('Final encoder config', encoder_config)
        self.encoder = VisionTransformer_freq(**encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder = VisionTransformer(**decoder_config)
                
        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )
        
        self.patch_size = encoder_config['patch_size']
        self.token_shape = (62, encoder_config['img_size'] // self.patch_size)

        ## Teacher model setting
        self.teacher_model_type = teacher_model_type
        self.decoder_out_dim = decoder_out_dim
        if self.teacher_model_type == 'clip':
            self.scaling_layer = ScalingLayerForClip()
            self.teacher_model, _ = clip.load("ViT-B/16", device='cpu', jit=False)
            self.decoder_out_dim = 512

        elif self.teacher_model_type == 'dino':
            self.scaling_layer = ScalingLayerForIM()
            self.teacher_model = get_dino_vit_base()
            self.decoder_out_dim = 768

        else:
            self.teacher_model = None

        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False # fix teacher_model model

            self.teacher_model.eval()
            self.teacher_input_size = kwargs.get('teacher_input_size', 224)

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim) # for quantize
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        # self.decode_task_layer_angle = nn.Sequential(
        #     nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
        #     nn.Tanh(),
        #     nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        # )
        self.decode_task_layer_raw = TemporalConv(out_chans=8)
        
        self.rec_loss_type = rec_loss_type

        print(f"process type for VQKD: {process_type}")
        self.process_type = process_type # in ['default', 'dall-e']
        self.logit_laplace_eps = 0.1
        self.kwargs = kwargs
        
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)
        # self.decode_task_layer_angle.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed', 
                'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed'}

    @property
    def device(self):
        return self.decoder.cls_token.device

    def pre_process(self, data):
        if self.process_type == 'default':
            # TODO: modify for adapt
            data = data.to(self.device)
            if data.max() <= 1.:
                data = data * 255.
            data = data / 127.5 - 1.0
        elif self.process_type == 'imagenet_norm':
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(self.device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(self.device)[None, :, None, None]
            data = (data - mean) / std
        return data
        
    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, input_chans=None, **kwargs):
        
        #data = self.pre_process(data)
        quantize, embed_ind, loss = self.encode(data, input_chans=input_chans)
        output = {}
        output['token'] = embed_ind.view(data.shape[0], -1)
        output['input_img'] = data
        output['quantize'] = rearrange(quantize, 'b d a c -> b (a c) d')

        return output

    def encode(self, x, input_chans=None):
        batch_size, n, a, t = x.shape
        encoder_features = self.encoder(x, input_chans, return_patch_tokens=True)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        N = to_quantizer_features.shape[1]
        h, w = n, N // n

        to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, loss
    
    def decode(self, quantize, input_chans=None, **kwargs):
        # reshape tokens to feature maps for patch embed in decoder
        # quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=self.token_shape[0], w=self.token_shape[1])
        b,c,h,w = quantize.shape
        decoder_features = self.decoder(quantize, input_chans, return_patch_tokens=True)
        rec = self.decode_task_layer(decoder_features)
        decoder_features = rearrange(decoder_features, 'b (h w) c -> b h w c', h=h, w=w)
        rec_raw = self.decode_task_layer_raw(decoder_features)
        # rec_angle = self.decode_task_layer_angle(decoder_features)
        return rec, rec_raw# , rec_angle
    
    def get_codebook_indices(self, x, input_chans=None, **kwargs):
        # for beit pre-training
        return self.get_tokens(x, input_chans, **kwargs)['token']

    @torch.no_grad()
    def get_regress_target(self, x, **kwargs):

        norm_imgs = self.scaling_layer(x)
        if self.teacher_model_type == 'clip':
            target = self.teacher_model.encode_image(norm_imgs, return_all_tokens=True) @ self.teacher_model.visual.proj
        elif self.teacher_model_type == 'dino':
            target = self.teacher_model.forward(norm_imgs, return_patch_tokens=True)
        else:
            raise NotImplementedError

        return target

    def calculate_rec_loss(self, rec, target):  
        if self.rec_loss_type == 'cosine':
            target = target / target.norm(dim=-1, keepdim=True)
            rec = rec / rec.norm(dim=-1, keepdim=True)
            rec_loss = (1 - (target * rec).sum(-1)).mean()
        else:
            raise NotImplementedError

        return rec_loss
    
    def cal_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c')
        rec_loss = self.loss_fn(rec, target)
        return rec_loss
    
    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / std
        return x

    def forward(self, x, input_chans=None, **kwargs):
        """
        x: shape [B, N, T]
        """
        #x = self.pre_process(x) # rescale to [-1, 1]

        #target = self.get_regress_target(x, **kwargs)
        x = rearrange(x, 'B N (A T) -> B N A T', T=200)
        x_fft = torch.fft.fft(x, dim=-1)
        amplitude = torch.abs(x_fft)
        amplitude = self.std_norm(amplitude)
        # angle = torch.angle(x_fft)
        # angle = self.std_norm(angle)

        quantize, embed_ind, emb_loss = self.encode(x, input_chans)
        
        xrec, xrec_raw = self.decode(quantize, input_chans)
        
        rec_loss = self.cal_rec_loss(xrec, amplitude)
        x = rearrange(x, 'B N A T -> B (N A) T')
        rec_raw_loss = self.calculate_rec_loss(xrec_raw, x)
        # rec_angle_loss = self.cal_rec_loss(xrec_angle, angle)
        loss = emb_loss + rec_raw_loss+ rec_loss# +rec_raw_loss#  + rec_angle_loss

        log = {}
        split="train" if self.training else "val"
        log[f'{split}/quant_loss'] = emb_loss.detach().mean()
        log[f'{split}/rec_loss'] = rec_loss.detach().mean()
        # log[f'{split}/rec_angle_loss'] = rec_angle_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()
        log[f'{split}/rec_raw_loss'] = rec_raw_loss.detach().mean()

        return loss, log



class VQKD_EYE_RAW(nn.Module):
    def __init__(self,
                 encoder_config,
                 decoder_config,
                 n_embed=8192, 
                 embed_dim=32,
                 decay=0.99,
                 process_type='default',
                 quantize_kmeans_init=True,
                 teacher_model_type='clip',
                 decoder_out_dim=200,
                 rec_loss_type='cosine',
                 smooth_l1_loss = False,
                 **kwargs
                 ):
        super().__init__()
        print(kwargs)
        if decoder_config['in_chans'] != embed_dim:
            print(f"Rewrite the in_chans in decoder from {decoder_config['in_chans']} to {embed_dim}")
            decoder_config['in_chans'] = embed_dim

        # encoder & decode params
        print('Final encoder config', encoder_config)
        self.encoder = VisionTransformer_EYE_attn_trace("encoder",**encoder_config)

        print('Final decoder config', decoder_config)
        self.decoder = VisionTransformer_EYE("decoder",**decoder_config)

        
                
        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )
        
        self.patch_size = encoder_config['patch_size']
        self.token_shape = (62, encoder_config['img_size'] // self.patch_size)

        ## Teacher model setting
        self.teacher_model_type = teacher_model_type
        self.decoder_out_dim = decoder_out_dim
        if self.teacher_model_type == 'clip':
            self.scaling_layer = ScalingLayerForClip()
            self.teacher_model, _ = clip.load("ViT-B/16", device='cpu', jit=False)
            self.decoder_out_dim = 512

        elif self.teacher_model_type == 'dino':
            self.scaling_layer = ScalingLayerForIM()
            self.teacher_model = get_dino_vit_base()
            self.decoder_out_dim = 768

        else:
            self.teacher_model = None

        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False # fix teacher_model model

            self.teacher_model.eval()
            self.teacher_input_size = kwargs.get('teacher_input_size', 224)

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim) # for quantize
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], 250),
        )
        self.decode_task_layer_feature = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], 31),
        )
        self.deconv = DeconvNet()
        
        self.rec_loss_type = rec_loss_type

        print(f"process type for VQKD: {process_type}")
        self.process_type = process_type # in ['default', 'dall-e']
        self.logit_laplace_eps = 0.1
        self.kwargs = kwargs
        
        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed', 
                'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed'}

    @property
    def device(self):
        return self.decoder.cls_token.device

    def pre_process(self, data):
        if self.process_type == 'default':
            # TODO: modify for adapt
            data = data.to(self.device)
            if data.max() <= 1.:
                data = data * 255.
            data = data / 127.5 - 1.0
        elif self.process_type == 'imagenet_norm':
            mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(self.device)[None, :, None, None]
            std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(self.device)[None, :, None, None]
            data = (data - mean) / std
        return data
        
    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, input_chans=None, **kwargs):
        
        #data = self.pre_process(data)
        quantize, embed_ind, loss = self.encode(data, input_chans=input_chans)
        output = {}
        output['token'] = embed_ind.view(data.shape[0], -1)
        output['input_img'] = data
        output['quantize'] = rearrange(quantize, 'b d a c -> b (a c) d')

        return output

    def encode(self, x, input_chans=None):
        batch_size, n, a, t = x.shape
        # n=n-1 # 注视位置xy编码到了同一个向量
        encoder_features = self.encoder(x, input_chans, return_patch_tokens=True)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        N = to_quantizer_features.shape[1]
        h, w = n, N // n

        to_quantizer_features = rearrange(to_quantizer_features, 'b (h w) c -> b c h w', h=h, w=w) # reshape for quantizer
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, loss
    
    def decode(self, quantize, input_chans=None, **kwargs):
        # reshape tokens to feature maps for patch embed in decoder
        # quantize = rearrange(quantize, 'b (h w) c -> b c h w', h=self.token_shape[0], w=self.token_shape[1])
        decoder_features = self.decoder(quantize, input_chans, return_patch_tokens=True)
        # decoder_features = torch.mean(decoder_features, dim=1)
        
        rec_raw = self.decode_task_layer(decoder_features)
        rec_raw = self.deconv(rec_raw)
        decoder_features = torch.mean(decoder_features, dim=1)
        rec_feature = self.decode_task_layer_feature(decoder_features)
        # rec = torch.mean(rec, dim=1)
        # rec_angle = self.decode_task_layer_angle(decoder_features)
        return rec_raw,rec_feature# , rec_angle
    
    def get_codebook_indices(self, x, input_chans=None, **kwargs):
        # for beit pre-training
        return self.get_tokens(x, input_chans, **kwargs)['token']

    @torch.no_grad()
    def get_regress_target(self, x, **kwargs):

        norm_imgs = self.scaling_layer(x)
        if self.teacher_model_type == 'clip':
            target = self.teacher_model.encode_image(norm_imgs, return_all_tokens=True) @ self.teacher_model.visual.proj
        elif self.teacher_model_type == 'dino':
            target = self.teacher_model.forward(norm_imgs, return_patch_tokens=True)
        else:
            raise NotImplementedError

        return target

    def calculate_rec_loss(self, rec, target):  
        if self.rec_loss_type == 'cosine':
            target = target / target.norm(dim=-1, keepdim=True)
            rec = rec / rec.norm(dim=-1, keepdim=True)
            rec_loss = (1 - (target * rec).sum(-1)).mean()
        else:
            raise NotImplementedError

        return rec_loss
    
    def cal_rec_loss(self, rec, target):
        target = rearrange(target, 'b n a c -> b (n a) c')
        rec = self.std_norm(rec)
        target = self.std_norm(target)
        rec_loss = self.loss_fn(rec, target)
        return rec_loss
    
    def cal_rec_loss_feature(self, rec, target):
        rec_loss = self.loss_fn(rec, target)
        return rec_loss
    
    def std_norm(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        x = (x - mean) / std
        return x

    def forward(self, x, input_chans=None, **kwargs):
        """
        x: shape [B, N, T]
        """
        #x = self.pre_process(x) # rescale to [-1, 1]

        #target = self.get_regress_target(x, **kwargs)
        EYE_feature = x[1]
        EYE_feature = EYE_feature.reshape(EYE_feature.shape[0],-1)
        # EYE_feature = self.std_norm(EYE_feature)
        x = x[0]
        x = rearrange(x, 'B N (A T) -> B N A T', T=250)

        quantize, embed_ind, emb_loss = self.encode(x, input_chans)
        
        xrec,xrec_feature = self.decode(quantize, input_chans)

        
        raw_rec_loss = self.cal_rec_loss(xrec, x)
        feature_rec_loss = self.cal_rec_loss_feature(xrec_feature, EYE_feature)
        # rec_angle_loss = self.cal_rec_loss(xrec_angle, angle)
        # loss = emb_loss + raw_rec_loss# +feature_rec_loss#  + rec_angle_loss
        # loss = emb_loss + feature_rec_loss # raw_rec_loss# +feature_rec_loss#  + rec_angle_loss
        # loss = emb_loss + raw_rec_loss +feature_rec_loss#  + rec_angle_loss
        loss = emb_loss + raw_rec_loss +feature_rec_loss#  + rec_angle_loss

        log = {}
        split="train" if self.training else "val"
        log[f'{split}/quant_loss'] = emb_loss.detach().mean()
        log[f'{split}/rec_loss'] = raw_rec_loss.detach().mean()
        log[f'{split}/feature_rec_loss'] = feature_rec_loss.detach().mean()
        # log[f'{split}/rec_angle_loss'] = rec_angle_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, log




class DeconvNet(nn.Module):
    def __init__(self):
        super(DeconvNet, self).__init__()
        # 使用较小的卷积核，输入维度: (B, 28, 200) 输出维度: (B, 28, 250)
        self.deconv = nn.ConvTranspose1d(
            in_channels=16,  # 输入通道数
            out_channels=16,  # 输出通道数
            kernel_size=5,  # 使用较小的卷积核
            stride=1,  # 步长为1
            padding=2,  # 填充，保持输出宽度正确
            output_padding=0,  # 输出额外的填充
        )

    def forward(self, x):
        return self.deconv(x)


class ScalingLayerForClip(nn.Module):
    def __init__(self):
        super(ScalingLayerForClip, self).__init__()
        self.register_buffer(
            "shift",
            torch.Tensor([0.48145466, 0.4578275, 0.40821073])[None, :, None, None],
        )
        self.register_buffer(
            "scale",
            torch.Tensor([0.26862954, 0.26130258, 0.27577711])[None, :, None, None],
        )

    def forward(self, inp):
        inp = ((inp + 1.0) * 127.5).clamp(0, 255.0) / 255.0  # rescale to [0, 1.]
        return (inp - self.shift) / self.scale


class ScalingLayerForIM(nn.Module):
    def __init__(self):
        super(ScalingLayerForIM, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )  # scale for tokenizer with default prosscess type \in [-1, 1]
        self.register_buffer(
            "scale", torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )

    def forward(self, inp):
        inp = ((inp + 1.0) * 127.5).clamp(0, 255.0) / 255.0  # rescale to [0, 1.]
        return (inp - self.shift) / self.scale


def get_model_default_params():
    return dict(
        img_size=1600,
        patch_size=200,
        in_chans=1,
        num_classes=1000,
        embed_dim=200,
        depth=12,
        num_heads=10,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_values=0.0,
        use_abs_pos_emb=True,
        use_rel_pos_bias=False,
        use_shared_rel_pos_bias=False,
        use_mean_pooling=True,
        init_scale=0.001,
    )


@register_model
def vqkd_encoder_base_decoder_3x200x12(
    pretrained=False,
    pretrained_weight=None,
    as_tokenzer=False,
    img_size=1600,
    n_code=8192,
    code_dim=32,
    **kwargs,
):
    encoder_config, decoder_config = (
        get_model_default_params(),
        get_model_default_params(),
    )

    # encoder settings
    encoder_config["img_size"] = img_size
    encoder_config["num_classes"] = 0
    # decoder settings
    decoder_config["img_size"] = img_size // decoder_config["patch_size"]
    decoder_config["patch_size"] = 1
    decoder_config["in_chans"] = code_dim
    decoder_config["num_classes"] = 0
    decoder_config["depth"] = 3
    # teacher settings
    _ = kwargs.pop("teacher_model_type", "clip")

    teacher_model_type = "None"
    decoder_out_dim = 200

    model = VQKD(
        encoder_config,
        decoder_config,
        n_code,
        code_dim,
        teacher_model_type=teacher_model_type,
        decoder_out_dim=decoder_out_dim,
        **kwargs,
    )

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None

        if pretrained_weight.startswith("https"):
            weights = torch.hub.load_state_dict_from_url(
                pretrained_weight, map_location="cpu", check_hash=True
            )
        else:
            weights = torch.load(pretrained_weight, map_location="cpu")

        if "model" in weights:
            weights = weights["model"]
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())

        for k in keys:
            if (
                k.startswith("loss")
                or k.startswith("teacher")
                or k.startswith("scaling")
            ):
                del weights[k]
        model.load_state_dict(weights)
    return model



@register_model
# 重建眼动特征和原始眼动信号
def vqkd_encoder_base_EYE_RAW(
    pretrained=False,
    pretrained_weight=None,
    as_tokenzer=False,
    img_size=1600,
    n_code=256,
    code_dim=32,
    **kwargs,
):
    n_code = 512
    encoder_config, decoder_config = (
        get_model_default_params(),
        get_model_default_params(),
    )

    # encoder settings
    encoder_config["img_size"] = img_size
    encoder_config["num_classes"] = 0
    # decoder settings
    decoder_config["img_size"] = img_size // decoder_config["patch_size"]
    decoder_config["patch_size"] = 1
    decoder_config["in_chans"] = code_dim
    decoder_config["num_classes"] = 0
    decoder_config["depth"] = 3
    # teacher settings
    _ = kwargs.pop("teacher_model_type", "clip")

    teacher_model_type = "None"
    decoder_out_dim = 200

    model = VQKD_EYE_RAW(
        encoder_config,
        decoder_config,
        n_code,
        code_dim,
        teacher_model_type=teacher_model_type,
        decoder_out_dim=decoder_out_dim,
        **kwargs,
    )

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None

        if pretrained_weight.startswith("https"):
            weights = torch.hub.load_state_dict_from_url(
                pretrained_weight, map_location="cpu", check_hash=True
            )
        else:
            weights = torch.load(pretrained_weight, map_location="cpu")

        if "model" in weights:
            weights = weights["model"]
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())

        for k in keys:
            if (
                k.startswith("loss")
                or k.startswith("teacher")
                or k.startswith("scaling")
            ):
                del weights[k]
        model.load_state_dict(weights)
    return model


if __name__ == "__main__":
    pass
