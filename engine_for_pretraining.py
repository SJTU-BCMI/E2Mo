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

from cgitb import enable
import math
import sys
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from einops import rearrange
from contextlib import nullcontext
import torch.distributed as dist

def random_masking(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # mask = np.hstack([
        #     np.zeros(len_keep),
        #     np.ones(L - len_keep),
        # ])
        # np.random.shuffle(mask)

        return mask.to(torch.bool)


def random_masking_EYE(x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        L=L-8
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        # mask = np.hstack([
        #     np.zeros(len_keep),
        #     np.ones(L - len_keep),
        # ])
        # np.random.shuffle(mask)
        # 将one hot代表向量展开
        indices_to_duplicate = [4, 9, 14, 19]
        offset = 0
        for idx in indices_to_duplicate:
            idx+=offset
            # 在 dim=1 维度上将对应的向量复制一遍，并接到它后面
            mask = torch.cat([mask[:, :idx+1], mask[:, idx:idx+1], mask[:, idx:idx+1], mask[:, idx+1:]], dim=1)
            offset+=2
        return mask.to(torch.bool)

def calculate_rec_loss(rec, target):  
    target = target / target.norm(dim=-1, keepdim=True)
    rec = rec / rec.norm(dim=-1, keepdim=True)
    rec_loss = (1 - (target * rec).sum(-1)).mean()
    return rec_loss


def cal_rec_loss(rec, target):
    rec_loss = F.mse_loss(rec, target)
    return rec_loss


def std_norm(x):
    mean = torch.mean(x, dim=(1, 2), keepdim=True)
    std = torch.std(x, dim=(1, 2), keepdim=True)
    x = (x - mean) / std
    return x


def train_one_epoch(model: torch.nn.Module, vqkd: torch.nn.Module,
                    data_loader_list: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, ch_names_list=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss()

    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        if len(data_loader) == 0:
            continue
        input_chans = utils.get_input_chans(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
            # assign learning rate & weight decay for each step
            it = start_steps + step + step_loader  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]

            samples = batch
            samples = samples.float().to(device, non_blocking=True) / 100
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
            bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=0.5).to(device, non_blocking=True)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    input_ids = vqkd.get_codebook_indices(samples, input_chans)
                #bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
                labels = input_ids[bool_masked_pos]
                labels_sym = input_ids[~bool_masked_pos]

            my_context = model.no_sync if args.distributed and (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
            with my_context():
                with torch.cuda.amp.autocast(): # enabled=False
                    outputs = model(samples, input_chans, bool_masked_pos=bool_masked_pos)

                    if args.enable_align:
                        x_rec, x_rec_sym = outputs
                        loss_rec = loss_fn(x_rec, labels)
                        loss_rec_sym = loss_fn(x_rec_sym, labels_sym)
                        loss = loss_rec + loss_rec_sym# + loss_align
                    else:
                        x_rec, x_rec_sym, x_rec_angle, x_rec_angle_sym = outputs
                        loss_1 = calculate_rec_loss(x_rec, labels)
                        loss_2 = calculate_rec_loss(x_rec_sym, labels_sym)
                        loss = loss_1 + loss_2


            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
                
                sys.exit(1)

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order, update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            
            if args.enable_align:
                mlm_acc = (x_rec.max(-1)[1] == labels).float().mean().item()
                mlm_acc_sym = (x_rec_sym.max(-1)[1] == labels_sym).float().mean().item()
                metric_logger.update(mlm_acc=mlm_acc)
                metric_logger.update(mlm_acc_sym=mlm_acc_sym)
                metric_logger.update(loss_rec=loss_rec.item() / 2)
                #metric_logger.update(loss_contra=loss_contra.item())
                #metric_logger.update(loss_align=loss_align.item() / 2)

                if log_writer is not None:
                    log_writer.update(mlm_acc=mlm_acc, head="loss")
                    log_writer.update(mlm_acc_sym=mlm_acc_sym, head="loss")
                    log_writer.update(loss_rec=loss_rec.item() / 2, head="loss")
                    #log_writer.update(loss_contra=loss_contra.item(), head="loss")
                    #log_writer.update(loss_align=loss_align.item() / 2, head="loss")
            else:
                #mlm_acc_1 = (outputs[0].max(-1)[1] == labels).float().mean().item()
                #mlm_acc_2 = (outputs[1].max(-1)[1] == labels).float().mean().item()
                #metric_logger.update(mlm_acc_1=mlm_acc_1)
                #metric_logger.update(mlm_acc_2=mlm_acc_2)
                metric_logger.update(loss_1=loss_1.item())
                metric_logger.update(loss_2=loss_2.item())

                if log_writer is not None:
                    #log_writer.update(mlm_acc_1=mlm_acc_1, head="loss")
                    #log_writer.update(mlm_acc_2=mlm_acc_2, head="loss")
                    log_writer.update(loss_1=loss_1.item(), head="loss")
                    log_writer.update(loss_2=loss_2.item(), head="loss")

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def train_one_epoch_EEMo(model: torch.nn.Module, vqkd: torch.nn.Module,
                    data_loader_list: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, ch_names_list=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss()

    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        if len(data_loader) == 0:
            continue
        input_chans = utils.get_input_chans(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
            # assign learning rate & weight decay for each step
            it = start_steps + step + step_loader  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]
            if args.modality == "EEG":
                samples = batch
                samples = samples.float().to(device, non_blocking=True) / 100
                samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
            elif args.modality == "EYE":
                samples = batch[0]
                samples = samples.float().to(device, non_blocking=True)
                samples = rearrange(samples, 'B N (A T) -> B N A T', T=250)
            if args.modality == "EYE":
                bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=0.5).to(device, non_blocking=True)
            else:
                bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=0.5).to(device, non_blocking=True)

            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    input_ids = vqkd.get_codebook_indices(samples, input_chans)
                #bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
                labels = input_ids[bool_masked_pos]
                labels_sym = input_ids[~bool_masked_pos]

            my_context = model.no_sync if args.distributed and (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
            with my_context():
                with torch.cuda.amp.autocast(): # enabled=False
                    outputs = model(samples, input_chans,args.modality, bool_masked_pos=bool_masked_pos)

                    if args.enable_align:
                        x_rec, x_rec_sym = outputs
                        loss_rec = loss_fn(x_rec, labels)
                        loss_rec_sym = loss_fn(x_rec_sym, labels_sym)
                        loss = loss_rec + loss_rec_sym# + loss_align
                    else:
                        x_rec, x_rec_sym, x_rec_angle, x_rec_angle_sym = outputs
                        loss_1 = calculate_rec_loss(x_rec, labels)
                        loss_2 = calculate_rec_loss(x_rec_sym, labels_sym)
                        loss = loss_1 + loss_2


            loss_value = loss.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
                
                sys.exit(1)

            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order, update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
            loss_scale_value = loss_scaler.state_dict()["scale"]
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            
            if args.enable_align:
                mlm_acc = (x_rec.max(-1)[1] == labels).float().mean().item()
                mlm_acc_sym = (x_rec_sym.max(-1)[1] == labels_sym).float().mean().item()
                metric_logger.update(mlm_acc=mlm_acc)
                metric_logger.update(mlm_acc_sym=mlm_acc_sym)
                metric_logger.update(loss_rec=loss_rec.item() / 2)
                #metric_logger.update(loss_contra=loss_contra.item())
                #metric_logger.update(loss_align=loss_align.item() / 2)

                if log_writer is not None:
                    log_writer.update(mlm_acc=mlm_acc, head="loss")
                    log_writer.update(mlm_acc_sym=mlm_acc_sym, head="loss")
                    log_writer.update(loss_rec=loss_rec.item() / 2, head="loss")
                    #log_writer.update(loss_contra=loss_contra.item(), head="loss")
                    #log_writer.update(loss_align=loss_align.item() / 2, head="loss")
            else:
                #mlm_acc_1 = (outputs[0].max(-1)[1] == labels).float().mean().item()
                #mlm_acc_2 = (outputs[1].max(-1)[1] == labels).float().mean().item()
                #metric_logger.update(mlm_acc_1=mlm_acc_1)
                #metric_logger.update(mlm_acc_2=mlm_acc_2)
                metric_logger.update(loss_1=loss_1.item())
                metric_logger.update(loss_2=loss_2.item())

                if log_writer is not None:
                    #log_writer.update(mlm_acc_1=mlm_acc_1, head="loss")
                    #log_writer.update(mlm_acc_2=mlm_acc_2, head="loss")
                    log_writer.update(loss_1=loss_1.item(), head="loss")
                    log_writer.update(loss_2=loss_2.item(), head="loss")

            metric_logger.update(loss=loss_value)
            metric_logger.update(loss_scale=loss_scale_value)
            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])

            metric_logger.update(lr=max_lr)
            metric_logger.update(min_lr=min_lr)
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]
            metric_logger.update(weight_decay=weight_decay_value)
            metric_logger.update(grad_norm=grad_norm)

            if log_writer is not None:
                log_writer.update(loss=loss_value, head="loss")
                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()

            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def compute_itc(model,batch,input_chans,aggregate = True):
    EEG_samples = batch[0]
    EYE_samples = batch[1]
    infer_eeg = model.module.infer_EEG(EEG_samples,input_chans=input_chans, bool_masked_pos=None)
    infer_eye = model.module.infer_EYE(EYE_samples,bool_masked_pos=None)

    eeg_features = infer_eeg["cls_feats"]
    eye_features = infer_eye["cls_feats"]
    logit_scale = model.module.logit_scale.exp().mean()

    eeg_multiffn_features = infer_eeg["cls_multiffn_feats"]
    eye_multiffn_features = infer_eye["cls_multiffn_feats"]
    logit_multi_scale = model.module.logit_multi_scale.exp().mean()



    aggregate = True
    if aggregate:
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more negatives to contrast with.
        gathered_eeg_features = [
            torch.zeros_like(eeg_features) for _ in range(world_size)
        ]
        gathered_eye_features = [
            torch.zeros_like(eye_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_eeg_features, eeg_features)
        dist.all_gather(gathered_eye_features, eye_features)

        all_eeg_features = torch.cat(
            [eeg_features]
            + gathered_eeg_features[:rank]
            + gathered_eeg_features[rank + 1 :]
        )
        all_eye_features = torch.cat(
            [eye_features]
            + gathered_eye_features[:rank]
            + gathered_eye_features[rank + 1 :]
        )

        # this is needed to send gradients back everywhere.
        logits_per_eeg = logit_scale * all_eeg_features @ all_eye_features.t()

        del gathered_eeg_features, gathered_eye_features
        logits_per_eye = logits_per_eeg.t()

        gathered_eeg_multiffn_features = [
            torch.zeros_like(eeg_multiffn_features) for _ in range(world_size)
        ]
        gathered_eye_multiffn_features = [
            torch.zeros_like(eye_multiffn_features) for _ in range(world_size)
        ]
        dist.all_gather(gathered_eeg_multiffn_features, eeg_multiffn_features)
        dist.all_gather(gathered_eye_multiffn_features, eye_multiffn_features)

        all_eeg_multiffn_features = torch.cat(
            [eeg_multiffn_features]
            + gathered_eeg_multiffn_features[:rank]
            + gathered_eeg_multiffn_features[rank + 1 :]
        )
        all_eye_multiffn_features = torch.cat(
            [eye_multiffn_features]
            + gathered_eye_multiffn_features[:rank]
            + gathered_eye_multiffn_features[rank + 1 :]
        )
        # this is needed to send gradients back everywhere.
        logits_per_multiffn_eeg = logit_multi_scale * all_eeg_multiffn_features @ all_eye_multiffn_features.t()
        logits_per_multiffn_eye = logits_per_multiffn_eeg.t()





    ground_truth = torch.arange(len(logits_per_eeg)).long().to(device=logits_per_eeg.get_device())

    itc_loss = (
        F.cross_entropy(logits_per_eeg.float(), ground_truth)
        + F.cross_entropy(logits_per_eye.float(), ground_truth)
    ) / 2

    itc_multiffn_loss = (
        F.cross_entropy(logits_per_multiffn_eeg.float(), ground_truth)
        + F.cross_entropy(logits_per_multiffn_eye.float(), ground_truth)
    ) / 2

    itc_total_loss = (itc_loss + itc_multiffn_loss) * 0.5

    ret = {
        "itc_loss": itc_total_loss,
        "itc_eeg2eye_logits": logits_per_eeg,
        "itc_eye2eeg_logits": logits_per_eye,
        "itc_labels": ground_truth,
        "itc_logit_scale": logit_scale,
        "itc_logit_multi_scale": logit_multi_scale,
    }
    return ret

def compute_itm(model,batch,input_chans,sim_eeg2eye,sim_eye2eeg):
    # ITM
    EEG_samples = batch[0]
    EYE_samples = batch[1]
    pos_len = EEG_samples.shape[0]
    neg_len = EEG_samples.shape[0]
    bsz = EEG_samples.shape[0]

    itm_labels = torch.cat([torch.ones(pos_len), torch.zeros(neg_len), torch.zeros(neg_len)]).to(
        EEG_samples.device
    )

    # batch = {k: v for k, v in batch.items()}
    infer_pos = model.module.infer(EEG_samples, EYE_samples, input_chans)
    batch_EEG = infer_pos["eeg_data"]
    batch_EYE = infer_pos["eye_data"]

    with torch.no_grad():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more hard negative candidates.
        gathered_eeg = [
            torch.zeros_like(batch_EEG) for _ in range(world_size)
        ]
        gathered_eye = [
            torch.zeros_like(batch_EYE) for _ in range(world_size)
        ]
        
        dist.all_gather(gathered_eeg, batch_EEG)
        dist.all_gather(gathered_eye, batch_EYE)

        all_eeg = torch.cat(
            [batch_EEG]
            + gathered_eeg[:rank]
            + gathered_eeg[rank + 1 :]
        )
        all_eye = torch.cat(
            [batch_EYE]
            + gathered_eye[:rank]
            + gathered_eye[rank + 1 :]
        )
  
        weights_eeg2eye = F.softmax(sim_eeg2eye[:bsz, :].float(), dim=1)
        weights_eye2eeg = F.softmax(sim_eye2eeg[:bsz, :].float(), dim=1)

        weights_eeg2eye.fill_diagonal_(0)
        weights_eye2eeg.fill_diagonal_(0)
    
    eeg_neg = []    
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_eye2eeg[b], 1).item()
        eeg_neg.append(all_eeg[neg_idx])
    eeg_neg = torch.stack(eeg_neg, dim=0)   

    # select a negative text for each image
    eye_neg = []
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_eeg2eye[b], 1).item()
        eye_neg.append(all_eye[neg_idx])
    eye_neg = torch.stack(eye_neg, dim=0)

    # text_labels is not used in ITM loss
    infer_eeg_neg = model.module.infer(eeg_neg, EYE_samples, input_chans)
    infer_eye_neg = model.module.infer(EEG_samples, eye_neg, input_chans)

    all_cls_feats = torch.cat([infer_pos["cls_feats"], infer_eeg_neg["cls_feats"], infer_eye_neg["cls_feats"]], dim=0)

    itm_logits = model.module.itm_score(all_cls_feats)
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }
    return ret


def compute_itm_huge_pos(model,batch,input_chans,sim_eeg2eye,sim_eye2eeg):
     # ITM
    EEG_samples = batch[0]
    EYE_samples = batch[1]
    pos_len = EEG_samples.shape[0]
    neg_len = EEG_samples.shape[0]
    bsz = EEG_samples.shape[0]

    # batch = {k: v for k, v in batch.items()}
    infer_pos = model.module.infer(EEG_samples, EYE_samples, input_chans)
    infer_pos_cls = infer_pos["cls_feats"]
    itm_labels = torch.ones(pos_len).to(EEG_samples.device)
    itm_logits = model.module.itm_score(infer_pos_cls)
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())
    return itm_loss

def compute_itm_huge_eeg_neg(model,batch,input_chans,sim_eeg2eye,sim_eye2eeg):
    # ITM
    EEG_samples = batch[0]
    EYE_samples = batch[1]
    pos_len = EEG_samples.shape[0]
    neg_len = EEG_samples.shape[0]
    bsz = EEG_samples.shape[0]

    # batch = {k: v for k, v in batch.items()}
    batch_EEG = EEG_samples
    batch_EYE = EYE_samples

    with torch.no_grad():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more hard negative candidates.
        gathered_eeg = [
            torch.zeros_like(batch_EEG) for _ in range(world_size)
        ]
        gathered_eye = [
            torch.zeros_like(batch_EYE) for _ in range(world_size)
        ]
        
        dist.all_gather(gathered_eeg, batch_EEG)
        dist.all_gather(gathered_eye, batch_EYE)

        all_eeg = torch.cat(
            [batch_EEG]
            + gathered_eeg[:rank]
            + gathered_eeg[rank + 1 :]
        )
        all_eye = torch.cat(
            [batch_EYE]
            + gathered_eye[:rank]
            + gathered_eye[rank + 1 :]
        )
  
        weights_eeg2eye = F.softmax(sim_eeg2eye[:bsz, :].float(), dim=1)
        weights_eye2eeg = F.softmax(sim_eye2eeg[:bsz, :].float(), dim=1)

        weights_eeg2eye.fill_diagonal_(0)
        weights_eye2eeg.fill_diagonal_(0)
    
    eeg_neg = []    
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_eye2eeg[b], 1).item()
        eeg_neg.append(all_eeg[neg_idx])
    eeg_neg = torch.stack(eeg_neg, dim=0)   

    # select a negative text for each image
    eye_neg = []
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_eeg2eye[b], 1).item()
        eye_neg.append(all_eye[neg_idx])
    eye_neg = torch.stack(eye_neg, dim=0)
    return eeg_neg,eye_neg,EYE_samples,EEG_samples,input_chans

    # text_labels is not used in ITM loss
    infer_eeg_neg = model.module.infer(eeg_neg, EYE_samples, input_chans)
    infer_eye_neg = model.module.infer(EEG_samples, eye_neg, input_chans)
    infer_eeg_neg_cls = infer_eeg_neg["cls_feats"]
    infer_eye_neg_cls = infer_eye_neg["cls_feats"]
    itm_labels = torch.zeros(neg_len).to(EEG_samples.device)
    itm_logits = model.module.itm_score(infer_eeg_neg_cls)
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())
    return itm_loss


def compute_itm_huge(model,batch,input_chans,sim_eeg2eye,sim_eye2eeg):
    # ITM
    EEG_samples = batch[0]
    EYE_samples = batch[1]
    pos_len = EEG_samples.shape[0]
    neg_len = EEG_samples.shape[0]
    bsz = EEG_samples.shape[0]

    # batch = {k: v for k, v in batch.items()}
    infer_pos = model.module.infer(EEG_samples, EYE_samples, input_chans)
    batch_EEG = infer_pos["eeg_data"]
    batch_EYE = infer_pos["eye_data"]

    with torch.no_grad():
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        # We gather tensors from all gpus to get more hard negative candidates.
        gathered_eeg = [
            torch.zeros_like(batch_EEG) for _ in range(world_size)
        ]
        gathered_eye = [
            torch.zeros_like(batch_EYE) for _ in range(world_size)
        ]
        
        dist.all_gather(gathered_eeg, batch_EEG)
        dist.all_gather(gathered_eye, batch_EYE)

        all_eeg = torch.cat(
            [batch_EEG]
            + gathered_eeg[:rank]
            + gathered_eeg[rank + 1 :]
        )
        all_eye = torch.cat(
            [batch_EYE]
            + gathered_eye[:rank]
            + gathered_eye[rank + 1 :]
        )
  
        weights_eeg2eye = F.softmax(sim_eeg2eye[:bsz, :].float(), dim=1)
        weights_eye2eeg = F.softmax(sim_eye2eeg[:bsz, :].float(), dim=1)

        weights_eeg2eye.fill_diagonal_(0)
        weights_eye2eeg.fill_diagonal_(0)
    
    eeg_neg = []    
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_eye2eeg[b], 1).item()
        eeg_neg.append(all_eeg[neg_idx])
    eeg_neg = torch.stack(eeg_neg, dim=0)   

    # select a negative text for each image
    eye_neg = []
    for b in range(bsz):
        neg_idx = torch.multinomial(weights_eeg2eye[b], 1).item()
        eye_neg.append(all_eye[neg_idx])
    eye_neg = torch.stack(eye_neg, dim=0)

    # text_labels is not used in ITM loss
    infer_eeg_neg = model.module.infer(eeg_neg, EYE_samples, input_chans)
    infer_eye_neg = model.module.infer(EEG_samples, eye_neg, input_chans)
    infer_pos_cls = infer_pos["cls_feats"]
    infer_eeg_neg_cls = infer_eeg_neg["cls_feats"]
    infer_eye_neg_cls = infer_eye_neg["cls_feats"]
    return infer_pos_cls,infer_eeg_neg_cls,infer_eye_neg_cls,torch.ones(pos_len).to(EEG_samples.device), torch.zeros(neg_len).to(EEG_samples.device), torch.zeros(neg_len).to(EEG_samples.device)

    all_cls_feats = torch.cat([infer_pos["cls_feats"], infer_eeg_neg["cls_feats"], infer_eye_neg["cls_feats"]], dim=0)

    itm_logits = model.module.itm_score(all_cls_feats)
    itm_loss = F.cross_entropy(itm_logits, itm_labels.long())

    ret = {
        "itm_loss": itm_loss,
        "itm_logits": itm_logits,
        "itm_labels": itm_labels,
    }
    return ret

def train_one_epoch_EEMo_multi_valid(model: torch.nn.Module, vqkd_EEG: torch.nn.Module,vqkd_EYE: torch.nn.Module,
                    data_loader_list: Iterable,valid_dataloader, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, ch_names_list=None, args=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    loss_fn = nn.CrossEntropyLoss()

    step_loader = 0
    for data_loader, ch_names in zip(data_loader_list, ch_names_list):
        if len(data_loader) == 0:
            continue
        input_chans = utils.get_input_chans(ch_names)
        for step, (batch) in enumerate(metric_logger.log_every(data_loader, print_freq * args.gradient_accumulation_steps, header)):
            # assign learning rate & weight decay for each step
            it = start_steps + step + step_loader  # global training iteration
            if lr_schedule_values is not None or wd_schedule_values is not None:
                for i, param_group in enumerate(optimizer.param_groups):
                    if lr_schedule_values is not None:
                        param_group["lr"] = lr_schedule_values[it] * param_group["lr_scale"]
                    if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                        param_group["weight_decay"] = wd_schedule_values[it]
            batch[0] = batch[0].float().to(device, non_blocking=True)/100
            batch[0] = rearrange(batch[0], 'B N (A T) -> B N A T', T=200)
            batch[1] = batch[1].float().to(device, non_blocking=True)
            batch[1] = rearrange(batch[1], 'B N (A T) -> B N A T', T=250)

            # 计算vq编码
            bool_masked_pos_EEG = random_masking(batch[0].flatten(1, 2), mask_ratio=0.5).to(device, non_blocking=True)
            bool_masked_pos_EYE = random_masking(batch[1].flatten(1, 2), mask_ratio=0.5).to(device, non_blocking=True)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    input_ids_EEG = vqkd_EEG.get_codebook_indices(batch[0], input_chans)
                    input_ids_EYE = vqkd_EYE.get_codebook_indices(batch[1], input_chans)
                labels_EEG = input_ids_EEG[bool_masked_pos_EEG]
                labels_EYE = input_ids_EYE[bool_masked_pos_EYE]
                labels_sym_EEG = input_ids_EEG[~bool_masked_pos_EEG]
                labels_sym_EYE = input_ids_EYE[~bool_masked_pos_EYE]
            
            my_context = model.no_sync if args.distributed and (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            
            with my_context():
                with torch.cuda.amp.autocast(): 
                    # ITC
                    ret_itc = compute_itc(model,batch,input_chans)
                    loss_itc = ret_itc["itc_loss"]
                    sim_eeg2eye = ret_itc["itc_eeg2eye_logits"]
                    sim_eye2eeg = ret_itc["itc_eye2eeg_logits"]
                    # ITM
                    ret_itm = compute_itm(model,batch,input_chans,sim_eeg2eye,sim_eye2eeg)
                    loss_itm = ret_itm["itm_loss"]
                    # MLM 同时对眼动和脑电做对称掩码并同时修复
                    EEG_rec,EYE_rec,EEG_rec_sym,EYE_rec_sym = \
                        model.module.multiMLM(batch[0],batch[1], input_chans,bool_masked_pos_EEG=bool_masked_pos_EEG,bool_masked_pos_EYE=bool_masked_pos_EYE)
                    
                    loss_rec_EEG = loss_fn(EEG_rec, labels_EEG)
                    loss_rec_EYE = loss_fn(EYE_rec, labels_EYE)
                    loss_rec_sym_EEG = loss_fn(EEG_rec_sym, labels_sym_EEG)
                    loss_rec_sym_EYE = loss_fn(EYE_rec_sym, labels_sym_EYE)
                    loss_MLM = loss_rec_EEG + loss_rec_EYE + loss_rec_sym_EEG + loss_rec_sym_EYE
                    loss = loss_itc + loss_itm + loss_MLM
                            

                loss_value = loss.item()
                if not math.isfinite(loss_value):
                    print(f"Loss is {loss_value}, stopping training at rank {utils.get_rank()}", force=True)
                    sys.exit(1)

                is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
                grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                        parameters=model.parameters(), create_graph=is_second_order, update_grad=(step + 1) % args.gradient_accumulation_steps == 0)
            
            
            loss_scale_value = loss_scaler.state_dict()["scale"]
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.zero_grad()

            torch.cuda.synchronize()
            # torch.cuda.empty_cache()
            if step % 50 == 0:
                eval_results = evaluate_EEMo_multi(vqkd_EEG,vqkd_EYE,valid_dataloader, model, device, header='Test:', ch_names=ch_names, metrics=['acc'], is_binary=True,args=args)
                if log_writer is not None:
                    log_writer.update(test_loss_itc=eval_results['loss_itc'], head="test_loss")
                    log_writer.update(test_loss_itm=eval_results['loss_itm'], head="test_loss")
                    log_writer.update(test_loss_MLM=eval_results['loss_MLM'], head="test_loss")
                    log_writer.update(test_loss=eval_results['loss'], head="test_loss")

            # 日志记录
            
            mlm_acc_EEG = (EEG_rec.max(-1)[1] == labels_EEG).float().mean().item()
            mlm_acc_EYE = (EYE_rec.max(-1)[1] == labels_EYE).float().mean().item()
            mlm_acc_sym_EEG = (EEG_rec_sym.max(-1)[1] == labels_sym_EEG).float().mean().item()
            mlm_acc_sym_EYE = (EYE_rec_sym.max(-1)[1] == labels_sym_EYE).float().mean().item()
            



            min_lr = 10.
            max_lr = 0.
            for group in optimizer.param_groups:
                min_lr = min(min_lr, group["lr"])
                max_lr = max(max_lr, group["lr"])
            weight_decay_value = None
            for group in optimizer.param_groups:
                if group["weight_decay"] > 0:
                    weight_decay_value = group["weight_decay"]

            metric_logger.update(mlm_acc_EEG=mlm_acc_EEG)
            metric_logger.update(mlm_acc_EYE=mlm_acc_EYE)
            # metric_logger.update(lr=max_lr)
            # metric_logger.update(min_lr=min_lr)
            # metric_logger.update(weight_decay=weight_decay_value)
            # metric_logger.update(grad_norm=grad_norm)
            metric_logger.update(loss_MLM=loss_MLM.item()/4)
            metric_logger.update(loss_itc=loss_itc.item())
            metric_logger.update(loss_itm=loss_itm.item())

            if log_writer is not None:
                log_writer.update(mlm_acc_EEG=mlm_acc_EEG, head="loss")
                log_writer.update(mlm_acc_EYE=mlm_acc_EYE, head="loss")
                log_writer.update(mlm_acc_sym_EEG=mlm_acc_sym_EEG, head="loss")
                log_writer.update(mlm_acc_sym_EYE=mlm_acc_sym_EYE, head="loss")
                log_writer.update(loss_rec_EEG=loss_rec_EEG.item(), head="loss")
                log_writer.update(loss_rec_EYE=loss_rec_EYE.item(), head="loss")
                log_writer.update(loss_rec_sym_EEG=loss_rec_sym_EEG.item(), head="loss")
                log_writer.update(loss_rec_sym_EYE=loss_rec_sym_EYE.item(), head="loss")
                log_writer.update(loss_MLM=loss_MLM.item()/4, head="loss")
                log_writer.update(loss_itc=loss_itc.item(), head="loss")
                log_writer.update(loss_itm=loss_itm.item(), head="loss")
                log_writer.update(loss=loss_value, head="loss")

                log_writer.update(loss_scale=loss_scale_value, head="opt")
                log_writer.update(lr=max_lr, head="opt")
                log_writer.update(min_lr=min_lr, head="opt")
                log_writer.update(weight_decay=weight_decay_value, head="opt")
                log_writer.update(grad_norm=grad_norm, head="opt")

                log_writer.set_step()
            if lr_scheduler is not None:
                lr_scheduler.step_update(start_steps + step + step_loader)
        step_loader += step
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_EEMo(vqkd,data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True,args=None):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #header = 'Test:'

    # switch to evaluation mode
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # assign learning rate & weight decay for each step
    
        if args.modality == "EEG":
            samples = batch
            samples = samples.float().to(device, non_blocking=True) / 100
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        elif args.modality == "EYE":
            samples = batch[0]
            samples = samples.float().to(device, non_blocking=True)
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=250)
        if args.modality == "EYE":
            bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=0.5).to(device, non_blocking=True)
        else:
            bool_masked_pos = random_masking(samples.flatten(1, 2), mask_ratio=0.5).to(device, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                input_ids = vqkd.get_codebook_indices(samples, input_chans)
            #bool_masked_pos = bool_masked_pos.flatten(1).to(torch.bool)
            labels = input_ids[bool_masked_pos]
            labels_sym = input_ids[~bool_masked_pos]

        my_context = model.no_sync if args.distributed and (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
        with my_context():
            with torch.cuda.amp.autocast(): # enabled=False
                outputs = model(samples, input_chans,args.modality, bool_masked_pos=bool_masked_pos)

                if args.enable_align:
                    x_rec, x_rec_sym = outputs
                    loss_rec = loss_fn(x_rec, labels)
                    loss_rec_sym = loss_fn(x_rec_sym, labels_sym)
                    loss = loss_rec + loss_rec_sym# + loss_align
                else:
                    x_rec, x_rec_sym, x_rec_angle, x_rec_angle_sym = outputs
                    loss_1 = calculate_rec_loss(x_rec, labels)
                    loss_2 = calculate_rec_loss(x_rec_sym, labels_sym)
                    loss = loss_1 + loss_2
        metric_logger.update(loss=loss.item())
        # for key, value in results.items():
        #     metric_logger.meters[key].update(value, n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}             




@torch.no_grad()
def evaluate_EEMo_multi(vqkd_EEG,vqkd_EYE,data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True,args=None):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    if is_binary:
        criterion = torch.nn.BCEWithLogitsLoss()
    else:
        criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    #header = 'Test:'

    # switch to evaluation mode
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # assign learning rate & weight decay for each step
    
        batch[0] = batch[0].float().to(device, non_blocking=True)/100
        batch[0] = rearrange(batch[0], 'B N (A T) -> B N A T', T=200)
        batch[1] = batch[1].float().to(device, non_blocking=True)
        batch[1] = rearrange(batch[1], 'B N (A T) -> B N A T', T=250)

        # 计算vq编码
        bool_masked_pos_EEG = random_masking(batch[0].flatten(1, 2), mask_ratio=0.5).to(device, non_blocking=True)
        bool_masked_pos_EYE = random_masking(batch[1].flatten(1, 2), mask_ratio=0.5).to(device, non_blocking=True)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                input_ids_EEG = vqkd_EEG.get_codebook_indices(batch[0], input_chans)
                input_ids_EYE = vqkd_EYE.get_codebook_indices(batch[1], input_chans)
            labels_EEG = input_ids_EEG[bool_masked_pos_EEG]
            labels_EYE = input_ids_EYE[bool_masked_pos_EYE]
            labels_sym_EEG = input_ids_EEG[~bool_masked_pos_EEG]
            labels_sym_EYE = input_ids_EYE[~bool_masked_pos_EYE]
        
        my_context = model.no_sync if args.distributed and (step + 1) % args.gradient_accumulation_steps != 0 else nullcontext
        with my_context():
            with torch.cuda.amp.autocast(): 
                # ITC
                ret_itc = compute_itc(model,batch,input_chans)
                loss_itc = ret_itc["itc_loss"]
                sim_eeg2eye = ret_itc["itc_eeg2eye_logits"]
                sim_eye2eeg = ret_itc["itc_eye2eeg_logits"]
                # ITM
                ret_itm = compute_itm(model,batch,input_chans,sim_eeg2eye,sim_eye2eeg)
                loss_itm = ret_itm["itm_loss"]
                # MLM 同时对眼动和脑电做对称掩码并同时修复
                EEG_rec,EYE_rec,EEG_rec_sym,EYE_rec_sym = \
                    model.module.multiMLM(batch[0],batch[1], input_chans,bool_masked_pos_EEG=bool_masked_pos_EEG,bool_masked_pos_EYE=bool_masked_pos_EYE)
                
                loss_rec_EEG = loss_fn(EEG_rec, labels_EEG)
                loss_rec_EYE = loss_fn(EYE_rec, labels_EYE)
                loss_rec_sym_EEG = loss_fn(EEG_rec_sym, labels_sym_EEG)
                loss_rec_sym_EYE = loss_fn(EYE_rec_sym, labels_sym_EYE)
                loss_MLM = loss_rec_EEG + loss_rec_EYE + loss_rec_sym_EEG + loss_rec_sym_EYE

                loss = loss_itc + loss_itm + loss_MLM

        metric_logger.update(loss=loss.item())
        metric_logger.update(loss_itc=loss_itc.item())
        metric_logger.update(loss_itm=loss_itm.item())
        metric_logger.update(loss_MLM=loss_MLM.item()/4)
        # for key, value in results.items():
        #     metric_logger.meters[key].update(value, n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}      