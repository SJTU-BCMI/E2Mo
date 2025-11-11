# --------------------------------------------------------
# BEIT: BERT Pre-Training of Image Transformers (https://arxiv.org/abs/2106.08254)
# Github source: https://github.com/microsoft/unilm/tree/master/beit
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# By Hangbo Bao
# Based on timm, DINO and DeiT code bases
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'
import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import utils
import numpy as np
from einops import rearrange
import time
import torch.nn.functional as F
from torch.autograd import Variable

def train_class_batch(model, samples, target, criterion, ch_names):
    outputs = model(samples, ch_names)
    loss = criterion(outputs, target)
    return loss, outputs

def train_regression_batch(model, samples, target, criterion, ch_names):
    outputs = model(samples, ch_names)
    loss = criterion(outputs.flatten(), target.float())
    return loss, outputs

def get_loss_scale_for_deepspeed(model):
    optimizer = model.optimizer
    return optimizer.loss_scale if hasattr(optimizer, "loss_scale") else optimizer.cur_scale


def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True,args=None):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        if args.modality=="multi":
            samples[0] = samples[0].float().to(device, non_blocking=True) / 100
            samples[0] = rearrange(samples[0], 'B N (A T) -> B N A T', T=200)
            samples[1] = samples[1].float().to(device, non_blocking=True)
            samples[1] = rearrange(samples[1], 'B N (A T) -> B N A T', T=250)
        elif args.modality=="EYE":
                samples = samples.float().to(device, non_blocking=True)
                samples = rearrange(samples, 'B N (A T) -> B N A T', T=250)
        elif args.modality=="EEG":
            samples = samples.float().to(device, non_blocking=True) / 100
            samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        targets = targets.to(device, non_blocking=True)

        if is_binary:
            targets = targets.float().unsqueeze(-1)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets, input_chans)

        if loss_scaler is None:
            samples = samples.half()
            loss, output = train_class_batch(
                model, samples, targets, criterion, input_chans)
        else:
            with torch.cuda.amp.autocast():
                loss, output = train_class_batch(
                    model, samples, targets, criterion, input_chans)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            if is_binary:
                balanced_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["balanced_accuracy"], is_binary)["balanced_accuracy"]
            else:
                balanced_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            balanced_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(balanced_acc=balanced_acc)
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
            log_writer.update(balanced_acc=balanced_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def train_one_epoch_cross_modal(model: torch.nn.Module,teacher: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True,args=None):
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        
        samples[0] = samples[0].float().to(device, non_blocking=True) / 100
        samples[0] = rearrange(samples[0], 'B N (A T) -> B N A T', T=200)
        samples[1] = samples[1].float().to(device, non_blocking=True)
        samples[1] = rearrange(samples[1], 'B N (A T) -> B N A T', T=250)
        
        targets = targets.to(device, non_blocking=True)
        
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets, input_chans)

        
        with torch.cuda.amp.autocast():
            # 学生模型仅输入眼动信号
            output = model(samples[1], input_chans)
            with torch.no_grad():
                # 教师模型输入眼动信号和脑电信号
                outputs_teacher = teacher.collect_layer_output(samples, input_chans,modality_type="multi")
        loss_cls = criterion(output[-1], targets)
        loss1 = F.mse_loss(output[-2],outputs_teacher[-2],reduction='mean')
        loss2 = F.mse_loss(output[-3],outputs_teacher[-3],reduction='mean')
        loss_teacher = (loss1+loss2)/2
        # for layID in range(len(output)-1):
        #     if layID == 0:
        #         loss_teacher = F.mse_loss(output[layID],outputs_teacher[layID])
        #     else:
        #         loss_teacher += F.mse_loss(output[layID],outputs_teacher[layID])
        # loss_teacher = loss_teacher / (len(output)-1)
        loss = loss_cls + loss_teacher
        # loss = loss_teacher
        loss_value = loss.item()
        output = output[-1]

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        if loss_scaler is None:
            loss /= update_freq
            model.backward(loss)
            model.step()

            if (data_iter_step + 1) % update_freq == 0:
                # model.zero_grad()
                # Deepspeed will call step() & model.zero_grad() automatic
                if model_ema is not None:
                    model_ema.update(model)
            grad_norm = None
            loss_scale_value = get_loss_scale_for_deepspeed(model)
        else:
            # this attribute is added by timm on one optimizer (adahessian)
            is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
            loss /= update_freq
            grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                    parameters=model.parameters(), create_graph=is_second_order,
                                    update_grad=(data_iter_step + 1) % update_freq == 0)
            if (data_iter_step + 1) % update_freq == 0:
                optimizer.zero_grad()
                if model_ema is not None:
                    model_ema.update(model)
            loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        if mixup_fn is None:
            if is_binary:
                balanced_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
            else:
                balanced_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            balanced_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(balanced_acc=balanced_acc)
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
            log_writer.update(loss_cls=loss_cls.item(), head="loss")
            log_writer.update(loss_teacher=loss_teacher.item(), head="loss")
            log_writer.update(balanced_acc=balanced_acc, head="loss")
            log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def _get_average_psd(energy_graph, freq_bands, sample_rate, stft_n=256):
    start_index = int(np.floor(freq_bands[0] / sample_rate * stft_n))
    end_index = int(np.floor(freq_bands[1] / sample_rate * stft_n))
    ave_psd = np.mean(energy_graph[:, start_index - 1:end_index] ** 2, axis=1)
    return ave_psd

def get_psd_feature(eeg: np.ndarray, sample_rate: int, window_size: int, stride_size:int, stft_n=256,
            freq_bands=[[1, 4], [4, 8], [8, 14], [14, 31], [31, 49]]) -> np.ndarray:
    """提取某段时序信号的PSD特征

    :param np.ndarray eeg: 信号 (n_channels, n_samples)
    :param int sample_rate: 采样率
    :param int window_size: 窗口长度 (s)
    :param int stride_size: 步长（s）
    :param int stft_n: fft参数, defaults to 256
    :param list freq_bands: 每个频段的范围, defaults to [[1, 4], [4, 8], [8, 14], [14, 31], [31, 49]]
    :return np.ndarray: PSD特征 (n_freq_bands, n_channels)
    """
    batch, n_channels, n_samples = eeg.shape

    psd = np.zeros((len(freq_bands), n_channels*batch))

    window_data = rearrange(eeg, 'b c t -> (b c) t')
    fft_data = np.fft.fft(window_data, n=stft_n)
    energy_graph = np.abs(fft_data[:, 0: int(stft_n / 2)])

    for band_index, band in enumerate(freq_bands):
        band_ave_psd = _get_average_psd(energy_graph, band, sample_rate, stft_n)
        psd[band_index, :] = band_ave_psd

    psd = rearrange(psd, 'f (b c) -> b c f', c=n_channels)
    return psd


def get_de_feature(eeg: np.ndarray, sample_rate: int, window_size: int, stride_size: int, stft_n=256,
            freq_bands=[[1, 4], [4, 8], [8, 14], [14, 31], [31, 49]]) -> np.ndarray:
    """提取时序信号的DE特征

    :param np.ndarray eeg: 信号 (n_channels, n_samples)
    :param int sample_rate: 采样率
    :param int window_size: 窗口长度 (s)
    :param int stride_size: 步长（s）
    :param int stft_n: fft参数, defaults to 256  大于序列长度的第一个2的幂次数
    :param list freq_bands: 每个频段的范围, defaults to [[1, 4], [4, 8], [8, 14], [14, 31], [31, 49]]
    :return np.ndarray: DE特征 (n_windows, n_freq_bands, n_channels)
    """

    psd = get_psd_feature(eeg, sample_rate, window_size, stride_size, stft_n,
            freq_bands)

    de = np.log2(100*psd)
    return de

def extract_DE(data):
    #data: b c nt
    data = data.cpu().numpy()
    DE = get_de_feature(data, sample_rate=200, window_size=1, stride_size=1)  #b c f
    DE = torch.tensor(DE)
    return DE


def cca_loss(H1, H2,device):  ##cca 损失
    r1 = 1e-3
    r2 = 1e-3
    eps = 1e-9
    H1, H2 = H1.t(), H2.t()
    o1 = H1.size(0)
    o2 = H2.size(0)
    m = H1.size(1)
    H1bar = H1 - H1.mean(dim=1).unsqueeze(dim=1).repeat(1,m)
    H2bar = H2 - H2.mean(dim=1).unsqueeze(dim=1).repeat(1,m)
    SigmaHat11 = (1.0 / (m-1)) * torch.matmul(H1bar,H1bar.t()) + r1 * torch.eye(o1).to(device)
    SigmaHat22 = (1.0 / (m-1)) * torch.matmul(H2bar,H2bar.t()) + r2 * torch.eye(o2).to(device)
    SigmaHat12 = (1.0 / (m-1)) * torch.matmul(H1bar,H2bar.t())
    SigmaHat11 = Variable(SigmaHat11, requires_grad=True)
    SigmaHat12 = Variable(SigmaHat12, requires_grad=True)
    SigmaHat22 = Variable(SigmaHat22, requires_grad=True)
    [D1, V1] = torch.symeig(SigmaHat11, eigenvectors=True)
    [D2, V2] = torch.symeig(SigmaHat22, eigenvectors=True)
    posInd1 = torch.gt(D1, eps).nonzero()[:, 0]
    D1 = D1[posInd1]
    V1 = V1[:, posInd1]
    posInd2 = torch.gt(D2, eps).nonzero()[:, 0]
    D2 = D2[posInd2]
    V2 = V2[:, posInd2]
    SigmaHat11RootInv = torch.matmul(torch.matmul(V1, torch.diag(D1 ** -0.5)), V1.t())
    SigmaHat22RootInv = torch.matmul(torch.matmul(V2, torch.diag(D2 ** -0.5)), V2.t())
    Tval = torch.matmul(torch.matmul(SigmaHat11RootInv,SigmaHat12), SigmaHat22RootInv)
    tmp = torch.trace(torch.matmul(Tval.t(), Tval))
    corr = torch.sqrt(tmp)
    return -corr


def train_one_epoch_lightning(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None, log_writer=None,
                    start_steps=None, lr_schedule_values=None, wd_schedule_values=None,
                    num_training_steps_per_epoch=None, update_freq=None, ch_names=None, is_binary=True,args=None):
    # start_time = time.time()
    input_chans = None
    if ch_names is not None:
        input_chans = utils.get_input_chans(ch_names)
    model.train(True)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('min_lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    if loss_scaler is None:
        model.zero_grad()
        model.micro_steps = 0
    else:
        optimizer.zero_grad()
    # print(time.time() - start_time)
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # print(time.time() - start_time)
        step = data_iter_step // update_freq
        if step >= num_training_steps_per_epoch:
            continue
        it = start_steps + step  # global training iteration
        # Update LR & WD for the first acc
        if lr_schedule_values is not None or wd_schedule_values is not None and data_iter_step % update_freq == 0:
            for i, param_group in enumerate(optimizer.param_groups):
                if lr_schedule_values is not None:
                    param_group["lr"] = lr_schedule_values[it] * param_group.get("lr_scale", 1.0)
                if wd_schedule_values is not None and param_group["weight_decay"] > 0:
                    param_group["weight_decay"] = wd_schedule_values[it]
        if args.modality=="multi":
            samples[0] = samples[0].float().to(device, non_blocking=True) / 100
            samples[0] = rearrange(samples[0], 'B N (A T) -> B N A T', T=200)
            samples[1] = samples[1].float().to(device, non_blocking=True)
            samples[1] = rearrange(samples[1], 'B N (A T) -> B N A T', T=250)
        elif args.modality=="EYE":
            samples = samples.float().to(device, non_blocking=True)
        elif args.modality=="EEG":
            samples = samples.float().to(device, non_blocking=True) / 100
        elif args.modality == "multi_feature":
            # # 眼动特征不变，脑电提取DE特征，并对两个信号分别做归一化
            # samples[0] = samples[0] / 100
            # samples[0] = extract_DE(samples[0])
            # samples[0] = samples[0].float().to(device, non_blocking=True)
            # samples[1] = samples[1].float().to(device, non_blocking=True)
            # samples[0] = samples[0].reshape(samples[0].shape[0], -1)
            # samples[0] = (samples[0] - torch.mean(samples[0], dim=0, keepdim=True))/ torch.std(samples[0], dim=0, keepdim=True)
            # samples[1] = samples[1].reshape(samples[1].shape[0], -1)
            # samples[1] = (samples[1] - torch.mean(samples[1], dim=0, keepdim=True))/ torch.std(samples[1], dim=0, keepdim=True)
            samples[0] = samples[0].float().to(device, non_blocking=True)
            samples[1] = samples[1].float().to(device, non_blocking=True)
            
            # samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        targets = targets.to(device, non_blocking=True)
        
        if is_binary:
            targets = targets.float().unsqueeze(-1)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets, input_chans)
        # print(time.time() - start_time)
        if args.model_name == "DCCA":
            alpha = 1
            x_eye,x_eeg,output = model(samples)
            loss_cls = criterion(output, targets)
            loss_cca = cca_loss(x_eye,x_eeg,device)
            loss = loss_cls+alpha*loss_cca
        elif args.model_name == "BDAE":
            decoder_eeg,decoder_eye,output= model(samples)
            loss_eeg_reconstruction = F.mse_loss(decoder_eeg,samples[0].reshape(samples[0].shape[0], -1))
            loss_eye_reconstruction = F.mse_loss(decoder_eye,samples[1].reshape(samples[1].shape[0], -1))
            loss_cls = criterion(output,targets)
            loss = loss_eeg_reconstruction+loss_eye_reconstruction+loss_cls
        elif args.model_name == "VNT":
            output,output1,output2 = model(samples)
            loss1 = criterion(output, targets)
            loss2 = criterion(output1, targets)
            loss3 = criterion(output2, targets)
            loss = (loss1+loss2+loss3)/3
            # loss = loss3
        elif args.model_name == "EEMo_concat":
            output = model(samples,input_chans)
            loss = criterion(output, targets)
        else:
            output = model(samples)
            loss = criterion(output, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        # if loss_scaler is None:
        # print(time.time() - start_time)
        loss /= update_freq
        
        loss.backward()
        optimizer.step()
        # print(time.time() - start_time)

        if (data_iter_step + 1) % update_freq == 0:
            # model.zero_grad()
            # Deepspeed will call step() & model.zero_grad() automatic
            if model_ema is not None:
                model_ema.update(model)
        grad_norm = None
        # loss_scale_value = get_loss_scale_for_deepspeed(model)

        torch.cuda.synchronize()

        if mixup_fn is None:
            if is_binary:
                balanced_acc = utils.get_metrics(torch.sigmoid(output).detach().cpu().numpy(), targets.detach().cpu().numpy(), ["accuracy"], is_binary)["accuracy"]
            else:
                balanced_acc = (output.max(-1)[-1] == targets).float().mean()
        else:
            balanced_acc = None
        metric_logger.update(loss=loss_value)
        metric_logger.update(balanced_acc=balanced_acc)
        # metric_logger.update(loss_scale=loss_scale_value)
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
            log_writer.update(balanced_acc=balanced_acc, head="loss")
            # log_writer.update(loss_scale=loss_scale_value, head="opt")
            log_writer.update(lr=max_lr, head="opt")
            log_writer.update(min_lr=min_lr, head="opt")
            log_writer.update(weight_decay=weight_decay_value, head="opt")
            log_writer.update(grad_norm=grad_norm, head="opt")

            log_writer.set_step()

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(data_loader, model, device, header='Test:', ch_names=None, metrics=['acc'], is_binary=True,args=None):
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
    pred = []
    true = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        samples = batch[0]
        target = batch[-1]
        if args.modality =="multi":
            samples[0] = samples[0].float().to(device, non_blocking=True) / 100
            samples[0] = rearrange(samples[0], 'B N (A T) -> B N A T', T=200)
            samples[1] = samples[1].float().to(device, non_blocking=True)
            samples[1] = rearrange(samples[1], 'B N (A T) -> B N A T', T=250)

        elif args.modality == "multi_feature":
            # 眼动特征不变，脑电提取DE特征，并对两个信号分别做归一化
            samples[0] = samples[0].float().to(device, non_blocking=True)
            samples[1] = samples[1].float().to(device, non_blocking=True)
            
        else:
            if args.modality=="EYE":
                samples = samples.float().to(device, non_blocking=True)
                if not args.is_lightning:
                    samples = rearrange(samples, 'B N (A T) -> B N A T', T=250)
            elif args.modality=="EEG":
                samples = samples.float().to(device, non_blocking=True) / 100
                if not args.is_lightning:
                    samples = rearrange(samples, 'B N (A T) -> B N A T', T=200)
        
        target = target.to(device, non_blocking=True)
        
        if is_binary:
            target = target.float().unsqueeze(-1)

        # compute output
        if args.is_lightning:
            if args.model_name in ["DCCA","BDAE"]:
                x_eye,x_eeg,output = model(samples)
            elif args.model_name == "VNT":
                output,_,_ = model(samples)
            elif args.model_name == "EEMo_concat":
                output = model(samples,input_chans)
            else:
                output = model(samples)
            loss = criterion(output, target)
        else:
            with torch.cuda.amp.autocast():
                if args.is_cross_modal:
                    output = model(samples[1], input_chans)
                    output = output[-1]
                    loss = criterion(output, target)
                else:
                    output = model(samples, input_chans)
                    loss = criterion(output, target)
        
        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()
        # results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        if args.modality == "multi" or args.modality == "multi_feature":
            batch_size = samples[0].shape[0]
        else:
            batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        # for key, value in results.items():
        #     metric_logger.meters[key].update(value, n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # gather the stats from all processes
    pred = torch.cat(pred, dim=0)
    true = torch.cat(true, dim=0)
    metric_logger.meters['pred'].update(pred)
    metric_logger.meters['true'].update(true)
    metric_logger.synchronize_between_processes()
    pred = np.array(metric_logger.meters['pred'].global_avg)
    true = np.array(metric_logger.meters['true'].global_avg)
    results = utils.get_metrics(pred, true, metrics, is_binary)
    for key, value in results.items():
        metric_logger.meters[key].update(value, n=batch_size)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

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
    pred = []
    true = []
    for step, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        images = batch[0]
        target = batch[-1]
        images = images.float().to(device, non_blocking=True) / 100
        images = rearrange(images, 'B N (A T) -> B N A T', T=200)
        if isinstance(target, dict):
            target = torch.tensor([config.downstream.label[label] for label in target['label']]).long().to(device, non_blocking=True)
        else:
            target = target.to(device, non_blocking=True)
        if is_binary:
            target = target.float().unsqueeze(-1)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images, input_chans=input_chans)
            loss = criterion(output, target)
        
        if is_binary:
            output = torch.sigmoid(output).cpu()
        else:
            output = output.cpu()
        target = target.cpu()
        results = utils.get_metrics(output.numpy(), target.numpy(), metrics, is_binary)
        pred.append(output)
        true.append(target)

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        for key, value in results.items():
            metric_logger.meters[key].update(value, n=batch_size)
        #metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.accuracy, losses=metric_logger.loss))
    
    pred = torch.cat(pred, dim=0)
    true = torch.cat(true, dim=0)
    # 计算混淆矩阵
    from sklearn.metrics import confusion_matrix
    pred = (pred.numpy()>0.5).astype(int)
    true = true.numpy()
    confusion_matrix = confusion_matrix(true, pred)
    return confusion_matrix