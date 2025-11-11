# --------------------------------------------------------
# Based on BEiT, BEiT v2, timm, DeiT and DINO code bases
# https://github.com/microsoft/unilm/tree/master/beitv2
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit/
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
import os
import random

from pathlib import Path

from timm.models import create_model
import torch.utils
from optim_factory import create_optimizer

from engine_for_pretraining import train_one_epoch,train_one_epoch_EEMo,evaluate_EEMo,evaluate_EEMo_multi,train_one_epoch_EEMo_multi_valid
from utils import NativeScalerWithGradNormCount as NativeScaler
import utils
import modeling_pretrain
import modeling_vqkd
import yaml
from addict import Dict
from collections import OrderedDict

def get_args():
    parser = argparse.ArgumentParser('EEMo pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--save_ckpt_freq', default=20, type=int)

    # tokenizer settings
    parser.add_argument("--tokenizer_weight", type=str)
    parser.add_argument("--tokenizer_model_EEG", type=str, default="vqkd_encoder_base_decoder_3x200x12")
    parser.add_argument("--tokenizer_model_EYE", type=str, default="vqkd_encoder_base_EYE_RAW")
    
    # Model parameters
    parser.add_argument('--model', default='EEMo_base', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--rel_pos_bias', action='store_true')
    parser.add_argument('--disable_rel_pos_bias', action='store_true', dest='rel_pos_bias')
    parser.set_defaults(rel_pos_bias=False)
    parser.add_argument('--abs_pos_emb', action='store_true')
    parser.set_defaults(abs_pos_emb=True)
    parser.add_argument('--layer_scale_init_value', default=0.1, type=float, 
                        help="0.1 for base, 1e-5 for large. set 0 to disable layer scale")

    parser.add_argument('--num_mask_patches', default=75, type=int,
                        help='number of the visual tokens/patches need be masked')
    parser.add_argument('--max_mask_patches_per_block', type=int, default=None)
    parser.add_argument('--min_mask_patches_per_block', type=int, default=16)

    parser.add_argument('--input_size', default=1600, type=int,
                        help='images input size for backbone')
    parser.add_argument('--second_input_size', default=1600, type=int,
                        help='images input size for discrete vae')

    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')

    # cls-pretraining settings
    parser.add_argument('--early_layers', default=9, type=int, help='early_layers, default 9 for base and 21 for large')
    parser.add_argument('--head_layers', default=2, type=int, help='head_layers')
    parser.add_argument('--shared_lm_head', default=True, type=utils.bool_flag, help='head_layers')

    # Tokenizer parameters
    parser.add_argument('--codebook_size', default=8192, type=int, help='number of codebook')
    parser.add_argument('--codebook_dim', default=32, type=int, help='number of codebook')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt_betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--weight_decay_end', type=float, default=None, help="""Final value of the
        weight decay. We use a cosine schedule for WD. 
        (Set the same value with args.weight_decay to keep weight decay no change)""")

    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--warmup_steps', type=int, default=-1, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')

    # Augmentation parameters
    parser.add_argument('--decoupling_aug', default=False, type=utils.bool_flag, help="use decoupling aug for tokenizer and vit")
    parser.add_argument('--color_jitter', type=float, default=0.4, metavar='PCT',
                        help='Color jitter factor (default: 0.4)')
    parser.add_argument('--train_interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--second_interpolation', type=str, default='lanczos',
                        help='Interpolation for discrete vae (random, bilinear, bicubic default: "bicubic")')
    parser.add_argument('--min_crop_scale', type=float, default=0.08, metavar='PCT',
                        help='min_crop_scale (default: 0.08)')


    # Dataset parameters
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--eval_data_path', default='', type=str, help='dataset path')
    parser.add_argument('--data_set', default='image_folder',  type=str, help='dataset path')

    parser.add_argument('--imagenet_default_mean_and_std', default=False, action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false', dest='auto_resume')
    parser.set_defaults(auto_resume=True)

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')    
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    
    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--enable_align', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)

    # EEMo parameters
    parser.add_argument('--modality', default="EEG", type=str, help='modality')
    parser.add_argument('--vqkd_pretrained_weight_EEG', default='',type=str)
    parser.add_argument('--vqkd_pretrained_weight_EYE', default='',type=str)
    parser.add_argument('--is_eval',default=0,type=int)
    parser.add_argument('--eval_model',default='',type=str)
    parser.add_argument('--eval_epoch',default=0,type=int)

    parser.add_argument('--load_from_pretrain',type=str)
    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    if 'cls_pt' in args.model:
        model = create_model(
            args.model,
            pretrained=False,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            use_shared_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            vocab_size=args.codebook_size,
            early_layers=args.early_layers,
            head_layers=args.head_layers,
            shared_lm_head=args.shared_lm_head,
        )
    else:
        model = create_model(
            args.model,
            pretrained=False,
            drop_path_rate=args.drop_path,
            drop_block_rate=None,
            use_shared_rel_pos_bias=args.rel_pos_bias,
            use_abs_pos_emb=args.abs_pos_emb,
            init_values=args.layer_scale_init_value,
            vocab_size=args.codebook_size
        )

    return model

def get_tokenizer_EEG(args):
    print(f"Creating tokenizer: {args.tokenizer_model_EEG}")
    model = create_model(
            args.tokenizer_model_EEG,
            pretrained=True,
            # pretrained_weight='./checkpoint/vqkd39/checkpoint-99.pth',#args.tokenizer_weight,
            # pretrained_weight='/data/jiangweibang/result_yhl/labram/checkpoint/vqkd1/checkpoint-99.pth',
            pretrained_weight=args.vqkd_pretrained_weight_EEG,#args.tokenizer_weight,
            # pretrained_weight='/data/jiangweibang/result_yhl/labram/checkpoint/vqkd2/checkpoint-99.pth',#args.tokenizer_weight,
            # pretrained_weight='/data/jiangweibang/result_yhl/labram/checkpoint/vqkd4/checkpoint-19.pth',
            as_tokenzer=True,
            n_code=args.codebook_size, 
            code_dim=args.codebook_dim,
        ).eval()
    return model

def get_tokenizer_EYE(args):
    print(f"Creating visual tokenizer: {args.tokenizer_model_EYE}")
    model = create_model(
            args.tokenizer_model_EYE,
            pretrained=True,
            # pretrained_weight='./checkpoint/vqkd39/checkpoint-99.pth',#args.tokenizer_weight,
            # pretrained_weight='/data/jiangweibang/result_yhl/labram/checkpoint/vqkd1/checkpoint-99.pth',
            pretrained_weight=args.vqkd_pretrained_weight_EYE,#args.tokenizer_weight,
            # pretrained_weight='/data/jiangweibang/result_yhl/labram/checkpoint/vqkd2/checkpoint-99.pth',#args.tokenizer_weight,
            # pretrained_weight='/data/jiangweibang/result_yhl/labram/checkpoint/vqkd4/checkpoint-19.pth',
            as_tokenzer=True,
            n_code=args.codebook_size, 
            code_dim=args.codebook_dim,
        ).eval()
    return model

def main(args):
    utils.init_distributed_mode(args)

    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    cudnn.benchmark = True

    model = get_model(args)
    patch_size = model.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (1, args.input_size // patch_size)
    args.patch_size = patch_size

    # get dataset
    dataset_root = "/data/jiangweibang/h5Data/"
    dataset_train_list= [] 
    train_ch_names_list = []
    time_window = []

    if args.modality == "EYE":
        if args.is_eval:
            from engine_get_dataset import get_pretrain_dataset
            train_dataset,ch_name = get_pretrain_dataset("A_depression_eye", "val")
            dataset_train_list.append(train_dataset)
            train_ch_names_list.append(ch_name)
            time_window.append(4)
        else:
            from engine_get_dataset import get_pretrain_dataset
            train_dataset,ch_name = get_pretrain_dataset("A_depression_eye", "train")

            dataset_train_list.append(train_dataset)
            train_ch_names_list.append(ch_name)
            time_window.append(4)
            valid_dataset,ch_name = get_pretrain_dataset("depression_multi", "light_val")
    elif args.modality == "EEG":
        if args.is_eval:
            from engine_get_dataset import get_pretrain_dataset
            train_dataset,ch_name = get_pretrain_dataset("A_depression_EEG", "train")
            dataset_train_list.append(train_dataset)
            train_ch_names_list.append(ch_name)
            time_window.append(4)
            valid_dataset,ch_name = get_pretrain_dataset("A_depression_multi", "light_val")
        else:
            # datasets_train = [
            #     [f"{dataset_root}raweegdata.hdf5", f"{dataset_root}restingstate.hdf5"
            #     ],
            #     [
            #         f"{dataset_root}seed-neg.hdf5", f"{dataset_root}seed-sleep.hdf5", f"{dataset_root}seed-sleep-emo3.hdf5", \
            #         f"{dataset_root}seed-french.hdf5", f"{dataset_root}seed-german.hdf5", f"{dataset_root}luoshuaiart.hdf5", f"{dataset_root}angersurprise.hdf5", \
            #         f"{dataset_root}seed-sleep2.hdf5", f"{dataset_root}confidence-figure-p1.hdf5", f"{dataset_root}confidence-figure-p2.hdf5", f"{dataset_root}confidence-text.hdf5"
            #     ],
            #     [f"{dataset_root}tuhepilepsy.hdf5", f"{dataset_root}tuhartifact.hdf5", f"{dataset_root}tuhslowing.hdf5", f"{dataset_root}tuhseizure.hdf5"],
            #     [f"{dataset_root}tuhepilepsy_19.hdf5", f"{dataset_root}tuhseizure_19.hdf5"],
            #     [f"{dataset_root}tuhseizure_19_2.hdf5"],
            #     [f"{dataset_root}tuhepilepsy_21.hdf5"],
            #     [f"{dataset_root}spisrestingstate.hdf5"],
            #     [f"{dataset_root}bcicompetitioniv1.hdf5"],
            #     [f"{dataset_root}grasplifteeg.hdf5"],
            #     [f"{dataset_root}motormovementimagery.hdf5"],
            #     [f"{dataset_root}sienascalpeeg.hdf5"],
            #     [f"{dataset_root}emobrain.hdf5"],
            #     [f"{dataset_root}inriabcichallenge.hdf5"],
            #     [f"{dataset_root}targetversusnon.hdf5"],
                
            # ]
            # time_window = [
            #     4,
            #     4,
            #     11,
            #     13,
            #     13,
            #     12,
            #     4,
            #     4,
            #     8,
            #     4,
            #     8,
            #     4,
            #     4,
            #     8
            # ] # to ensure the total sequence length be around 256 for each dataset
            # dataset_train_list, train_ch_names_list = utils.build_pretraining_dataset(datasets_train, time_window, stride_size=800, start_percentage=0, end_percentage=0.8)
            
            dataset_train_list = []
            train_ch_names_list = []
            time_window = []
            from engine_get_dataset import get_pretrain_dataset
            train_dataset,ch_name = get_pretrain_dataset("A_depression_EEG", "train")

            dataset_train_list.append(train_dataset)
            train_ch_names_list.append(ch_name)
            time_window.append(4)
    elif args.modality == "multi":
        if args.is_eval:
            from engine_get_dataset import get_pretrain_dataset
            train_dataset,ch_name = get_pretrain_dataset("A_depression_multi", "val")
            dataset_train_list.append(train_dataset)
            train_ch_names_list.append(ch_name)
            time_window.append(4)
            valid_dataset,ch_name = get_pretrain_dataset("A_depression_multi", "light_val")
        else:
            from engine_get_dataset import get_pretrain_dataset
            train_dataset,ch_name = get_pretrain_dataset("A_depression_multi", "train")

            dataset_train_list.append(train_dataset)
            train_ch_names_list.append(ch_name)
            time_window.append(4)
            valid_dataset,ch_name = get_pretrain_dataset("A_depression_multi", "light_val")
    valid_dataset,ch_name = get_pretrain_dataset("A_depression_multi", "light_val")
    
    if args.modality == "multi":
        vqkd_EEG = get_tokenizer_EEG(args).to(device)
        vqkd_EYE = get_tokenizer_EYE(args).to(device)
    elif args.modality == "EEG":
        vqkd = get_tokenizer_EEG(args).to(device)
    elif args.modality == "EYE":
        vqkd = get_tokenizer_EYE(args).to(device)

    if True:  # args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_rank = global_rank
        num_training_steps_per_epoch = sum([len(dataset) for dataset in dataset_train_list]) // args.batch_size // num_tasks

        sampler_train_list = []
        for dataset in dataset_train_list:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=True
            )
            sampler_train_list.append(sampler_train)
        print("Sampler_train = %s" % str(sampler_train))
        sampler_valid = torch.utils.data.DistributedSampler(
            valid_dataset, num_replicas=num_tasks, rank=sampler_rank, shuffle=False
        )
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train_list = []
    for dataset, sampler in zip(dataset_train_list, sampler_train_list):
        data_loader_train = torch.utils.data.DataLoader(
            dataset, sampler=sampler,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        data_loader_train_list.append(data_loader_train)
    data_loader_valid = torch.utils.data.DataLoader(
        valid_dataset, sampler=sampler_valid,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    model.to(device)
    # if args.eye_unfree:
    #     pass
    # else:
    #     if args.modality == "EYE": # 冻结模型中除了MLP_EYE以外的所有参数
    #         for name, param in model.named_parameters():
    #             if "mlp_EYE" not in name:
    #                 param.requires_grad = False
    #             else:
    #                 print(name)


    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Model = %s" % str(model_without_ddp))
    print('number of params:', n_parameters)

    # print("Tokenizer = %s" % str(vqkd))
    total_batch_size = args.batch_size * utils.get_world_size() * args.gradient_accumulation_steps
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    print("Number of training steps = %d" % num_training_steps_per_epoch)
    print("Number of training examples per epoch = %d" % (total_batch_size * num_training_steps_per_epoch))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    optimizer = create_optimizer(
        args, model_without_ddp)
    loss_scaler = NativeScaler()

    print("Use step level LR & WD scheduler!")
    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=args.warmup_steps,
    )
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end, args.epochs, num_training_steps_per_epoch)
    print("Max WD = %.7f, Min WD = %.7f" % (max(wd_schedule_values), min(wd_schedule_values)))

    utils.auto_load_model(
        args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    
    if not args.resume and args.modality == "EYE":
        
        checkpoint = torch.load(args.load_from_pretrain, map_location='cpu')
        checkpoint = checkpoint["model"]
    
        model_without_ddp.load_state_dict(checkpoint,strict=False)
    
    if not args.resume and args.modality == "multi":
        
            if args.load_from_pretrain is not None:
                checkpoint = torch.load(args.load_from_pretrain, map_location='cpu')
                checkpoint_model = checkpoint['model']
                try:
                    model_without_ddp.load_state_dict(checkpoint_model,strict=True)
                except:
                    print("load model error\n"*50)
                    model_without_ddp.load_state_dict(checkpoint_model,strict=False)

    if args.is_eval:
        if args.modality == "multi":
            checkpoint = torch.load(args.eval_model, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])

            eval_results = evaluate_EEMo_multi(vqkd_EEG,vqkd_EYE,data_loader_train_list[0], model, device, header='Test:', ch_names=train_ch_names_list[0], metrics=['acc'], is_binary=True,args=args)
            if global_rank == 0:
                if args.epochs == 11:
                    with open(f"results/multi_valid_{args.model}_ab3.txt", mode="a", encoding="utf-8") as f:
                        f.write(f"{args.eval_epoch}:"+str(eval_results["loss"]) + "\n")
                        f.write(f"{args.eval_epoch}:"+str(eval_results["loss_itc"]) + "\n")
                        f.write(f"{args.eval_epoch}:"+str(eval_results["loss_itm"]) + "\n")
                        f.write(f"{args.eval_epoch}:"+str(eval_results["loss_MLM"]) + "\n \n")
                else:
                    with open(f"results/multi_valid_{args.model}.txt", mode="a", encoding="utf-8") as f:
                        f.write(f"{args.eval_epoch}:"+str(eval_results["loss"]) + "\n")
                        f.write(f"{args.eval_epoch}:"+str(eval_results["loss_itc"]) + "\n")
                        f.write(f"{args.eval_epoch}:"+str(eval_results["loss_itm"]) + "\n")
                        f.write(f"{args.eval_epoch}:"+str(eval_results["loss_MLM"]) + "\n \n")
        else:
            checkpoint = torch.load(args.eval_model, map_location='cpu')
            model_without_ddp.load_state_dict(checkpoint['model'])

            eval_results = evaluate_EEMo(vqkd,data_loader_train_list[0], model, device, header='Test:', ch_names=train_ch_names_list[0], metrics=['acc'], is_binary=True,args=args)
            if global_rank == 0:
                with open(f"results/valid_{args.model}.txt", mode="a", encoding="utf-8") as f:
                    f.write(f"{args.eval_epoch}:"+str(eval_results["loss"]) + "\n")
        return
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()




    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            for data_loader_train in data_loader_train_list:
                data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)
        if args.modality == "multi":
            train_stats = train_one_epoch_EEMo_multi_valid(
                model, vqkd_EEG,vqkd_EYE, data_loader_train_list,data_loader_valid,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                ch_names_list=train_ch_names_list,
                args=args,
            )
        else:
            train_stats = train_one_epoch_EEMo(
                model, vqkd, data_loader_train_list,
                optimizer, device, epoch, loss_scaler,
                args.clip_grad, log_writer=log_writer,
                start_steps=epoch * num_training_steps_per_epoch,
                lr_schedule_values=lr_schedule_values,
                wd_schedule_values=wd_schedule_values,
                ch_names_list=train_ch_names_list,
                args=args,
            )
        if args.output_dir:
            utils.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, save_ckpt_freq=args.save_ckpt_freq)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, 'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
