export NCCL_DEBUG=ERROR

# only depression
OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 --master_port=29504 main_pretraining.py \
        --output_dir /data/jiangweibang/result_yhl/labram/checkpoint/pretrain_EEMo_multi_demo3 \
        --log_dir log/pretrain/EEMo_multi_demo3 \
        --model EEMo_base \
        --shared_lm_head True \
        --early_layers 9 \
        --head_layers 4 \
        --num_mask_patches 75 \
        --second_input_size 224 \
        --second_interpolation bicubic \
        --min_crop_scale 0.2 \
        --batch_size 64 \
        --lr 5e-4 \
        --warmup_epochs 5 \
        --clip_grad 3.0 \
        --drop_path 0. \
        --layer_scale_init_value 0.1 \
        --imagenet_default_mean_and_std \
        --opt_betas 0.9 0.98 \
        --opt_eps 1e-8  \
        --epochs 50 \
        --save_ckpt_freq 1 \
        --enable_align \
        --codebook_dim 64 \
        --gradient_accumulation_steps 1 \
        --modality multi \
        --tokenizer_model_EEG vqkd_encoder_base_decoder_3x200x12 \
        --vqkd_pretrained_weight_EEG /data/jiangweibang/result_yhl/labram/checkpoint/vqkd_demo/checkpoint.pth \
        --tokenizer_model_EYE vqkd_encoder_base_EYE_RAW \
        --vqkd_pretrained_weight_EYE /data/jiangweibang/result_yhl/labram/checkpoint/vqkd_eye_demo/checkpoint.pth \