export NCCL_DEBUG=ERROR
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 main_tokenizer_training.py \
    --output_dir /data/jiangweibang/result_yhl/labram/checkpoint/vqkd_demo \
    --log_dir "log/vqkd_demo/" \
    --process_type default \
    --train_interpolation bicubic \
    --min_crop_scale 0.08 \
    --model vqkd_encoder_base_decoder_3x200x12 \
    --codebook_n_emd 8192 \
    --codebook_emd_dim 64 \
    --quantize_kmeans_init \
    --rec_loss_type cosine \
    --batch_size 128 \
    --opt adamw \
    --opt_betas 0.9 0.99 \
    --weight_decay 1e-4  \
    --warmup_epochs 10 \
    --epochs 50 \
    --save_ckpt_freq 5 \
