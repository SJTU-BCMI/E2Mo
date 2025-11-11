export NCCL_DEBUG=ERROR
dataset_name="depression_multi"
# for seed in {0}
# do
seed=0
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 --master_port=29515 main_finetuning.py \
        --output_dir /data/jiangweibang/result_yhl/labram/checkpoint/finetune_${dataset_name}_base_multi_demo_${seed}/ \
        --log_dir log/finetune/${dataset_name}_base_multi_demo_${seed}/ \
        --model EEMo_multi_base_finetune \
        --weight_decay 0.05 \
        --finetune /data/jiangweibang/result_yhl/labram/checkpoint/pretrain_EEMo_multi_base16/checkpoint-19.pth \
        --batch_size 64 \
        --lr 1e-6 \
        --update_freq 1 \
        --warmup_epochs 5 \
        --epochs 50 \
        --layer_decay 0.65 \
        --drop_path 0.1 \
        --mixup 0. \
        --cutmix 0. \
        --imagenet_default_mean_and_std \
        --dist_eval \
        --save_ckpt_freq 20 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset ${dataset_name} \
        --disable_qkv_bias \
        --model_filter_name gzp \
        --modality "multi" \
        --seed ${seed}
