
export NCCL_DEBUG=ERROR


# EYE base
dataset_name="depression_eye"
for seed in 0
do
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=1 torchrun --nnodes=1 --nproc_per_node=8 --master_port=29512 main_finetuning.py \
        --output_dir /data/jiangweibang/result_yhl/labram/checkpoint/finetune_${dataset_name}_base_eye_demo_${seed}/ \
        --log_dir log/finetune/${dataset_name}_base_eye_demo_${seed}/ \
        --model EEMo_EYE_base_finetune \
        --weight_decay 0.05 \
        --batch_size 64 \
        --lr 1e-6 \
        --finetune /data/jiangweibang/result_yhl/labram/checkpoint/pretrain_EEMo_EYE_base11_unfree/checkpoint-49.pth \
        --update_freq 1 \
        --warmup_epochs 5 \
        --drop 0. \
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
        --modality "EYE" \
        --seed ${seed}
wait
done