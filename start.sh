NCCL_IB_CUDA_SUPPORT=1 NCCL_NET_GDR_LEVEL=SYS NCCL_IB_GID_INDEX=3  \
torchrun --nproc_per_node=4 \
    finetune_llama2.py \
    --model_name_or_path /home/zhongyuting/model/Sheared-LLaMA-1.3B \
    --data_path /mnt/jfs/wangbinluo/Finetune_llama2/1024.json \
    --num_train_epochs 2 \
    --per_device_train_batch_size 2 \
    --optimizer "adam" \
    --learning_rate 2e-5 \
    --adam_beta1 "0.9" \
    --adam_beta2 "0.95" \
    --weight_decay 0.1 \
    --lr_scheduler "linear" \
    --gradient_checkpointing True \
    --auto_cast False \
    --model_max_length 2048 \
    --use_ddp False\
    --tp 2 \
    --tensorboard_path /home/wangbinluo/Finetune_llama2/tensorboard \