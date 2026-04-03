export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export MODEL_PATH=''
export DATA_PATH=''
export SAVE_PATH=''


swift rlhf \
    --rlhf_type dpo \
    --model $MODEL_PATH \
    --dataset $DATA_PATH \
    --output_dir $SAVE_PATH \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --save_strategy 'steps' \
    --save_steps 20 \
    --save_only_model true \
    --save_total_limit 1 \
    --eval_strategy 'steps' \
    --eval_steps 20 \
    --logging_steps 1 \
    --max_length 8192 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --deepspeed zero3 \
    --report_to tensorboard \
