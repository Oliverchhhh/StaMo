CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fabric run model train_renderer.py \
    --config_path configs/libero.yaml \
    --strategy='deepspeed_stage_2' \
    --devices=8 \
    --accelerator=cuda \
    --precision="bf16-true" \
    --main-port=52444
