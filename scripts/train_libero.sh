CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 fabric run model train_renderer.py \
    --config_path configs/libero.yaml \
    --strategy='configs/ds_stage2.json' \
    --devices=8 \
    --accelerator=cuda \
    --precision="bf16-true" \
    --main-port=52444
