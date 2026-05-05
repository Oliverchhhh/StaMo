CUDA_VISIBLE_DEVICES=0 fabric run model validate_renderer.py \
    --config_path configs/eval.yaml \
    --devices=1 \
    --accelerator=cuda \
    --precision="32"
