## 第一步：下载代码、数据和权重

1. 下载代码

   ```
   git clone https://github.com/Oliverchhhh/StaMo.git
   ```

2. 下载权重

   ```
   cd StaMo/checkpoints
   第一个权重:
   git clone https://huggingface.co/timm/vit_base_patch14_reg4_dinov2.lvd142m
   
   第二个权重：
   git clone https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers
   
   cd ..
   ```

3. 下载数据

   ```
   ## 此时处于StaMo/目录下
   conda create -n stamo python=3.10 -y
   conda activate stamo
   pip install -e .
   pip install opencv-python
   pip install oss2
   
   # 在StaMo目录下运行
   python scripts/download_data.py
   ```

## 第二步：数据处理

__1. 生成图片数据集__

```
# 在StaMo目录下运行
python scripts/create_datasets.py
```

__2. 生成图片数据集索引__

```
# 在StaMo目录下运行
python scripts/create_jsons.py
```

## 第三步：检验

此时`StaMo`目录下应该存在`jsons`，`cuphead`，`datasets`目录，此时数据下载完毕



## 第四步：训练

```
# 在StaMo/目录下

bash scripts/train_libero.sh
```

