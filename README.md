# 🤖 StaMo: Unsupervised Learning of Generalizable Robot Motion from Compact State Representation



## 🚀 Quick Start

### 🛠️ Installation

1. **Create and activate the conda environment:**
   ```bash
   conda create -n stamo python=3.10 -y
   conda activate stamo
   ```

2. **Install the package:**
   ```bash
   cd StaMo && pip install -e .
   ```

## 🎯 Usage

### 🎨 Diffusion AutoEncoder



#### 📊 Step 1: Data Format Conversion

1. **Download robotic data** in advance and extract them into image format
2. **Convert to JSON format** using our provided script:

   ```bash
   python scripts/create_jsons.py
   ```

#### 🏋️ Step 2: Model Training

1. **Configure your setup** (optional):
   - Modify configuration files according to your VRAM requirements
   - Adjust training parameters as needed

2. **Start training:**
   ```bash
   bash scripts/train_libero.sh
   ```

3. **Monitor training progress:**
   ```bash
   tensorboard --logdir .
   ```


#### 📈 Step 3: Validation

Validate your trained model and results:

```bash
python validate_renderer.py
```


## 📚 Citation

If you use this work in your research, please cite our paper:

```bibtex
@article{liu2025stamo,
  title={StaMo: Unsupervised Learning of Generalizable Robotic Motions from Static Images},
  author={Liu, Mingyu and Shu, Jiuhe and Chen, Hui and Li, Zeju and Zhao, Canyu and Yang, Jiange and Gao, Shenyuan and Chen, Hao and Shen, Chunhua},
  journal={arXiv preprint arXiv:2510.05057},
  year={2025}
}

@article{zhao2024moviedreamer,
  title={Moviedreamer: Hierarchical generation for coherent long visual sequence},
  author={Zhao, Canyu and Liu, Mingyu and Wang, Wen and Chen, Weihua and Wang, Fan and Chen, Hao and Zhang, Bo and Shen, Chunhua},
  journal={arXiv preprint arXiv:2407.16655},
  year={2024}
}
```

## 🎫 License

For academic use, this project is licensed under [the 2-clause BSD License](https://opensource.org/license/bsd-2-clause). 
For commercial use, please contact [Chunhua Shen](mailto:chhshen@gmail.com).
