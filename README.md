# Hydra+MultiROCKET 时间序列分类

这个项目是一个使用 Hydra+MultiROCKET 进行时间序列分类的演示项目。它基于 aeon 库中的 MultiRocketHydraClassifier 实现，提供了完整的训练和推理流程。

## 项目结构

```
.
├── config/                 # 配置文件目录
│   └── default.yaml        # 默认配置文件
├── src/                    # 源代码目录
│   ├── config/             # 配置模块
│   ├── data/               # 数据处理模块
│   ├── models/             # 模型实现模块
│   ├── utils/              # 工具函数模块
│   └── visualization/      # 可视化模块
├── train.py                # 训练脚本
├── predict.py              # 推理脚本
└── requirements.txt        # 依赖项
```

## 安装

1. 克隆仓库：

```bash
git clone https://github.com/yourusername/hydra-multirocket.git
cd hydra-multirocket
```

2. 创建虚拟环境并安装依赖：

```bash
# 使用 conda
conda create -n hydra-multirocket python=3.8
conda activate hydra-multirocket
pip install -r requirements.txt

# 或者使用 venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## 使用方法

### 训练模型

使用默认配置训练模型：

```bash
python train.py
```

使用自定义配置文件：

```bash
python train.py --config path/to/your/config.yaml
```

覆盖配置文件中的某些参数：

```bash
python train.py --dataset basic_motions --model_type multirocket_hydra --output_dir output/my_experiment
```

### 推理

使用训练好的模型进行推理：

```bash
python predict.py --model_path output/multirocket_hydra_20230101_120000/model.pkl
```

使用自定义数据集：

```bash
python predict.py --model_path output/multirocket_hydra_20230101_120000/model.pkl --custom_dataset_path path/to/your/dataset.ts
```

## 配置文件

配置文件使用 YAML 格式，包含以下主要部分：

- `data`: 数据配置
  - `dataset_name`: 数据集名称，可选 "basic_motions" 或 "italy_power_demand"
  - `custom_dataset_path`: 自定义数据集路径
  - `test_size`: 测试集比例
  - `random_state`: 随机种子

- `model`: 模型配置
  - `model_type`: 模型类型，可选 "hydra", "multirocket" 或 "multirocket_hydra"
  - `n_kernels`: 卷积核数量
  - `max_dilations_per_kernel`: 每个卷积核的最大扩张数
  - `n_features_per_kernel`: 每个卷积核的特征数
  - `random_state`: 随机种子

- `training`: 训练配置
  - `epochs`: 训练轮数
  - `verbose`: 是否显示详细信息
  - `save_best`: 是否保存最佳模型

- `output_dir`: 输出目录

## 模型说明

### Hydra

Hydra (HYbrid Dictionary-Rocket Architecture) 是一种使用竞争卷积核的时间序列分类方法。它将输入时间序列通过一组随机卷积核进行转换，这些卷积核被安排成多个组，每组有多个核。在每个时间点，Hydra 计算每个组中与输入时间序列最接近的卷积核。

### MultiROCKET

MultiROCKET 是 ROCKET (RandOm Convolutional KErnel Transform) 的扩展版本，它添加了更多的池化操作，从每个卷积核中提取更多特征。除了最大值和正值比例外，MultiROCKET 还添加了正值均值、正值索引均值和最长正值序列等特征。

### MultiRocketHydra

MultiRocketHydra 是将 Hydra 和 MultiROCKET 的结果结合起来的混合模型。它将两者的特征连接起来，然后使用 RidgeClassifierCV 进行训练。这种组合利用了两种方法的优势，通常能够获得更好的分类性能。

## 参考文献

1. Dempster, A., Schmidt, D.F. and Webb, G.I. (2023) Hydra: Competing convolutional kernels for fast and accurate time series classification. [Journal Paper](https://link.springer.com/article/10.1007/s10618-023-00939-3)

2. Cahng Wei T, Dempster A, Bergmeir C and Webb G (2022) MultiRocket: multiple pooling operators and transformations for fast and effective time series classification [Journal Paper](https://link.springer.com/article/10.1007/s10618-022-00844-1)

3. Dempster A, Petitjean F and Webb GI (2019) ROCKET: Exceptionally fast and accurate time series classification using random convolutional kernels. [Journal Paper](https://link.springer.com/article/10.1007/s10618-020-00701-z)
