# Hydra+MultiROCKET 项目工作流程指南

本文档详细描述了Hydra+MultiROCKET项目的完整工作流程，从数据获取到模型训练再到预测，包括每个步骤需要运行的脚本和参数。

## 目录

- [完整工作流程](#完整工作流程)
  - [第一步：数据准备](#第一步数据准备)
  - [第二步：数据处理（可选）](#第二步数据处理可选)
  - [第三步：模型训练](#第三步模型训练)
  - [第四步：模型预测](#第四步模型预测)
  - [第五步：模型比较（可选）](#第五步模型比较可选)
- [完整流程示例](#完整流程示例)
- [配置文件说明](#配置文件说明)
- [项目结构](#项目结构)

## 完整工作流程

### 第一步：数据准备

**脚本**：`prepare_data.py`

**功能**：
- 从币安API获取历史K线数据
- 计算移动平均线
- 识别金叉/死叉信号
- 标注信号成功/失败/无效
- 提取特征
- 划分数据集
- 保存处理后的数据集

**运行命令**：
```bash
python prepare_data.py --output_dir output/prepared_data
```

**主要参数**：
- `--symbol`：交易对，默认为"BTCUSDT"
- `--interval`：K线间隔，默认为"15m"
- `--start_date`：开始日期，默认为"2023-01-01"
- `--end_date`：结束日期，默认为"2023-12-31"
- `--ma_short`：短期移动平均线周期，默认为20
- `--ma_long`：长期移动平均线周期，默认为50
- `--profit_target`：盈利目标，默认为0.005（0.5%）
- `--stop_loss`：止损水平，默认为0.005（0.5%）
- `--segment_length`：时间序列片段长度，默认为40
- `--output_dir`：输出目录，默认为"output/prepared_data"

**输出**：
- `output/prepared_data/train_dataset.npz`：训练集
- `output/prepared_data/val_dataset.npz`：验证集
- `output/prepared_data/test_dataset.npz`：测试集
- `output/prepared_data/BTCUSDT_15m_signals_distribution.png`：信号分布图
- `output/prepared_data/config.txt`：配置信息

### 第二步：数据处理（可选）

#### 2.1 修复数据中的缺失值

**脚本**：`check_and_fix_data.py`

**功能**：
- 检查数据中是否有缺失值
- 修复缺失值（使用前向填充或均值填充）

**运行命令**：
```bash
python check_and_fix_data.py
```

**输出**：
- 修复后的数据集（覆盖原文件）

#### 2.2 转换为二分类数据（可选）

**脚本**：`convert_to_binary.py`

**功能**：
- 将三分类数据（成功/失败/无效）转换为二分类数据（成功/失败）
- 移除标签为2（无效）的样本

**运行命令**：
```bash
python convert_to_binary.py
```

**输出**：
- `output/prepared_data/binary/train_dataset.npz`：二分类训练集
- `output/prepared_data/binary/val_dataset.npz`：二分类验证集
- `output/prepared_data/binary/test_dataset.npz`：二分类测试集

<!-- 已移除数据转置步骤，因为数据生成时已经是正确格式 -->

### 第三步：模型训练

#### 3.1 三分类模型训练

**脚本**：`train.py`

**功能**：
- 加载数据
- 创建并训练Hydra+MultiROCKET模型
- 评估模型性能
- 保存模型

**运行命令**：
```bash
python train.py --config config/default.yaml
```

**输出**：
- `output/models/multirocket_hydra_[timestamp]/model.pkl`：训练好的模型
- `output/models/multirocket_hydra_[timestamp]/confusion_matrix.png`：混淆矩阵
- `output/models/multirocket_hydra_[timestamp]/time_series_samples.png`：时间序列样本图
- `output/models/multirocket_hydra_[timestamp]/config.yaml`：配置文件

#### 3.2 二分类模型训练

**脚本**：`train.py`

**功能**：
- 加载二分类数据
- 创建并训练Hydra+MultiROCKET模型
- 评估模型性能
- 保存模型

**运行命令**：
```bash
python train.py --config config/binary.yaml
```

**输出**：
- `output/models/binary/multirocket_hydra_[timestamp]/model.pkl`：训练好的模型
- `output/models/binary/multirocket_hydra_[timestamp]/confusion_matrix.png`：混淆矩阵
- `output/models/binary/multirocket_hydra_[timestamp]/time_series_samples.png`：时间序列样本图
- `output/models/binary/multirocket_hydra_[timestamp]/config.yaml`：配置文件

#### 3.3 优化参数的二分类模型训练

**脚本**：`train.py`

**功能**：
- 使用优化的参数训练二分类模型

**运行命令**：
```bash
python train.py --config config/binary_optimized.yaml
```

**输出**：
- `output/models/binary_optimized/multirocket_hydra_[timestamp]/model.pkl`：训练好的模型
- 其他输出文件（同上）

### 第四步：模型预测

**脚本**：`predict.py`

**功能**：
- 加载训练好的模型
- 对测试数据进行预测
- 评估预测结果

**运行命令**：
```bash
python predict.py --model_path [model_path] --custom_dataset_path [test_data_path]
```

**参数**：
- `--model_path`：模型路径，例如 "output/models/binary/multirocket_hydra_[timestamp]/model.pkl"
- `--custom_dataset_path`：测试数据路径，例如 "output/prepared_data/binary/test_dataset.npz"

**输出**：
- 控制台输出：评估指标（准确率、精确率、召回率、F1分数）
- 控制台输出：分类报告
- 控制台输出：混淆矩阵

### 第五步：模型比较（可选）

#### 5.1 比较三分类和二分类模型

**脚本**：`compare_models.py`

**功能**：
- 评估三分类模型和二分类模型的性能
- 比较两种模型的优缺点

**运行命令**：
```bash
python compare_models.py
```

**输出**：
- 控制台输出：两种模型的评估指标
- `output/comparison/model_comparison.png`：比较图表

#### 5.2 比较不同参数的二分类模型

**脚本**：`compare_binary_models.py`

**功能**：
- 比较原始二分类模型和优化参数后的二分类模型

**运行命令**：
```bash
python compare_binary_models.py
```

**输出**：
- 控制台输出：两种模型的评估指标
- `output/comparison/binary_models_comparison.png`：比较图表

## 完整流程示例

如果您想从头到尾完整运行一次项目，可以按照以下步骤操作：

1. **准备数据**：
   ```bash
   python prepare_data.py --output_dir output/prepared_data
   ```

2. **检查并修复数据中的缺失值**：
   ```bash
   python check_and_fix_data.py
   ```

3. **转换为二分类数据**（如果需要）：
   ```bash
   python convert_to_binary.py
   ```

<!-- 已移除数据转置步骤，因为数据生成时已经是正确格式 -->

5. **训练模型**（选择一种配置）：
   ```bash
   # 三分类模型
   python train.py --config config/default.yaml

   # 或者二分类模型
   python train.py --config config/binary.yaml

   # 或者优化参数的二分类模型
   python train.py --config config/binary_optimized.yaml
   ```

6. **使用模型进行预测**：
   ```bash
   # 使用最新训练的模型（需要替换为实际的模型路径）
   python predict.py --model_path output/models/binary/multirocket_hydra_[timestamp]/model.pkl --custom_dataset_path output/prepared_data/binary/test_dataset.npz
   ```

7. **比较不同模型**（可选）：
   ```bash
   # 比较三分类和二分类模型
   python compare_models.py

   # 比较不同参数的二分类模型
   python compare_binary_models.py
   ```

## 配置文件说明

项目中使用了多个配置文件，它们位于 `config/` 目录下：

1. **default.yaml**：默认配置，用于三分类模型
2. **binary.yaml**：二分类模型配置
3. **binary_optimized.yaml**：优化参数的二分类模型配置

您可以根据需要修改这些配置文件中的参数，如卷积核数量、特征数等。

## 项目结构

```
HydraMultiROCKET/
├── config/                       # 配置文件目录
│   ├── default.yaml              # 默认配置（三分类）
│   ├── binary.yaml               # 二分类配置
│   └── binary_optimized.yaml     # 优化参数的二分类配置
├── data_processing/              # 数据处理模块
│   ├── __init__.py
│   ├── binance_data_fetcher.py   # 获取币安历史数据
│   ├── signal_generator.py       # 生成交易信号和标签
│   ├── feature_extractor.py      # 提取特征
│   └── data_normalizer.py        # 数据标准化
├── src/                          # 源代码目录
│   ├── data/                     # 数据加载模块
│   ├── models/                   # 模型定义
│   ├── utils/                    # 工具函数
│   ├── visualization/            # 可视化模块
│   └── config/                   # 配置模块
├── output/                       # 输出目录
│   ├── prepared_data/            # 准备好的数据
│   │   ├── binary/               # 二分类数据
│   │   └── ...
│   ├── models/                   # 训练好的模型
│   │   ├── binary/               # 二分类模型
│   │   ├── binary_optimized/     # 优化参数的二分类模型
│   │   └── ...
│   └── comparison/               # 模型比较结果
├── prepare_data.py               # 数据准备脚本
├── check_and_fix_data.py         # 检查和修复数据脚本
├── convert_to_binary.py          # 转换为二分类数据脚本
├── train.py                      # 模型训练脚本
├── predict.py                    # 模型预测脚本
├── compare_models.py             # 比较三分类和二分类模型脚本
├── compare_binary_models.py      # 比较不同参数的二分类模型脚本
├── requirements.txt              # 项目依赖
└── WorkflowGuide.md              # 本工作流程指南
```

---

*注：本文档中的时间戳 `[timestamp]` 表示模型训练时的时间戳，实际使用时需要替换为具体的时间戳。*
