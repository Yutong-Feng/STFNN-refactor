# STFNN

[English](README.md) | 中文

STFNN（时空场神经网络）用于站点级空气质量插值：用目标站点在时间和空间上最近的已观测邻居来估计它的值。模型学习时空场的梯度，沿每个邻居到目标站点的路径对梯度积分，再用学到的权重把各个邻居的估计值加权组合起来。

论文：[Spatio-Temporal Field Neural Networks for Air Quality Inference](https://doi.org/10.24963/ijcai.2024/803)，IJCAI 2024（AI for Good track）。

> [!NOTE]
> **本代码是 STFNN 论文作者发布的官方重构版本**，重构过程借助了 AI，并与原始实现做过对照检查：
>
> - **模型**：加载同一份原始权重后，两份实现的预测结果一致（最大差约 1e-5，属于浮点误差），对照的是原始实验全部使用的配置（`st_embed=simple`、`dym_graph=true`、GELU）。
> - **邻居采样**：在相同的随机掩码下，采样出的数据逐元素相同。
>
> 数据处理和评估流程有意做了改动，因此结果不能和原始代码的结果直接比较：
>
> - **修复**：邻居的时间戳与其数据行对应；训练集与验证集的时间窗口不再重叠；归一化只用训练时段的统计量；评估指标对所有查询点统一计算，不再按 batch 平均（原来的 RMSE 偏低）；评估掩码由种子固定；IDW 权重改为与距离成反比；布尔配置项可以在命令行关闭；坐标重合的站点只保留一个。
> - **删除**：`poly` / `linear` 时空嵌入、SELU 激活、36 站点的小数据集，原始实验都没有用到；`mm_size` 分块计算，改用 `data.batch_size` 加 `train.acc_steps` 代替。
> - **新增**：Hydra 配置管理、`inductive` 划分方式、early stopping、每个 epoch 输出验证集和测试集指标、`evaluate.py`。

## 安装

```bash
pip install -r requirements.txt
```

需要 Python 3.10 及以上版本。按默认 batch size 训练需要一块 CUDA GPU，约占 11 GiB 显存。

### 数据

| 文件 | 内容 |
|---|---|
| `air_2018.npy` | (时间, 站点, 16) 的浮点数组，缺失值为 NaN |
| `air_2018_locations.csv` | 各站点的 `longitude`、`latitude`，每行一个站点，顺序与 npy 的站点维一致 |

`air_2018.npy` 的 16 个特征：

| 下标 | 类别 | 特征 |
|---|---|---|
| 0-5 | 空气质量 | PM2.5、PM10、NO2、CO、O3、SO2 |
| 6-12 | 气象 | 降雨、温度、气压、湿度、风速、风向、天气 |
| 13-15 | 时间信息 | 小时、星期、是否节假日 |

数据集不包含在本仓库中。默认从 `~/dataset/air_2018.npy` 和 `~/dataset/air_2018_locations.csv` 读取；数据放在其他位置时，在 `configs/config.yaml` 或命令行里修改 `data.data_path` 和 `data.loc_path`。

## 运行

用默认配置训练：

```bash
python train.py
```

所有超参数都在 `configs/config.yaml` 里，可以用 Hydra 语法在命令行覆盖：

```bash
python train.py data.batch_size=2 train.acc_steps=8   # 少占显存，有效 batch size 不变
python train.py data.split=inductive                  # 除了时间，也留出站点
python train.py -m model.max_step=2,4,8,16            # 批量实验，每个取值跑一次
```

常用配置项：

| 配置项 | 默认值 | 含义 |
|---|---|---|
| `data.label` | `pm25` | 预测目标：`pm25`、`pm10`、`no2`、`co`、`o3`、`so2`、`rain`、`temp`、`press` |
| `data.split` | `transductive` | `transductive`：训练、验证、测试只按时间划分，所有站点都参与训练；`inductive`：验证站点和测试站点在训练时也完全不出现 |
| `data.mask_ratio` | `0.25` | 每个样本中被掩码、作为预测目标的站点比例 |
| `train.epochs` | `200` | 最大 epoch 数 |
| `train.patience` | `20` | 验证集 MAE 连续这么多个 epoch 没有改善就提前停止 |
| `seed` | `42` | 固定数据划分、掩码、数据打乱和模型初始化的随机性 |

数据按时间顺序划分为训练、验证、测试三段，各占 60%、20%、20%。每个 epoch 都会在日志里输出验证集和测试集的 MAE、MAPE、RMSE；只有验证集的 MAE 用来挑选模型和触发提前停止。训练结束后，用最佳模型在测试集上再评估一次。

每次运行会把配置（`.hydra/`）、日志 `train.log` 和最佳模型 `model.pth` 保存到 `outputs/<日期>/<时间>/`（批量实验保存到 `multirun/`）。

用某次运行的配置重新评估它保存的模型：

```bash
python evaluate.py run_dir=outputs/<日期>/<时间>
```

## 目录结构

```
configs/config.yaml   超参数配置
data.py               数据读取、归一化、数据划分、邻居采样
model.py              STFNN 模型
train.py              训练入口
evaluate.py           评估已保存的模型
requirements.txt      依赖
```

## 引用

如果使用了本代码，请引用原论文：

```bibtex
@inproceedings{ijcai2024p803,
  title     = {Spatio-Temporal Field Neural Networks for Air Quality Inference},
  author    = {Feng, Yutong and Wang, Qiongyan and Xia, Yutong and Huang, Junlin and Zhong, Siru and Liang, Yuxuan},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {7260--7268},
  year      = {2024},
  month     = {8},
  note      = {AI for Good},
  doi       = {10.24963/ijcai.2024/803},
  url       = {https://doi.org/10.24963/ijcai.2024/803},
}
```
