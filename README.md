# 面向隐私保护的车联网入侵检测 代码文档

本仓库是论文 **Towards Privacy-Preserving Intrusion Detection in IoV: a Federated Learning Approach with Adaptive Differential Privacy and Dynamic Aggregation** 的实验代码。

本项目面向车联网入侵检测，支持多客户端联邦训练及以下实验组件：

1. **动态聚合**：使用 FedDFW 调整客户端聚合权重。
2. **参数更新一致性筛选（PUC）**：根据参数更新方向筛选参数更新。
3. **差分隐私训练**：采用固定阈值裁剪，或启用 AdaCliP 自适应裁剪。

同时提供 FedAvg、FedDisco 和 FedProx 基线配置，方便开展对比实验。




---

## 1. 环境依赖

项目依赖统一列在 `requirements.txt` 中。创建并激活独立环境后，在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

> 说明：GPU 环境需要安装与本机 CUDA 环境相匹配的 PyTorch 和 torchvision。依赖版本目前未锁定；没有可用 GPU 时，将命令中的 `--device cuda` 改为 `--device cpu`。

DP 和 AdaCliP 分支需要 `backpack-for-pytorch` 计算逐样本梯度。`opacus` 用于保留的 `DPReparamWrapper` 类，不替代当前示例所需的 BackPACK。



## 2. 数据集说明

本项目适用于多种车联网入侵检测数据集，请自行准备数据集并进行数据清洗与特征选择。当前数据路径定义在 `datasets.py` 中，相对于**运行命令时的工作目录**解析，请从 `FedDFW` 仓库根目录启动训练。


以下示例使用 `CIC-IoV2024`：

```text
FL/
|-- FedDFW/
|   |-- main.py
|   |-- args.py
|   |-- datasets.py
|   `-- README.md
`-- dataset/
    `-- CICIoV2024_Decimal/
        `-- CIC_IOV.csv
```

加载路径为 `../dataset/CICIoV2024_Decimal/CIC_IOV.csv`。若使用其他目录，请修改 `datasets.py` 对应分支的 `data_path`。



---

## 3. 快速开始

准备好环境与数据后，可以先运行少量通信轮次检查基础训练流程：

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method fedavg --T 3 --E 1 --device cuda --puc 0 --random_seed 1 --iid 0 --dirichlet_alpha 0.5 --node_num 20 --use_dp 0 --use_sparse_adaclip 0
```

此命令只减少通信轮数，仍加载完整数据。

---

## 4. 联邦学习


**训练流程：**

1. 加载数据集并划分训练与测试数据，按 IID 或 Dirichlet Non-IID 配置划分客户端。
2. 初始化服务端和客户端模型；启用 AdaCliP 时执行重要性预热。
3. 每轮同步全局参数，执行客户端本地训练及可选的 PUC、DP 处理。
4. 根据 FedAvg、FedDisco 或 FedDFW 配置聚合客户端模型；FedProx 在本地训练阶段生效。
5. 评估模型并记录训练损失、准确率和分类指标。


入口：`main.py`。

```bash
python main.py [选项]
```

### 4.1 方法与配置

本项目通过命令行参数组合不同方法，而不是为每种方法提供单独的训练入口：

| 方法或组件 | 配置 | 说明 |
| --- | --- | --- |
| FedAvg | `--server_method fedavg` | 基于数据量的模型聚合基线 |
| FedDisco | `--server_method fedavg --disco 1` | 根据客户端标签分布差异调整聚合权重 |
| FedProx | `--server_method fedavg --client_method fedprox` | 在客户端目标中加入近端正则项 |
| FedDFW | `--server_method feddfw` | 动态调整服务端聚合权重 |
| PUC | `--server_method feddfw --puc 1` | Parameter Update Consistency，根据参数更新方向生成一致性筛选掩码 |
| DP | `--use_dp 1 --use_sparse_adaclip 0` | 使用固定裁剪阈值的裁剪与加噪训练分支 |
| AdaClip | `--use_dp 1 --use_sparse_adaclip 1` | 对应代码中的 AdaCliP，包含稀疏化与自适应裁剪 |

注意：

- PUC 仅在 FedDFW 分支中生效。`fedavg --puc 1` 不会启用 PUC，FedAvg 示例统一设置 `--puc 0`。
- AdaCliP 必须同时设置 `--use_dp 1`，单独设置 `--use_sparse_adaclip 1` 不会进入该训练分支。
- 以下命令用于说明配置与消融组合，不代表论文所有实验的最终超参数。

### 4.2 命令行参数

默认值来自 `args.py`，并不等同于下面示例的实验设置。

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--dataset` | str | `CIC-IDS2017` | 数据集名称 |
| `--local_model` | str | `MLP` | 本地模型；以下示例使用 `CNN` |
| `--server_method` | str | `feddfw` | 聚合方法：`fedavg` 或 `feddfw` |
| `--client_method` | str | `local_train` | 本地训练方法：`local_train` 或 `fedprox` |
| `--T` | int | `3` | 通信轮数 |
| `--E` | int | `1` | 每轮本地训练 epoch 数 |
| `--node_num` | int | `10` | 客户端数量 |
| `--random_seed` | int | `1` | 实验随机种子 |
| `--iid` | int | `1` | `1` 为 IID，`0` 为 Non-IID |
| `--dirichlet_alpha` | float | `1.0` | Dirichlet分布划分参数 |
| `--batchsize` | int | `32` | 本地训练批大小 |
| `--lr` | float | `0.001` | 客户端学习率 |
| `--optimizer` | str | `sgd` | 客户端优化器 |
| `--puc` | int | `1` | FedDFW 的 PUC 开关 |
| `--threshold` | float | `0.4` | PUC 一致性阈值 |
| `--mu` | float | `0.1` | FedProx 近端正则项系数 |
| `--disco` | int | `0` | FedDisco 权重调整开关 |
| `--measure_difference` | str | `kl` | 分布差异度量 |


### 4.3 常用示例

以下示例使用同一组基础配置：CIC-IoV2024、CNN、20 个客户端、30 轮通信、每轮 1 个本地 epoch，以及 `alpha=0.5` 的 Dirichlet Non-IID 划分。


#### FedAvg


```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method fedavg --T 30 --E 1 --server_epochs 1 --device cuda --threshold 0.4 --puc 0 --random_seed 1 --iid 0 --dirichlet_alpha 0.5 --node_num 20 --use_dp 0 --use_sparse_adaclip 0
```

#### FedDisco

在 FedAvg 聚合前启用分布差异权重调整，不使用 `--server_method feddisco`。

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method fedavg --client_method local_train --T 30 --E 1 --server_epochs 1 --device cuda --puc 0 --random_seed 1 --iid 0 --dirichlet_alpha 0.5 --node_num 20 --use_dp 0 --use_sparse_adaclip 0 --disco 1 --measure_difference kl --disco_a 0.5 --disco_b 0.1
```

#### FedProx

服务端采用 FedAvg，客户端加入近端正则项，不使用 `--server_method fedprox`。


```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method fedavg --client_method fedprox --mu 0.1 --T 30 --E 1 --server_epochs 1 --device cuda --puc 0 --random_seed 1 --iid 0 --dirichlet_alpha 0.5 --node_num 20 --use_dp 0 --use_sparse_adaclip 0 --disco 0
```

#### FedDFW，不启用 PUC

仅启用动态聚合：

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method feddfw --T 30 --E 1 --server_epochs 1 --device cuda --threshold 0.4 --puc 0 --random_seed 1 --iid 0 --dirichlet_alpha 0.5 --node_num 20 --use_dp 0 --use_sparse_adaclip 0
```

#### FedDFW + PUC

启用动态聚合与参数更新一致性筛选：

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method feddfw --T 30 --E 1 --server_epochs 1 --device cuda --threshold 0.4 --puc 1 --random_seed 1 --iid 0 --dirichlet_alpha 0.5 --node_num 20 --use_dp 0 --use_sparse_adaclip 0
```

---

## 5. 差分隐私与自适应裁剪

通过 `--use_dp 1` 开启客户端裁剪与加噪训练。`--use_sparse_adaclip 0` 使用固定裁剪阈值，`--use_sparse_adaclip 1` 进入 AdaCliP 分支。

### 5.1 相关参数

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `--use_dp` | int | `0` | DP 训练总开关 |
| `--dp_max_grad_norm` | float | `4.0` | 裁剪阈值 |
| `--dp_noise_multiplier` | float | `0.8` | 噪声乘子 |
| `--dp_delta` | float | `1e-5` | 隐私会计目标 delta |
| `--dp_epsilon_limit` | float | `10.0` | 隐私报告使用的预算参考上限 |
| `--use_sparse_adaclip` | int | `0` | 是否使用 AdaCliP |
| `--topk_ratio` | float | `0.6` | 初始掩码及梯度 Top-k 保留比例 |
| `--use_mask_release` | int | `0` | 是否逐轮释放初始屏蔽参数 |
| `--release_ratio` | float | `0.2` | 每轮释放剩余初始屏蔽参数的比例 |

### 5.2 常用示例

#### FedAvg + DP

启用差分隐私：

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method fedavg --node_num 20 --use_dp 1 --use_sparse_adaclip 0 --dp_epsilon_limit 10 --dp_noise_multiplier 0.5 --dp_max_grad_norm 4.0 --dp_delta 1e-5
```


#### FedDFW + PUC + DP

同时启用动态聚合、PUC 和固定阈值裁剪：

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method feddfw --puc 1  --node_num 20 --use_dp 1 --use_sparse_adaclip 0 --dp_epsilon_limit 10 --dp_noise_multiplier 0.5 --dp_max_grad_norm 4.0 --dp_delta 1e-5
```

#### FedAvg + DP + AdaClip

在 FedAvg 上使用 AdaCliP：

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method fedavg  --puc 0 --node_num 20 --use_dp 1 --use_sparse_adaclip 1 --dp_epsilon_limit 10 --dp_noise_multiplier 0.5 --dp_max_grad_norm 4.0 --dp_delta 1e-5 --topk_ratio 0.6 --use_mask_release 0 --release_ratio 0.2
```

#### FedDFW + PUC + DP + AdaClip

启用动态聚合、参数一致性筛选和 AdaCliP：

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method feddfw --puc 1 --node_num 20 --use_dp 1 --use_sparse_adaclip 1 --dp_epsilon_limit 10 --dp_noise_multiplier 0.5 --dp_max_grad_norm 4.0 --dp_delta 1e-5 --topk_ratio 0.6 --use_mask_release 0 --release_ratio 0.2
```







### 5.3 运行说明

- AdaCliP 在通信轮次前为每个客户端执行 10 个 epoch 的重要性预热，预热不包含在 `--T` 与 `--E` 中。
- 上述 AdaClip 示例保留 `--use_mask_release 0`，即关闭逐轮掩码释放；此时 `--release_ratio` 不参与释放调度。需要逐轮释放时，改为 `--use_mask_release 1`。
- DP 需要计算逐样本梯度，训练速度和内存开销与普通训练不同。

---





## 注意事项

- 实验命令展示不同方法的配置，不代表所有论文实验的最终超参数或已复现结果。
- 数据集名称区分大小写，例如 `CIC-IoV2024`。
