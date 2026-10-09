# Privacy-Preserving Intrusion Detection in IoV


**English** | [简体中文](README.zh-CN.md)


This repository contains the experimental code for **Towards Privacy-Preserving Intrusion Detection in IoV: a Federated Learning Approach with Adaptive Differential Privacy and Dynamic Aggregation**.

The project supports multi-client federated training for intrusion detection in the Internet of Vehicles (IoV), with the following experimental components:

1. **Dynamic aggregation**: FedDFW adjusts client aggregation weights.
2. **Parameter Update Consistency (PUC)**: filters parameter updates based on their directions.
3. **Differentially private training**: uses fixed-threshold clipping or AdaCliP adaptive clipping.

FedAvg, FedDisco, and FedProx baseline configurations are also provided for comparative experiments.

---

## 1. Dependencies

Dependencies are listed in `requirements.txt`. After creating and activating a dedicated environment, run the following command from the repository root:

```bash
python -m pip install -r requirements.txt
```

> For GPU training, install PyTorch and torchvision versions compatible with your local CUDA environment. Dependency versions are currently not pinned. If no GPU is available, replace `--device cuda` with `--device cpu`.

The DP and AdaCliP branches require `backpack-for-pytorch` to compute per-sample gradients. `opacus` is used by the retained `DPReparamWrapper` class; it does not replace BackPACK for the examples below.

## 2. Datasets

The project supports multiple IoV intrusion detection datasets. Prepare the datasets and perform data cleaning and feature selection yourself. Dataset paths are currently defined in `datasets.py` and resolved relative to the **working directory from which the command is run**. Start training from the `FedDFW` repository root.

The examples below use `CIC-IoV2024`:

```text
FL/
|-- FedDFW/
|   |-- main.py
|   |-- args.py
|   |-- datasets.py
|   |-- README.md
|   `-- README.zh-CN.md
`-- dataset/
    `-- CICIoV2024_Decimal/
        `-- CIC_IOV.csv
```

The loading path is `../dataset/CICIoV2024_Decimal/CIC_IOV.csv`. To use a different directory, update `data_path` in the corresponding branch of `datasets.py`.

---

## 3. Quick Start

Once the environment and dataset are ready, run a few communication rounds to check the basic training workflow:

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method fedavg --T 3 --E 1 --device cuda --puc 0 --random_seed 1 --iid 0 --dirichlet_alpha 0.5 --node_num 20 --use_dp 0 --use_sparse_adaclip 0
```

This command reduces only the number of communication rounds; it still loads the full dataset.

---

## 4. Federated Learning

**Training workflow:**

1. Load the dataset and split it into training and test sets, then partition the training data among clients using IID or Dirichlet Non-IID settings.
2. Initialize server and client models; perform importance warm-up when AdaCliP is enabled.
3. Synchronize global parameters each round, run client-side local training, and apply optional PUC and DP processing.
4. Aggregate client models according to the FedAvg, FedDisco, or FedDFW configuration. FedProx takes effect during local training.
5. Evaluate the model and record training loss, accuracy, and classification metrics.

Entry point: `main.py`.

```bash
python main.py [options]
```

### 4.1 Methods and Configurations

Methods are configured by combining command-line arguments rather than using separate training entry points:

| Method or component | Configuration | Description |
| --- | --- | --- |
| FedAvg | `--server_method fedavg` | Baseline aggregation weighted by data volume |
| FedDisco | `--server_method fedavg --disco 1` | Adjusts aggregation weights using differences in client label distributions |
| FedProx | `--server_method fedavg --client_method fedprox` | Adds a proximal regularization term to the client objective |
| FedDFW | `--server_method feddfw` | Dynamically adjusts server-side aggregation weights |
| PUC | `--server_method feddfw --puc 1` | Parameter Update Consistency; generates a consistency filtering mask from parameter update directions |
| DP | `--use_dp 1 --use_sparse_adaclip 0` | Training branch with fixed-threshold clipping and noise addition |
| AdaClip | `--use_dp 1 --use_sparse_adaclip 1` | Corresponds to AdaCliP in the code, including sparsification and adaptive clipping |

Notes:

- PUC takes effect only in the FedDFW branch. `fedavg --puc 1` does not enable PUC.
- AdaCliP also requires `--use_dp 1`. Setting only `--use_sparse_adaclip 1` does not activate this training branch.
- The commands below illustrate configurations and ablation combinations, not the final hyperparameters for every experiment in the paper.

### 4.2 Command-Line Arguments

Defaults come from `args.py` and do not necessarily match the experimental settings in the examples below.

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--dataset` | str | `CIC-IDS2017` | Dataset name |
| `--local_model` | str | `MLP` | Local model; the examples below use `CNN` |
| `--server_method` | str | `feddfw` | Aggregation method: `fedavg` or `feddfw` |
| `--client_method` | str | `local_train` | Local training method: `local_train` or `fedprox` |
| `--T` | int | `3` | Number of communication rounds |
| `--E` | int | `1` | Number of local training epochs per round |
| `--node_num` | int | `10` | Number of clients |
| `--random_seed` | int | `1` | Experiment random seed |
| `--iid` | int | `1` | `1` for IID; `0` for Non-IID |
| `--dirichlet_alpha` | float | `1.0` | Dirichlet partitioning parameter |
| `--batchsize` | int | `32` | Local training batch size |
| `--lr` | float | `0.001` | Client learning rate |
| `--optimizer` | str | `sgd` | Client optimizer |
| `--puc` | int | `1` | PUC switch for FedDFW |
| `--threshold` | float | `0.4` | PUC consistency threshold |
| `--mu` | float | `0.1` | FedProx proximal regularization coefficient |
| `--disco` | int | `0` | FedDisco weight adjustment switch |
| `--measure_difference` | str | `kl` | Distribution difference metric |

### 4.3 Examples

The following examples share the same base settings: CIC-IoV2024, CNN, 20 clients, 30 communication rounds, 1 local epoch per round, and a Dirichlet Non-IID partition with `alpha=0.5`.

#### FedAvg

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method fedavg --T 30 --E 1 --server_epochs 1 --device cuda --threshold 0.4 --puc 0 --random_seed 1 --iid 0 --dirichlet_alpha 0.5 --node_num 20 --use_dp 0 --use_sparse_adaclip 0
```

#### FedDisco

Enable distribution-based weight adjustment before FedAvg aggregation. Do not use `--server_method feddisco`.

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method fedavg --client_method local_train --T 30 --E 1 --server_epochs 1 --device cuda --puc 0 --random_seed 1 --iid 0 --dirichlet_alpha 0.5 --node_num 20 --use_dp 0 --use_sparse_adaclip 0 --disco 1 --measure_difference kl --disco_a 0.5 --disco_b 0.1
```

#### FedProx

Use FedAvg on the server and add a proximal regularization term on clients. Do not use `--server_method fedprox`.

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method fedavg --client_method fedprox --mu 0.1 --T 30 --E 1 --server_epochs 1 --device cuda --puc 0 --random_seed 1 --iid 0 --dirichlet_alpha 0.5 --node_num 20 --use_dp 0 --use_sparse_adaclip 0 --disco 0
```

#### FedDFW Without PUC

Enable dynamic aggregation only:

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method feddfw --T 30 --E 1 --server_epochs 1 --device cuda --threshold 0.4 --puc 0 --random_seed 1 --iid 0 --dirichlet_alpha 0.5 --node_num 20 --use_dp 0 --use_sparse_adaclip 0
```

#### FedDFW + PUC

Enable dynamic aggregation and parameter update consistency filtering:

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method feddfw --T 30 --E 1 --server_epochs 1 --device cuda --threshold 0.4 --puc 1 --random_seed 1 --iid 0 --dirichlet_alpha 0.5 --node_num 20 --use_dp 0 --use_sparse_adaclip 0
```

---

## 5. Differential Privacy and Adaptive Clipping

Use `--use_dp 1` to enable client-side clipping and noise addition. `--use_sparse_adaclip 0` selects fixed-threshold clipping, while `--use_sparse_adaclip 1` selects the AdaCliP branch.

### 5.1 Arguments

| Argument | Type | Default | Description |
| --- | --- | --- | --- |
| `--use_dp` | int | `0` | Master switch for DP training |
| `--dp_max_grad_norm` | float | `4.0` | Clipping threshold |
| `--dp_noise_multiplier` | float | `0.8` | Noise multiplier |
| `--dp_delta` | float | `1e-5` | Target delta for privacy accounting |
| `--dp_epsilon_limit` | float | `10.0` | Reference privacy budget upper limit used in privacy reports |
| `--use_sparse_adaclip` | int | `0` | Whether to use AdaCliP |
| `--topk_ratio` | float | `0.6` | Top-k retention ratio for the initial mask and gradients |
| `--use_mask_release` | int | `0` | Whether to progressively release initially masked parameters each round |
| `--release_ratio` | float | `0.2` | Fraction of remaining initially masked parameters released each round |

### 5.2 Examples

#### FedAvg + DP

Enable differential privacy:

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method fedavg --node_num 20 --use_dp 1 --use_sparse_adaclip 0 --dp_epsilon_limit 10 --dp_noise_multiplier 0.5 --dp_max_grad_norm 4.0 --dp_delta 1e-5
```

#### FedDFW + PUC + DP

Enable dynamic aggregation, PUC, and fixed-threshold clipping:

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method feddfw --puc 1  --node_num 20 --use_dp 1 --use_sparse_adaclip 0 --dp_epsilon_limit 10 --dp_noise_multiplier 0.5 --dp_max_grad_norm 4.0 --dp_delta 1e-5
```

#### FedAvg + DP + AdaClip

Use AdaCliP with FedAvg:

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method fedavg  --puc 0 --node_num 20 --use_dp 1 --use_sparse_adaclip 1 --dp_epsilon_limit 10 --dp_noise_multiplier 0.5 --dp_max_grad_norm 4.0 --dp_delta 1e-5 --topk_ratio 0.6 --use_mask_release 0 --release_ratio 0.2
```

#### FedDFW + PUC + DP + AdaClip

Enable dynamic aggregation, parameter consistency filtering, and AdaCliP:

```bash
python main.py --dataset CIC-IoV2024 --local_model CNN --server_method feddfw --puc 1 --node_num 20 --use_dp 1 --use_sparse_adaclip 1 --dp_epsilon_limit 10 --dp_noise_multiplier 0.5 --dp_max_grad_norm 4.0 --dp_delta 1e-5 --topk_ratio 0.6 --use_mask_release 0 --release_ratio 0.2
```

### 5.3 Runtime Notes

- AdaCliP performs 10 importance warm-up epochs for each client before the communication rounds. These warm-up epochs are not included in `--T` or `--E`.
- The AdaClip examples above retain `--use_mask_release 0`, disabling progressive mask release. In this case, `--release_ratio` does not affect the release schedule. Set `--use_mask_release 1` to enable progressive release.
- DP requires per-sample gradients, so its training speed and memory usage differ from standard training.

---

## Notes

- The experimental commands illustrate method configurations; they do not represent the final hyperparameters or reproduced results for all experiments in the paper.
- Dataset names are case-sensitive, for example, `CIC-IoV2024`.
