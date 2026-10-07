import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import copy
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib import rcParams
from collections import Counter
import matplotlib.ticker as mtick



# Subset function
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
       

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, index):
        feature, label = self.dataset[self.idxs[index]] 
        return feature, label


class IntrusionDataset3D(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.X = self.X.view(self.X.shape[0], 1, self.X.shape[1])   # 3维
        self.y = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)
        self.targets = self.y
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
class IntrusionDataset2D(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y.values if hasattr(y, 'values') else y, dtype=torch.long)
        self.targets = self.y
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Main data loader
class Data(object):
    def __init__(self, args, rank=-1):
        self.args = args
        self.node_num = args.node_num 
        self.rank = rank
     

        if args.dataset == 'CIC-IDS2017':
            data_path = '../dataset/CIC-IDS2017/process_clean_7.csv'
            data = pd.read_csv(data_path, low_memory=False)
            # label_column = ' Label'
            label_column = 'Attack Type'
            X = data.drop(columns=[label_column])
            y = data[label_column]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
            self.train_set = IntrusionDataset3D(X_train, y_train)
            self.test_set = IntrusionDataset3D(X_test, y_test)
            self.num_classes = len(np.unique(y_train))
            self.class_names = ['Benign','Port Scanning', 'Web Attacks', 'Brute Force', 'DDoS', 'Bots', 'DoS']
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)

                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha1=args.dirichlet_alpha,
                        non_iid_alpha2=args.dirichlet_alpha2,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num
                    )
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha=args.dirichlet_alpha,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num,
                        args=args,
                        class_names=self.class_names
                    )
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion

            else:
                total_len = len(self.train_set)
                base_len = total_len // self.node_num
                data_num = [base_len for _ in range(self.node_num)]
                # 将剩余部分补到前几个划分中
                for i in range(total_len - sum(data_num)):
                    data_num[i] += 1

                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set
            self._count_client_distribution()
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])

        elif args.dataset == 'CIC-IoV2024':
            data_path = '../dataset/CICIoV2024_Decimal/CIC_IOV.csv'
            data = pd.read_csv(data_path, low_memory=False)
            # 如果是其他列名，请修改下方变量
            label_column = 'label'
            X = data.drop(columns=[label_column]).values
            y = data[label_column].values
            if len(y.shape) > 1:
                y = y.ravel()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
            self.train_set = IntrusionDataset3D(X_train, y_train)
            self.test_set = IntrusionDataset3D(X_test, y_test)
            self.num_classes = len(np.unique(y_train))
            self.class_names = ['BENIGN', 'DoS', 'GAS', 'RPM', 'SPEED', 'STEERING_WHEEL']
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)

                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha1=args.dirichlet_alpha,
                        non_iid_alpha2=args.dirichlet_alpha2,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num
                    )
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha=args.dirichlet_alpha,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num,
                        args=args,
                        class_names=self.class_names
                    )
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion

            else:
                total_len = len(self.train_set)
                base_len = total_len // self.node_num
                data_num = [base_len for _ in range(self.node_num)]
                # 将剩余部分补到前几个划分中
                for i in range(total_len - sum(data_num)):
                    data_num[i] += 1

                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self._count_client_distribution()
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))]) 

        elif args.dataset == 'car-hacking':
            data_path = '../dataset/car-hacking-dataset/car-hacking-dataset.csv'
            data = pd.read_csv(data_path, low_memory=False)
            # 如果是其他列名，请修改下方变量
            label_column = 'label'
            X = data.drop(columns=[label_column]).values
            y = data[label_column].values
            if len(y.shape) > 1:
                y = y.ravel()

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
            self.train_set = IntrusionDataset3D(X_train, y_train)
            self.test_set = IntrusionDataset3D(X_test, y_test)
            self.num_classes = len(np.unique(y_train))
            self.class_names = [str(c) for c in np.unique(y_train)]
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)

                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha1=args.dirichlet_alpha,
                        non_iid_alpha2=args.dirichlet_alpha2,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num
                    )
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha=args.dirichlet_alpha,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num,
                        args=args,
                        class_names=self.class_names
                    )
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion

            else:
                total_len = len(self.train_set)
                base_len = total_len // self.node_num
                data_num = [base_len for _ in range(self.node_num)]
                # 将剩余部分补到前几个划分中
                for i in range(total_len - sum(data_num)):
                    data_num[i] += 1

                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self._count_client_distribution()
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))]) 

        elif args.dataset == 'NSL-KDD':
            data_path = '../dataset/NSL-KDD/process_5.csv'
            data = pd.read_csv(data_path, low_memory=False)
            label_column = 'subclass'
            X = data.drop(columns=[label_column])
            y = data[label_column]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
            self.train_set = IntrusionDataset3D(X_train, y_train)
            self.test_set = IntrusionDataset3D(X_test, y_test)
            self.num_classes = len(np.unique(y_train))
            self.class_names = ['Normal' ,'DoS', 'R2L', 'Probe', 'U2R']
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)

                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha1=args.dirichlet_alpha,
                        non_iid_alpha2=args.dirichlet_alpha2,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num
                    )
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha=args.dirichlet_alpha,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num,
                        args=args,
                        class_names=self.class_names
                    )
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion

            else:
                total_len = len(self.train_set)
                base_len = total_len // self.node_num
                data_num = [base_len for _ in range(self.node_num)]
                # 将剩余部分补到前几个划分中
                for i in range(total_len - sum(data_num)):
                    data_num[i] += 1

                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self._count_client_distribution()
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])            

        elif args.dataset == 'UNSW':
            data_path = '../dataset/UNSW_NB15/process_9.csv'
            data = pd.read_csv(data_path, low_memory=False)
            label_column = 'attack_cat'
            X = data.drop(columns=[label_column, 'id', 'label'])
            y = data[label_column]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
            self.train_set = IntrusionDataset3D(X_train, y_train)
            self.test_set = IntrusionDataset3D(X_test, y_test)
            self.num_classes = len(np.unique(y_train))
            self.class_names = ['Normal', 'Reconnaissance', 'Backdoor', 'DoS', 'Exploits', 'Analysis', 'Fuzzers', 'Shellcode', 'Generic']
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)

                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha1=args.dirichlet_alpha,
                        non_iid_alpha2=args.dirichlet_alpha2,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num
                    )
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha=args.dirichlet_alpha,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num,
                        args=args,
                        class_names=self.class_names
                    )
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion

            else:
                total_len = len(self.train_set)
                base_len = total_len // self.node_num
                data_num = [base_len for _ in range(self.node_num)]
                # 将剩余部分补到前几个划分中
                for i in range(total_len - sum(data_num)):
                    data_num[i] += 1

                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])              
            self._count_client_distribution()
            


        elif args.dataset == 'EDGE-IIoT':
            data_path = '../dataset/EDGE-IIoT/EDGE-IIoT.csv'
            data = pd.read_csv(data_path, low_memory=False)
            label_column = 'label'
            X = data.drop(columns=[label_column])
            y = data[label_column]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
            self.train_set = IntrusionDataset3D(X_train, y_train)
            self.test_set = IntrusionDataset3D(X_test, y_test)
            self.num_classes = len(np.unique(y_train))
            self.class_names = ['Normal','SQL_injection','Backdoor','DDOS_TCP', 'DDOS_HTTP','DDoS_ICMP','Port_Scanning','DDoS_UDP','Vulnerability','XSS']
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)

                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha1=args.dirichlet_alpha,
                        non_iid_alpha2=args.dirichlet_alpha2,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num
                    )
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha=args.dirichlet_alpha,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num,
                        args=args,
                        class_names=self.class_names
                    
                    )
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion

            else:
                total_len = len(self.train_set)
                base_len = total_len // self.node_num
                data_num = [base_len for _ in range(self.node_num)]
                # 将剩余部分补到前几个划分中
                for i in range(total_len - sum(data_num)):
                    data_num[i] += 1

                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])              
            self._count_client_distribution()

        elif args.dataset == 'VeRemi':
            data_path = '../dataset/VeRemi/VeRemi.csv'
            data = pd.read_csv(data_path, low_memory=False)
            label_column = 'label'
            X = data.drop(columns=[label_column])
            y = data[label_column]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
            self.train_set = IntrusionDataset3D(X_train, y_train)
            self.test_set = IntrusionDataset3D(X_test, y_test)
            self.num_classes = len(np.unique(y_train))
            self.class_names = ['Normal','Constant','Constant Offset', 'Random', 'Random Offset', 'Eventual Stop']
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)

                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha1=args.dirichlet_alpha,
                        non_iid_alpha2=args.dirichlet_alpha2,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num
                    )
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha=args.dirichlet_alpha,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num,
                        args=args,
                        class_names=self.class_names
                    
                    )
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion

            else:
                total_len = len(self.train_set)
                base_len = total_len // self.node_num
                data_num = [base_len for _ in range(self.node_num)]
                # 将剩余部分补到前几个划分中
                for i in range(total_len - sum(data_num)):
                    data_num[i] += 1

                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])              
            self._count_client_distribution()

        elif args.dataset == 'TON-IoT':
            data_path = '../dataset/TON-IoT/process_10.csv'
            data = pd.read_csv(data_path, low_memory=False)
            label_column = 'type'
            X = data.drop(columns=[label_column, 'label'])
            y = data[label_column]

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
            self.train_set = IntrusionDataset3D(X_train, y_train)
            self.test_set = IntrusionDataset3D(X_test, y_test)
            self.num_classes = len(np.unique(y_train))
            self.class_names = ['normal', 'backdoor','ddos','dos', 'injection', 'mitm', 'password', 'ransomware', 'scanning', 'xss']
            if args.iid == 0:  # noniid
                random_state = np.random.RandomState(int(args.random_seed))
                num_indices = len(self.train_set)

                if args.dirichlet_alpha2:
                    groups, proportion = build_non_iid_by_dirichlet_hybrid(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha1=args.dirichlet_alpha,
                        non_iid_alpha2=args.dirichlet_alpha2,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num
                    )
                else:
                    groups, proportion = build_non_iid_by_dirichlet_new(
                        random_state=random_state,
                        dataset=self.train_set,
                        non_iid_alpha=args.dirichlet_alpha,
                        num_classes=self.num_classes,
                        num_indices=num_indices,
                        n_workers=self.node_num,
                        args=args,
                        class_names=self.class_names
                    
                    )
                self.train_loader = groups
                self.groups = groups
                self.proportion = proportion

            else:
                total_len = len(self.train_set)
                base_len = total_len // self.node_num
                data_num = [base_len for _ in range(self.node_num)]
                # 将剩余部分补到前几个划分中
                for i in range(total_len - sum(data_num)):
                    data_num[i] += 1

                splited_set = torch.utils.data.random_split(self.train_set, data_num)
                self.train_loader = splited_set

            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])              
            self._count_client_distribution()

        elif args.dataset == 'IOTdataset':
            data_dir = '../dataset/IOTdataset/'
            all_train_sets = [None] * 8
            all_test_sets = [None] * 8
            
            # --- 分布式分片按需加载优化 ---
            load_indices = [self.rank - 1] if self.rank > 0 else range(8)
            print(f"[Loading] IOTdataset | Role: {'Client '+str(self.rank) if self.rank>0 else 'Server/Root'}", flush=True)
            
            for i in load_indices:
                file_path = f"{data_dir}{i}.csv"
                print(f"  - Reading partition {i} from {file_path}...", end="", flush=True)
                df = pd.read_csv(file_path, low_memory=False)
                label_column = 'label'
                X = df.drop(columns=[label_column])
                y = df[label_column]
                
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=0.2, random_state=42, stratify=y
                )
                
                all_train_sets[i] = IntrusionDataset3D(X_train, y_train)
                all_test_sets[i] = IntrusionDataset3D(X_test, y_test)
                print(" Done.", flush=True)
            
            self.train_loader = all_train_sets
            
            # 聚合全局测试集 (仅 Server)
            if self.rank <= 0:
                loaded_test_sets = [ts for ts in all_test_sets if ts is not None]
                combined_test_X = torch.cat([ts.X for ts in loaded_test_sets], dim=0)
                combined_test_y = torch.cat([ts.y for ts in loaded_test_sets], dim=0)
            else:
                combined_test_X = all_test_sets[self.rank-1].X
                combined_test_y = all_test_sets[self.rank-1].y
            
            self.test_set = IntrusionDataset3D(combined_test_X.squeeze(1).numpy(), combined_test_y.numpy())
            self.test_set.X = combined_test_X 
            self.test_set.y = combined_test_y
            self.test_set.targets = combined_test_y
            
            self.num_classes = 9
            self.class_names = ["normal", "ransomware", "thetick", "bashlite", "httpbackdoor", "beurk", "backdoor", "bdvl", "xmrig"]
            self.train_set = self.test_set 
            
            self._count_client_distribution()
            self.test_loader = torch.utils.data.random_split(self.test_set, [int(len(self.test_set))])

    def _count_client_distribution(self):
        """通用统计逻辑：自动适配索引列表(Non-IID)或数据集列表(IID/IOTdataset)"""
        self.traindata_cls_counts = np.zeros((self.node_num, self.num_classes), dtype=int)
        
        for client_id in range(self.node_num):
            # 获取当前客户端的数据片段
            client_data = self.train_loader[client_id]
            
            # 如果是 None (分布式环境下跳过了非本级分片的加载)，则跳过统计
            if client_data is None:
                continue
            
            # 情况 A: client_data 是索引列表 (例如 Dirichlet 生成的分片)
            if isinstance(client_data, (list, np.ndarray, torch.Tensor)):
                for idx in client_data:
                    label = self.train_set[idx][1].item()
                    self.traindata_cls_counts[client_id][label] += 1
            
            # 情况 B: client_data 是数据集对象 (Subset 或 Dataset)
            else:
                for idx in range(len(client_data)):
                    label = client_data[idx][1].item()
                    self.traindata_cls_counts[client_id][label] += 1




### Dirichlet noniid functions ###


def build_non_iid_by_dirichlet_new(
    random_state = np.random.RandomState(0), dataset = 0, non_iid_alpha = 10, num_classes = 10, num_indices = 60000, n_workers = 10, args = None, class_names = None
):

    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []
    
    for idx, target in enumerate(dataset.targets):
        indicesbyclass[int(target)].append(idx)
    
    for i in range(num_classes):
        random_state.shuffle(indicesbyclass[i])
    
    client_partition = random_state.dirichlet(np.repeat(non_iid_alpha, n_workers), num_classes).transpose()

    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))
    
    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]
            
    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])


    def plot_client_label_distribution(dict_users, dataset, num_classes, args, summary=False, class_names=None):
        """
        dict_users: {client_id: [data_index, ...]}
        dataset: 支持 dataset[i][1] 返回标签（整数索引）
        num_classes: 标签类别数量
        """
        # 检查 class_names 是否匹配
        if class_names is not None:
            if len(class_names) != num_classes:
                raise ValueError(f"自定义标签名数量（{len(class_names)}）与类别数（{num_classes}）不匹配！")
        else:
            class_names = [f'label {i}' for i in range(num_classes)]

        n_clients = len(dict_users)
        client_ids = sorted(dict_users.keys())  # 原始 client_id
        client_id_to_display = {cid: i + 1 for i, cid in enumerate(client_ids)}  # 显示id从1开始

        # 统计每个类别在各客户端的分布
        label_distribution = [[] for _ in range(num_classes)]
        client_label_counts = np.zeros((n_clients, num_classes), dtype=int)

        for i, cid in enumerate(client_ids):
            display_id = client_id_to_display[cid]
            indices = dict_users[cid]
            for idx in indices:
                label = int(dataset[idx][1])  # 保持整数索引
                label_distribution[label].append(display_id)
                client_label_counts[i, label] += 1

        # 绘制堆叠直方图
        plt.figure(figsize=(12, 8))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        bins = np.arange(0.5, n_clients + 1.5, 1)
        plt.hist(label_distribution,
                stacked=True,
                bins=bins,
                label=class_names,
                rwidth=0.5,
                color=colors[:num_classes])

        plt.xticks(range(1, n_clients + 1), [str(i) for i in range(1, n_clients + 1)])
        plt.xlabel("Client Id", fontsize=14)
        plt.ylabel("Number of Samples", fontsize=14)
        plt.title(f"Client Data Distribution ({args.dataset})", fontsize=16)
        ax = plt.gca()
        ax.yaxis.set_major_formatter(mtick.ScalarFormatter(useMathText=False))
        ax.ticklabel_format(style='plain', axis='y')  # 禁用科学计数法
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'{args.dataset}_client_Data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

        # 打印摘要信息
        if summary:
            print("\nLabel distribution per client:")
            for i in range(n_clients):
                display_id = i + 1  # 显示id从1开始
                nz_indices = np.nonzero(client_label_counts[i])[0]
                nz_names = [class_names[idx] for idx in nz_indices]
                nz_counts = client_label_counts[i, nz_indices].tolist()
                print(f"Client {display_id}: classes {nz_names}, counts {nz_counts}")

            full_class_clients = np.sum(np.all(client_label_counts > 0, axis=1))
            print(f"\nClients containing all classes: {full_class_clients}/{n_clients}")
    plot_client_label_distribution(dict_users, dataset, num_classes=num_classes, args=args, summary=True, class_names=class_names)




    return dict_users, client_partition


def build_non_iid_by_dirichlet_LT(
    random_state = np.random.RandomState(0), dataset = 0,  lt_rho = 10.0, non_iid_alpha = 10, num_classes = 10, num_indices = 60000, n_workers = 10
):
    # generate indicesbyclass list
    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []
    for idx, target in enumerate(dataset.targets):
        indicesbyclass[int(target)].append(idx)

    # calculate the image per class for LT
    # reformulate the indicesbyclass according to the image per class
    imb_factor = 1/float(lt_rho)
    for _classes_idx in range(num_classes):
        num = int(len(indicesbyclass[_classes_idx]) * (imb_factor**(_classes_idx / (num_classes - 1.0))))
        random_state.shuffle(indicesbyclass[_classes_idx])
        indicesbyclass[_classes_idx] = indicesbyclass[_classes_idx][:num]
    
    client_partition = random_state.dirichlet(np.repeat(non_iid_alpha, n_workers), num_classes).transpose()

    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))
    
    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]
            
    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])

    return dict_users, client_partition

def build_non_iid_by_dirichlet_hybrid(
    random_state = np.random.RandomState(0), dataset = 0, non_iid_alpha1 = 10, non_iid_alpha2 = 1, num_classes = 10, num_indices = 60000, n_workers = 10
):

    indicesbyclass = {}
    for i in range(num_classes):
        indicesbyclass[i] = []
    
    for idx, target in enumerate(dataset.targets):
        indicesbyclass[int(target)].append(idx)
    
    for i in range(num_classes):
        random_state.shuffle(indicesbyclass[i])
    
    partition = random_state.dirichlet(np.repeat(non_iid_alpha1, n_workers), num_classes).transpose()

    partition2 = random_state.dirichlet(np.repeat(non_iid_alpha2, n_workers/2), num_classes).transpose()

    new_partition1 = copy.deepcopy(partition[:int(n_workers/2)])

    sum_distr1 = np.sum(new_partition1, axis=0)

    diag_mat = np.diag(1 - sum_distr1)

    new_partition2 = np.dot(diag_mat, partition2.T).T

    client_partition = np.vstack((new_partition1, new_partition2))

    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            client_partition[i][j] = int(round(client_partition[i][j]*len(indicesbyclass[j])))
    
    client_partition_index = copy.deepcopy(client_partition)
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                client_partition_index[i][j] = client_partition_index[i][j]
            elif i == len(client_partition) - 1:
                client_partition_index[i][j] = len(indicesbyclass[j])
            else:
                client_partition_index[i][j] = client_partition_index[i-1][j] + client_partition_index[i][j]
            
    dict_users = {}
    for i in range(n_workers):
        dict_users[i] = []
    
    for i in range(len(client_partition)):
        for j in range(len(client_partition[i])):
            if i == 0:
                dict_users[i].extend(indicesbyclass[j][:int(client_partition_index[i][j])])
            else:
                dict_users[i].extend(indicesbyclass[j][int(client_partition_index[i-1][j]) : int(client_partition_index[i][j])])
    
    for i in range(len(dict_users)):
        random_state.shuffle(dict_users[i])

    return dict_users, client_partition