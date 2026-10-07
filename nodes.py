import copy
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import DatasetSplit
from utils import init_model
from utils import init_optimizer, model_parameter_vector

try:
    from backpack import extend
except ImportError:
    extend = lambda x: x


class Node(object):
    def __init__(self, num_id, local_data, train_set, args):
        self.num_id = num_id
        self.args = args
        self.node_num = self.args.node_num
        if num_id == -1:
            self.valid_ratio = args.server_valid_ratio
        else:
            self.valid_ratio = args.client_valid_ratio

        if self.args.dataset == 'cifar10' or self.args.dataset == 'fmnist':
            self.num_classes = 10
        elif self.args.dataset == 'CIC-IDS2017':
            self.num_classes = 7
        elif self.args.dataset == 'CIC-IoV2024':
            self.num_classes = 6
        elif self.args.dataset == 'NSL-KDD':
            self.num_classes = 5
        elif self.args.dataset == 'UNSW':
            self.num_classes = 9
        elif self.args.dataset == 'car-hacking':
            self.num_classes = 5
        elif self.args.dataset == 'EDGE-IIoT':
            self.num_classes = 10
        elif self.args.dataset == 'VeRemi':
            self.num_classes = 6
        elif self.args.dataset == 'TON-IoT':
            self.num_classes = 10
        elif self.args.dataset == 'IOTdataset':
            self.num_classes = 9
        # 适配 IID (Subset) 与 Non-IID (List) 不同的返回格式
        if hasattr(local_data, 'indices'):
            client_indices = local_data.indices
        elif isinstance(local_data, (list, np.ndarray)):
            client_indices = list(local_data)
        else:
            client_indices = []

        if num_id == -1:
            # 中央节点
            self.local_data, self.validate_set = self.train_val_split_forServer(client_indices, train_set, self.valid_ratio, self.num_classes)
        elif args.dataset == 'IOTdataset' and num_id != -1:
            # IOTdataset 特殊独享 DataLoader
            self.local_data = DataLoader(local_data, batch_size=args.batchsize, shuffle=True)
            self.validate_set = DataLoader(local_data, batch_size=args.validate_batchsize, shuffle=False)
        elif args.iid == 1 and num_id != -1:
            # 独立同分布 IID 节点的测试与验证集划分
            self.local_data, self.validate_set = self.train_val_split_foriid(client_indices, train_set, self.valid_ratio, self.num_classes)
        elif args.iid == 0 and num_id != -1:
            # ==【关键修复】== 非独立同分布 Non-IID 下的 Dataloader 划分
            self.local_data, self.validate_set = self.train_val_split(client_indices, train_set, self.valid_ratio)

        self.model = init_model(self.args.local_model, self.args).to(args.device)

        if getattr(args, "use_dp", False):
                    try:
                        self.model = extend(self.model)
                    except:
                        # 如果模型已经是 ReparamModule，可能需要 extend base_model，这里视具体实现而定
                        pass
        # 判断1：是否开启了 DP
        use_dp = getattr(args, "use_dp", False)
        
        # 判断2：当前是客户端还是服务器？
        is_client = (num_id != -1)

        if use_dp and is_client and ('feddfw' in args.server_method):
            if args.local_model == 'CNN':
                from models_dict.cnn import CNN_FedLaw_Standard
                # 设置 flatten_dim
                if args.dataset == 'CIC-IDS2017': flatten_dim = 3328
                elif args.dataset == 'NSL-KDD': flatten_dim = 7680
                elif args.dataset == 'VeRemi': flatten_dim = 1024
                elif args.dataset == 'UNSW': flatten_dim = 12544
                elif args.dataset == 'CIC-IoV2024': flatten_dim = 512
                elif args.dataset == 'TON-IoT': flatten_dim = 2304
                elif args.dataset == 'IOTdataset': flatten_dim = 1792
                else: flatten_dim = 512 # Fallback
                
                self.model = CNN_FedLaw_Standard(
                    norm_method=args.norm_methods_per_layer, 
                    device=args.device, 
                    flatten_dim=flatten_dim, 
                    num_classes=self.num_classes
                ).to(args.device)
                
                # FedDFW 下 Client 必须 extend
                self.model = extend(self.model)
            else:
                # Fallback for MLP in FedDFW
                self.model = init_model(self.args.local_model, self.args).to(args.device)
                try: self.model = extend(self.model)
                except: pass
        
        else:
            # === 其他情况 (FedAvg+DP, FedProx+DP, Non-DP, Server等) ===
            # 直接使用 init_model 初始化原生模型 (如 CNN_NSLKDD)
            self.model = init_model(self.args.local_model, self.args).to(args.device)
            
            # 如果是 DP Client (但在上面的 if 没命中，说明是 FedAvg/FedProx)
            # 我们仍然需要 extend 它以支持 BackPACK
            if use_dp and is_client:
                self.model = extend(self.model)


        self.optimizer = init_optimizer(self.num_id, self.model, args)
        self.increase_history = {}   
        self.mask_dict = {}          
        self.param_names = [name for name, _ in self.model.named_parameters()]  
        self.client_update = {}       
        self.active = True
        if use_dp:
            dataset_len = len(self.local_data.dataset)
            batch_size = args.batchsize
            self.sample_rate = batch_size / float(max(1, dataset_len))
            self.dp_noise_multiplier = getattr(args, "dp_noise_multiplier", 1.0)
            self.epsilon_limit = getattr(args, "dp_epsilon_limit", 10.0)
            self.dp_delta = getattr(args, "dp_delta", 1e-5)
            self.clip_norm = getattr(args, "dp_max_grad_norm", 1.0)
            # 【修复】使用全局 step 计数器，确保跨轮次累积隐私消耗
            self.train_steps = 0 
            self.rdp_orders = np.arange(2, 64, 0.5) # 扩充 order 范围
            
            # SparseAdaCliP 初始化
            self.use_sparse_adaclip = getattr(args, "use_sparse_adaclip", False)
            if self.use_sparse_adaclip:
                total_dim = sum(p.numel() for p in self.model.parameters())
                self.m_vec = torch.zeros(total_dim, device=args.device)
                self.s_vec = torch.ones(total_dim, device=args.device)
                self.topk_ratio = getattr(args, "topk_ratio", 0.6)
                self.mask_scheduler = None
        # node init for feddyn
        if args.client_method == 'feddyn':
            self.old_grad = None
            self.old_grad = copy.deepcopy(self.model)
            self.old_grad = model_parameter_vector(args, self.old_grad)
            self.old_grad = torch.zeros_like(self.old_grad)
        if 'feddyn' in args.server_method:
            self.server_state = copy.deepcopy(self.model)
            for param in self.server_state.parameters():
                param.data = torch.zeros_like(param.data)
        
        # node init for fedadam's server
        if args.server_method == 'fedadam' and num_id == -1:
            m = copy.deepcopy(self.model)
            self.zero_weights(m)
            self.m = m
            v = copy.deepcopy(self.model)
            self.zero_weights(v)
            self.v = v

    def zero_weights(self, model):
        for n, p in model.named_parameters():
            p.data.zero_()

    def train_val_split(self, idxs, train_set, valid_ratio): 

        np.random.shuffle(idxs)

        validate_size = valid_ratio * len(idxs)
        # print(validate_size)
        # exit()
        idxs_test = idxs[:int(validate_size)]
        idxs_train = idxs[int(validate_size):]

        train_loader = DataLoader(DatasetSplit(train_set, idxs_train),
                                  batch_size=self.args.batchsize, num_workers=0, shuffle=True)

        test_loader = DataLoader(DatasetSplit(train_set, idxs_test),
                                 batch_size=self.args.validate_batchsize,  num_workers=0, shuffle=True)
        

        return train_loader, test_loader



    def train_val_split_forServer(self, idxs, train_set, valid_ratio, num_classes=None):
        # 将所有索引都用于训练集，验证集为空
        train_loader = DataLoader(DatasetSplit(train_set, idxs),
                                batch_size=self.args.batchsize, num_workers=0, shuffle=True)
        test_loader = None  # 或者 DatasetSplit(train_set, []) 也行

        return train_loader, test_loader

    def train_val_split_foriid(self, idxs, train_set, valid_ratio, num_classes=None):
        # 随机打乱样本的索引
        np.random.shuffle(idxs)

        # 如果未指定num_classes，自动推断类别数
        if num_classes is None:
            label_set = set()
            for idx in idxs:
                label = train_set[idx][1]
                if isinstance(label, torch.Tensor):
                    label = label.item()
                label_set.add(label)
            num_classes = max(label_set) + 1  # 类别数

        # 按类别分配样本到训练集和验证集
        class_samples = {i: [] for i in range(num_classes)}
        for idx in idxs:
            label = train_set[idx][1]
            if isinstance(label, torch.Tensor):
                label = label.item()
            class_samples[label].append(idx)

        # 划分每个类别的训练集和验证集
        idxs_train = []
        idxs_test = []

        for class_id, class_idxs in class_samples.items():
            # 计算该类别样本的验证集大小
            validate_size = int(valid_ratio * len(class_idxs))
            
            # 按照比例划分该类别的样本到训练集和验证集
            np.random.shuffle(class_idxs)
            class_test = class_idxs[:validate_size]
            class_train = class_idxs[validate_size:]

            idxs_train.extend(class_train)
            idxs_test.extend(class_test)

        # 生成训练和验证数据加载器
        train_loader = DataLoader(DatasetSplit(train_set, idxs_train),
                                batch_size=self.args.batchsize, num_workers=0, shuffle=True)
        test_loader = DataLoader(DatasetSplit(train_set, idxs_test),
                                batch_size=self.args.validate_batchsize, num_workers=0, shuffle=True)

        return train_loader, test_loader



# Tools for long-tailed functions
def label_indices2indices(list_label2indices):
    indices_res = []
    for indices in list_label2indices:
        indices_res.extend(indices)

    return indices_res

def _get_img_num_per_cls(list_label2indices_train, num_classes, imb_factor, imb_type):
    img_max = len(list_label2indices_train) / num_classes
    img_num_per_cls = []
    if imb_type == 'exp':
        for _classes_idx in range(num_classes):
            num = img_max * (imb_factor**(_classes_idx / (num_classes - 1.0)))
            img_num_per_cls.append(int(num))

    return img_num_per_cls

def train_long_tail(list_label2indices_train, num_classes, imb_factor, imb_type):
    new_list_label2indices_train = label_indices2indices(copy.deepcopy(list_label2indices_train))
    img_num_list = _get_img_num_per_cls(copy.deepcopy(new_list_label2indices_train), num_classes, imb_factor, imb_type)
    print('img_num_class')
    print(img_num_list)

    list_clients_indices = []
    classes = list(range(num_classes))
    for _class, _img_num in zip(classes, img_num_list):
        indices = list_label2indices_train[_class]
        np.random.shuffle(indices)
        idx = indices[:_img_num]
        list_clients_indices.append(idx)
    num_list_clients_indices = label_indices2indices(list_clients_indices)
    print('All num_data_train')
    print(len(num_list_clients_indices))

    return img_num_list, list_clients_indices
