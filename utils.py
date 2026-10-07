
import numpy as np
import torch
import torch.nn.functional as F
import random
from torch.backends import cudnn
from torch.optim import Optimizer
from models_dict import  cnn

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from models_dict.SE_ResNet18 import SEResNet18
##############################################################################
# Tools
##############################################################################

class RunningAverage():
    """A simple class that maintains the running average of a quantity

    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """

    def __init__(self):
        self.steps = 0
        self.total = 0

    def update(self, val):
        self.total += val
        self.steps += 1

    def value(self):
        return self.total / float(self.steps)


def model_parameter_vector(args, model):
    if ('fedlaw' in args.server_method) or ('feddfw' in args.server_method):
        vector = model.flat_w
    else:
        param = [p.view(-1) for p in model.parameters()]
        vector = torch.cat(param, dim=0)
    return vector

def get_model_size(model):
    """计算模型参数量与物理体积 (MB)"""
    param_size = 0
    param_count = 0
    for param in model.parameters():
        param_count += param.nelement()
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    total_size_mb = (param_size + buffer_size) / 1024**2
    return f"Parameters: {param_count/1e6:.2f} M | Size: {total_size_mb:.2f} MB"

##############################################################################
# Initialization function
##############################################################################

def init_model(model_type, args):
    if args.dataset == 'cifar10':
        num_classes = 10
    elif args.dataset == 'car-hacking':
        num_classes = 5
    elif args.dataset == 'CIC-IDS2017':
        num_classes = 7
    elif args.dataset == 'CIC-IoV2024':
        num_classes = 6
    elif args.dataset == 'NSL-KDD':
        num_classes = 5
    elif args.dataset == 'UNSW':
        num_classes = 9
    elif args.dataset == 'VeRemi':
        num_classes = 6
    elif args.dataset == 'TON-IoT':
        num_classes = 10
    elif args.dataset == 'IOTdataset':
        num_classes = 9
    elif args.dataset == 'EDGE-IIoT':
        num_classes = 10
    else:
        num_classes = 100

    if('feddfw' in args.server_method):

        # if 'fedlaw' in args.server_method:
        if model_type == 'CNN':

                # print(model.state_dict())
                # exit()
            
            if args.dataset == 'CIC-IDS2017':
                model = cnn.CNN_FedLaw(norm_method=args.norm_methods_per_layer, device=args.device, flatten_dim=3328, num_classes=7)
            elif args.dataset == 'CIC-IoV2024':
                model = cnn.CNN_FedLaw(norm_method=args.norm_methods_per_layer, device=args.device, flatten_dim=512, num_classes=6)

            elif args.dataset == 'NSL-KDD':
                model = cnn.CNN_FedLaw(norm_method=args.norm_methods_per_layer, device=args.device, flatten_dim=7680, num_classes=5)
            elif args.dataset == 'UNSW':
                model = cnn.CNN_FedLaw(norm_method=args.norm_methods_per_layer, device=args.device, flatten_dim=12544, num_classes=9)
            elif args.dataset == 'car-hacking':
                model = cnn.CNN_FedLaw(norm_method=args.norm_methods_per_layer, device=args.device, flatten_dim=512, num_classes=5)
            elif args.dataset == 'VeRemi':
                model = cnn.CNN_FedLaw(norm_method=args.norm_methods_per_layer, device=args.device, flatten_dim=1024, num_classes=6)
            elif args.dataset == 'TON-IoT':
                model = cnn.CNN_FedLaw(norm_method=args.norm_methods_per_layer, device=args.device, flatten_dim=2304, num_classes=10)
            elif args.dataset == 'IOTdataset':
                model = cnn.CNN_FedLaw(norm_method=args.norm_methods_per_layer, device=args.device, flatten_dim=1792, num_classes=9)


        elif model_type == 'MLP':
            if args.dataset == 'CIC-IDS2017':
                model = cnn.MLPIDS2017_fedlaw()
            elif args.dataset == 'NSL-KDD':
                model = cnn.MLP_NSLKDD_fedlaw()
            elif args.dataset == 'CIC-IoV2024':
                model = cnn.MLP_IOV2024_TwoHead_fedlaw()
            elif args.dataset == 'UNSW':
                model = cnn.MLP_UNSW_fedlaw()
            elif args.dataset == 'car-hacking':
                model = cnn.MLP_CarHacking_TwoHead_fedlaw()
        
        elif model_type == 'SE_ResNet18':
            model = SEResNet18(num_classes=num_classes)

    else:
        if model_type == 'CNN':

            if args.dataset == 'CIC-IDS2017':
                model = cnn.CNNIDS2017()
            
            elif args.dataset == 'CIC-IoV2024':
                model = cnn.CNN_IOV2024()

            elif args.dataset == 'NSL-KDD':
                model = cnn.CNN_NSLKDD(num_classes=num_classes)
            elif args.dataset == 'UNSW':
                model = cnn.CNN_UNSW()
            elif args.dataset == 'car-hacking':
                model = cnn.CNN_carhacking()
            elif args.dataset == 'VeRemi':
                model = cnn.CNN_VeRemi()
            elif args.dataset == 'TON-IoT':
                model = cnn.CNN_TON_IoT()
            elif args.dataset == 'IOTdataset':
                # Re-using CNN_FedLaw standard without Reparam if needed, or define specific
                # For consistency with other datasets, using standard model if server_method not feddfw
                model = cnn.CNN_IOTdataset()
            elif args.dataset == 'EDGE-IIoT':
                model = cnn.CNN_EDGE_IIoT(num_classes=num_classes)

        elif model_type == 'MLP':
            if args.dataset == 'CIC-IDS2017':
                model = cnn.MLPIDS2017()
            elif args.dataset == 'CIC-IoV2024':
                model = cnn.MLP_IOV2024_TwoHead()
            elif args.dataset == 'NSL-KDD':
                model = cnn.MLP_NSLKDD(num_classes=num_classes)
            elif args.dataset == 'VeRemi':
                model = cnn.MLP_VeRemi(num_classes=num_classes)
            elif args.dataset == 'UNSW':
                model = cnn.MLP_UNSW()
            elif args.dataset == 'car-hacking':
                model = cnn.MLP_CarHacking_TwoHead()
            elif args.dataset == 'EDGE-IIoT':
                model = cnn.MLP_EDGE_IIoT(num_classes=num_classes)
        
        elif model_type == 'SE_ResNet18':
            model = SEResNet18(num_classes=num_classes)


    return model

def init_optimizer(num_id, model, args):
    optimizer = []

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.local_wd_rate)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.local_wd_rate)

    return optimizer

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True

##############################################################################
# Training function
##############################################################################

def generate_selectlist(client_node, ratio = 0.5):
    candidate_list = [i for i in range(len(client_node))]
    select_num = int(ratio * len(client_node))
    select_list = np.random.choice(candidate_list, select_num, replace = False).tolist()
    return select_list

def lr_scheduler(rounds, node_list, args):
    # learning rate scheduler for decaying
    if rounds != 0:
        args.lr *= 0.99 #0.99
        for i in range(len(node_list)):
            node_list[i].args.lr = args.lr
            node_list[i].optimizer.param_groups[0]['lr'] = args.lr
    # print('Learning rate={:.4f}'.format(args.lr))


##############################################################################
# Validation function
##############################################################################


def validate(args, node, which_dataset = 'validate'):

    node.model.to(args.device).eval() 
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')
    
    correct = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            
            # if which_dataset == 'local':
            #     output = node.model(data,True)
            #     exit()
            # else:
            output = node.model(data)
            if isinstance(output, tuple):
                output = output[0]

            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset) * 100
    return acc



def testloss(args, node, which_dataset = 'validate'):
    node.model.to(args.device).eval()  
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    loss = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            output = node.model(data)
            if isinstance(output, tuple):
                output = output[0]
            loss_local =  F.cross_entropy(output, target, reduction='mean')
            loss.append(loss_local.item())
    loss_value = sum(loss)/len(loss)
    return loss_value



def report_metrics(args, node, which_dataset='validate', class_names=None):
    node.model.to(args.device).eval()

    # 选择验证集 or 本地集
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined dataset choice')

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(args.device), target.to(args.device)
            output = node.model(data)
            if isinstance(output, tuple):
                output = output[0]
            preds = output.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    acc = accuracy_score(all_targets, all_preds)
    f1_macro = f1_score(all_targets, all_preds, average='macro')
    precision = precision_score(all_targets, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='macro', zero_division=0)

    print(f"\n== Overall Metrics ==")
    print(f"Accuracy     : {acc:.4f}")
    print(f"F1 (Macro)   : {f1_macro:.4f}")
    print(f"Precision    : {precision:.4f}")
    print(f"Recall       : {recall:.4f}")    
    
    # 混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True)  # 每一类的预测分布
    print("\nConfusion Matrix:\n", cm)
    # 画混淆矩阵图
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=class_names if class_names else "auto",
            yticklabels=class_names if class_names else "auto")
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'{args.dataset} Confusion Matrix')
    plt.savefig(f"{args.server_method}_{args.client_method}_puc{args.puc}_{args.dataset}_{args.local_model}_disco{args.disco}_dp{args.use_dp}_adaclip{args.use_sparse_adaclip}_confusion_matrix_predicted.png", dpi=300, bbox_inches='tight')
    # plt.show()


    # 分类报告
    report = classification_report(all_targets, all_preds,
                                   target_names=class_names if class_names else None,
                                   digits=4, zero_division=0)
    print("\nClassification Report:\n", report)



def plot_accuracy_curve(acc_list,args=None, catagory=None,save_path="accuracy_curve.png"):
    rounds = list(range(1, len(acc_list) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(rounds, acc_list, linestyle='-', color='blue', label=args.server_method)

    plt.title(f"{catagory}_Accuracy Curve of {args.server_method} on {args.dataset}")
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# Functions for FedLAW with param as an input
def validate_with_param(args, node, param, which_dataset = 'validate'):
    node.model.to(args.device).eval()
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    correct = 0.0
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            output = node.model.forward_with_param(data, param)
            if isinstance(output, tuple):
                output = output[0]
            pred = output.argmax(dim=1)
            correct += pred.eq(target.view_as(pred)).sum().item()
        acc = correct / len(test_loader.dataset) * 100
    return acc

def testloss_with_param(args, node, param, which_dataset = 'validate'):
    node.model.to(args.device).eval()  
    if which_dataset == 'validate':
        test_loader = node.validate_set
    elif which_dataset == 'local':
        test_loader = node.local_data
    else:
        raise ValueError('Undefined...')

    loss = []
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(args.device), target.to(args.device)
            output = node.model.forward_with_param(data, param)
            if isinstance(output, tuple):
                output = output[0]
            loss_local =  F.cross_entropy(output, target, reduction='mean')
            loss.append(loss_local.item())
    loss_value = sum(loss)/len(loss)
    return loss_value


def flatten_model_params(model):
    """
    将标准模型的参数展平为 flat_w。
    关键：遍历顺序必须与 ReparamModule (即 Server 端) 完全一致，
    否则参数位置会错乱。
    """
    params = []
    # ReparamModule 的构造逻辑是遍历 modules() -> named_parameters(recurse=False)
    for m in model.modules():
        for n, p in m.named_parameters(recurse=False):
            # 通常只聚合 requires_grad 的参数，或者所有参数
            # ReparamModule 默认处理所有非 None 参数
            if p is not None:
                params.append(p.view(-1))
    
    if len(params) == 0:
        return torch.tensor([]).to(next(model.parameters()).device)
        
    return torch.cat(params)

def load_flat_params_to_model(model, flat_w):
    """
    将 flat_w 切片并加载回标准模型。
    """
    offset = 0
    # 必须使用 no_grad，避免修改参数时被记录进计算图
    with torch.no_grad():
        for m in model.modules():
            for n, p in m.named_parameters(recurse=False):
                if p is not None:
                    numel = p.numel()
                    # 检查越界（防止服务器发来的参数长度不匹配）
                    if offset + numel > flat_w.numel():
                        raise ValueError(f"flat_w 长度 {flat_w.numel()} 不足，无法填充参数 {n} (需要 {offset+numel})")
                    
                    # 从 flat_w 中切出对应部分，重塑并赋值
                    # 注意：flat_w 可能在不同设备上，需确保 to(p.device)
                    p.data.copy_(flat_w[offset : offset + numel].view(p.shape).to(p.device))
                    offset += numel
