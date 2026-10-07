import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import flatten_model_params, load_flat_params_to_model
from torch.autograd import Variable
from scipy import special

##############################################################################
# General server function
##############################################################################

def receive_client_models(args, client_nodes, select_list, size_weights):
    client_params = []
    for cid in select_list:
        node = client_nodes[cid]
        
        # === DP 模式适配 ===
        if getattr(args, "use_dp", False):
            # 【修改】仅当算法是 FedDFW 时，才进行扁平化
            if 'feddfw' in args.server_method:
                flat_w = flatten_model_params(node.model).detach().clone()
                client_params.append({'flat_w': flat_w})
            else:
                # FedAvg / FedProx: 直接传输 state_dict
                client_params.append(copy.deepcopy(node.model.state_dict()))
        # =================
        else:
            # 非 DP 模式保持原有逻辑
            if hasattr(node.model, "get_param"):
                client_params.append(node.model.get_param(clone=True))
            else:
                client_params.append(copy.deepcopy(node.model.state_dict()))

    agg_weights = [size_weights[cid] for cid in select_list]
    agg_weights = [w / sum(agg_weights) for w in agg_weights]

    return agg_weights, client_params


def receive_client_models_pool(args, client_nodes, select_list, size_weights):
    client_params = []
    for cid in select_list:
        node = client_nodes[cid]
        
        # === DP 模式适配 ===
        if getattr(args, "use_dp", False):
            # 仅 FedDFW 扁平化
            if 'feddfw' in args.server_method:
                flat_w = flatten_model_params(node.model).detach().clone()
                param_dict = {'flat_w': flat_w}
            else:
                # FedAvg: 保持 state_dict
                param_dict = copy.deepcopy(node.model.state_dict())
            
            client_params.append(param_dict)
        # =================
        else:
            # 非 DP 模式
            model = node.model
            if hasattr(model, "base_model"):
                base_model = model.base_model
            else:
                base_model = model

            if hasattr(base_model, "get_param"):
                param_dict = base_model.get_param(clone=True)
            else:
                param_dict = copy.deepcopy(base_model.state_dict())
            
            client_params.append(param_dict)

    agg_weights = [size_weights[cid] for cid in select_list]
    agg_weights = [w / sum(agg_weights) for w in agg_weights]
    return agg_weights, client_params


def flatten_params(param_dict):
    # 这是 server 内部处理 dict 的函数，保留
    flat_w = torch.cat([v.flatten() for k, v in param_dict.items() if v is not None])
    return flat_w

def get_model_updates(client_params, prev_para):
    prev_param = copy.deepcopy(prev_para)
    client_updates = []
    for param in client_params:
        client_updates.append(param.sub(prev_param))
    return client_updates

def get_client_params_with_serverlr(server_lr, prev_param, client_updates):
    client_params = []
    with torch.no_grad():
        for update in client_updates:
            param = prev_param.add(update*server_lr)
            client_params.append(param)
    return client_params

def print_float_list(title, value, precision=4):
    """
    通用浮点数打印函数，支持 list / tensor / list[tensor] / float。
    会保留指定的小数位数（默认4位）。
    
    Args:
        title (str): 输出说明标题。
        value: 要格式化的对象（tensor/list/list[tensor]/float）。
        precision (int): 小数保留位数。
    """
    def format_number(x):
        return round(float(x), precision)

    if isinstance(value, torch.Tensor):
        value = value.tolist()
    
    if isinstance(value, float) or isinstance(value, int):
        formatted = format_number(value)

    elif isinstance(value, list):
        # 如果是 list[tensor] 或 list[list[tensor]]
        formatted = []
        for item in value:
            if isinstance(item, torch.Tensor):
                formatted.append(format_number(item.item()))
            elif isinstance(item, list):
                formatted.append([format_number(x) for x in item])
            else:
                formatted.append(format_number(item))
    else:
        formatted = value  # fallback
    
    print(f"{title}: {formatted}\n")

global_T_weights_dict={}

def get_distribution_difference(client_cls_counts, participation_clients, metric, hypo_distribution):
    local_distributions = client_cls_counts[np.array(participation_clients),:]
    local_distributions = local_distributions / local_distributions.sum(axis=1)[:,np.newaxis]
    
    if metric=='cosine':
        similarity_scores = local_distributions.dot(hypo_distribution)/ (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = 1.0 - similarity_scores
    elif metric=='only_iid':
        similarity_scores = local_distributions.dot(hypo_distribution)/ (np.linalg.norm(local_distributions, axis=1) * np.linalg.norm(hypo_distribution))
        difference = np.where(similarity_scores>0.9, 0.01, float('inf'))
    elif metric=='l1':
        difference = np.linalg.norm(local_distributions-hypo_distribution, ord=1, axis=1)
    elif metric=='l2':
        difference = np.linalg.norm(local_distributions-hypo_distribution, axis=1)
    elif metric=='kl':
        difference = special.kl_div(local_distributions, hypo_distribution)
        difference = np.sum(difference, axis=1)

        difference = np.array([0 for _ in range(len(difference))]) if np.sum(difference) == 0 else difference / np.sum(difference)
    return difference

def disco_weight_adjusting(old_weight, distribution_difference, a, b):
    weight_tmp = old_weight - a*distribution_difference + b

    if np.sum(weight_tmp>0)>0:
        new_weight = np.copy(weight_tmp)
        new_weight[new_weight<0.0]=0.0

    total_normalizer = sum([new_weight[r] for r in range(len(old_weight))])
    new_weight = [new_weight[r] / total_normalizer for r in range(len(old_weight))]
    return new_weight



def clean_state_dict(state_dict):
    """
    去掉 Opacus GradSampleModule 加的 _module. 前缀
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_module."):
            new_state_dict[k[len("_module."):]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict





def Server_update(args, central_node, client_nodes, select_list, size_weights, rounds_num=None, traindata_cls_counts=None, global_dist=None, change=0):
    '''
    server update functions for baselines（适配 size_weights 为列表）
    '''
    global size_weights_global  # 改为列表：索引=客户端ID，值=权重
    global global_T_weights

    
    # 初始化：size_weights_global 为列表（与 size_weights 一致）
    if rounds_num == change:
        size_weights_global = size_weights  # 直接赋值列表

    # 兜底：若未初始化，强制初始化为列表（长度与客户端数量一致）
    if 'size_weights_global' not in globals() or size_weights_global is None:
        size_weights_global = size_weights  # 用当前 size_weights 初始化

    # 接收客户端模型（逻辑不变，函数已适配列表权重）
    if args.server_method == 'feddfw':
        agg_weights, client_params = receive_client_models_pool(args, client_nodes, select_list, size_weights_global)
    else:
        agg_weights, client_params = receive_client_models(args, client_nodes, select_list, size_weights)


    # Disco权重调整（逻辑不变）
    if args.disco:
        distribution_difference = get_distribution_difference(
            client_cls_counts=traindata_cls_counts,
            participation_clients=select_list,
            metric=args.measure_difference,
            hypo_distribution=global_dist
        )
        
        original_weights = np.array(agg_weights)
        agg_weights = disco_weight_adjusting(
            old_weight=original_weights,
            distribution_difference=distribution_difference,
            a=args.disco_a,
            b=args.disco_b
        )
    print_float_list("agg_weights", agg_weights)
    # FedAvg 聚合（打印列表时直接传递）
    if args.server_method == 'fedavg':
        print_float_list("Global size weights", size_weights_global)  # 直接传列表
        avg_global_param = fedavg(client_params, agg_weights)
        central_node.model.load_state_dict(avg_global_param)

    # FedDFW + PUC（Parameter Update Consistency，参数更新一致性）
    # 启用 PUC 时，根据客户端参数更新方向的一致性调整聚合权重。
    elif args.server_method == 'feddfw' and args.puc == 1:
        mask_dict = {}
        for cid in select_list:
            client = client_nodes[cid]
            if hasattr(client, 'mask_dict'):
                mask_dict[cid] = client.mask_dict

        if rounds_num == change:
            global_T_weights = torch.tensor(agg_weights, dtype=torch.float32).to(args.device)


        avg_global_param, cur_global_T_weight = feddfw(args, client_params, agg_weights, central_node, rounds_num, global_T_weights, mask_dict, select_list)
        global_T_weights = cur_global_T_weight
        # 关键：按客户端ID（列表索引）更新 size_weights_global
        for i in range(len(select_list)):
            cid = select_list[i]  # 客户端ID=列表索引
            size_weights_global[cid] = global_T_weights[i].item()  # 列表索引赋值
        
        print_float_list("Global size weights", size_weights_global)  # 直接传列表
        central_node.model.load_param(avg_global_param)
    
    # FedDFW（不启用 PUC）：不使用参数更新一致性掩码进行聚合。
    elif args.server_method == 'feddfw' and args.puc != 1:
        if rounds_num == change:       
            global_T_weights = torch.tensor(agg_weights, dtype=torch.float32).to(args.device)
        
        avg_global_param, cur_global_T_weight = feddfw(args, client_params, agg_weights, central_node, rounds_num, global_T_weights)
        global_T_weights = cur_global_T_weight
        # 按客户端ID（列表索引）更新
        for i in range(len(select_list)):
            cid = select_list[i]
            size_weights_global[cid] = global_T_weights[i].item()
        
        print_float_list("Global size weights", size_weights_global)
        central_node.model.load_param(avg_global_param)    
    else:
        raise ValueError('Undefined server method...')

    return central_node



# FedAvg
def fedavg(parameters, list_nums_local_data):
    fedavg_global_params = copy.deepcopy(parameters[0])
    # d=[]
    for name_param in parameters[0]:
        list_values_param = []
        for dict_local_params, num_local_data in zip(parameters, list_nums_local_data):
            # print(dict_local_params[name_param])
            val = dict_local_params[name_param]
            if val is None:
                continue
            list_values_param.append(val * num_local_data)
        
        if not list_values_param:
            fedavg_global_params[name_param] = None
            continue

        # print("list_values_param:",list_values_param)
        value_global_param = sum(list_values_param) / sum(list_nums_local_data)
        # print("value_global_param:",value_global_param)
   
        # print("name_param:"+name_param+':',fedavg_global_params[name_param]-value_global_param)


        # print("name_param:"+name_param+':',torch.mean(torch.abs(fedavg_global_params[name_param]-value_global_param)))
        # if name_param[-6:]=="weight":
        # a=1-torch.mean(torch.abs(fedavg_global_params[name_param]-value_global_param))
        # d.append(a.item())
        # d=0.999
        fedavg_global_params[name_param] = value_global_param
    # exit()
    # print(d)
    return fedavg_global_params









def unflatten_weight(M, flat_w):
 
    ws = (t.view(s) for (t, s) in zip(flat_w.split(M._weights_numels), M._weights_shapes))
    
    for (m, n), w in zip(M._weights_module_names, ws):
        # print(type(m))
        # exit()
        # print(m,n,w)
        if 'Batch' in str(type(m)):
            print(m,n,w)
        setattr(m, n, w)
    # exit()
    # yield
    # for m, n in M._weights_module_names:
    #     setattr(m, n, None)




def to_var(x, requires_grad=True):
    if isinstance(x, dict):
        return {k: to_var(v, requires_grad) for k, v in x.items()}
    elif torch.is_tensor(x):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, requires_grad=requires_grad)
    else:
        return x

def _cost_matrix(x, y, dis, p=2):
        d_cosine = nn.CosineSimilarity(dim=-1, eps=1e-8)
    
        
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)

        if dis == 'cos':
            # print('cos_dis')
            C = 1-d_cosine(x_col, y_lin)
        elif dis == 'euc':
            # print('euc_dis')
            C= torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C




def feddfw(args, parameters, list_nums_local_data, central_node, rounds, global_T_weight, mask_dict=None, select_list=None):
    param = central_node.model.get_param()
    global_params = copy.deepcopy(param)    # 当前服务器全局模型参数
    flat_w_list = [dict_local_params['flat_w'] for dict_local_params in parameters] # 所有客户端的扁平化模型参数列表
    local_param_list = torch.stack(flat_w_list)         # 所有客户端的扁平化模型参数
    T_weights = to_var(global_T_weight)      #当前服务器维护的客户端权重向量

    


    if args.server_optimizer == 'sgd':
        Attoptimizer = torch.optim.SGD([T_weights], lr=0.01, momentum=0.9, weight_decay=1e-4)
    elif args.server_optimizer == 'adam':
        Attoptimizer = torch.optim.Adam([T_weights], lr=0.01)

    
    print_float_list("T_weights_before update", T_weights/torch.sum(T_weights, dim=0, keepdim=True))

    server_epochs = args.server_epochs      # 服务器更新轮数


    for i in range(server_epochs):
        print(f"server weight update: {i + 1}")



        scale = getattr(args, "scale", 8)

        probability_train = torch.nn.functional.softmax(T_weights * scale  , dim=0)
        # probability_train = (T_weights )/torch.sum(T_weights , dim=0, keepdim=True)
        print("probability_train", probability_train)


        C = _cost_matrix(global_params['flat_w'].detach().unsqueeze(0), local_param_list.detach(), args.reg_distance)
        reg_loss = torch.sum(probability_train * C, dim=(-2, -1))
        
        client_grad = local_param_list - global_params['flat_w']
        column_sum = torch.matmul(probability_train.unsqueeze(0), client_grad)

        diff = client_grad - column_sum
        l2_distance = torch.norm(diff, p=2, dim=1)
        sim_loss = torch.sum(probability_train * l2_distance)
        print_float_list("L2_distance", l2_distance)
        print_float_list("Sim_loss", sim_loss)   

        # cos
        # cos_sim = torch.nn.functional.cosine_similarity(
        #     client_grad,  # [N, D]
        #     column_sum.expand_as(client_grad),  # [N, D]（将聚合方向扩展到与客户端梯度同形）
        #     dim=1  # 在参数维度（D维）上计算相似度
        # )  # cos_sim shape: [N]（每个客户端对应一个相似度）

        # sim_loss = torch.sum(probability_train * (1 - cos_sim))  # 1 - cos_sim为余弦距离
        # print_float_list("cos_sim", cos_sim)
        # print_float_list("Sim_loss", sim_loss)



        Loss = sim_loss + reg_loss * args.Lambda

        Attoptimizer.zero_grad()
        Loss.backward()
        Attoptimizer.step()
        print_float_list("step {} Loss ".format(i+1), Loss.item())

    def project_weights(w,min_val=0.02 , max_val=0.3):
        
        if min_val * len(w) > 1.0:
            min_val = 1.0 / len(w) # 如果设置过大，强制调整为平均值       
        # Step 1: 所有值裁剪到 [0, max_val]
        w = torch.clamp(w, min=min_val, max=max_val)

        # Step 2: 如果 sum 超过 1，则重新归一化
        total = w.sum()
        if total > 0:
            w = w / total
        else:
            # 如果全 0，就平均分
            w = torch.ones_like(w) / len(w)


        w = torch.clamp(w, min=0, max=max_val)
        w /= w.sum()

        return w
    # global_T_weight = T_weights.detach()
    global_T_weight = project_weights(T_weights.detach(), max_val=0.3)
    # relu_w = torch.relu(T_weights)  # 保证非负
    # global_T_weight = relu_w / relu_w.sum()

    print_float_list("T_weights_after update", global_T_weight)

    # 对权重进行归一化（确保概率总和为 1）  
    # probability_train = torch.nn.functional.softmax(T_weights * scale + bias, dim=0)
    probability_train = global_T_weight
    # probability_train = (T_weights )/torch.sum(T_weights , dim=0, keepdim=True)
    print_float_list("probability_train_after update", probability_train)


    fedavg_global_params = copy.deepcopy(parameters[0])



    # PUC（Parameter Update Consistency）用于过滤历史更新方向不一致的参数。
    if args.puc == 1:
        
        # 根据参数更新一致性掩码，对每个参数位置的客户端聚合权重进行筛选。
        for name_param in parameters[0]:
            list_delta = []
            effective_weights = []
            

            for idx, (dict_local_params, prob_weight) in enumerate(zip(parameters, probability_train)):
                client_id = select_list[idx] if select_list else idx  # 如果没有传 select_list，则默认索引一致
                local_param = dict_local_params[name_param]
                global_param = global_params[name_param]
                delta_param = local_param - global_param

                if mask_dict is not None and client_id in mask_dict and name_param in mask_dict[client_id]:
                    mask = mask_dict[client_id][name_param]
                    prob_weight_effective = prob_weight * mask
                else:
                    # 如果没有对应的掩码（例如 BatchNorm 的 running_mean 等缓冲区），
                    # 或者该客户端没有掩码数据，则 fallback 到原始权重。
                    # if mask_dict is not None and client_id in mask_dict:
                    #     print(f"No mask for client {client_id}, param: {name_param}")
                    prob_weight_effective = prob_weight



                list_delta.append(delta_param * args.gamma)
                effective_weights.append(prob_weight_effective)


            # 堆叠为 [K, ...]
            stacked_deltas = torch.stack(list_delta)
            stacked_weights = torch.stack(effective_weights)

            # 步骤 1：对 stacked_weights 做归一化，使得不同客户端在每个参数位置的权重和为 1
            total_weight = torch.sum(stacked_weights, dim=0)     # shape: [...]
            safe_total_weight = total_weight.clone()
            mask_zero = (safe_total_weight == 0)
            safe_total_weight[mask_zero] = 1.0      # 防止除0
            normalized_weights = stacked_weights / safe_total_weight  # shape: [K, ...]

            # 确保 normalized_weights 可以广播到 stacked_deltas 的形状 [K, ...]
            # 如果是标量权重（[K]），补齐为 [K, 1, 1...]
            if normalized_weights.dim() < stacked_deltas.dim():
                view_shape = [normalized_weights.shape[0]] + [1] * (stacked_deltas.dim() - 1)
                normalized_weights = normalized_weights.view(*view_shape)

            weighted_delta = stacked_deltas * normalized_weights
            aggregated_delta = torch.sum(weighted_delta, dim=0)

            aggregated_delta[mask_zero] = 0.0


            if mask_zero.all():
                print(f"警告：{name_param} 的有效权重总和为零！")
                fedavg_global_params[name_param] = global_params[name_param]  # 或保持不变
            else:
                # fedavg_global_params[name_param] = sum(list_values_param) / total_weight
                fedavg_global_params[name_param] = global_param + aggregated_delta
        



    else:
        for name_param in parameters[0]:
            list_values_param = []
            for dict_local_params, num_local_data in zip(parameters, probability_train):
                list_values_param.append(dict_local_params[name_param] * num_local_data * args.gamma)

            value_global_param = sum(list_values_param) / sum(probability_train)
            
            fedavg_global_params[name_param] = value_global_param




    return fedavg_global_params, global_T_weight





