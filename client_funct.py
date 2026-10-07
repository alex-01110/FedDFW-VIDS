import copy
import numpy as np
import torch
import torch.nn.functional as F
import gc
import math
import sys

try:
    from backpack import backpack, extend
    from backpack.extensions import BatchGrad
except ImportError:
    class DummyBackpack:
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    backpack = lambda x: DummyBackpack()
    extend = lambda x: x
    BatchGrad = lambda: None

from scipy import special
import six
from utils import validate, model_parameter_vector, flatten_model_params, load_flat_params_to_model

##############################################################################
# General client function 
##############################################################################



def receive_server_model(args, client_nodes, central_node):
    """
    接收并同步模型：自动判断 Server 和 Client 的类型进行适配。
    """
    # 1. 尝试从 Server 获取参数
    if hasattr(central_node.model, 'get_param'):
        # Server 是 ReparamModule (FedDFW)
        server_param = central_node.model.get_param(clone=True)
    else:
        # Server 是 Standard Model (FedAvg)
        server_param = copy.deepcopy(central_node.model.state_dict())

    # 2. 检查是否有 flat_w (FedDFW 特征)
    server_flat_w = None
    if 'flat_w' in server_param:
        server_flat_w = server_param['flat_w'].detach().clone()
        
    use_dp = getattr(args, "use_dp", False)

    for client_id, node in client_nodes.items():
        if not node.active: continue

        node.model.to(args.device)
        
        if use_dp:
            # DP 模式：Client 必是 Standard Model
            # 如果 Server 给的是 flat_w (FedDFW)，则需要转换
            if server_flat_w is not None:
                load_flat_params_to_model(node.model, server_flat_w)
            else:
                # 否则直接加载 state_dict (FedAvg)
                # 注意：需处理参数名匹配（如 _module 前缀），这里假设对齐或使用 strict=False
                node.model.load_state_dict(server_param, strict=False)
            
            # 确保已 extend
            if not hasattr(node.model, "autograd_backward"):
                node.model = extend(node.model)
        else:
            # 非 DP 模式：
            # 如果 Client 是 ReparamModule 且 Server 也是，直接 load_param
            if hasattr(node.model, 'load_param') and server_flat_w is not None:
                node.model.load_param(server_param)
            else:
                node.model.load_state_dict(server_param, strict=False)

    del server_param
    if server_flat_w is not None: del server_flat_w
    gc.collect()
    torch.cuda.empty_cache()
    return client_nodes



def Client_update(args, client_nodes, central_node, rounds_num):
    # 1. 接收模型
    client_nodes = receive_server_model(args, client_nodes, central_node)
    
    # 准备全局参数用于 Update Diff 或 FedProx
    if hasattr(central_node.model, 'get_param'):
        global_param_dict = central_node.model.get_param(clone=True)
    else:
        global_param_dict = central_node.model.state_dict()
        
    # 提取全局 flat_w，用于计算模型更新差分。
    global_flat_w = None
    if 'flat_w' in global_param_dict:
        global_flat_w = global_param_dict['flat_w'].detach().clone()
    else:
        # 【关键修复】兼容非 ReparamModule 模型，手动展平以计算 Diff
        from utils import flatten_model_params
        global_flat_w = flatten_model_params(central_node.model).detach().clone()
    
    use_dp = getattr(args, "use_dp", False)
    client_losses = []

    for client_id, node in client_nodes.items():
        if not node.active: continue
        node.model.to(args.device)
        
        # 准备 FedProx 参考模型
        global_ref = None
        if args.client_method == 'fedprox':
            if use_dp:
                # DP 模式下 Client 已加载最新参数，直接 Clone
                global_ref = {k: v.detach().clone() for k, v in node.model.named_parameters()}
            else:
                global_ref = global_param_dict

        # === 训练循环 ===
        epoch_losses = []
        for epoch in range(args.E):
            if use_dp:
                if args.client_method == "fedprox":
                    loss = client_fedprox_DP(global_ref, args, node)
                else:
                    loss = client_localTrain_DP(args, node)
            else:
                if args.client_method == "fedprox":
                    loss = client_fedprox(global_ref, args, node)
                else:
                    loss = client_localTrain(args, node)
            epoch_losses.append(loss)
        
        # 封装为单节点更新逻辑
        node_loss = sum(epoch_losses)/len(epoch_losses) if epoch_losses else 0.0
        post_process_client_update(args, node, global_flat_w, rounds_num, client_id)
        client_losses.append(node_loss)

    train_loss = sum(client_losses) / len(client_losses) if client_losses else 0.0
    return client_nodes, train_loss

def Client_update_single(args, node, rounds_num):
    """用于 RPC 远程调用的单节点训练函数"""
    node.model.to(args.device)
    use_dp = getattr(args, "use_dp", False)
    
    # 模拟从 central_node 获取全局参数 (在分布式模式下，参数已通过 set_model_params 加载)
    global_flat_w = node.model.flat_w.detach().clone() if hasattr(node.model, 'flat_w') else flatten_model_params(node.model)

    # 刚挂载传来的 Server 权重还是热乎的，作为 FedProx 的固定锚点
    global_ref = None
    if args.client_method == 'fedprox':
        global_ref = {k: v.detach().clone() for k, v in node.model.named_parameters()}

    epoch_losses = []
    for epoch in range(args.E):
        if use_dp:
            if args.client_method == "fedprox":
                loss = client_fedprox_DP(global_ref, args, node)
            else:
                loss = client_localTrain_DP(args, node)
        else:
            if args.client_method == "fedprox":
                loss = client_fedprox(global_ref, args, node)
            else:
                loss = client_localTrain(args, node)
        epoch_losses.append(loss)
    
    node_loss = sum(epoch_losses)/len(epoch_losses) if epoch_losses else 0.0
    post_process_client_update(args, node, global_flat_w, rounds_num, node.num_id)
    return node_loss

def post_process_client_update(args, node, global_flat_w, rounds_num, client_id):
    """计算训练后的模型更新差分和 PUC 掩码；DP 裁剪加噪在训练分支内完成。"""
    
    # 基础差异提取
    if hasattr(node.model, "flat_w"):
        local_flat_w = node.model.flat_w.detach().clone()
        update_diff_dict = {'flat_w': local_flat_w - global_flat_w.to(local_flat_w.device)}
    else:
        # 非 FedDFW 场景，使用 state_dict 级联比较 (此处简化为 flat)
        local_flat_w = flatten_model_params(node.model)
        update_diff_dict = {'flat_w': local_flat_w - global_flat_w.to(local_flat_w.device)}

    # 最终确保 node.client_update 被赋值
    if 'feddfw' in args.server_method:
        # PUC（Parameter Update Consistency）根据参数历史更新方向生成筛选掩码。
        mask_map = consistency_mask(args, update_diff_dict, node.increase_history, client_id, rounds_num)
        node.client_update = {'flat_w': update_diff_dict['flat_w'] * (mask_map['flat_w'] if args.puc == 1 else 1.0)}
        node.mask_dict = mask_map
    else:
        node.client_update = update_diff_dict



def Client_validate(args, client_nodes):
    '''
    client validation functions, for testing local personalization
    '''
    client_acc = []
    for idx in range(len(client_nodes)):
        acc = validate(args, client_nodes[idx])
        # print('client ', idx, ', after  training, acc is', acc)
        client_acc.append(acc)
    avg_client_acc = sum(client_acc) / len(client_acc)
    # print("client personalization acc is ", client_acc)
    # exit()
    return avg_client_acc,client_acc





def DKL(_p, _q):
    # print(_p)
    # print(_q)
    # print(_p.log())
    # print(_q.log())
    # print(_p.log() - _q.log())
    return  torch.sum(_p * (_p.log() - _q.log()), dim=-1)


# Vanilla local training

def client_localTrain_DP(args, node):

    model = node.model 
    device = args.device
    loss_fn = F.cross_entropy

    model.train()
    optimizer = node.optimizer
    
    # RDP 参数
    rdp_orders = node.rdp_orders
    target_delta = node.dp_delta
    
    # Mask Scheduler 步进
    current_mask = None
    if getattr(node, "use_sparse_adaclip", False) and getattr(node, "mask_scheduler", None) is not None:
        current_mask = node.mask_scheduler.step(args.current_round)
    
    total_loss = 0.0
    total_samples = 0
    
    # === 修正：直接遍历 DataLoader ===
    for batch_idx, (data, target) in enumerate(node.local_data):
        data, target = data.to(device), target.to(device)
        B = data.size(0)
        total_samples += B
        
        optimizer.zero_grad()
        
        if getattr(node, "use_sparse_adaclip", False):

            final_flat_grad, recovered_grad, loss_val = SparseAdaCliP(
                model=model,
                batch_x=data,
                batch_y=target,
                loss_fn=loss_fn,
                m_vec=node.m_vec,
                s_vec=node.s_vec,
                clip_norm=node.clip_norm,
                noise_multiplier=node.dp_noise_multiplier,
                topk_ratio=node.topk_ratio,
                mask=current_mask
            )
            
            # 更新统计量
            if recovered_grad is not None:
                batch_rec_mean = recovered_grad.mean(dim=0).detach()
                node.m_vec, node.s_vec = update_m_s(
                    batch_rec_mean, node.m_vec, node.s_vec, node.dp_noise_multiplier
                )            
        # 分支 2：标准 DP-SGD 
        else:
            outputs = model(data)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = loss_fn(outputs, target)
            loss_val = loss.item() # 记录 loss

            with backpack(BatchGrad()):
                loss.backward()

            # 收集梯度
            all_grads = []
            for param in model.parameters():
                if param.requires_grad:
                    if hasattr(param, "grad_batch"):
                        all_grads.append(param.grad_batch.reshape(B, -1))
                    else:
                      
                        g = torch.zeros(B, param.numel(), device=device)
                        all_grads.append(g)
            
            if not all_grads:
                optimizer.step() 
                continue
                
            grad_matrix = torch.cat(all_grads, dim=1) # [B, Total_Dim]

            # 标准 DP-SGD 裁剪与加噪
            norms = torch.norm(grad_matrix, dim=1)
            clip_factors = (node.clip_norm / (norms + 1e-6)).clamp(max=1.0)
            clipped_grads = grad_matrix * clip_factors.unsqueeze(1)
            
            # Mean Aggregation
            clipped_mean = clipped_grads.sum(dim=0) / float(max(1, B))
            sigma = node.dp_noise_multiplier * node.clip_norm / float(max(1, B))
            noise = torch.randn_like(clipped_mean) * sigma
            final_flat_grad = clipped_mean + noise

        # === 统一写回梯度 ===
        pointer = 0
        for param in model.parameters():
            if param.requires_grad:
                num_param = param.numel()
                grad_segment = final_flat_grad[pointer : pointer + num_param]
                
                if param.grad is None:
                    param.grad = torch.zeros_like(param.data)
                    
                param.grad.data.copy_(grad_segment.view_as(param.data))
                pointer += num_param
        
        # 优化器更新
        optimizer.step()
        
        total_loss += loss_val * B
        
        # 累积隐私预算
        node.train_steps += 1
        eps, _, _ = compute_client_privacy(
            batch_size=B,
            dataset_size=len(node.local_data.dataset),
            noise_multiplier=node.dp_noise_multiplier,
            steps=node.train_steps,
            orders=rdp_orders,
            target_delta=target_delta
        )
        node.epsilon_spent = eps

    return total_loss / float(max(1, total_samples))


def client_fedprox_DP(global_ref_dict, args, node):

    model = node.model 
    device = args.device
    loss_fn = F.cross_entropy

    model.train()
    optimizer = node.optimizer
    
    rdp_orders = node.rdp_orders
    target_delta = node.dp_delta

    # Mask Scheduler
    current_mask = None
    if getattr(node, "use_sparse_adaclip", False) and getattr(node, "mask_scheduler", None) is not None:
        current_mask = node.mask_scheduler.step(args.current_round)

    total_loss = 0.0
    total_samples = 0


    for batch_idx, (data, target) in enumerate(node.local_data):
        data, target = data.to(device), target.to(device)
        B = data.size(0)
        total_samples += B
        optimizer.zero_grad()

        # 计算 FedProx 正则项 (仅用于 Loss 统计，梯度后续手动加)
        prox_reg = 0.0
        for name, param in model.named_parameters():
            if name in global_ref_dict:
                global_w = global_ref_dict[name]
                prox_reg += ((param - global_w) ** 2).sum()

        # 分支 1：使用 SparseAdaCliP
        if getattr(node, "use_sparse_adaclip", False):

            final_flat_grad, recovered_grad, loss_val = SparseAdaCliP(
                model=model,
                batch_x=data,
                batch_y=target,
                loss_fn=loss_fn,
                m_vec=node.m_vec,
                s_vec=node.s_vec,
                clip_norm=node.clip_norm,
                noise_multiplier=node.dp_noise_multiplier,
                topk_ratio=node.topk_ratio,
                mask=current_mask
            )
       
            # 更新统计量 (增加非空检查)
            if recovered_grad is not None:
                batch_rec_mean = recovered_grad.mean(dim=0).detach()
                node.m_vec, node.s_vec = update_m_s(
                    batch_rec_mean, node.m_vec, node.s_vec, node.dp_noise_multiplier
                )
            
            # 加上正则项 Loss 用于显示
            loss_val += (args.mu / 2) * prox_reg.item()

        # 分支 2：标准 DP-SGD (手动实现)
        else:
            outputs = model(data)
            loss_task = loss_fn(outputs, target)
            loss = loss_task + (args.mu / 2) * prox_reg
            loss_val = loss.item()

            with backpack(BatchGrad()):
                loss_task.backward() # 只对 task loss 反向传播获取 grad_batch

            all_grads = []
            for param in model.parameters():
                if param.requires_grad:
                    if hasattr(param, "grad_batch"):
                        all_grads.append(param.grad_batch.reshape(B, -1))
                    else:

                        all_grads.append(torch.zeros(B, param.numel(), device=device))
            
            if not all_grads:
                optimizer.step()
                continue
                
            grad_matrix = torch.cat(all_grads, dim=1)
            
            # Standard DP-SGD Logic
            norms = torch.norm(grad_matrix, dim=1)
            clip_factors = (node.clip_norm / (norms + 1e-6)).clamp(max=1.0)
            clipped_grads = grad_matrix * clip_factors.unsqueeze(1)
            sigma = node.dp_noise_multiplier * node.clip_norm / float(max(1, B))
            noise = torch.randn_like(clipped_grads.sum(dim=0)) * sigma 
            final_flat_grad = clipped_grads.sum(dim=0) / float(max(1, B)) + noise

        # === 统一写回梯度 ===
        pointer = 0
        for param in model.parameters():
            if param.requires_grad:
                num_param = param.numel()
                param.grad.data.copy_(final_flat_grad[pointer:pointer+num_param].view_as(param.data))
                pointer += num_param

        # === 补回 FedProx 正则梯度 ===
        if args.mu > 0:
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in global_ref_dict and param.requires_grad:
                        global_w = global_ref_dict[name]
                        prox_grad = args.mu * (param.data - global_w)
                        if param.grad is not None:
                            param.grad.data.add_(prox_grad)

        # 优化器步进
        optimizer.step()
        
        total_loss += loss_val * B
        node.train_steps += 1
        eps, _, _ = compute_client_privacy(
            batch_size=B, dataset_size=len(node.local_data.dataset),
            noise_multiplier=node.dp_noise_multiplier, steps=node.train_steps,
            orders=rdp_orders, target_delta=target_delta
        )
        node.epsilon_spent = eps

    return total_loss / float(max(1, total_samples))






def client_localTrain(args, node, loss=0.0):

    node.model.train()
    loss = 0.0
    train_loader = node.local_data

    for data, target in train_loader:
        node.optimizer.zero_grad()
        data, target = data.to(args.device), target.to(args.device)
        

        output_local = node.model(data)
        if isinstance(output_local, tuple):
            output_local = output_local[0]

        loss_local = F.cross_entropy(output_local, target)
        loss_local.backward()
        

        node.optimizer.step()
        
        loss += loss_local.item()

    return loss / len(train_loader)


def client_fedprox(global_params, args, node, loss=0.0):

    node.model.train()
    loss = 0.0
    train_loader = node.local_data


    global_dict = {k: v.detach().clone().to(args.device) for k, v in global_params.items()}

    for data, target in train_loader:
        node.optimizer.zero_grad()
        data, target = data.to(args.device), target.to(args.device)
        
        output_local = node.model(data)
        if isinstance(output_local, tuple):
            output_local = output_local[0]
        loss_local = F.cross_entropy(output_local, target)
        
        # FedProx正则项
        prox_reg = 0.0
        for name, param in node.model.named_parameters():
            if name in global_dict:
                prox_reg += ((param - global_dict[name]) ** 2).sum()
        loss_local += (args.mu / 2) * prox_reg
        

        loss_local.backward()
        node.optimizer.step()
        
        loss += loss_local.item()

    return loss / len(train_loader)

def consistency_mask(args, update_diff, increase_history, client_id, epoch_index):
    """
    参数一致性掩码计算
    """
    # 计算参数更新方向

    if epoch_index == 0:
        # 如果没有历史记录，则初始化
        increase_history[client_id] = {key: torch.zeros_like(val) for key, val in update_diff.items()}

        for key in update_diff:
            increase_history[client_id][key] = (update_diff[key] >= 0).float()
        return {key: torch.ones_like(val) for key, val in update_diff.items()}

    mask = {}

    for key in update_diff:
        pos_consistent = increase_history[client_id][key]
        neg_consistent = 1 - pos_consistent

        consistency = torch.where(update_diff[key] >= 0, pos_consistent, neg_consistent)

        mask[key] = (consistency > args.threshold).float()

    for key in update_diff:
        increase = (update_diff[key] >= 0).float()
        increase_history[client_id][key] = (
            increase_history[client_id][key] * epoch_index + increase
        ) / (epoch_index + 1)

    return mask


def SparseAdaCliP(model, batch_x, batch_y, loss_fn,
                  m_vec, s_vec, clip_norm, noise_multiplier,
                  topk_ratio=1.0, mask=None):
    """
    仅包含 SparseAdaCliP 逻辑的函数。
    """
    device = batch_x.device
    model.train()

    outputs = model(batch_x)
    if isinstance(outputs, tuple):
        outputs = outputs[0]
    loss = loss_fn(outputs, batch_y)

    with backpack(BatchGrad()):
        loss.backward()

    B = batch_x.size(0)
    all_grads = []
    
    # 严格顺序遍历 + 零填充
    for param in model.parameters():
        if param.requires_grad:
            if hasattr(param, 'grad_batch'):
                g = param.grad_batch.reshape(B, -1)
                all_grads.append(g)
            else:
                g = torch.zeros(B, param.numel(), device=device)
                all_grads.append(g)
    
    grad_matrix = torch.cat(all_grads, dim=1)  # [B, D]

    # --- SparseAdaCliP 核心流程 ---
    
    # 1. Mask (应用预热阶段生成的掩码)
    if mask is not None:
        flat_mask = torch.cat([m.flatten().to(device) for m in mask])
        grad_matrix = grad_matrix * flat_mask.unsqueeze(0)
    
    # 2. Norm (标准化)
    centered_grad = (grad_matrix - m_vec) / (s_vec.sqrt() + 1e-6)
    
    # 3. Top-k (动态稀疏化)
    if topk_ratio < 1.0:
        total_dim = centered_grad.shape[1]
        k = max(1, int(total_dim * topk_ratio))
        abs_centered = centered_grad.abs()
        val, _ = torch.kthvalue(abs_centered, total_dim - k + 1, dim=1, keepdim=True)
        centered_grad = centered_grad * (abs_centered >= val)
    
    # 4. Clip (自适应裁剪)
    norms = torch.norm(centered_grad, dim=1)
    clip_factors = (clip_norm / (norms + 1e-6)).clamp(max=1.0)
    centered_grad = centered_grad * clip_factors.unsqueeze(1)
    
    # 5. Noise (加噪)
    clipped_mean = centered_grad.mean(dim=0)
    sigma = noise_multiplier * clip_norm / float(max(1, B))
    noise = torch.randn_like(clipped_mean) * sigma
    noisy_mean = clipped_mean + noise
    
    # 6. Recover (恢复尺度)
    final_grad = noisy_mean * (s_vec.sqrt() + 1e-6) + m_vec
    
    # 计算用于更新统计量的梯度 (Un-noised)
    recovered_grad = centered_grad * (s_vec.sqrt() + 1e-6) + m_vec
    
    # 再次应用掩码 (防御性操作)
    if mask is not None:
        final_grad = final_grad * flat_mask

    return final_grad, recovered_grad, loss.item()

def update_m_s(recovered_mean, m_vec, s_vec, noise_multiplier, beta_1=0.9, beta_2=0.999, h_1=1e-6, h_2=1e6):
    """保留原有update_m_s逻辑，无修改"""
    m_vec_old = m_vec.clone()
    m_vec = beta_1 * m_vec + (1 - beta_1) * recovered_mean
    grad_diff = (recovered_mean - m_vec_old) ** 2
    var_est = grad_diff
    var_est = torch.clamp(var_est, min=h_1, max=h_2)
    s_vec = beta_2 * s_vec + (1 - beta_2) * var_est
    return m_vec, s_vec

# ===================== 新增：Mask相关函数（适配联邦学习） =====================
class MaskScheduler:
    """按重要性释放初始屏蔽参数；调度表为相对初始屏蔽数量的累计释放比例。"""
    def __init__(self, base_mask, importance_scores, release_schedule, use_mask_release=True):
        self.base_mask = [m.clone() for m in base_mask]
        self.use_mask_release = bool(use_mask_release)
        self.release_schedule = list(release_schedule)
        # 确保schedule非递减
        for i in range(1, len(self.release_schedule)):
            if self.release_schedule[i] < self.release_schedule[i-1]:
                self.release_schedule[i] = self.release_schedule[i-1]

        # 只收集初始屏蔽参数，已保留的参数始终可训练。
        flat_scores = []
        flat_indices = []
        offset = 0
        for mask, scores in zip(self.base_mask, importance_scores):
            mask_flat = mask.flatten()
            scores_flat = scores.flatten()
            numel = mask_flat.numel()
            for j in range(numel):
                if mask_flat[j].item() == 0:
                    flat_scores.append(scores_flat[j].item())
                    flat_indices.append(offset + j)
            offset += numel

        if len(flat_scores) == 0:
            self.sorted_release_indices = torch.tensor([], dtype=torch.long)
        else:
            flat_scores = torch.tensor(flat_scores)
            flat_indices = torch.tensor(flat_indices, dtype=torch.long)
            sorted_idx = torch.argsort(flat_scores, descending=True)
            self.sorted_release_indices = flat_indices[sorted_idx]

        self.total_release_count = self.sorted_release_indices.numel()
        self.current_mask = [m.clone() for m in self.base_mask]
        self.current_round = -1

    def step(self, round_num):
        """按联邦学习轮次更新mask"""
        self.current_round = round_num
        # 关闭释放时始终返回初始掩码，release_ratio 不影响训练掩码。
        if not self.use_mask_release:
            return [m.clone() for m in self.base_mask]
        if self.total_release_count == 0:
            return [m.clone() for m in self.current_mask]
        
        # 累计释放比例的分母是初始屏蔽参数数量，不是模型总参数数量。
        if round_num < len(self.release_schedule):
            target_fraction = self.release_schedule[round_num]
        else:
            target_fraction = self.release_schedule[-1]
        
        release_count = int(self.total_release_count * target_fraction)
        release_count = min(release_count, self.total_release_count)

        # 构建新mask
        flat_mask = torch.cat([m.flatten() for m in self.base_mask]).clone()
        if release_count > 0:
            indices_to_unmask = self.sorted_release_indices[:release_count]
            flat_mask[indices_to_unmask] = 1

        # 恢复mask结构
        new_mask_list = []
        offset = 0
        for mask in self.base_mask:
            numel = mask.numel()
            new_mask = flat_mask[offset: offset+numel].view(mask.shape)
            new_mask_list.append(new_mask)
            offset += numel
        
        self.current_mask = new_mask_list
        return [m.clone() for m in self.current_mask]

def generate_topk_mask(importance_scores, prune_fraction=0.1):
    """保留原有generate_topk_mask逻辑，无修改"""
    flat_scores = torch.cat([scores.flatten() for scores in importance_scores])
    total_params = flat_scores.numel()
    k = int(total_params * prune_fraction)
    if k <= 0:
        return [torch.ones_like(scores) for scores in importance_scores]
    
    topk_values, _ = torch.topk(flat_scores, k, largest=False)
    threshold = topk_values.max()

    mask_list = []
    for scores in importance_scores:
        mask = torch.ones_like(scores)
        mask[scores <= threshold] = 0
        mask_list.append(mask)
    return mask_list

def accumulate_importance(node, pretrain_epochs=15, args=None):
    # 移除 wrapper
    model = node.model 
    train_loader = node.local_data
    loss_fn = F.cross_entropy
    optimizer = node.optimizer
    device = args.device
    clip_norm = node.clip_norm
    
    model.to(device)
    model.train()
    
    importance_scores = [torch.zeros_like(p, device=device) for p in model.parameters() if p.requires_grad]
    print(f'[Client {node.num_id}] Accumulating Importance Score...')
    
    for epoch in range(pretrain_epochs):
        for batch_idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            with backpack(BatchGrad()):
                loss.backward()
            B = batch_x.size(0)
            grad_norms = torch.zeros(B, device=device)
            for param in model.parameters():
                if param.requires_grad and hasattr(param, "grad_batch"):
                    grad_batch = param.grad_batch.reshape(B, -1)
                    grad_norms += (grad_batch ** 2).sum(dim=1)
            grad_norms = grad_norms.sqrt()
            clip_factors = (clip_norm / (grad_norms + 1e-6)).clamp(max=1.0)

            idx = 0
            for param in model.parameters():
                if param.requires_grad:
                    if hasattr(param, "grad_batch"):
                        grad_batch = param.grad_batch.reshape(B, -1)
                        clipped = grad_batch * clip_factors.view(-1, 1)
                        clipped_mean = clipped.mean(dim=0).view(param.shape)
                        importance_scores[idx] += clipped_mean.abs()
                    idx += 1
    return importance_scores


def _compute_delta(orders, rdp, eps):
  """Compute delta given a list of RDP values and target epsilon.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    eps: The target epsilon.
  Returns:
    Pair of (delta, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  deltas = np.exp((rdp_vec - eps) * (orders_vec - 1))
  idx_opt = np.argmin(deltas)
  return min(deltas[idx_opt], 1.), orders_vec[idx_opt]


def _compute_eps(orders, rdp, delta):
  """Compute epsilon given a list of RDP values and target delta.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.
  Returns:
    Pair of (eps, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  eps = rdp_vec - math.log(delta) / (orders_vec - 1)

  idx_opt = np.nanargmin(eps)  # Ignore NaNs
  return eps[idx_opt], orders_vec[idx_opt]


def _compute_rdp(q, sigma, alpha):
  """Compute RDP of the Sampled Gaussian mechanism at order alpha.
  Args:
    q: The sampling rate.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.
  Returns:
    RDP at alpha, can be np.inf.
  """
  if q == 0:
    return 0

  if q == 1.:
    return alpha / (2 * sigma**2)

  if np.isinf(alpha):
    return np.inf

  return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp(q, noise_multiplier, steps, orders):
  """Compute RDP of the Sampled Gaussian Mechanism.
  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
        to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders, can be np.inf.
  """
  if np.isscalar(orders):
    rdp = _compute_rdp(q, noise_multiplier, orders)
  else:
    rdp = np.array([_compute_rdp(q, noise_multiplier, order)
                    for order in orders])

  return rdp * steps


def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
  """Compute delta (or eps) for given eps (or delta) from RDP values.
  Args:
    orders: An array (or a scalar) of RDP orders.
    rdp: An array of RDP values. Must be of the same length as the orders list.
    target_eps: If not None, the epsilon for which we compute the corresponding
              delta.
    target_delta: If not None, the delta for which we compute the corresponding
              epsilon. Exactly one of target_eps and target_delta must be None.
  Returns:
    eps, delta, opt_order.
  Raises:
    ValueError: If target_eps and target_delta are messed up.
  """
  if target_eps is None and target_delta is None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (Both are).")

  if target_eps is not None and target_delta is not None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (None is).")

  if target_eps is not None:
    delta, opt_order = _compute_delta(orders, rdp, target_eps)
    return target_eps, delta, opt_order
  else:
    eps, opt_order = _compute_eps(orders, rdp, target_delta)
    return eps, target_delta, opt_order


def _log_add(logx, logy):
  """Add two numbers in the log space."""
  a, b = min(logx, logy), max(logx, logy)
  if a == -np.inf:  # adding 0
    return b
  # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
  return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx, logy):
  """Subtract two numbers in the log space. Answer must be non-negative."""
  if logx < logy:
    raise ValueError("The result of subtraction must be non-negative.")
  if logy == -np.inf:  # subtracting 0
    return logx
  if logx == logy:
    return -np.inf  # 0 is represented as -np.inf in the log space.

  try:
    # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
    return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
  except OverflowError:
    return logx


def _log_print(logx):
  """Pretty print."""
  if logx < math.log(sys.float_info.max):
    return "{}".format(math.exp(logx))
  else:
    return "exp({})".format(logx)


def _compute_log_a_int(q, sigma, alpha):
  """Compute log(A_alpha) for integer alpha. 0 < q < 1."""
  assert isinstance(alpha, six.integer_types)

  # Initialize with 0 in the log space.
  log_a = -np.inf

  for i in range(alpha + 1):
    log_coef_i = (
        math.log(special.binom(alpha, i)) + i * math.log(q) +
        (alpha - i) * math.log(1 - q))

    s = log_coef_i + (i * i - i) / (2 * (sigma**2))
    log_a = _log_add(log_a, s)

  return float(log_a)


def _compute_log_a_frac(q, sigma, alpha):
  """Compute log(A_alpha) for fractional alpha. 0 < q < 1."""
  # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
  # initialized to 0 in the log space:
  log_a0, log_a1 = -np.inf, -np.inf
  i = 0

  z0 = sigma**2 * math.log(1 / q - 1) + .5

  while True:  # do ... until loop
    coef = special.binom(alpha, i)
    log_coef = math.log(abs(coef))
    j = alpha - i

    log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
    log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

    log_e0 = math.log(.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
    log_e1 = math.log(.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

    log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
    log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

    if coef > 0:
      log_a0 = _log_add(log_a0, log_s0)
      log_a1 = _log_add(log_a1, log_s1)
    else:
      log_a0 = _log_sub(log_a0, log_s0)
      log_a1 = _log_sub(log_a1, log_s1)

    i += 1
    if max(log_s0, log_s1) < -30:
      break

  return _log_add(log_a0, log_a1)


def _compute_log_a(q, sigma, alpha):
  """Compute log(A_alpha) for any positive finite alpha."""
  if float(alpha).is_integer():
    return _compute_log_a_int(q, sigma, int(alpha))
  else:
    return _compute_log_a_frac(q, sigma, alpha)


def _log_erfc(x):
  """Compute log(erfc(x)) with high accuracy for large x."""
  try:
    return math.log(2) + special.log_ndtr(-x * 2**.5)
  except NameError:
    # If log_ndtr is not available, approximate as follows:
    r = special.erfc(x)
    if r == 0.0:
      # Using the Laurent series at infinity for the tail of the erfc function:
      #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
      # To verify in Mathematica:
      #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
      return (-math.log(math.pi) / 2 - math.log(x) - x**2 - .5 * x**-2 +
              .625 * x**-4 - 37. / 24. * x**-6 + 353. / 64. * x**-8)
    else:
      return math.log(r)
    
def compute_client_privacy(batch_size, dataset_size, noise_multiplier, steps, orders, target_delta=1e-5):
    """
    计算客户端的累计隐私预算
    Args:
        batch_size: 每批大小
        dataset_size: 本地训练样本数
        noise_multiplier: Gaussian噪声倍数
        steps: 已训练步数
        orders: RDP阶数列表
        target_delta: DP目标delta
    Returns:
        eps: 计算得到的epsilon
        delta: target_delta
        opt_order: 最优RDP阶数
    """
    q = batch_size / dataset_size
    rdp = compute_rdp(q, noise_multiplier, steps, orders)
    eps, delta, opt_order = get_privacy_spent(orders, rdp, target_delta=target_delta)
    return eps, delta, opt_order
