import os
import argparse
import torch
import torch.distributed.rpc as rpc
from datasets import Data
from nodes import Node
from args import args_parser
from utils import setup_seed, RunningAverage, model_parameter_vector
from comm_monitor import NetworkMonitor
import time
import numpy as np
import distributed_utils
from server_funct import fedavg, feddfw

def run_server(args, data):
    num_clients = args.world_size - 1
    client_names = [f"client_{i}" for i in range(num_clients)]
    
    # 网络监控
    monitor = NetworkMonitor(nic_name=args.nic)
    total_net_sent, total_net_recv = 0, 0
    total_comm_time = 0.0

    central_node = Node(-1, data.test_loader[0], data.test_set, args)
    
    # --- [新增] 模型大小探测与开销对账 ---
    from utils import get_model_size
    print(f"\n[Model Info] Architecture: {args.local_model} | Dataset: {args.dataset}", flush=True)
    print(f"[Model Info] {get_model_size(central_node.model)}", flush=True)
    model_ele = central_node.model.flat_w.nelement() if hasattr(central_node.model, 'flat_w') \
                else sum(p.numel() for p in central_node.model.parameters())
    print(f"[Traffic Info] Expected Per-Round Sent (to {num_clients} clients): ≈ {model_ele*4/1024**2 * num_clients:.2f} MB", flush=True)
    
    # 获取聚合所需的 size_weights
    client_samples = np.sum(data.traindata_cls_counts, axis=1)
    size_weights = client_samples / np.sum(client_samples)
    global_T_weights = torch.tensor(size_weights, dtype=torch.float32).to(args.device)

    print(f"Starting Distributed Training on {args.nic or 'All'} for {args.T} rounds...")
    
    # RPC Warm-up
    print(f"\n--- [Pre-flight] Warming up RPC ---")
    warm_futures = []
    for name in client_names:
        warm_futures.append(rpc.rpc_async(name, distributed_utils.rpc_ping))
    [f.wait() for f in warm_futures]
    monitor.get_diff()
    
    comm_history = []
    test_acc_recorder = []

    for r in range(args.T):
        print(f"\n--- Round {r+1} / {args.T} ---")
        monitor.get_diff() 
        
        # 1. 广播全局模型参数，根据模式仅发送单一权重。
        broadcast_start = time.time()
        if hasattr(central_node.model, "get_param"):
            global_params_payload = central_node.model.get_param(clone=True)
            server_flat_w = global_params_payload.get('flat_w', None)
            # 关键优化：如果 server_flat_w 存在，则 global_params_payload 设为 None，避免重复发送模型字典
            send_payload = None 
        else:
            send_payload = {k: v.cpu() for k, v in central_node.model.state_dict().items()}
            server_flat_w = None
        
        sync_futures = []
        for name in client_names:
            sync_futures.append(rpc.rpc_async(
                name, 
                distributed_utils.remote_set_params, 
                args=(send_payload, server_flat_w)
            ))
        [f.wait() for f in sync_futures]
        broadcast_time = time.time() - broadcast_start

        # 2. 触发训练与聚合；DP 裁剪加噪由客户端的 use_dp 分支控制。
        train_start = time.time()
        print(f"  [Standard] Round {r}: Requesting weights from clients...", flush=True)
        train_futures = []
        for name in client_names:
            train_futures.append(rpc.rpc_async(name, distributed_utils.remote_train, args=(r,)))
        results = [f.wait() for f in train_futures]

        client_params = []
        for res in results:
            gpu_state = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in res['state_dict'].items()}
            client_params.append(gpu_state)

        agg_weights = [size_weights[i] for i in range(len(results))]
        if args.server_method == 'fedavg':
            avg_params = fedavg(client_params, agg_weights)
            central_node.model.load_state_dict(avg_params, strict=False)
        elif args.server_method == 'feddfw':
            mask_dict = {}
            for i, res in enumerate(results):
                if 'mask_dict' in res:
                    mask_dict[i] = {k: v.to(args.device) if hasattr(v, 'to') else v for k, v in res['mask_dict'].items()}
            avg_params, cur_T_weight = feddfw(
                args, client_params, agg_weights, central_node, r, global_T_weights,
                mask_dict=mask_dict if mask_dict else None, select_list=list(range(num_clients))
            )
            global_T_weights = cur_T_weight
            central_node.model.load_state_dict(avg_params, strict=False)

        # 性能统计
        train_total_wall_time = time.time() - train_start
        train_avg_comp_time = sum([res['train_time'] for res in results]) / num_clients
        sent, recv = monitor.get_diff()
        total_net_sent += sent
        total_net_recv += recv
        max_client_train_time = max([res['train_time'] for res in results])
        round_c_time = max(0, train_total_wall_time - max_client_train_time) + broadcast_time
        
        comm_history.append({
            'round': r + 1, 'sent_bytes': sent, 'recv_bytes': recv,
            'absolute_comm_bytes': sent + recv, 'comm_time_sec': round_c_time,
            'max_client_train_time_sec': max_client_train_time
        })
        print(f"  Round {r} Communication Overhead: {round_c_time:.2f}s", flush=True)
        print(f"  [Traffic] Sent: {NetworkMonitor.format_bytes(sent)}, Recv: {NetworkMonitor.format_bytes(recv)}", flush=True)

        # 验证
        from utils import validate
        acc = validate(args, central_node, which_dataset='local')
        test_acc_recorder.append(acc)
        print(f"  [Result] Global Test Acc: {acc:.4f} | Current Best: {max(test_acc_recorder):.4f}")

    # 最终报告
    print("\n" + "="*20 + " Final Communication Report " + "="*20)
    print("--- Detailed Per-Round Absolute Communication Cost ---")
    for item in comm_history:
        # 使用 NetworkMonitor.format_bytes 使输出更易读
        comm_size_str = NetworkMonitor.format_bytes(item['absolute_comm_bytes'])
        print(f" Round {item['round']:3d}: {comm_size_str:10s}  (Time: {item['comm_time_sec']:.4f} s)")
    
    print("\n--- Summary ---")
    print(f"Total Physical Sent       : {NetworkMonitor.format_bytes(total_net_sent)}")
    print(f"Total Physical Recv       : {NetworkMonitor.format_bytes(total_net_recv)}")
    avg_comm_time = sum([item['comm_time_sec'] for item in comm_history]) / len(comm_history) if comm_history else 0
    print(f"Avg Comm Time / Round     : {avg_comm_time:.4f} s")
    print("="*60)
    
    if args.comm_log_path:
        import pandas as pd
        pd.DataFrame(comm_history).to_csv(args.comm_log_path, index=False)

def main():
    import sys
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master_port', type=str, default='29500')
    parser.add_argument('--nic', type=str, default='eth0')
    parser.add_argument('--world_size', type=int, default=9)
    parser.add_argument('--comm_log_path', type=str, default='comm_log.csv')
    
    dist_args, remaining_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining_args
    temp_args = args_parser()
    temp_args.nic = dist_args.nic
    temp_args.world_size = dist_args.world_size
    temp_args.node_num = dist_args.world_size - 1
    temp_args.comm_log_path = dist_args.comm_log_path
    
    os.environ['MASTER_ADDR'] = dist_args.master_addr
    os.environ['MASTER_PORT'] = dist_args.master_port
    rank = dist_args.rank
    
    print(f"[Rank {rank}] Loading dataset...", flush=True)
    data = Data(temp_args, rank=rank)
    if rank != 0:
        import distributed_utils
        distributed_utils.init_worker(rank-1, data.train_loader[rank-1], data.train_set, temp_args)
    
    # 增加 RPC 全局超时时间：仅在支持的环境下设置
    options = None
    if hasattr(rpc, "TensorPipeRpcBackendOptions"):
        options = rpc.TensorPipeRpcBackendOptions(rpc_timeout=86400.0)

    if rank == 0:
        print(f"Initializing Server [Rank 0] at {dist_args.master_addr}:{dist_args.master_port}", flush=True)
        if options is not None:
            rpc.init_rpc("server", rank=0, world_size=dist_args.world_size, rpc_backend_options=options)
        else:
            rpc.init_rpc("server", rank=0, world_size=dist_args.world_size)
        run_server(temp_args, data)
    else:
        print(f"Initializing Client_{rank-1} [Rank {rank}] connecting to {dist_args.master_addr}", flush=True)
        if options is not None:
            rpc.init_rpc(f"client_{rank-1}", rank=rank, world_size=dist_args.world_size, rpc_backend_options=options)
        else:
            rpc.init_rpc(f"client_{rank-1}", rank=rank, world_size=dist_args.world_size)
    
    rpc.shutdown()

if __name__ == '__main__':
    main()
