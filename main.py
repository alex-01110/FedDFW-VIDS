from datasets import Data
from nodes import Node
from args import args_parser
from utils import *
from server_funct import *
from client_funct import *
import os
import pandas as pd
args = args_parser()
import time
from collections import Counter
import os


if __name__ == '__main__':

    # Set random seeds
    setup_seed(args.random_seed)
    print(args)

    # Loading data (必须在分布式节点握手前完成读取装载)
    if args.dataset == 'IOTdataset':
        args.node_num = 8
    from datasets import Data
    data = Data(args)

    # ================== Distributed Init =================
    if args.distributed:
        import torch.distributed.rpc as rpc
        import distributed_utils
        import os
        
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = str(args.master_port)
        
        # ================== 强制走物理网卡的终极封杀令 ==================
        nic_name = 'lo' if args.master_addr in ['127.0.0.1', 'localhost'] else 'eth0'
        os.environ['GLOO_SOCKET_IFNAME'] = nic_name
        os.environ['TP_SOCKET_IFNAME'] = nic_name  # 强制 TensorPipe 也绑定网卡
        
        # 如果你有用到 GPU，彻底禁用 NCCL 的共享内存和 P2P 直接通信
        os.environ['NCCL_SHM_DISABLE'] = '1'
        os.environ['NCCL_P2P_DISABLE'] = '1'
        # ================================================================

        # 配置 TensorPipe 强制使用 TCP ('uv')，禁用内存通道 ('shm', 'cma')
        options = rpc.TensorPipeRpcBackendOptions(
            _transports=["uv"],
            _channels=["basic"]
        )
        
        if args.rank != 0:
            print(f"Initializing Client_{args.rank-1} connecting to {args.master_addr}:{args.master_port} (NIC: {nic_name})")
            # 关键修复：所有 Client 无论如何必须在 init_rpc() 全局大门开启前，完全初始化完备底层的 Dataset 参数与代理！
            distributed_utils.init_worker(args.rank-1, data.train_loader[args.rank-1], data.train_set, args)
            rpc.init_rpc(f"client_{args.rank-1}", rank=args.rank, world_size=args.node_num + 1, rpc_backend_options=options)
            rpc.shutdown()
            exit(0) # 客户端完成生命周期直接退出
        else:
            print(f"Initializing Server at {args.master_addr}:{args.master_port} and waiting for clients... (NIC: {nic_name})")
            rpc.init_rpc("server", rank=0, world_size=args.node_num + 1, rpc_backend_options=options)
    # ====================================================
    # 全局计数 = 所有客户端的类别计数之和
    global_counts = np.sum(data.traindata_cls_counts, axis=0)  # 形状：[n_classes]
    # 归一化为概率分布（总和为1）
    global_dist = global_counts / np.sum(global_counts)
    if args.disco:
        print(f"全局参考分布(global_dist):{global_dist}")
        print(f"全局分布总和：{np.sum(global_dist):.4f}")
  
    # Data-size-based aggregation weights
    sample_size = []
    for i in range(args.node_num): 
        sample_size.append(len(data.train_loader[i]))
    size_weights = [i/sum(sample_size) for i in sample_size]
    
    from utils import get_model_size
    central_node = Node(args.node_num, data.test_loader[0], data.test_set, args)
    print(f"\n[Model Info] Node -1 (Server) Model. {get_model_size(central_node.model)}")
    
    # Initialize the client nodes
    client_nodes = {}
    # Track communication cost per client (in bytes)
    client_comm_bytes = {i: 0 for i in range(args.node_num)}
    model_params = sum(p.numel() for p in central_node.model.parameters())
    model_size_bytes = model_params * 4 # float32

    for i in range(args.node_num): 
        client_nodes[i] = Node(i, data.train_loader[i], data.train_set, args) 
    if getattr(args, "use_dp", False) and getattr(args, "use_sparse_adaclip", False):
        print("\n=============== SparseAdaCliP Warmup Stage ===============")
        
        # topk_ratio 控制初始保留比例，release_ratio 控制每轮释放剩余屏蔽参数的比例。
        target_ratio = getattr(args, "topk_ratio", 0.6)
        release_ratio = getattr(args, "release_ratio", 0.6)
        release_schedule = [1.0 - (1.0 - release_ratio) ** (r + 1) for r in range(args.T)]
        for client_id, node in client_nodes.items():
            print(f"Initializing MaskScheduler for Client {client_id}...")
            
            # 1. 计算重要性得分 (跑 2-5 个 epoch)
            importance_scores = accumulate_importance(node, pretrain_epochs=10, args=args)
            
            # 2. 生成初始掩码 (Base Mask)
            # 计算需要剪枝的比例 = 1.0 - 保留比例
            base_masks = generate_topk_mask(importance_scores, prune_fraction=1.0 - target_ratio)
            
            # 3. 初始化调度器
            node.mask_scheduler = MaskScheduler(
                base_mask=base_masks, 
                importance_scores=importance_scores, 
                release_schedule=release_schedule,
                use_mask_release=getattr(args, "use_mask_release", True)
            )
            
            # 清理
            del importance_scores, base_masks
            torch.cuda.empty_cache()
            
        print("=============== Warmup Finished ===============\n")    
    final_test_acc_recorder = RunningAverage()

    avgtime=[]
    test_acc_recorder = []
    train_loss_recorder = [] # 新增：存储每轮平均训练 Loss

    for rounds in range(args.T):

        print('===============Stage 1 The {:d}-th round==============='.format(rounds + 1))
        start = time.time()
        args.current_round = rounds # 确保轮次信息同步到 args 中
        lr_scheduler(rounds, client_nodes, args)

        if args.distributed:
            import torch.distributed.rpc as rpc
            from comm_monitor import NetworkMonitor
            import copy
            import distributed_utils
            
            if not hasattr(args, 'monitor'): 
                nic = 'lo' if args.master_addr in ['127.0.0.1', 'localhost'] else 'eth0'
                args.monitor = NetworkMonitor(nic_name=nic)
                args.total_net_sent, args.total_net_recv = 0, 0
                args.round_traffic_list = [] # 新增：分轮次流量账单存储器
            
            args.monitor.get_diff() # reset loop
            
            # 0. 抽样逻辑 (分布式与非分布式对齐)
            if args.select_ratio == 1.0:
                select_list = [idx for idx in range(len(client_nodes))]
            else:
                select_list = generate_selectlist(client_nodes, args.select_ratio)

            # 1. 广播全局参数
            if hasattr(central_node.model, 'get_param'):
                server_param = central_node.model.get_param(clone=True)
            else:
                server_param = copy.deepcopy(central_node.model.state_dict())
            
            # 必须把下发参数提前剥离到系统主内存 CPU 上，才能使用底层 TensorPipe 进行万能网络互传
            for k, v in server_param.items():
                if hasattr(v, 'cpu'):
                    server_param[k] = v.cpu()
                    
            server_flat_w = server_param.get('flat_w', None)
            if server_flat_w is not None: server_flat_w = server_flat_w.detach().cpu().clone()
            
            sync_futures = []
            for client_id in select_list: # 原来是全量 client_nodes.keys()，现在修改为受限的 select_list
                if not client_nodes[client_id].active: continue
                sync_futures.append(rpc.rpc_async(f"client_{client_id}", distributed_utils.remote_set_params, args=(server_param, server_flat_w)))
            [f.wait() for f in sync_futures]
            
            # DP 裁剪加噪由客户端的 use_dp 训练分支执行。
            train_futures = []
            for client_id in select_list:
                if not client_nodes[client_id].active: continue
                train_futures.append(rpc.rpc_async(f"client_{client_id}", distributed_utils.remote_train, args=(rounds,)))
                client_comm_bytes[client_id] += 2 * model_size_bytes
            results = [f.wait() for f in train_futures]
            
            for i, client_id in enumerate([cid for cid in select_list if client_nodes[cid].active]):
                payload = results[i]
                node = client_nodes[client_id]
                node.model.load_state_dict(payload['state_dict'], strict=False)
            train_loss = sum([res['loss'] for res in results]) / len(results) if results else 0.0
            
            # 记录流量
            sent, recv = args.monitor.get_diff()
            args.total_net_sent += sent; args.total_net_recv += recv
            divisor = 1.0
            round_cost = (sent + recv) / divisor
            args.round_traffic_list.append(round_cost)
            print(f"Round Traffic [Master: {args.master_addr} | Divisor: {divisor}]: Sent {NetworkMonitor.format_bytes(sent/divisor)}, Recv {NetworkMonitor.format_bytes(recv/divisor)} | Absolute Round Cost: {NetworkMonitor.format_bytes(round_cost)}", flush=True)
        else:
            client_nodes, train_loss = Client_update(args, client_nodes, central_node, rounds)
        
        train_loss_recorder.append(train_loss)
        print(f"Training Loss: {train_loss:.4f}") # 每轮输出 Loss
        if (rounds + 1) % 10 == 0 and getattr(args, "use_dp", False):
            print(f"\n[Round {rounds + 1}] Client Privacy Budget Report:")
            # 遍历所有客户端（或者只遍历参与本轮的 select_list 里的客户端）
            # 这里示例打印所有活跃客户端的预算
            for client_id, node in client_nodes.items():
                if node.active:
                    print(f"  Client {client_id}: ε (epsilon) = {node.epsilon_spent:.4f} / {node.epsilon_limit}")

        avg_client_acc,client_acc = Client_validate(args, client_nodes)
        print(f"client_acc: {client_acc}")
        print(f"{args.server_method},{args.client_method}, averaged clients personalization acc is {avg_client_acc:.4f}")

        

        
        # Partial select function (已经在分布式逻辑中集成，这里仅为非分布式模式保留逻辑)
        if not args.distributed:
            if args.select_ratio == 1.0:
                select_list = [idx for idx in range(len(client_nodes))]
            else:
                select_list = generate_selectlist(client_nodes, args.select_ratio)

            # Track communication for selected clients
            for client_id in select_list:
                # Each selected client:
                # 1. Downloads the global model (model_size)
                # 2. Uploads the local update (model_size)
                client_comm_bytes[client_id] += 2 * model_size_bytes

        # Server update
    
    

        central_node = Server_update(args, central_node, client_nodes, select_list, size_weights,rounds_num=rounds, traindata_cls_counts=data.traindata_cls_counts, global_dist=global_dist)
        
        if torch.cuda.is_available():    
            torch.cuda.synchronize() 
        end = time.time()    

        
        acc = validate(args, central_node, which_dataset = 'local')
        print(f"{args.server_method},{args.client_method}, global model test acc is {acc:.4f}" )
        test_acc_recorder.append(acc)






        print(f'Running time: {(end - start)/60:.4f} Min')
        avgtime.append((end - start)/60)
        best_acc = max(test_acc_recorder)
        print(f"Current_Best test acc is:{best_acc:.4f}")
        # Final acc recorder
        if rounds >= args.T - 10:
            final_test_acc_recorder.update(acc)


    # 保存准确率记录
    pd.DataFrame({
        'round': list(range(1, len(test_acc_recorder) + 1)), 
        'accuracy': test_acc_recorder
    }).to_csv(f"accuracy_record_{args.server_method}_{args.client_method}_puc{args.puc}_{args.dataset}_{args.local_model}_disco{args.disco}_dp{args.use_dp}_adaclip{args.use_sparse_adaclip}.csv", index=False)

   
    
    
    # 保存 Loss 记录
    pd.DataFrame({
        'round': list(range(1, len(train_loss_recorder) + 1)), 
        'loss': train_loss_recorder
    }).to_csv(f"loss_record_{args.server_method}_{args.client_method}_puc{args.puc}_{args.dataset}_{args.local_model}_disco{args.disco}_dp{args.use_dp}_adaclip{args.use_sparse_adaclip}.csv", index=False)

    report_metrics(args, central_node, which_dataset='local')


    if len(test_acc_recorder) >= 10:
        top_10_std = np.std(test_acc_recorder[-10:])
        print(f"Top 10 test acc std is:{top_10_std:.4f}")
    else:
        top_10_std = np.std(test_acc_recorder)
        print(f"Test acc std (all rounds) is:{top_10_std:.4f}")

    # Communication Cost Measurement
    model_params = sum(p.numel() for p in central_node.model.parameters())
    # Each parameter is float32 (4 bytes)
    model_size_mb = (model_params * 4) / (1024 * 1024)
    
    # In each round, selected clients:
    # 1. Download global model (model_size)
    # 2. Upload local updates (model_size)
    # Total per round per client = 2 * model_size
    
    total_comm_mb = 0
    # Current rounds is range(args.T)
    # select_list depends on round, let's track it or estimate if fixed
    # For simplicity, if we don't have per-round track here, we can use the formula:
    # (sum of select_list lengths across all rounds) * 2 * model_size
    # Since select_list depends on rounds, I should have added a counter in the loop.
    # Let's add a counter in the loop or calculate based on select_ratio.
    
    # Assuming select_ratio is used:
    total_selected_clients = 0
    if args.select_ratio == 1.0:
        total_selected_clients = args.node_num * args.T
    else:
        # Number of clients selected per round
        total_selected_clients = int(args.select_ratio * args.node_num) * args.T
        
    total_comm_mb = total_selected_clients * 2 * model_size_mb
    
    print("\n=============== Communication Report ===============")
    print(f"Model Parameters: {model_params:,}")
    print(f"Single Model Size: {model_size_bytes / (1024 * 1024):.2f} MB")
    print(f"Total Communication Rounds: {args.T}")
    
    print("\nPer-Client Communication Cost:")
    total_comm_bytes = 0
    for client_id, comm_bytes in client_comm_bytes.items():
        total_comm_bytes += comm_bytes
        print(f"  Client {client_id:2d}: {comm_bytes / (1024 * 1024):.2f} MB")
    
    print(f"\nTotal Absolute Communication Cost: {total_comm_bytes / (1024 * 1024):.2f} MB ({total_comm_bytes / (1024 * 1024 * 1024):.4f} GB)")
    print("====================================================\n")

    if args.distributed:
        from comm_monitor import NetworkMonitor
        print("\n============= Final Physical Network Traffic ==============")
        
        # 针对本地单机多进程模拟的流量翻倍进行自动除以 2 的修正
        divisor = 1.0
        
        real_total_bytes = (args.total_net_sent + args.total_net_recv) / divisor
        real_avg_client_bytes = real_total_bytes / args.node_num

        print(f"Total Sent (Server Downlink): {NetworkMonitor.format_bytes(args.total_net_sent / divisor)}")
        print(f"Total Recv (Server Uplink):   {NetworkMonitor.format_bytes(args.total_net_recv / divisor)}")
        print(f"*** REAL System Total Comm Cost:   {NetworkMonitor.format_bytes(real_total_bytes)} ***")
        print(f"*** REAL Avg Client Comm Cost:     {NetworkMonitor.format_bytes(real_avg_client_bytes)} ***")
        
        print("\n--- Detailed Per-Round Absolute Communication Cost ---")
        for r, cost in enumerate(args.round_traffic_list):
            print(f" Round {r+1:3d}: {NetworkMonitor.format_bytes(cost)}")
        
        if args.round_traffic_list:
            avg_round_cost = sum(args.round_traffic_list) / len(args.round_traffic_list)
            print(f"\n*** REAL Avg Round Comm Cost:      {NetworkMonitor.format_bytes(avg_round_cost)} ***")
        
        if divisor == 2.0:
            print("======== (Localhost loopback traffic detected, divided by 2 to match true physical traffic) ========")
        else:
            print("======== (This is the real physical RPC traffic) ========")
            
        import torch.distributed.rpc as rpc
        rpc.shutdown()
