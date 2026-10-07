import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # Data
    parser.add_argument('--noniid_type', type=str, default='dirichlet',
                        help="iid or dirichlet")
    parser.add_argument('--iid', type=int, default=1,  
                        help='set 1 for iid, 0 for non-iid')
    parser.add_argument('--batchsize', type=int, default=32, 
                        help="batchsize")
    parser.add_argument('--validate_batchsize', type=int, default=64, 
                        help="batchsize")
    parser.add_argument('--dirichlet_alpha', type=float, default=1.0, 
                    help="dirichlet_alpha")
    parser.add_argument('--dirichlet_alpha2', type=float, default=False, 
                    help="dirichlet_alpha2")
    parser.add_argument('--longtail_proxyset', type=str, default='none',
                    help="longtail_proxyset")
    parser.add_argument('--longtail_clients', type=str, default='none', 
                    help="longtail_clients")

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        help=".to(args.device): {cuda, cpu}")
    
    # Distributed
    parser.add_argument('--distributed', type=int, default=0,
                        help="设置 1 以开启真实的 RPC 多机分布式通信训练")
    parser.add_argument('--rank', type=int, default=0,
                        help="分布式节点的级别 (0代表中央 Server, 1~N 代表真实的 Client边缘节点)")
    parser.add_argument('--master_addr', type=str, default='127.0.0.1',
                        help="分布式主节点(Server)监听的 IP 地址")
    parser.add_argument('--master_port', type=str, default='29500',
                        help="分布式主节点(Server)监听的端口号")
    

    


    
    parser.add_argument('--node_num', type=int, default=10, # 200
                        help="Number of nodes")
    parser.add_argument('--T', type=int, default=3,  # 100 
                        help="Number of communication rounds")
    parser.add_argument('--E', type=int, default=1, # 3
                        help="Number of local epochs: E")
    parser.add_argument('--dataset', type=str, default='CIC-IDS2017',
                        help="Type of algorithms:{CIC-IDS2017, IOV2024, NSLKDD, UNSW, car-hacking}") 
    parser.add_argument('--select_ratio', type=float, default=1,
                    help="the ratio of client selection in each round")
    parser.add_argument('--local_model', type=str, default='MLP',
                        help='Type of local model: {CNN, MLP, SE_ResNet18}')
    parser.add_argument('--random_seed', type=int, default=12,
                        help="random seed for the whole experiment")
    parser.add_argument('--exp_name', type=str, default='FirstTable',
                        help="experiment name")
    


    # Server function
    parser.add_argument('--server_method', type=str, default='feddfw', 
                        help="fedavg, feddfw")
    parser.add_argument('--server_valid_ratio', type=float, default=0.02, 
                    help="the ratio of validate set (proxy dataset) in the central server")
    parser.add_argument('--server_epochs', type=int, default=1,
                        help="optimizer epochs on server")
    parser.add_argument('--server_optimizer', type=str, default='sgd',
                        help="type of server optimizer, adam or sgd")
    parser.add_argument('--gamma', type=float, default=1.0,
                        help="vector_scale")
    parser.add_argument('--reg_distance', type=str, default='euc',  
                        help="cos or euc")
    # PUC (Parameter Update Consistency)：参数更新一致性机制开关。
    parser.add_argument('--puc', type=int, default=1,
                        help='参数更新一致性筛选')
    parser.add_argument('--threshold', type=float, default=0.4, 
                        help='mask 判断的一致性阈值')
    parser.add_argument('--Lambda', type=float, default=1.0, 
                        help='reg_loss的权重')
    parser.add_argument('--disco', type=int, default=0, 
                        help='whether to use disco aggregation')
    parser.add_argument('--measure_difference', type=str, default='kl', 
                        help='how to measure difference.{kl, cosine, only_iid, l1, l2}')
    parser.add_argument('--disco_a', type=float, default=0.5, 
                        help='under sub mode, n_k-disco_a*d_k+disco_b')
    parser.add_argument('--disco_b', type=float, default=0.1)
    parser.add_argument('--norm_methods_per_layer', type=str, default='[g,g,g]',
                        help='''for CNN model: uses astring list to detrimne the function per layer
                                (s:mean shift, n: l2 norm, v:l2 norm with variance scale, c: multipy by the numbers after c
                                g: group normalization, b: batch normalization, x:learnable parameter) ''')                    
    # Client function
    parser.add_argument('--client_method', type=str, default='local_train',
                        help="local_train, fedprox")
    parser.add_argument('--optimizer', type=str, default='sgd',
                        help="optimizer: {sgd, adam}")
    parser.add_argument('--client_valid_ratio', type=float, default=0.2,
                    help="the ratio of validate set in the clients")  
    parser.add_argument('--lr', type=float, default=0.001,
                        help='clients loca learning rate')
    parser.add_argument('--local_wd_rate', type=float, default=1e-4,
                        help='clients local wd rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='clients SGD momentum')
    parser.add_argument('--mu', type=float, default=0.1,
                        help="clients proximal term mu for FedProx")    
    parser.add_argument('--alpha', type=float, default=0.8,
                        help='attack loss权重系数')
    # DP 总开关；启用后可通过 use_sparse_adaclip 选择自适应裁剪分支。
    parser.add_argument('--use_dp', type=int, default=0 ,
                        help="是否启用差分隐私")  
    parser.add_argument('--dp_max_grad_norm', type=float, default=4.0,
                        help="梯度裁剪阈值") 
    parser.add_argument('--dp_noise_multiplier', type=float, default=0.8,
                        help="噪声系数（越大隐私越强）") 
    parser.add_argument('--dp_delta', type=float, default=1e-5,
                        help="隐私参数δ")     
    parser.add_argument('--dp_epsilon_limit', type=float, default=10.0,
                        help="隐私预算")     
    parser.add_argument('--use_sparse_adaclip', type=int, default=0,
                        help="在 use_dp=1 时启用 SparseAdaCliP 自适应裁剪分支")
    parser.add_argument('--topk_ratio', type=float, default=0.6,
                        help="SparseAdaCliP 初始保留参数比例")
    parser.add_argument('--use_mask_release', type=int, choices=[0, 1], default=1,
                        help="是否逐轮释放初始屏蔽参数")
    parser.add_argument('--release_ratio', type=float, default=0.2,
                        help="use_mask_release=1 时每轮释放剩余屏蔽参数的比例")
    parser.add_argument('--current_round', type=int, default=0,
                        help="当前轮次")    



    args = parser.parse_args()

    return args
