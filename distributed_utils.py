from nodes import Node
from utils import model_parameter_vector
import time


class RPCNodeWrapper:
    def __init__(self, node_id, local_loader, train_set, args):
        self.node = Node(node_id, local_loader, train_set, args)

        # SparseAdaCliP 是 use_dp 控制的 DP 训练子分支。
        if getattr(args, "use_dp", False) and getattr(args, "use_sparse_adaclip", False):
            print(f"  [Client {node_id}] Starting SparseAdaCliP Warmup (Importance Scoring)...", flush=True)
            from client_funct import accumulate_importance, generate_topk_mask, MaskScheduler

            # 1. 计算重要性得分 (预热)
            importance_scores = accumulate_importance(self.node, pretrain_epochs=5, args=args)

            # 2. 生成初始掩码
            target_ratio = getattr(args, "topk_ratio", 0.6)
            base_masks = generate_topk_mask(importance_scores, prune_fraction=1.0 - target_ratio)

            # 每轮释放剩余屏蔽参数的 release_ratio，转换为累计释放比例。
            release_ratio = getattr(args, "release_ratio", 0.6)
            self.node.mask_scheduler = MaskScheduler(
                base_mask=base_masks,
                importance_scores=importance_scores,
                release_schedule=[1.0 - (1.0 - release_ratio) ** (r + 1) for r in range(args.T)],
                use_mask_release=getattr(args, "use_mask_release", True)
            )
            print(f"  [Client {node_id}] MaskScheduler initialized (Ratio: {target_ratio}).", flush=True)

    def get_model_params(self):
        return model_parameter_vector(self.node.args, self.node.model)

    def set_model_params(self, server_param, server_flat_w=None):
        """同步模型参数，支持仅传输单张量模式。"""
        # 如果 server_param 为 None，说明使用了单一 Flat 传输模式
        if server_param is None and server_flat_w is not None:
            from utils import load_flat_params_to_model
            load_flat_params_to_model(self.node.model, server_flat_w)
        elif server_param is not None:
            # 兼容模式：传输的是字典
            self.node.model.load_state_dict(server_param, strict=False)

    def local_train(self, rounds):
        from client_funct import Client_update_single
        # RPC 下发的轮次必须同步到 DP 训练读取的调度轮次。
        self.node.args.current_round = rounds
        start_time = time.time()
        loss = Client_update_single(self.node.args, self.node, rounds)
        train_time = time.time() - start_time
        return {
            'state_dict': {k: v.cpu() for k, v in self.node.model.state_dict().items()},
            'loss': loss, 'train_time': train_time
        }


global_rpc_node = None


def init_worker(node_id, local_loader, train_set, args):
    global global_rpc_node
    global_rpc_node = RPCNodeWrapper(node_id, local_loader, train_set, args)
    print(f"Client {node_id} initialized and waiting...")


def remote_train(rounds):
    return global_rpc_node.local_train(rounds)


def remote_get_params():
    return global_rpc_node.get_model_params()


def remote_set_params(server_param, server_flat_w=None):
    """同步服务端下发的模型参数。"""
    global_rpc_node.set_model_params(server_param, server_flat_w)


def rpc_ping(): pass
