import torch
import torch.nn.functional as F
import math

# todo
class ImportanceTokenMerging:
    def __init__(self, p=0.4, k=0.25, **kwargs):
        self.pool_param = p
        self.dst_ratio_k = k

    def reduce(self, x, r, global_context, **kwargs):
        """
        根据重要性分数缩减 token。

        参数:
        - x (Tensor): 输入 token，形状 (B, N, C)。B 必须为 2 (uncond, cond)。
        - r (float): 要合并/移除的 token 比例 (merging ratio)。
        - global_context: 包含 'last_cfg_hidden_state' 的全局上下文。

        返回:
        - x_reduced (Tensor): 缩减后的 token，形状 (B, N_final, C)。
        """

        # 如果合并率为 0，则不执行任何操作
        if r == 0:
            global_context.importance_unmerge = None
            return x

        B, N, C = x.shape

        # 1. 获取重要性分数 (来自论文 Eq. 2)
        cfg_hidden_state = global_context.last_cfg_hidden_state

        # (1, N, C) -> (1, N) -> (N,)
        importance = cfg_hidden_state.sum(dim=-1)
        importance_scores_orig = importance[0]

        # 2. 计算各种大小
        N_final = int(N * (1 - r))
        N_dst = int(N * self.dst_ratio_k)
        N_ind = N_final - N_dst

        # 3. 处理大小的边缘情况
        # 如果 k (dst_ratio_k) 太大，导致 N_ind 为负，则调整
        if N_ind < 0:
            N_ind = 0
            N_dst = N_final

        pool_size = min(int(N_final * (1 + self.pool_param)), N)

        # 确保 token 池至少能容纳 N_dst 个 token
        if pool_size < N_dst:
            pool_size = min(N_dst, N)

        # 确保我们不会试图从 src_pool 中选择比它本身还多的 ind token
        N_src_pool_max = pool_size - N_dst
        if N_ind > N_src_pool_max:
            N_ind = max(0, N_src_pool_max)
            N_dst = N_final - N_ind  # 重新计算 N_dst 以保持 N_final 不变

        # 4. 根据重要性划分 token

        # 4.1. 获取重要 token 池 (Important Token Pool)
        pool_indices = torch.topk(
            importance_scores_orig, k=pool_size, dim=-1
        ).indices  # (pool_size,)

        # 4.2. 从池中随机选择 dst 和 src_pool
        rand_indices = torch.randperm(pool_size, device=x.device)
        dst_indices_in_pool = rand_indices[:N_dst]
        src_indices_in_pool = rand_indices[N_dst:]  # (pool_size - N_dst)

        # 4.3. 获取这些 token 的 *原始* 索引
        dst_indices_orig = pool_indices[dst_indices_in_pool]  # (N_dst,)
        src_pool_indices_orig = pool_indices[
            src_indices_in_pool
        ]  # (pool_size - N_dst,)

        # 4.4. 获取 "不重要" token (不在池中的)
        pool_mask = torch.zeros(N, device=x.device, dtype=torch.bool)
        pool_mask.scatter_(dim=0, index=pool_indices, value=True)
        unimportant_indices_orig = (~pool_mask).nonzero(as_tuple=True)[
            0
        ]  # (N - pool_size,)

        # 4.5. 组合完整的 src 列表 (src_pool + unimportant)
        src_indices_orig = torch.cat(
            [src_pool_indices_orig, unimportant_indices_orig]
        )  # (N - N_dst,)

        # 4.6. 根据原始索引 gathering 张量
        x_dst = x.gather(
            dim=1, index=dst_indices_orig.unsqueeze(0).unsqueeze(-1).expand(B, -1, C)
        )
        x_src = x.gather(
            dim=1, index=src_indices_orig.unsqueeze(0).unsqueeze(-1).expand(B, -1, C)
        )

        # 5. 计算相似度并找到最佳分配 (论文 3. Method)
        with torch.no_grad():  # 相似度计算不需要梯度
            x_src_norm = F.normalize(x_src, dim=-1)
            x_dst_norm = F.normalize(x_dst, dim=-1)
            # (B, N_src, N_dst)
            sim = x_src_norm @ x_dst_norm.transpose(-2, -1)

        # sim_idx 是每个 src token 应该分配到的 dst token 的*索引* (在 dst 集合内)
        sim_max, sim_idx = sim.max(dim=-1)  # (B, N_src)

        # 6. 找到独立 (ind) Token
        N_src_pool = src_pool_indices_orig.shape[0]
        if N_ind > 0 and N_src_pool > 0:
            # 只在 src_pool 中寻找 ind token
            sim_max_pool = sim_max[:, :N_src_pool]  # (B, N_src_pool)

            # 使用 batch 0 的相似度作为索引 (B=2 时两者相同)
            # 找到相似度 *最小* 的 N_ind 个 token
            ind_indices_in_src_pool = torch.topk(
                sim_max_pool[0], k=N_ind, dim=-1, largest=False
            ).indices  # (N_ind,)

            # 获取 ind token 的原始索引
            ind_indices_orig = src_pool_indices_orig[
                ind_indices_in_src_pool
            ]  # (N_ind,)

            # 获取 ind token 的张量
            x_ind = x.gather(
                dim=1,
                index=ind_indices_orig.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
            )
        else:
            # 没有 ind token
            ind_indices_in_src_pool = torch.tensor(
                [], device=x.device, dtype=torch.long
            )
            ind_indices_orig = torch.tensor([], device=x.device, dtype=torch.long)
            x_ind = torch.empty(B, 0, C, device=x.device, dtype=x.dtype)

        # 7. 找到需要合并 (merge) 的 Token

        # 7.1. 在 src_pool 中, 找到 *不是* ind 的 token
        ind_mask_in_src_pool = torch.zeros(
            N_src_pool, device=x.device, dtype=torch.bool
        )
        if N_ind > 0:
            ind_mask_in_src_pool.scatter_(
                dim=0, index=ind_indices_in_src_pool, value=True
            )

        merge_indices_in_src_pool = (~ind_mask_in_src_pool).nonzero(as_tuple=True)[0]
        src_pool_merge_indices_orig = src_pool_indices_orig[merge_indices_in_src_pool]

        # 7.2. 所有要合并的 token = (src_pool - ind) + unimportant
        all_merge_indices_orig = torch.cat(
            [src_pool_merge_indices_orig, unimportant_indices_orig]
        )  # (N - N_final,)

        # 7.3. 找到这些要合并的 token 各自的分配 (assignments)
        sim_idx_pool = sim_idx[:, :N_src_pool]
        sim_idx_unimportant = sim_idx[:, N_src_pool:]

        if merge_indices_in_src_pool.shape[0] > 0:
            assignments_pool_merge = sim_idx_pool.gather(
                dim=1, index=merge_indices_in_src_pool.unsqueeze(0).expand(B, -1)
            )
        else:
            assignments_pool_merge = torch.empty(
                B, 0, device=x.device, dtype=torch.long
            )

        all_merge_assignments = torch.cat(
            [assignments_pool_merge, sim_idx_unimportant], dim=1
        )  # (B, N - N_final)

        # 8. 执行平均合并 (Merging via averaging)
        x_dst_merged = x_dst.clone()

        if all_merge_indices_orig.shape[0] > 0:
            # (B, N_merge, C)
            x_merge = x.gather(
                dim=1,
                index=all_merge_indices_orig.unsqueeze(0)
                .unsqueeze(-1)
                .expand(B, -1, C),
            )

            # (B, N_dst, 1)
            counts = torch.ones(B, N_dst, 1, device=x.device, dtype=x.dtype)

            # 必须按 batch 循环执行 index_add_
            for i in range(B):
                counts[i].index_add_(
                    dim=0,
                    index=all_merge_assignments[i],
                    source=torch.ones(
                        all_merge_assignments.shape[1],
                        1,
                        device=x.device,
                        dtype=x.dtype,
                    ),
                )
                x_dst_merged[i].index_add_(
                    dim=0, index=all_merge_assignments[i], source=x_merge[i]
                )

            x_dst_merged = x_dst_merged / counts

        # 9. 组合最终的缩减张量并保存 unmerge 信息
        x_reduced = torch.cat([x_dst_merged, x_ind], dim=1)

        # 使用 batch 0 的分配，因为它们是相同的
        unmerge_info = (
            dst_indices_orig,  # (N_dst,)
            ind_indices_orig,  # (N_ind,)
            all_merge_indices_orig,  # (N - N_final,)
            all_merge_assignments[0],  # (N - N_final,) - 映射到 dst_indices 的索引
            N,  # 原始 N
            N_dst,  # Dst 数量
        )
        global_context.importance_unmerge = unmerge_info

        return x_reduced

    def unmerge(self, x_reduced, global_context, **kwargs):
        """
        将缩减后的 token 广播回原始大小。
        """
        unmerge_info = global_context.importance_unmerge

        # 如果没有 unmerge 信息 (例如 r=0)，直接返回
        if unmerge_info is None:
            return x_reduced

        dst_indices, ind_indices, merged_indices, merged_assignments, N, N_dst = (
            unmerge_info
        )

        B, N_final, C = x_reduced.shape

        # 分离 dst 和 ind 的输出
        x_dst_out = x_reduced[:, :N_dst, :]
        x_ind_out = x_reduced[:, N_dst:, :]

        # 创建一个空的完整张量
        x_full = torch.empty(B, N, C, device=x_reduced.device, dtype=x_reduced.dtype)

        # 1. 将 dst token 的输出 scatter 回原始位置
        x_full.scatter_(
            dim=1,
            index=dst_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
            src=x_dst_out,
        )

        # 2. 将 ind token 的输出 scatter 回原始位置
        if ind_indices.shape[0] > 0:
            x_full.scatter_(
                dim=1,
                index=ind_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
                src=x_ind_out,
            )

        # 3. 将 merged token 的输出 scatter 回原始位置
        if merged_indices.shape[0] > 0:
            # 从 dst 的输出中 gather 那些被合并的 token 的输出
            x_merged_out = x_dst_out.gather(
                dim=1,
                index=merged_assignments.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
            )
            # 将它们 scatter 回去
            x_full.scatter_(
                dim=1,
                index=merged_indices.unsqueeze(0).unsqueeze(-1).expand(B, -1, C),
                src=x_merged_out,
            )

        return x_full
