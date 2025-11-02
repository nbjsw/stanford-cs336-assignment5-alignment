import torch
from einops import rearrange, reduce
from typing import Callable


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
):
    """
    Compute rewards for each group of rollout responses, normalized by the group size.

    Args:
        reward_fn: Scores the rollout responses against the ground truths,
            producing a dict with keys "reward", "format_reward", and "answer_reward".
        rollout_responses: Rollouts from the policy. The length of this list is
            rollout_batch_size = n_prompts_per_rollout_batch * group_size.
        repeated_ground_truths: The ground truths for the examples. The length of this
            list is rollout_batch_size, because the ground truth for each example is repeated
            group_size times.
        group_size: Number of responses per question (group).
        advantage_eps: Small constant to avoid division by zero in normalization.
        normalize_by_std: If True, divide by the per-group standard deviation; otherwise
            subtract only the group mean.

    Returns:
        tuple[torch.Tensor, torch.Tensor, dict[str, float]].
            advantages shape (rollout_batch_size,). Group-normalized rewards for each rollout
                response.
            raw_rewards shape (rollout_batch_size,). Unnormalized rewards for each rollout response.
            metadata your choice of other statistics to log (e.g. mean, std, max/min of rewards).
    """
    rollout_batch_size = len(rollout_responses)
    if rollout_batch_size % group_size != 0:
        raise ValueError("Rollout batch size must be divisible by group_size.")        
    num_groups = rollout_batch_size // group_size

    # 步骤 1: 计算原始回报 (r^(i))
    raw_rewards_list = []
    for response, gt in zip(rollout_responses, repeated_ground_truths):
        # 假设 reward_fn 返回的字典中，'reward' 键包含了用于 RL 的最终分数
        score = reward_fn(response, gt)['reward']
        raw_rewards_list.append(score)

    # 转换为 PyTorch Tensor，并移动到默认设备 (通常是 CPU/GPU)
    # shape (rollout_batch_size, )
    raw_rewards = torch.tensor(raw_rewards_list, dtype=torch.float32)

    # 使用 einops 库的 reduce 函数，一步完成了张量的分组、求平均（归约）和保持维度三个复杂的任务
    # shape: (num_groups, 1)
    grouped_mean_rewards = reduce(
        raw_rewards, '(g i) -> g 1', 'mean', g=num_groups, i=group_size)

    # shape: (num_groups, group_size)
    grouped_rewards = rearrange(
        raw_rewards, '(g i) -> g i', g=num_groups, i=group_size)

    # Broadcasting
    advantage_numerator = grouped_rewards - grouped_mean_rewards

    if normalize_by_std:
        # 计算标准差
        # shape: (num_groups, 1)
        std_rewards = torch.std(grouped_rewards, dim=1, keepdim=True)

        # 计算分母: std(...) + advantage_eps
        advantage_denominator = std_rewards + advantage_eps

        # 计算优势
        advantages = advantage_numerator / advantage_denominator
    else:
        # Dr. GRPO 简化变体
        advantages = advantage_numerator

    # 将优势变平 (Flatten) 回到 batch shape (rollout_batch_size)
    # 'g i -> (g i)' 表示将 (组数 g, 组内索引 i) 重新展平
    advantages = rearrange(advantages, 'g i -> (g i)')

    # 收集元数据
    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "reward_std": raw_rewards.std().item(),
        "reward_max": raw_rewards.max().item(),
        "reward_min": raw_rewards.min().item(),
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std().item(),
        "advantage_max": advantages.max().item(),
        "advantage_min": advantages.min().item(),
    }

    return advantages, raw_rewards, metadata

