import torch
from einops import rearrange, reduce
from typing import Callable, Literal


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


# 在 LLM 强化学习 (RLHF) 的这个阶段（如 GRPO 或 PPO），回报 (Reward) 或优势 (Advantage) 是对整个生成的 Response 序列的一次性评价，
# 是一个标量，而非针对序列中的每个 Token 的。1. 奖励（Reward）的性质在您查看的这个作业背景中（使用 MATH 数据集进行推理 RL），
# 奖励函数的定义是稀疏的 (Sparse) 和终端的 (Terminal) 1：稀疏奖励：模型在生成 Response 的中间步骤（即 $t=0$ 到 $T-1$）获得的奖励 $r_t$ 被设为 零 (0) 
# 2。终端奖励：只有在 Response 结束时（即采取终端动作 $a_T$ 时），奖励 $r_T$ 才会被计算
# 作业中的 Advantage 是一个特例（序列不变）在您这份 CS336 Assignment 5 的具体实现中，这个矛盾是由于 LLM RLHF 的特殊设定 导致的。
# 在您的作业中，Advantage $A_t$ 被简化并视为在序列上是不变的。
# 在传统的策略梯度（如 REINFORCE 或带 V-Function 的 A2C）中，优势函数是时变的：$A_t = R(\tau) - V(s_t)$。
def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the policy-gradient loss at every token, where raw_rewards_or_advantages is either
    the raw reward or an already-normalized advantage.

    Args:
        raw_rewards_or_advantages: Shape (batch_size, 1), scalar reward/advantage for each rollout response.
        policy_log_probs: Shape (batch_size, sequence_length), logprobs for each token.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length), the per-token policy-gradient loss (to
        be aggregated across the batch and sequence dimensions in the training loop).
    """
    per_token_gradient_term = raw_rewards_or_advantages * policy_log_probs
    naive_policy_gradient_loss = -per_token_gradient_term
    return naive_policy_gradient_loss


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
   cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Computes the per-token GRPO-Clip loss.

    Args:
        advantages: torch.Tensor Shape (batch_size, 1), per-example advantages A.
        policy_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log
        probs from the policy being trained.
        old_log_probs: torch.Tensor Shape (batch_size, sequence_length), per-token log probs
            from the old policy.
        cliprange: float Clip parameter ϵ (e.g. 0.2).

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss torch.Tensor of shape (batch_size, sequence_length), the per-token clipped loss.
        metadata dict containing whatever you want to log. We suggest logging whether each 
            token was clipped or not, i.e., whether the clipped policy gradient loss on the RHS of
            the min was lower than the LHS.
    """
    # 1. 计算新旧策略的对数概率之差 (log(pi_new) - log(pi_old))
    # 形状: (batch_size, sequence_length)
    log_ratio = policy_log_probs - old_log_probs

    # 2. 计算策略比率 (Ratio): r_t(theta) = exp(log_ratio)
    # 形状: (batch_size, sequence_length)
    ratio = torch.exp(log_ratio)

    # --- PPO/GRPO Clipping Objective (最大化目标 J) 的右项 ---
    # 3a. 计算裁剪项: clip(r_t(theta), 1-eps, 1+eps)
    clipped_ratio = torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)

    # 4. 计算最终的 PPO/GRPO Objective J (最大化目标)
    # PPO Objective: min(J_LHS, J_RHS)
    # 形状: (batch_size, sequence_length)
    ppo_objective = torch.min(ratio * advantages, clipped_ratio * advantages)

    # 5. 计算最终的 Loss L (最小化损失)
    # Loss = -J (为了使用梯度下降最小化 Loss 来实现最大化 Objective)
    loss = -ppo_objective

    # 6. 计算元数据 (Metadata)
    # 我们需要知道有多少 token 被裁剪了。
    # 对于 A >= 0: J = min(unclipped, clipped)
    #   如果 unclipped > clipped，说明被裁剪了 (即 ratio > 1 + cliprange)
    # 对于 A < 0: J = min(unclipped, clipped)
    #   如果 unclipped < clipped，说明被裁剪了 (即 ratio < 1 - cliprange)

    # Note: 这里的逻辑比单纯比较 unclipped_term 和 clipped_term 更健壮，
    # 因为它直接基于 ratio 和 cliprange 检查。

    # 检查 ratio 是否超出 [1-eps, 1+eps] 范围
    is_clipped = (ratio > 1.0 + cliprange) | (ratio < 1.0 - cliprange)

    # 计算裁剪率 (Clip Fraction): 占总数的比例 (这里返回的是布尔张量，后续求平均)
    clip_fraction = is_clipped.float()

    metadata = {
        "clip_fraction": clip_fraction,
    }

    return loss, metadata


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """
    Select and compute the desired policy-gradient loss.

    Args:
        policy_log_probs (batch_size, sequence_length), per-token log-probabilities from the
        policy being trained.
        loss_type One of "no_baseline", "reinforce_with_baseline", or "grpo_clip".
        raw_rewards Required if loss_type == "no_baseline"; shape (batch_size, 1).
        advantages Required for "reinforce_with_baseline" and "grpo_clip"; shape (batch_size, 1).
        old_log_probs Required for "grpo_clip"; shape (batch_size, sequence_length).
        cliprange Required for "grpo_clip"; scalar ϵ used for clipping.

    Returns:
        tuple[torch.Tensor, dict[str, torch.Tensor]].
        loss (batch_size, sequence_length), per-token loss.
        metadata dict, statistics from the underlying routine (e.g., clip fraction for GRPO-Clip).
    """
    # 初始化空的元数据字典
    metadata: dict[str, torch.Tensor | float | Any] = {"loss_type": loss_type}

    if loss_type == "no_baseline":
        # REINFORCE (无基线)
        if raw_rewards is None:
            raise ValueError(
                "For loss_type='no_baseline', raw_rewards is required."
            )
        loss = compute_naive_policy_gradient_loss(
            raw_rewards,
            policy_log_probs,
        )
        # Naive loss 不返回额外的 metadata，在这里添加一个空的
        metadata["used_input"] = "raw_rewards"

    elif loss_type == "reinforce_with_baseline":
        # REINFORCE with Baseline (使用优势值 A)
        if advantages is None:
            raise ValueError(
                "For loss_type='reinforce_with_baseline', advantages is required."
            )
        
        loss = compute_naive_policy_gradient_loss(
            advantages,
            policy_log_probs,
        )
        metadata["used_input"] = "advantages"

    elif loss_type == "grpo_clip":
        # PPO / GRPO Clip Loss
        if advantages is None or old_log_probs is None or cliprange is None:
            raise ValueError(
                "For loss_type='grpo_clip', advantages, old_log_probs, and cliprange are all required."
            )

        # 调用前面修复好的 GRPO Clip Loss 函数
        loss, grpo_metadata = compute_grpo_clip_loss(
            advantages,
            policy_log_probs,
            old_log_probs,
            cliprange,
        )
        metadata.update(grpo_metadata)
        metadata["used_input"] = "advantages"
        
    else:
        # 兜底：处理非法的 loss_type
        raise ValueError(
            f"Unknown loss_type: {loss_type}. Must be one of 'no_baseline', 'reinforce_with_baseline', or 'grpo_clip'."
        )

    # 记录最终的平均损失（通常用于训练循环的日志）
    metadata["mean_loss"] = loss.mean().item()

    return loss, metadata

