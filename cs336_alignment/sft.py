import torch
import torch.nn.functional as F
from transformers import PreTrainedModel, PreTrainedTokenizer

# 为什么这些指标在预训练时相对不那么重要或不被直接优化的原因：
# 1. 预训练的目标：最大化似然（最大化平均对数概率）预训练阶段（例如，使用大量文本数据进行下一个词预测）的目标非常简单和基础：
# 让模型学会语言的基本规律和知识。优化的指标： 模型通常直接优化负对数似然损失 (Negative Log-Likelihood, NLL)，
# 这等价于最小化平均对数概率的负值。$$\text{Loss}_{\text{Pre-train}} = -\frac{1}{N} \sum_{i=1}^N \log p_\theta(y_i | x_i)$$
# Log-Probability (对数概率) 的重要性： 至关重要！ 预训练的核心就是不断提高这个平均对数概率。如果平均 $\log p$ 高，说明模型在整个训练集上拟合得好。

# 2. 熵（Entropy）在预训练时是自然涌现的，而非直接优化目标
# 数据驱动： 预训练数据（如网页、书籍）本身具有巨大的多样性和不确定性。例如，对于句子“我喜欢吃...”，后面可以是“苹果”、“香蕉”、“面条”等等，分布是相对平坦的。

# 自然结果： 模型在学习拟合这个数据分布时，它会自动学会在不确定的地方产生较高的熵，在确定的地方（如句末标点）产生较低的熵。

# 为什么不直接优化熵？ 如果我们强行加入一个目标函数来最小化熵（即鼓励模型过度自信），这会抑制模型的学习能力和多样性。
# 模型会倾向于只记住训练数据中最常见的答案，而无法泛化到新颖的、合理的但训练数据中不常见的答案上。这会导致欠拟合或生成文本的贫乏。

# 3. 微调阶段（SFT/RL）关注的额外因素
# 在微调阶段，我们不再只是要求模型“学会语言”，而是要求它**“根据人类偏好或特定任务进行调整”**：

# SFT（监督微调）： 目标是模仿人类的回答风格。此时，Log-Probability（对正确回答的概率）依然是优化的核心。

# RL（强化学习）： 这是熵变得更重要的阶段。

# 平衡准确性和多样性： 仅仅追求最高的 Log-Probability（即只模仿奖励模型给出的“最佳”答案）会导致模型行为僵化，探索性差，容易陷入局部最优。

# 熵作为正则化项： 在 RL 损失中加入一个基于熵的项（通常是正的，鼓励高熵），就是为了防止策略（Policy）过快地收敛到一个过于尖锐（低熵）的分布上，从而保持模型的灵活性和多样性。

def tokenize_prompt_and_output(prompt_strs: list[str], output_strs: list[str], tokenizer: PreTrainedTokenizer):
    """
    Tokenize prompt and output and then concatenate and response mask.
    
    Args:
        prompts_strs: List of prompt strings
        output_strs: List of output strings
        tokenizer: Tokenizer to use for tokenization

    Returns:
        dict[str, torch.Tensor]. Let prompt_and_output_lens be a list containing the lengths of
        the tokenized prompt and output strings. Then the returned dictionary should have the
        following keys:
            input_ids, shape (batch_size, max(prompt_and_output_lens) - 1)
            labels, shape (batch_size, max(prompt_and_output_lens) - 1)
            response_mask, shape (batch_size, max(prompt_and_output_lens) - 1)
    """
    # 对 Prompt 进行分词
    tokenized_prompts: dict[str, list[list[int]]] = tokenizer(
        prompt_strs,
        return_tensors=None,
        padding=False, # 先不填充，我们之后手动处理
        truncation=True,
        add_special_tokens=False
    )

    # 对 Output 进行分词 (注意：这里我们只关心 ID 和长度，不关心 Padding)
    tokenized_outputs: dict[str, list[list[int]]] = tokenizer(
        output_strs,
        return_tensors=None,
        padding=False, # 先不填充，我们之后手动处理
        truncation=True,
        add_special_tokens=False
    )

    batch_size = len(prompt_strs)
    prompt_lens: list[int] = [len(p) for p in tokenized_prompts['input_ids']]
    output_lens: List[int] = [len(o) for o in tokenized_outputs['input_ids']]

    # 确定最大长度 (用于后续的 Padding)
    max_len = max(p + o for p, o in zip(prompt_lens, output_lens))

    # ----------------------------------------------------------------------
    # 3. 拼接 input_ids 和 构建 response_mask
    # ----------------------------------------------------------------------

    final_input_ids: list[torch.Tensor] = []
    final_labels: list[torch.Tensor] = []
    final_response_mask: list[torch.Tensor] = []

    # 用于表示填充的 Token ID (通常是 tokenizer.pad_token_id)
    PAD_ID = tokenizer.pad_token_id

    for i in range(batch_size):
        prompt_ids: torch.Tensor = torch.tensor(tokenized_prompts['input_ids'][i], dtype=torch.long)
        output_ids: torch.Tensor = torch.tensor(tokenized_outputs['input_ids'][i], dtype=torch.long)

        current_prompt_len = prompt_lens[i]
        current_output_len = output_lens[i]

        # --- A. 拼接 (Concatenate) ---
        # 完整的序列：[Prompt Tokens] + [Output Tokens]
        full_ids = torch.cat([prompt_ids, output_ids])

        # --- B. 构建 Mask ---
        # 掩码：[0, 0, ..., 0 (Prompt)] + [1, 1, ..., 1 (Output)]
        prompt_mask = torch.zeros(current_prompt_len, dtype=torch.bool)
        response_mask = torch.ones(current_output_len, dtype=torch.bool)
        mask = torch.cat([prompt_mask, response_mask])

        # --- C. 填充 (Padding) ---
        padding_len: int = max_len - len(full_ids)
        if padding_len > 0:
            # 创建一个全为 PAD_ID 的张量
            padding: torch.Tensor = torch.full((padding_len,), PAD_ID, dtype=torch.long)

            full_ids = torch.cat([full_ids, padding])
            # 填充部分在掩码中对应 0 (False)
            mask = torch.cat([mask, torch.zeros(padding_len, dtype=torch.bool)])

        # --- D. 最终输出和标签的移位 (Shift for CLM) ---

        # input_ids: 切掉最后一个 Token (L-1)
        current_input_ids: torch.Tensor = full_ids[:-1] 

        # labels: 切掉第一个 Token (L-1)
        current_labels: torch.Tensor = full_ids[1:] 

        # 掩码也要相应地被切掉第一个元素（因为 input_ids 的第一个元素没有对应的 label）
        current_mask: torch.Tensor = mask[1:] 

        final_input_ids.append(current_input_ids)
        final_labels.append(current_labels)
        final_response_mask.append(current_mask)

    # 4. 堆叠成批次张量
    return {
        "input_ids": torch.stack(final_input_ids),
        "labels": torch.stack(final_labels), 
        "response_mask": torch.stack(final_response_mask)
    }


# H(p) = - sum(p(x) * log(p(x)))
def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Get the entropy of the next-token predictions (i.e., entropy over the vocabulary dimension).

    Args:
        logits: Tensor of shape (batch_size, sequence_length, vocab_size) containing unnormalized logits.

    Returns:
        torch.Tensor Shape (batch_size, sequence_length). The entropy for each next-token prediction.
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy_per_token = - probs * log_probs
    entropy_per_token = torch.sum(entropy_per_token, dim=-1)
    return entropy_per_token



# log pθ(y | x) = log [softmax(fθ(x))]y
def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Get per-token conditional log-probabilities (given the previous tokens) from a causal language model,
    and optionally the entropy of the model’s next-token distribution.
    
    Args:
        model: PreTrainedModel HuggingFace model used for scoring (placed on the correct device
               and in inference mode if gradients should not be computed).
        input_ids: torch.Tensor shape (batch_size, sequence_length), concatenated prompt +
                   response tokens as produced by your tokenization method.
        labels: torch.Tensor shape (batch_size, sequence_length), labels as produced by your tokenization method.
        return_token_entropy: bool If True, also return per-token entropy by calling compute_entropy.

    Returns:
        dict[str, torch.Tensor].
        "log_probs" shape (batch_size, sequence_length), conditional log-probabilities log pθ(xt | x<t).
        "token_entropy" optional, shape (batch_size, sequence_length), per-token entropy for each position
        (present only if return_token_entropy=True).
    """
    model.eval()
    with torch.no_grad():
        # Shape (batch_size, sequence_length, vocab)
        logits = model(input_ids).logits
    log_probs = F.log_softmax(logits, dim=-1)
    # 扩展 labels 以便能用于 gather 操作，目标索引维度是最后一维 (-1)
    # labels_expanded 形状: (batch_size, sequence_length, 1)
    # labels = torch.tensor([
    #     [10, 25, 300],  # 批次 1
    #     [5,  15, 80]    # 批次 2
    # ])
    #
    # tensor([[[ 10],
    #          [ 25],
    #          [300]],
    #
    #         [[  5],
    #          [ 15],
    #          [ 80]]])
    labels_expanded = labels.unsqueeze(-1)

    # 使用 torch.gather 提取：log_probs 形状: (batch_size, sequence_length, 1)
    log_probs_gathered = torch.gather(log_probs, dim=-1, index=labels_expanded)
    # 挤压掉最后一个维度，得到最终的 (batch_size, sequence_length) 形状
    log_probs = log_probs_gathered.squeeze(-1)

    results: dict[str, torch.Tensor] = {
        "log_probs": log_probs
    }

    # 5. 可选：计算并返回 token 熵
    if return_token_entropy:
        token_entropy = compute_entropy(logits)
        results["token_entropy"] = token_entropy

    return results 

def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    # 1. 确保 mask 是浮点型，以便进行乘法操作 (mask 1 或 0)
    mask_float = mask.to(tensor.dtype)

    # 2. 将 tensor 中被 mask 掉 (mask == 0) 的部分置为 0
    # 只有 mask_float 为 1 的元素会保留其原始值，其他都被置零
    masked_tensor = tensor * mask_float

    # 3. 求和
    if dim is not None:
        # 沿着指定的维度求和
        summed_result = torch.sum(masked_tensor, dim=dim)
    else:
        # 对所有维度求和
        summed_result = torch.sum(masked_tensor)

    # 3. 求和
    if dim is not None:
        # 沿着指定的维度求和
        summed_result = torch.sum(masked_tensor, dim=dim)
    else:
        # 对所有维度求和
        summed_result = torch.sum(masked_tensor)

    normalized_sum = summed_result / normalize_constant
    return normalized_sum


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    loss = (-masked_normalize(policy_log_probs, response_mask, normalize_constant, -1)).mean()
    loss /= gradient_accumulation_steps

    loss.backward()

    loss_metadata = {
        'gradient_accumulation_steps': gradient_accumulation_steps,
        'normalize_constant': normalize_constant
    }

    return (loss, loss_metadata)

