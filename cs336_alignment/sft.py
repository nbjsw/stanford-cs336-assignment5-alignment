import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizer


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


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes the per-token entropy of next-token predictions.
    H(p) = - sum(p(x) * log(p(x)))
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    entropy_per_token = - probs * log_probs
    entropy_per_token = torch.sum(entropy_per_token, dim=-1)
    return entropy_per_token

