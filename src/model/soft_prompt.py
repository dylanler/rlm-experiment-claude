"""
Soft-prompt injection and generation utilities.
Prepends soft-prompt embeddings to question token embeddings,
then generates via the frozen LM.
"""

import torch
from torch import Tensor


def inject_soft_prompt_and_generate(
    model,
    tokenizer,
    soft_prompt_embeds: Tensor,
    question_text: str,
    max_new_tokens: int = 256,
) -> str:
    """
    Prepends soft-prompt embeddings to the question's token embeddings,
    then generates via the frozen LM.

    Args:
        model: Frozen Qwen3-1.7B model
        tokenizer: Corresponding tokenizer
        soft_prompt_embeds: [num_soft_tokens, D_model]
        question_text: The question to answer
        max_new_tokens: Maximum tokens to generate

    Returns: Generated answer string
    """
    model_dtype = next(model.parameters()).dtype

    question_ids = tokenizer(
        question_text, return_tensors="pt"
    ).input_ids.to(model.device)

    with torch.no_grad():
        question_embeds = model.model.embed_tokens(question_ids)  # [1, q_len, D_model]

    soft_prompt = soft_prompt_embeds.unsqueeze(0).to(
        device=model.device, dtype=model_dtype
    )  # [1, num_soft, D_model]

    combined_embeds = torch.cat(
        [soft_prompt, question_embeds], dim=1
    )  # [1, num_soft + q_len, D_model]

    attn_mask = torch.ones(
        1, combined_embeds.shape[1], device=model.device, dtype=torch.long
    )

    with torch.no_grad():
        outputs = model.generate(
            inputs_embeds=combined_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            repetition_penalty=1.3,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Truncate repetitive output: if a sentence repeats, stop there
    sentences = text.split('. ')
    seen = set()
    result_parts = []
    for s in sentences:
        s_clean = s.strip().lower()
        if s_clean in seen and len(s_clean) > 10:
            break
        seen.add(s_clean)
        result_parts.append(s)
    return '. '.join(result_parts)


def compute_soft_prompt_loss(
    model,
    tokenizer,
    soft_prompt_embeds: Tensor,
    question_text: str,
    gold_answer: str,
) -> Tensor:
    """
    Computes cross-entropy loss for training the compressor + aggregator.
    The soft prompt is prepended to the question, and loss is computed
    only on the gold answer tokens.

    Args:
        model: Frozen Qwen3-1.7B model
        tokenizer: Corresponding tokenizer
        soft_prompt_embeds: [num_soft_tokens, D_model]
        question_text: The question
        gold_answer: The gold answer to train towards

    Returns: scalar loss tensor (with grad through soft_prompt_embeds)
    """
    model_dtype = next(model.parameters()).dtype

    # Tokenize question and answer
    question_ids = tokenizer(
        question_text, return_tensors="pt", add_special_tokens=True
    ).input_ids.to(model.device)
    answer_ids = tokenizer(
        gold_answer, return_tensors="pt", add_special_tokens=False
    ).input_ids.to(model.device)

    # Get embeddings (no_grad for frozen model's embedding layer weights,
    # but soft_prompt_embeds carries grad)
    with torch.no_grad():
        question_embeds = model.model.embed_tokens(question_ids)  # [1, q_len, D]
        answer_embeds = model.model.embed_tokens(answer_ids)  # [1, a_len, D]

    # Cast soft prompt to model dtype (e.g. bfloat16) for compatibility
    soft_prompt = soft_prompt_embeds.unsqueeze(0).to(
        device=model.device, dtype=model_dtype
    )  # [1, num_soft, D]

    # Combine: [soft_prompt | question | answer]
    combined_embeds = torch.cat(
        [soft_prompt, question_embeds, answer_embeds], dim=1
    )

    num_soft = soft_prompt.shape[1]
    q_len = question_ids.shape[1]
    a_len = answer_ids.shape[1]
    total_len = num_soft + q_len + a_len

    attn_mask = torch.ones(1, total_len, device=model.device, dtype=torch.long)

    # Build labels: -100 for soft prompt and question tokens, actual ids for answer
    labels = torch.full(
        (1, total_len), -100, dtype=torch.long, device=model.device
    )
    labels[0, num_soft + q_len :] = answer_ids[0]

    # Forward pass through frozen model body but grad flows through soft_prompt_embeds
    outputs = model(
        inputs_embeds=combined_embeds,
        attention_mask=attn_mask,
        labels=labels,
    )

    return outputs.loss
