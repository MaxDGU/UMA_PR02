"""Rollout generation for GRPO fine-tuning."""

import torch
import torch.nn.functional as F

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_transformer_phase3 import extract_region


def build_prompt(prob, student_config, tokenizer):
    """Build prompt token IDs for a (problem, student_config) pair.

    Uses coarse bins: g_low/mid/high, d_low/mid/high, rt_3..6, ice_0..100.
    Returns prompt_ids (list of ints, includes BOS, no EOS).
    """
    g_bin, d_bin, rt_bin, ice_bin = student_config
    prompt_text = f"<student> {g_bin} {d_bin} {rt_bin} {ice_bin} </student> <problem> {prob} </problem>"
    ids = tokenizer.encode(prompt_text)  # includes BOS and EOS
    return ids[:-1]  # remove EOS, keep BOS


def generate_rollout(model, prompt_ids, tokenizer, device, temperature=1.0, max_new_tokens=60):
    """Generate a single rollout from the model.

    Returns (full_ids, answer_str, full_text).
    """
    model.eval()
    input_ids = torch.tensor([prompt_ids], device=device)
    eos_id = tokenizer.token_to_id.get("</answer>", tokenizer.eos_token_id)

    with torch.no_grad():
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = model(generated)
            # Clamp logits to prevent inf/nan from softmax after gradient updates
            logits = torch.clamp(logits, -100, 100)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            # Guard against residual numerical issues
            if torch.isnan(probs).any() or torch.isinf(probs).any():
                break
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == eos_id:
                break

    full_ids = generated[0].tolist()
    full_text = tokenizer.decode(full_ids)
    answer_str = extract_region(full_text, "<answer>", "</answer>").replace(" ", "")

    return full_ids, answer_str, full_text


def generate_group_rollouts(model, prob, student_config, tokenizer, device,
                            group_size, temperature, max_new_tokens=60):
    """Generate G rollouts for one (problem, student_config) pair."""
    prompt_ids = build_prompt(prob, student_config, tokenizer)
    rollouts = []
    for _ in range(group_size):
        full_ids, answer_str, full_text = generate_rollout(
            model, prompt_ids, tokenizer, device, temperature, max_new_tokens
        )
        rollouts.append({
            "prompt_ids": prompt_ids,
            "full_ids": full_ids,
            "answer": answer_str,
            "text": full_text,
        })
    return rollouts
