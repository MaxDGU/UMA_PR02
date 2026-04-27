from pathlib import Path
import sys

import pytest


pytest.importorskip("torch")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "results" / "transformer_replication"))

from finetune_humandata import compute_response_sft_kl_loss  # type: ignore  # noqa: E402


def test_sft_kl_beta_zero_matches_response_token_cross_entropy() -> None:
    logits = torch.tensor(
        [
            [
                [4.0, 0.0, 0.0],
                [0.0, 3.0, 0.0],
                [0.0, 0.0, 2.0],
            ]
        ]
    )
    labels = torch.tensor([[-100, 1, 2]])

    loss, metrics = compute_response_sft_kl_loss(logits, labels, kl_beta=0.0)
    expected = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, logits.shape[-1]),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    assert torch.allclose(loss, expected)
    assert torch.allclose(metrics["sft_nll"], expected)
    assert metrics["kl"].item() == 0.0
    assert metrics["token_count"] == 2


def test_sft_kl_masks_prompt_tokens_and_identical_reference_has_zero_kl() -> None:
    logits = torch.tensor(
        [
            [
                [10.0, 0.0, 0.0],
                [0.0, 10.0, 0.0],
                [0.0, 0.0, 10.0],
            ]
        ]
    )
    labels = torch.tensor([[-100, -100, 2]])

    loss, metrics = compute_response_sft_kl_loss(
        logits,
        labels,
        ref_logits=logits.clone(),
        kl_beta=0.05,
    )
    expected = F.cross_entropy(
        logits[:, 1:2, :].reshape(-1, logits.shape[-1]),
        labels[:, 2:3].reshape(-1),
        ignore_index=-100,
    )

    assert metrics["token_count"] == 1
    assert torch.allclose(metrics["sft_nll"], expected)
    assert abs(metrics["kl"].item()) < 1e-7
    assert torch.allclose(loss, expected, atol=1e-7)


def test_sft_kl_positive_when_policy_moves_from_reference() -> None:
    policy_logits = torch.tensor([[[0.0, 3.0], [3.0, 0.0]]])
    ref_logits = torch.tensor([[[3.0, 0.0], [0.0, 3.0]]])
    labels = torch.tensor([[-100, 1]])

    loss, metrics = compute_response_sft_kl_loss(
        policy_logits,
        labels,
        ref_logits=ref_logits,
        kl_beta=0.05,
    )

    assert metrics["token_count"] == 1
    assert metrics["kl"].item() > 0
    assert loss.item() > metrics["sft_nll"].item()
