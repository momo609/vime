from argparse import Namespace

import pytest
import torch

from vime.backends.speculative_training.draft_trainer import (
    _architecture_fingerprint,
    _load_target_embedding,
)


class _Draft(torch.nn.Module):
    def __init__(self, rows=4, hidden=3):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(rows, hidden)
        self.proj = torch.nn.Linear(hidden, hidden, bias=False)


@pytest.mark.unit
def test_generic_target_embedding_loader_reads_pytorch_checkpoint(tmp_path):
    source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    torch.save({"model.embed_tokens.weight": source}, tmp_path / "pytorch_model.bin")
    model = _Draft()
    args = Namespace(
        draft_target_embedding_path=str(tmp_path),
        draft_target_embedding_key="model.embed_tokens.weight",
        hf_checkpoint=None,
    )

    _load_target_embedding(model, args)

    assert torch.equal(model.embed_tokens.weight, source)


@pytest.mark.unit
def test_architecture_fingerprint_changes_with_parameter_layout():
    assert _architecture_fingerprint(_Draft(rows=4)) != _architecture_fingerprint(_Draft(rows=5))
