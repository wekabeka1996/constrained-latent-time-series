"""
tests/test_vae.py
=================
Minimal pytest suite for Phase 1A — verifies:

  1. VAE forward-pass produces the expected tensor shapes.
  2. Every known checkpoint can be loaded (strict state-dict match).
  3. decode_latent() returns the correct output shape.

Run from the project root:

    python -m pytest tests/test_vae.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Make sure ``src/`` is importable when tests are run from the project root
# (no ``pip install -e .`` required).
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.vae import (
    D_INPUT,
    D_LATENT,
    VAE,
    decode_latent,
    load_checkpoint,
)


# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

BATCH = 4
DEVICE = "cpu"


def _checkpoint_candidates() -> list[Path]:
    """Return existing checkpoint paths in priority order."""
    candidates = [
        PROJECT_ROOT / "models" / "vae_beta50.pth",
        PROJECT_ROOT / "models" / "vae_model_weights.pth",
    ]
    return [p for p in candidates if p.exists()]


ALL_CHECKPOINTS = {
    "models/vae_model_weights.pth": PROJECT_ROOT / "models" / "vae_model_weights.pth",
    "models/vae_beta50.pth":        PROJECT_ROOT / "models" / "vae_beta50.pth",
}


# ---------------------------------------------------------------------------
# test_vae_forward_pass_shape
# ---------------------------------------------------------------------------

class TestVAEForwardPassShape:
    """Verify output shapes of the baseline VAE (default dimensions)."""

    def test_reconstruction_shape(self) -> None:
        model = VAE()
        x = torch.randn(BATCH, D_INPUT)
        x_hat, mu, logvar = model(x)
        assert x_hat.shape == (BATCH, D_INPUT), (
            f"Expected reconstruction shape ({BATCH}, {D_INPUT}), got {x_hat.shape}"
        )

    def test_mu_shape(self) -> None:
        model = VAE()
        x = torch.randn(BATCH, D_INPUT)
        _, mu, _ = model(x)
        assert mu.shape == (BATCH, D_LATENT), (
            f"Expected mu shape ({BATCH}, {D_LATENT}), got {mu.shape}"
        )

    def test_logvar_shape(self) -> None:
        model = VAE()
        x = torch.randn(BATCH, D_INPUT)
        _, _, logvar = model(x)
        assert logvar.shape == (BATCH, D_LATENT), (
            f"Expected logvar shape ({BATCH}, {D_LATENT}), got {logvar.shape}"
        )

    def test_encode_returns_two_tensors(self) -> None:
        model = VAE()
        x = torch.randn(BATCH, D_INPUT)
        result = model.encode(x)
        assert len(result) == 2, "encode() must return (mu, logvar)"

    def test_decode_output_in_range(self) -> None:
        """Tanh output must be in [-1, 1]."""
        model = VAE()
        z = torch.randn(BATCH, D_LATENT)
        x_hat = model.decode(z)
        assert x_hat.min().item() >= -1.0 - 1e-6
        assert x_hat.max().item() <=  1.0 + 1e-6


# ---------------------------------------------------------------------------
# test_load_available_checkpoint
# ---------------------------------------------------------------------------

class TestLoadAvailableCheckpoint:
    """Load checkpoints from disk and verify forward-pass compatibility."""

    def test_at_least_one_checkpoint_found(self) -> None:
        found = _checkpoint_candidates()
        if not found:
            pytest.skip(
                "No checkpoint files found on disk.  "
                "Place vae_model_weights.pth or models/vae_beta50.pth "
                "in the project root to enable this test."
            )
        assert len(found) >= 1

    def test_first_priority_checkpoint_loads(self) -> None:
        found = _checkpoint_candidates()
        if not found:
            pytest.skip("No checkpoint files found.")
        ckpt_path = found[0]
        model = load_checkpoint(ckpt_path, map_location=DEVICE)
        assert isinstance(model, VAE), "load_checkpoint must return a VAE instance"

    def test_first_priority_checkpoint_forward_pass(self) -> None:
        found = _checkpoint_candidates()
        if not found:
            pytest.skip("No checkpoint files found.")
        ckpt_path = found[0]
        model = load_checkpoint(ckpt_path, map_location=DEVICE)
        # Infer d_input from the loaded model
        d_input = model.encoder[0].in_features
        x = torch.randn(BATCH, d_input)
        x_hat, mu, logvar = model(x)
        assert x_hat.shape == (BATCH, d_input)
        assert mu.shape[0] == BATCH
        assert logvar.shape[0] == BATCH

    @pytest.mark.parametrize("label,path", list(ALL_CHECKPOINTS.items()))
    def test_individual_checkpoint_loads(self, label: str, path: Path) -> None:
        """Each known checkpoint must load with strict=True and no key errors."""
        if not path.exists():
            pytest.skip(f"Checkpoint not found: {label}")
        model = load_checkpoint(path, map_location=DEVICE)
        assert isinstance(model, VAE)
        d_input  = model.encoder[0].in_features
        d_latent = model.fc_mu.out_features
        x = torch.randn(2, d_input)
        x_hat, mu, logvar = model(x)
        assert x_hat.shape == (2, d_input)
        assert mu.shape    == (2, d_latent)
        assert logvar.shape == (2, d_latent)


# ---------------------------------------------------------------------------
# test_decode_latent_shape
# ---------------------------------------------------------------------------

class TestDecodeLatent:
    """Verify the decode_latent() convenience function."""

    def test_batched_decode_shape(self) -> None:
        model = VAE()
        z = torch.randn(BATCH, D_LATENT)
        x_hat = decode_latent(model, z)
        assert x_hat.shape == (BATCH, D_INPUT)

    def test_single_vector_decode_shape(self) -> None:
        """1-D latent tensor should be auto-expanded to (1, d_latent)."""
        model = VAE()
        z = torch.randn(D_LATENT)
        x_hat = decode_latent(model, z)
        assert x_hat.shape == (1, D_INPUT)

    def test_decode_latent_no_gradient(self) -> None:
        """decode_latent must not require or produce gradients."""
        model = VAE()
        z = torch.randn(BATCH, D_LATENT)
        x_hat = decode_latent(model, z)
        assert not x_hat.requires_grad

    def test_decode_latent_deterministic(self) -> None:
        """Two calls with the same z must return identical results (no sampling)."""
        model = VAE()
        model.eval()
        z = torch.randn(BATCH, D_LATENT)
        out1 = decode_latent(model, z)
        out2 = decode_latent(model, z)
        assert torch.allclose(out1, out2), "decode_latent should be deterministic"
