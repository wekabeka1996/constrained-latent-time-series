"""
src/vae.py
==========
Canonical definition of the Variational Autoencoder (VAE) used throughout
this research project.

All trained checkpoints in ``models/`` are raw ``OrderedDict`` state-dicts
whose keys follow PyTorch's automatic numbering of ``nn.Sequential`` layers:

    encoder.0.weight, encoder.2.weight, ...
    fc_mu.weight, fc_logvar.weight
    decoder.0.weight, decoder.2.weight, decoder.4.weight

The layer numbering must match exactly for ``load_state_dict`` to succeed.
Do **not** rename or reorder layers without retraining.

Checkpoint dimensions confirmed by inspection (2026-06-18):

    Baseline VAE  : d_input=40, d_hidden=64/32, d_latent=8
    vae_latent16  : d_input=40, d_hidden=64/32, d_latent=16
    vae_deeper    : d_input=40, d_hidden=64/64/32, d_latent=8  (extra layer)
    vae_wider     : d_input=40, d_hidden=128/128/64, d_latent=8 (wider + extra)
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Default dimensions shared by the baseline and all beta-variant checkpoints.
# These constants are referenced by training scripts and tests.
# ---------------------------------------------------------------------------
D_STRUCT: int = 10   # structural indicator sub-vector dimension
D_PARAM:  int = 20   # parameter sub-vector dimension
D_STAT:   int = 10   # statistical-feature sub-vector dimension
D_INPUT:  int = D_STRUCT + D_PARAM + D_STAT   # 40
D_LATENT: int = 8    # baseline latent space dimension
D_HIDDEN1: int = 64  # first hidden layer width
D_HIDDEN2: int = 32  # second hidden layer width (encoder bottleneck / decoder start)


# ---------------------------------------------------------------------------
# VAE
# ---------------------------------------------------------------------------

class VAE(nn.Module):
    """Variational Autoencoder for econometric model-structure representation.

    The network maps a fixed-length parameter vector ``θ`` (encoding model
    type, lag orders, and numeric coefficients) to a low-dimensional latent
    space and reconstructs it.

    Architecture (baseline, matches all ``vae_beta*.pth`` checkpoints):

        Encoder:  Linear(d_input → 64) → ReLU → Linear(64 → 32) → ReLU
        μ head:   Linear(32 → d_latent)
        σ² head:  Linear(32 → d_latent)
        Decoder:  Linear(d_latent → 32) → ReLU → Linear(32 → 64) → ReLU
                  → Linear(64 → d_input) → Tanh

    The Tanh output normalises reconstructed values to [-1, 1], which matches
    the normalisation used during data generation.

    Parameters
    ----------
    d_input:
        Dimension of the input/output parameter vector θ. Default: 40.
    d_latent:
        Dimension of the latent space z.  Default: 8.
    d_hidden1:
        Width of the first encoder/last decoder hidden layer. Default: 64.
    d_hidden2:
        Width of the second encoder/first decoder hidden layer. Default: 32.
    """

    def __init__(
        self,
        d_input:  int = D_INPUT,
        d_latent: int = D_LATENT,
        d_hidden1: int = D_HIDDEN1,
        d_hidden2: int = D_HIDDEN2,
    ) -> None:
        super().__init__()

        # ---- Encoder --------------------------------------------------------
        # nn.Sequential numbering: Linear=0, ReLU=1, Linear=2, ReLU=3
        # Checkpoint keys: encoder.0.weight, encoder.2.weight
        self.encoder = nn.Sequential(
            nn.Linear(d_input, d_hidden1),
            nn.ReLU(),
            nn.Linear(d_hidden1, d_hidden2),
            nn.ReLU(),
        )

        # ---- Latent projections --------------------------------------------
        self.fc_mu    = nn.Linear(d_hidden2, d_latent)
        self.fc_logvar = nn.Linear(d_hidden2, d_latent)

        # ---- Decoder --------------------------------------------------------
        # nn.Sequential numbering: Linear=0, ReLU=1, Linear=2, ReLU=3,
        #                          Linear=4, Tanh=5
        # Checkpoint keys: decoder.0.weight, decoder.2.weight, decoder.4.weight
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden2),
            nn.ReLU(),
            nn.Linear(d_hidden2, d_hidden1),
            nn.ReLU(),
            nn.Linear(d_hidden1, d_input),
            nn.Tanh(),
        )

    # ------------------------------------------------------------------
    # Core VAE operations
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input ``x`` to Gaussian parameters (μ, log σ²).

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, d_input)``.

        Returns
        -------
        mu:
            Mean of the approximate posterior, shape ``(batch, d_latent)``.
        logvar:
            Log-variance of the approximate posterior, same shape.
        """
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(
        self, mu: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample z via the reparameterisation trick: z = μ + ε·σ.

        During inference (``model.eval()``), sampling still occurs.  Pass
        ``mu`` directly when you want the deterministic mean embedding.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector ``z`` to reconstructed parameter space.

        Parameters
        ----------
        z:
            Latent tensor of shape ``(batch, d_latent)``.

        Returns
        -------
        x_hat:
            Reconstructed parameter vector, shape ``(batch, d_input)``,
            values in [-1, 1] due to the final Tanh.
        """
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full VAE forward pass.

        Parameters
        ----------
        x:
            Input tensor of shape ``(batch, d_input)``.

        Returns
        -------
        x_hat:
            Reconstruction, shape ``(batch, d_input)``.
        mu:
            Posterior mean, shape ``(batch, d_latent)``.
        logvar:
            Posterior log-variance, shape ``(batch, d_latent)``.
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def __repr__(self) -> str:  # pragma: no cover
        d_in  = self.encoder[0].in_features
        d_lat = self.fc_mu.out_features
        d_h1  = self.encoder[0].out_features
        d_h2  = self.encoder[2].out_features
        return (
            f"VAE(d_input={d_in}, d_hidden1={d_h1}, d_hidden2={d_h2}, "
            f"d_latent={d_lat})"
        )


# ---------------------------------------------------------------------------
# Public helper functions
# ---------------------------------------------------------------------------

def vae_loss(
    x: torch.Tensor,
    x_hat: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """β-VAE ELBO loss (MSE reconstruction + β · KL divergence).

    Parameters
    ----------
    x:
        Original input, shape ``(batch, d_input)``.
    x_hat:
        Reconstruction from the decoder.
    mu, logvar:
        Approximate posterior parameters from the encoder.
    beta:
        KL weight.  ``beta=1`` is standard VAE; higher values push the
        posterior closer to 𝒩(0,I) at the cost of reconstruction fidelity.

    Returns
    -------
    loss:
        Scalar loss tensor.
    """
    recon = F.mse_loss(x_hat, x, reduction="mean")
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
    return recon + beta * kl


def load_checkpoint(
    checkpoint_path: Union[str, Path],
    model: Union[VAE, None] = None,
    map_location: Union[str, torch.device] = "cpu",
) -> VAE:
    """Load a trained VAE from a checkpoint file.

    All checkpoints in this project are raw ``OrderedDict`` state-dicts
    (not wrapped in a metadata dict).  The function inspects the first key to
    infer the required dimensions automatically and falls back to defaults if
    the model argument is already provided.

    Parameters
    ----------
    checkpoint_path:
        Path to the ``.pth`` file.
    model:
        An already-instantiated ``VAE``.  If ``None``, a new model is created
        whose dimensions are inferred from the checkpoint.
    map_location:
        Device string or ``torch.device`` passed to ``torch.load``.

    Returns
    -------
    model:
        ``VAE`` with weights loaded, placed on ``map_location``.

    Raises
    ------
    FileNotFoundError:
        If ``checkpoint_path`` does not exist.
    RuntimeError:
        If the state-dict keys are incompatible with the model.
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state: dict = torch.load(
        checkpoint_path, map_location=map_location, weights_only=True
    )

    # Handle potential wrapper dict (e.g. {"model_state_dict": ..., "epoch": ...})
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]
    # else: assume state IS the raw OrderedDict

    if model is None:
        model = _infer_model_from_state(state)

    model.load_state_dict(state, strict=True)
    model = model.to(map_location)
    model.eval()
    return model


def decode_latent(model: VAE, z: torch.Tensor) -> torch.Tensor:
    """Convenience wrapper: decode a latent tensor without gradient tracking.

    Parameters
    ----------
    model:
        A ``VAE`` instance (trained or random).
    z:
        Latent tensor of shape ``(batch, d_latent)`` or ``(d_latent,)``.

    Returns
    -------
    x_hat:
        Decoded parameter vector, shape ``(batch, d_input)``.
    """
    if z.dim() == 1:
        z = z.unsqueeze(0)
    with torch.no_grad():
        return model.decode(z)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_model_from_state(state: dict) -> VAE:
    """Infer VAE constructor arguments from a raw state-dict."""
    # d_input: output dim of the last decoder linear layer
    d_input  = state["decoder.4.weight"].shape[0]
    # d_latent: output dim of fc_mu
    d_latent = state["fc_mu.weight"].shape[0]
    # d_hidden1: output dim of the first encoder Linear (index 0)
    d_hidden1 = state["encoder.0.weight"].shape[0]
    # d_hidden2: output dim of the second encoder Linear (index 2)
    d_hidden2 = state["encoder.2.weight"].shape[0]
    return VAE(
        d_input=d_input,
        d_latent=d_latent,
        d_hidden1=d_hidden1,
        d_hidden2=d_hidden2,
    )
