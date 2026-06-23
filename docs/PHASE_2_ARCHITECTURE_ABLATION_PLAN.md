# Phase 2 Architecture Ablation Plan

This document specifies the required ablation studies designed to isolate and evaluate the sources of model validity, novelty, and composition performance. 

*Note: No ablation is implemented in P24.*

---

## Baselines (A0 - A2)

### A0: Copy/Reference Baseline
- **Purpose**: Establishes the lower bound of novelty. It simply replicates the reference specifications used as input source.
- **What it Tests**: The default performance of copying reference dynamics without any generative model.
- **Expected Evidence Fields**: Replicated `ModelSpec` records with zero novelty.
- **Risks**: High validity rate masking a complete lack of learning or generalization.
- **Interpretation Limits**: Useful only as a control group for novelty.

### A1: Random Valid Baseline
- **Purpose**: Establishes the baseline probability of generating valid specifications through random parameter sampling.
- **What it Tests**: The density of the valid parameter space under simple random generation.
- **Expected Evidence Fields**: Randomly initialized `ModelSpec` records with evaluated validity and novelty rates.
- **Risks**: Can generate highly valid but structurally simple or uninteresting specifications.
- **Interpretation Limits**: Does not test conditional learning or structured composition.

### A2: Structural composition oracle
- **Purpose**: Establishes the theoretical upper bound of composition performance. It creates perfect composite structures using the ground truth generators.
- **What it Tests**: The maximum achievable metric scores under perfect ARMA-GARCH composition.
- **Expected Evidence Fields**: Hand-crafted composite `ModelSpec` records with high validity and composition rates.
- **Risks**: Represents an un-learned oracle rather than a learned model.
- **Interpretation Limits**: Used as the target ceiling for evaluating trained candidates.

---

## Model Candidates (M0 - M4)

### M0: Unconstrained Single-Latent VAE
- **Purpose**: Evaluates a standard, un-factorized VAE without structural constraints.
- **What it Tests**: Whether simple latent space interpolation is sufficient to spontaneously discover valid composition (Family C) without explicit architectural partitioning or rules.
- **Expected Evidence Fields**: Latent dimension of size `z_dim`; untyped float outputs parsed post-hoc.
- **Risks**: High rates of mathematically invalid specifications and posterior collapse.
- **Interpretation Limits**: Serves as a baseline to demonstrate the necessity of explicit factorization and constraints.

### M1: Single-Latent Constrained VAE
- **Purpose**: Isolates the impact of the typed `ModelSpec` decoder on a single latent space representation.
- **What it Tests**: Whether enforcing constraints and structural grammar during decoding is sufficient on its own, without latent factorization.
- **Expected Evidence Fields**: Single latent space (`z_dim`); typed decoder output parsed into `ModelSpec`.
- **Risks**: The model may produce valid specs but fail to learn disentangled or clean compositions of mean and volatility dynamics.
- **Interpretation Limits**: Shows if the decoder boundary alone resolves validity but fails at structured composition.

### M2: Factorized Latent Unconstrained Decoder
- **Purpose**: Isolates the impact of latent space factorization without explicit structural constraints in the decoder.
- **What it Tests**: Whether partitioning the latent space into mean and volatility dimensions is sufficient on its own to generate valid compositions, without using a typed decoder.
- **Expected Evidence Fields**: Partitioned latent spaces (`z_mean`, `z_volatility`, `z_shared`); post-hoc parsed float parameter outputs.
- **Risks**: May produce factorized but mathematically invalid configurations due to the lack of constraint boundaries.
- **Interpretation Limits**: Establishes if factorization alone guarantees composition but fails on validity.

### M3: Factorized Constrained VAE (FC-VAE)
- **Purpose**: Integrates both latent space factorization and the typed `ModelSpec` decoder. This is the intended first serious neural model candidate.
- **What it Tests**: Whether the combination of factorized latent partitions and strict decoder validation gates enables the model to successfully learn and generate valid, novel, and composed specs (Family C) in a zero-shot manner.
- **Expected Evidence Fields**: Partitioned latents (`z_mean`, `z_volatility`, `z_shared`); typed decoder outputs parsed into `ModelSpec`.
- **Risks**: Increased optimization complexity; potential training instability due to multi-task objectives.
- **Interpretation Limits**: Represents the primary candidate for testing the Phase 2 research question.

### M4: Grammar-Constrained Decoder
- **Purpose**: Implements a hard-coded parser or grammar mask in the decoder that guarantees the generation of valid composite (Family C) structures.
- **What it Tests**: The extent to which validity is forced by pre-coded decoder priors versus being actively discovered by the latent space.
- **Expected Evidence Fields**: Masked or grammar-guided decoder output streams.
- **Risks**: High validity rate masking poor latent learning or representation collapse.
- **Interpretation Limits**: Must be clearly reported as grammar-prior validity rather than spontaneous latent discovery.
