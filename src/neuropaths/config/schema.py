"""Dataclass schemas for experiment configs.

Designed to exactly cover the hyperparameters that appear hardcoded in
the legacy scripts plus the knobs the dissertation sweeps over:

    * Section 5 "Training the Neural Operator" -- 1000 / 5000 PDE
      instances, 100 epochs, batch sizes 32/64, k_max in {12, 16},
      d_c in {128, 256}, input channel width 256, lr = 0.001,
      grid sizes s in {16, 32, 63} for training and s' for testing.
    * Section 4 "Generating Training Data" -- fine FD grid 311x311,
      Reynolds = 10, L = 0.2, nu = 1 (note: legacy code uses Re=8).
    * Section 4.1 boundary generator -- N_max = 8, epsilon tolerance.

Defaults here track the dissertation's canonical "square 32x32" run so
that `ExperimentConfig()` without overrides reproduces the baseline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DomainKind = Literal["square", "curved"]
DeviceStr = Literal["auto", "cuda", "mps", "cpu"]


@dataclass
class PDEConfig:
    """FD solver + velocity-field parameters."""

    # Fine FD grid used to solve the committor PDEs before subsampling.
    # Dissertation fixes this to 311x311 so 16/32/63 subsample cleanly.
    fine_grid: int = 311

    # Coarse grid the dataset is subsampled onto; also the FNO input grid.
    coarse_grid: int = 32

    # Turbulence: Reynolds number, characteristic length, viscosity.
    # Dissertation picks Re=10, L=0.2, nu=1 (=> U = nu*Re/L = 50).
    # Legacy `training_data_1_generator.py` passes Reynolds=8.0 -- flag.
    reynolds: float = 10.0
    char_length: float = 0.2
    viscosity: float = 1.0

    # Whether to use curved domains (random trig-polynomial boundaries)
    # or the identity boundaries phi=0, psi=1 that the legacy code runs.
    domain: DomainKind = "square"

    # Trig-polynomial boundary generator (section 4.1).
    boundary_n_max: int = 8
    boundary_eps: float = 0.1

    # Epsilon neighbourhood padding used by turbulent_velocity_field.
    eps_1: float = 0.0
    eps_2: float = 0.0

    # Velocity-field Fourier-mode cap. Notes chapter §2.4 introduces this
    # as the "laminar vs turbulent" knob: k_max=8 laminar, 16 turbulent.
    # Interpreted as |k| <= velocity_kmax * (2 pi / L_domain), i.e. the
    # top-end of the scaling band in Algorithm 2 step 3. The legacy code
    # hardcodes `kfmax = 30 * kfmin` regardless; this field replaces it.
    velocity_kmax: int = 8

    # Bottom of the same scaling band. Dissertation uses the domain
    # fundamental (kfmin = 2 pi / L_domain => velocity_kmin=1).
    velocity_kmin: int = 1


@dataclass
class DataConfig:
    """Dataset generation + loader config."""

    num_train_solutions: int = 1000
    num_test_solutions: int = 200

    train_csv: str = "data/train.csv"
    test_csv: str = "data/test.csv"

    # CSV column layout produced by the generator. Matches legacy schema
    # ['solution_id', 'x', 'y', 'b1', 'b2', 'rho']; extended by 'finv'
    # for the curved-domain experiment.
    include_finv_column: bool = False

    batch_size: int = 32
    num_workers: int = 0  # keep 0 by default; DataLoader workers interact
    # poorly with some MPS setups.

    # PRNG seed -- passed to seed_everything and the velocity-field draws.
    seed: int = 2203_2025

    # Rejection sampling at generation time. We use two complementary
    # criteria to drop "no-transition" or near-degenerate samples that
    # detonate relative-L2 losses (the per-sample ||y||_2 sits in the
    # denominator and small targets blow it up):
    #
    #   * `rho_min_max`: discard if the coarse-grid rho.max() is below
    #     this peak threshold. Cheap, kills the obvious zero-everywhere
    #     cases. 0.0 disables.
    #   * `rho_min_l2`:  discard if ||rho||_2 (over the coarse grid) is
    #     below this threshold. This is the principled criterion --
    #     it matches the quantity that actually appears in LpLoss's
    #     denominator. 0.0 disables.
    #
    # A candidate must clear BOTH thresholds to be accepted.
    rho_min_max: float = 0.01
    rho_min_l2: float = 0.0

    # When rejection is enabled the generator draws ceil(N * oversample_factor)
    # candidates and keeps the first N that pass both thresholds.
    # If too few pass, generate_dataset raises a RuntimeError. 1.0 disables
    # oversampling (fail fast on any rejection).
    oversample_factor: float = 1.5


@dataclass
class ModelConfig:
    """FNO2D architecture (wraps neuralop.models.FNO)."""

    # Number of channels in the input *function*. The (x, y) grid
    # coordinates are NOT counted here -- neuralop's FNO appends a 2-channel
    # grid positional embedding internally before lifting.
    # Square domain: (b1, b2) -> 2.
    # Curved domain: (b1, b2, f^{-1}) -> 3.
    in_channels: int = 2
    out_channels: int = 1

    # Fourier mode truncation k_max (same in both spatial dims).
    modes: int = 12

    # Fourier channel width d_c (neuralop: hidden_channels).
    width: int = 128

    # Depth L (number of Fourier layers; neuralop: n_layers).
    depth: int = 4

    # Projection block hidden width (Q). Forwarded to neuralop as
    # projection_channel_ratio = projection_hidden / width.
    projection_hidden: int = 128

    activation: Literal["gelu", "relu"] = "gelu"

    # Per-axis domain padding (fraction of grid size). Required for the
    # FFT-based spectral convolution when an axis is non-periodic --
    # otherwise the FFT sees a hard discontinuity at the Dirichlet
    # boundary. fno-explained.pdf §3.6 recommends padding only the
    # non-periodic axes. For our (Dirichlet x, periodic y) setup the
    # natural choice is e.g. [0.125, 0.0]. ``None`` disables padding.
    domain_padding: list[float] | None = None

    # Pathway controls inside each FNO block. neuralop's defaults bolt
    # on a per-block channel MLP (with its own skip path) on top of the
    # standard Li et al. 2021 architecture. These extras can train the
    # operator to lean on grid-specific feature interactions, which
    # then breaks zero-shot super-resolution at unseen grid sizes (we
    # observed this empirically: ok at G=16, fails at G=63).
    #
    # Set ``use_channel_mlp: false`` and ``channel_mlp_skip: "none"``
    # to recover the vanilla architecture; keep ``fno_skip: "linear"``
    # for the original W_l skip from Li et al.
    #
    # Use "none" (string) to map to Python None.
    use_channel_mlp: bool = True
    channel_mlp_skip: Literal["linear", "identity", "soft-gating", "none"] = "soft-gating"
    fno_skip: Literal["linear", "identity", "soft-gating", "none"] = "linear"


@dataclass
class TrainConfig:
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 0.0

    # Loss function. Options:
    #   * "relative_l2": neuralop.LpLoss(d=2, p=2).rel  -- the dissertation
    #     metric ||x-y||_2 / (||y||_2 + loss_eps); long tail when ||y||->0.
    #   * "absolute_l2": neuralop.LpLoss(d=2, p=2).abs  -- pure ||x-y||_2,
    #     immune to the small-target singularity. Recommended for committor
    #     problems where ||y||_2 varies several orders of magnitude.
    #   * "h1": neuralop.H1Loss(d=2)  -- Sobolev norm penalising both
    #     value and gradient errors. fno-explained.pdf §5.1 recommends
    #     this for elliptic PDEs whose solutions are smooth.
    #   * "mse": torch.nn.MSELoss() -- legacy fallback.
    loss: Literal["relative_l2", "absolute_l2", "h1", "mse"] = "relative_l2"

    # Additive eps in the relative-L2 denominator: ||x-y||_2 / (||y||_2 + eps).
    # Caps the amplification when ||y||_2 is small. Only used by the relative
    # and h1 losses. Set to ~1.0 if relative-L2 has visible long-tail outliers
    # in the validation distribution; 0.0 falls through to neuralop's default
    # (1e-8, effectively no stabilisation).
    loss_eps: float = 0.0

    scheduler: Literal["none", "cosine", "step"] = "none"
    step_size: int = 30
    gamma: float = 0.5

    # Checkpoint output.
    checkpoint_path: str = "checkpoints/fno2d.pth"

    # Device.
    device: DeviceStr = "auto"
    deterministic: bool = False


@dataclass
class EvalConfig:
    checkpoint_path: str = "checkpoints/fno2d.pth"
    output_dir: str = "results"

    # Zero-shot super-resolution study (Table 1/2 in the dissertation):
    # train at `s`, evaluate at each of these.
    test_resolutions: list[int] = field(default_factory=lambda: [16, 32, 63])

    # Number of PDEs visualised in the qualitative prediction plots.
    num_plot_samples: int = 3


@dataclass
class ExperimentConfig:
    """Top-level config for a single reproducible experiment."""

    name: str = "square_32_baseline"
    output_dir: str = "runs/${name}"

    pde: PDEConfig = field(default_factory=PDEConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
