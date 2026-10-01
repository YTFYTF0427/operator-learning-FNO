"""Offline dataset generation: solve committor PDEs, subsample, write CSV.

Pipeline (mirrors the legacy `training_data_1_generator.py` but driven
entirely by ExperimentConfig and a passed-in RNG):

    for sid in range(num_solutions):
        1. Draw phi, psi (square -> identity; curved -> Algorithm 1).
        2. Draw b via turbulent_velocity_field on the 311x311 FFT grid.
        3. Build btilde_x, btilde_y (b evaluated on the transformed grid).
        4. Solve q^+ and q^- on a (fine_grid) x (fine_grid) grid.
        5. rho = q^+ * q^- / |Omega|   (square: |Omega|=1).
        6. Subsample to coarse_grid via linspace index selection
           (dissertation: fine_grid = (coarse_grid - 1) * step + 1, so
           the subsample is exact — e.g. 311 -> 32 takes every 10th).
        7. Append rows (sid, x, y, b1, b2, rho) (+ finv for curved).

The CSV schema matches the legacy format so existing notebooks continue
to read training sets. Switching to Parquet/Zarr is a Phase 3 concern
once dataset sizes exceed a few GB.
"""

from __future__ import annotations

import multiprocessing as mp
import os
from dataclasses import dataclass
from pathlib import Path

import autograd.numpy as anp
import numpy as np
import pandas as pd
from tqdm import tqdm

from neuropaths.config import DataConfig, PDEConfig
from neuropaths.pde.boundaries import generate_boundary_pair
from neuropaths.pde.solvers import (
    reactive_density,
    solve_backward_committor,
    solve_forward_committor,
)
from neuropaths.pde.transforms import inverse_map
from neuropaths.pde.velocity import turbulent_velocity_field


def _domain_area(phi, psi, n_probe: int = 1024) -> float:
    """Numerical |Omega| = integral_0^1 (psi(y) - phi(y)) dy (trapz)."""
    y = np.linspace(0.0, 1.0, n_probe)
    gap = np.asarray(psi(y)) - np.asarray(phi(y))
    return float(np.trapezoid(gap, y))


def _subsample_indices(n_fine: int, n_coarse: int) -> np.ndarray:
    """Linspace index selection including both boundaries.

    Matches legacy: `np.linspace(0, n_fine - 1, n_coarse, dtype=int)`.
    For n_fine=311, n_coarse=32 this is exactly `arange(0, 311, 10)` plus
    the endpoint — a clean stride-10 subsample.
    """
    return np.linspace(0, n_fine - 1, n_coarse, dtype=int)


@dataclass(frozen=True)
class _SolutionJob:
    """Pickle-friendly per-solution payload for multiprocessing workers.

    Carries only plain data — configs are frozen dataclasses; SeedSequence
    is picklable. Every closure (boundaries, velocity field, solvers)
    is built *inside* the worker, not passed across the pool boundary.
    """

    sid: int
    pde_cfg: PDEConfig
    data_cfg: DataConfig
    seed_seq: np.random.SeedSequence


def _generate_one_solution(
    job: _SolutionJob,
) -> tuple[int, list[list[float]], float, float]:
    """Run the full per-solution pipeline and return its coarse-grid rows.

    Returns ``(sid, rows, rho_max, rho_l2)``. The orchestrator uses both
    summary statistics for rejection sampling (`DataConfig.rho_min_max`
    and `DataConfig.rho_min_l2`) and re-sorts by ``sid`` so that the
    output CSV is identical across worker counts.
    """
    pde_cfg = job.pde_cfg
    data_cfg = job.data_cfg
    rng = np.random.default_rng(job.seed_seq)

    fine_grid = pde_cfg.fine_grid
    coarse_grid = pde_cfg.coarse_grid
    n_interior = fine_grid - 2
    coarse_idx = _subsample_indices(fine_grid, coarse_grid)

    # 1. Boundaries.
    phi, psi = generate_boundary_pair(
        rng,
        kind=pde_cfg.domain,
        n_max=pde_cfg.boundary_n_max,
        eps=pde_cfg.boundary_eps,
    )

    # 2. Velocity field.
    field = turbulent_velocity_field(
        reynolds=pde_cfg.reynolds,
        char_length=pde_cfg.char_length,
        viscosity=pde_cfg.viscosity,
        k_min=pde_cfg.velocity_kmin,
        k_max=pde_cfg.velocity_kmax,
        eps_1=pde_cfg.eps_1,
        eps_2=pde_cfg.eps_2,
        n_grid=fine_grid,
        rng=rng,
    )
    bx = field.bx
    by = field.by

    # 3. Solve committor PDEs.
    q_plus, X, Y = solve_forward_committor(phi, psi, bx, by, n_interior)
    q_minus, _, _ = solve_backward_committor(phi, psi, bx, by, n_interior)

    # 4. Reactive density.
    omega = 1.0 if pde_cfg.domain == "square" else _domain_area(phi, psi)
    rho = reactive_density(q_plus, q_minus, domain_area=omega)
    #rho = q_plus

    # 5. Subsample.
    ii, jj = np.meshgrid(coarse_idx, coarse_idx, indexing="ij")
    x_c = np.asarray(X)[ii, jj]
    y_c = np.asarray(Y)[ii, jj]
    rho_c = rho[ii, jj]

    # 6. b on the coarse grid in original coordinates.
    if pde_cfg.domain == "square":
        bx_c = bx(x_c, y_c)
        by_c = by(x_c, y_c)
    else:
        f_inv = inverse_map(phi, psi)
        xprime = np.asarray(f_inv(anp.asarray(x_c), anp.asarray(y_c)))
        bx_c = bx(xprime, y_c)
        by_c = by(xprime, y_c)

    finv_c = None
    if data_cfg.include_finv_column:
        f_inv = inverse_map(phi, psi)
        finv_c = np.asarray(f_inv(anp.asarray(x_c), anp.asarray(y_c)))

    rows: list[list[float]] = []
    for i in range(coarse_grid):
        for j in range(coarse_grid):
            row = [
                float(job.sid),
                float(x_c[i, j]),
                float(y_c[i, j]),
                float(bx_c[i, j]),
                float(by_c[i, j]),
                float(rho_c[i, j]),
            ]
            if finv_c is not None:
                row.append(float(finv_c[i, j]))
            rows.append(row)

    return job.sid, rows, float(rho_c.max()), float(np.linalg.norm(rho_c))


def generate_dataset(
    pde_cfg: PDEConfig,
    data_cfg: DataConfig,
    *,
    split: str = "train",
    output_path: str | Path | None = None,
    num_workers: int = 1,
) -> Path:
    """Generate a single CSV of (solution_id, x, y, b1, b2, rho[, finv]).

    Reproducibility is anchored in ``data_cfg.seed`` (plus a split offset);
    a ``SeedSequence`` is spawned per solution so the output is identical
    regardless of ``num_workers``. Pass ``num_workers=-1`` to use all
    cores reported by ``os.cpu_count()``.

    The hot path is ``scipy.sparse.linalg.spsolve`` on an ~N²×N² system
    (CPU-bound, no GPU payoff) but per-solution work is independent, so
    an ``mp.Pool`` fans out near-linearly until the box runs out of cores.
    Set single-threaded BLAS env vars (``OMP_NUM_THREADS=1`` etc.) before
    launching to avoid oversubscription.
    """
    if split not in ("train", "test"):
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")

    raw_path = output_path if output_path is not None else (
        data_cfg.train_csv if split == "train" else data_cfg.test_csv
    )
    out_path = Path(raw_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    num_target = (
        data_cfg.num_train_solutions if split == "train" else data_cfg.num_test_solutions
    )
    rho_min_max = float(data_cfg.rho_min_max)
    rho_min_l2 = float(data_cfg.rho_min_l2)
    rejection_active = rho_min_max > 0.0 or rho_min_l2 > 0.0
    oversample = max(1.0, float(data_cfg.oversample_factor))
    num_candidates = (
        int(np.ceil(num_target * oversample)) if rejection_active else num_target
    )

    # Deterministic per-candidate seeds. Split offset keeps train/test draws
    # disjoint even though they share data_cfg.seed.
    base_seed = data_cfg.seed + (0 if split == "train" else 1)
    child_sequences = np.random.SeedSequence(base_seed).spawn(num_candidates)
    jobs = [
        _SolutionJob(sid=sid, pde_cfg=pde_cfg, data_cfg=data_cfg, seed_seq=ss)
        for sid, ss in enumerate(child_sequences)
    ]

    if num_workers < 0:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, min(num_workers, num_candidates))

    columns = ["solution_id", "x", "y", "b1", "b2", "rho"]
    if data_cfg.include_finv_column:
        columns.append("finv")

    desc = f"generate[{split}]"
    if num_workers == 1:
        results = [
            _generate_one_solution(job)
            for job in tqdm(jobs, desc=desc, unit="pde")
        ]
    else:
        # ``spawn`` context avoids inheriting parent state on Linux (matches
        # macOS default); imap_unordered streams results so tqdm updates live.
        ctx = mp.get_context("spawn")
        with ctx.Pool(num_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_generate_one_solution, jobs),
                    total=num_candidates,
                    desc=f"{desc} (x{num_workers})",
                    unit="pde",
                )
            )

    # Sort by candidate sid for reproducibility before filtering.
    results.sort(key=lambda item: item[0])

    if rejection_active:
        accepted = [
            r for r in results
            if r[2] >= rho_min_max and r[3] >= rho_min_l2
        ]
        n_accepted = len(accepted)
        n_rejected = num_candidates - n_accepted
        print(
            f"[{split}] rejection: {n_accepted}/{num_candidates} accepted "
            f"(rho_min_max={rho_min_max:g}, rho_min_l2={rho_min_l2:g}); "
            f"{n_rejected} discarded as no-transition or near-degenerate draws."
        )
        if n_accepted < num_target:
            raise RuntimeError(
                f"generate_dataset[{split}]: only {n_accepted} of {num_candidates} "
                f"candidates passed rejection (rho_min_max={rho_min_max:g}, "
                f"rho_min_l2={rho_min_l2:g}); need {num_target}. Increase "
                f"data.oversample_factor (currently {oversample:g}) or relax "
                f"a threshold."
            )
        kept = accepted[:num_target]
    else:
        kept = results[:num_target]

    # Renumber to a contiguous 0..num_target-1 range so downstream consumers
    # can rely on solution_id being a dense index.
    records: list[list[float]] = []
    for new_sid, (_, rows, _, _) in enumerate(kept):
        for row in rows:
            row[0] = float(new_sid)
            records.append(row)

    df = pd.DataFrame(records, columns=columns)
    df.to_csv(out_path, index=False)
    return out_path
