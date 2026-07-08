"""Scalar-transport example: Sherwood number from an assimilated interior flow (paper §IV).

Loads a reconstructed flow field (the assimilation .npz produced by the data-assimilation
stage, next to this file), solves the axisymmetric advection-diffusion of a passive scalar
for a range of Peclet numbers, and produces:

  1. Sh vs diffusion time (t/Pe), one line per Pe                -> sherwood_vs_time.png
  2. Sh vs Pe at the last time point                            -> sherwood_vs_pe.png
  3. the final scalar field per Pe, normalized by its max       -> final_concentration.png
  4. the Sh(time, Pe) matrix + the Pe and diffusion-time axes    -> sherwood.npz

Run (a few seconds per Pe on the bundled medium mesh; scales up with mesh size):

    python examples/scalar_transport/example_scalar_transport.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pyinnerbubble.transport import (
    plot_final_concentration,
    plot_sherwood_vs_pe,
    plot_sherwood_vs_time,
    simulate_scalar_transport,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent

# Assimilation output (reconstructed flow field) to run transport on.
ASSIMILATION_FILE = SCRIPT_DIR / "assimilation.npz"

T_END_DIFFUSION = 0.08   # simulation end time in DIFFUSION-time units (t/Pe)
N_TIME = 1001            # number of time points on the shared diffusion-time axis
PE_VEC = [10, 30,60, 100, 300, 600, 1000]   # Peclet numbers, one Sh(t) curve each

OUT_DIR = REPO_ROOT / "output" / "scalar_transport"
# ---------------------------------------------------------------------------


def main():
    if not ASSIMILATION_FILE.exists():
        raise FileNotFoundError(
            f"Assimilation file not found at {ASSIMILATION_FILE}. Put the data-assimilation "
            f"output (assimilation.npz with r_mat/z_mat/u_r/u_z) next to this script."
        )

    result = simulate_scalar_transport(
        ASSIMILATION_FILE, t_end_diffusion=T_END_DIFFUSION, n_time=N_TIME,
        pe_vec=PE_VEC, verbose=True,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_sherwood_vs_time(result, save_path=OUT_DIR / "sherwood_vs_time.png")
    plot_sherwood_vs_pe(result, save_path=OUT_DIR / "sherwood_vs_pe.png")
    plot_final_concentration(result, save_path=OUT_DIR / "final_concentration.png")

    # Sh(time, Pe) matrix and the two axes.
    np.savez(
        OUT_DIR / "sherwood.npz",
        pe_vec=result.pe_vec,
        t_diffusion=result.t_diffusion,
        sh_matrix=result.sh_matrix,
        sh_final=result.sh_final,
    )
    print(f"[transport] Sh at last time per Pe: {dict(zip(result.pe_vec, result.sh_final))}")
    print(f"[transport] results saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
