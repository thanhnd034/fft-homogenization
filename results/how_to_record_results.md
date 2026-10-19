# How to reproduce and record the coupled FFT results

The tracked [coupled_model_JAX.py](../coupled_model_JAX.py) solves a specific periodic thermoelastic RVE with Fourier-generated heterogeneous material parameters.

## Demonstration settings coded in the script

| Variable | Value |
| --- | --- |
| Grid | 512×512 |
| Microstructure sample index | 8281 |
| Fourier-mode mask | Radial |
| Reference temperature | 20 |
| Mean temperature | 250 |
| Prescribed macroscopic temperature gradient | [0.1, 0.02] |
| Prescribed strain tensor | [[0.05, 0], [0, 0]] |
| Thermal and mechanical convergence tolerances | 1e-8 each |
| Maximum thermal/mechanical iterations | 1000 each |

## What to report after executing

Record the convergence status and last equilibrium residual for both solvers; the macroscopic heat flux; the effective conductivity component; mean temperature gradient; mean strain and stress; and the local field images and convergence histories. The program renders these outputs to the console and with Matplotlib.

No numerical values are presented in this file because a complete verified execution output was not available in the repository. The table above describes **input settings**, not results.
