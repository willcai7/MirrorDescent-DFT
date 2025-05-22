# Mirror Descent DFT

This repository provides the official implementation for the paper:

**Toward optimal-scaling DFT: stochastic Hartree theory in the thermodynamic and complete basis set limits at arbitrary temperature**

Yuhang Cai and Michael J. Lindsey

Paper: https://arxiv.org/abs/2504.15816

## Overview

This project implements a novel approach to Density Functional Theory (DFT) using stochastic methods and mirror descent optimization. The implementation achieves optimal scaling in both thermodynamic and complete basis set limits at arbitrary temperatures.

## Setup

We use `uv` to organize the python packages. To get started, create a new `uv` environment and install all required packages:

```bash
uv venv --python 3.10
source .venv/bin/activate
uv pip install -e .
chmod +x eg_plot.sh
chmod +x eg_MD.sh
chmod +x exp_MD.sh
```

### Repository Structure

- `eg_MD.sh` - Script for a single run
- `exp_MD.sh` - Script for multiple runs
- `eg_plot.sh` - Script to generate plots for a specific run
- `src/mirrordft/` - Source code directory:
    - `models/` - Code for hamiltonian, contour method, and mirror descent
    - `trainings/` - Specifies training routines
    - `plots/` - Contains code for plot styles and scripts
    - `utils/` - Contains utility scripts for configuration initialization and logging
- `notebooks/` - Contains notebooks for demonstrating each ingredient in our algorithm

## Training

To run for a specific DFT problem, use:

```bash
./eg_MD.sh
```

### Configuration Parameters

| Parameter | Type | Description |
|:----------|:-----|:------------|
| `job_name` | str | Name of the job |
| `dim` | int | Dimension of the system |
| `N` | int | Number of grid points |
| `L` | float | System size |
| `beta` | float | Inverse temperature |
| `alpha` | float | Yukawa potential parameter |
| `cheat` | bool | Whether to use deterministic gradient |
| `N_samples` | int | Number of Gaussian samples |
| `N_poles` | int | Number of poles |
| `max_iter` | int | Maximum number of iterations |
| `raw` | bool | Whether to use raw mode |
| `ratio` | int | Ratio parameter for external potential |
| `eval_iter` | int | Evaluation interval |
| `update_poles_iter` | int | Poles update interval |
| `lr` | float | Learning rate |
| `scf_compare` | bool | Whether to compare with SCF |
| `mu` | float | Chemical potential |
| `tol` | float | Tolerance |
| `decay` | str | Decay type |
| `decay_iter` | int | Decay interval |
| `plot` | bool | Whether to plot results |

For more details about each configuration, refer to:
- `src/mirrordft/utils/config.py`
- `src/mirrordft/trainings/train_SMD.py`

The `exp_MD.sh` is a multi-runs version of `eg_MD.sh`.

## Citation

If you find this code useful in your research, please cite our paper:

```bibtex
@article{cai2025toward,
  title={Toward optimal-scaling DFT: stochastic Hartree theory in the thermodynamic and complete basis set limits at arbitrary temperature},
  author={Cai, Yuhang and Lindsey, Michael},
  journal={arXiv preprint arXiv:2504.15816},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

We welcome contributions! If you have any questions or encounter any issues, feel free to:
1. Open an issue to report bugs or request features
2. Submit a pull request with your improvements
3. Contact the authors directly

## Acknowledgments

We thank all contributors and users of this project for their valuable feedback and support.

