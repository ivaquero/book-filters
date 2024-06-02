# Filter Simulation Toolkit for Education

![code size](https://img.shields.io/github/languages/code-size/ivaquero/book-filters.svg)
![repo size](https://img.shields.io/github/repo-size/ivaquero/book-filters.svg)

This project is the reorganization of the code in the book [Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) and draws on some of the content in [EKF/UKF Toolbox for Matlab](https://github.com/EEA-sensors/ekfukf).

<p align="left">
<a href="README.md">English</a> |
<a href="README-CN.md">简体中文</a>
</p>

## Goals

- Build a filter simulation test tool that is friendly to beginners
- Form a set of easy-to-understand introductory tutorials

## Requirements

To build the environment, there are 3 options

- Install [Anaconda](https://www.anaconda.com/download/success)
- Install [Miniconda](https://docs.anaconda.com/free/miniconda/miniconda-install/)
- Install [Miniforge](https://conda-forge.org/miniforge/)

For option 2&3, you need to run the following command after installation

```bash
conda install scipy matplotlib pandas
```

## Structure

- `filters`: Filter-related module
  - `bayes`: Bayesian statistics
  - `fusion`: data fusion
  - `ghk`: α-β-γ filtering
  - `ghq`: Gaussian-Hermite numerical integration
  - `imm`: interactive multiple models
  - `kalman_ckf`: cubature Kalman filter
  - `kalman_ekf`: extended Kalman filter
  - `kalman_enkf`: ensemble Kalman filter
  - `kalman_fm`: fading-memory filter
  - `kalman_hinf`: H∞ filter
  - `kalman_ukf`: unscented Kalman filter
  - `kalman`: linear Kalman filter
  - `lsq`: least squares filter
  - `particle`: particle filter
  - `resamplers`: sampler
  - `sigma_points`: Sigma point
  - `smoothers`: smoother
  - `solvers`: equation solvers (such as Runge-Kutta)
  - `stats`: statistical indicators
  - `helpers`: auxiliary tools
- `models`: Model-related module
  - `const_acc`: constant acceleration model
  - `const_vel`: constant velocity model
  - `coord_ture`: coordinated rotation model
  - `singer`: Singer model
  - `noise`: model noise
- `ssmodel*`: model base class
- `plots`: Plot-related module
  - `plot_common`: common plot (measurement, trajectory, residual)
  - `plot_bayes`: Bayes statistical plot
  - `plot_nonlinear`: nonlinear statistical plot
  - `plot_gh`: α-β-γ filter plot
  - `plot_kf`: Kalman filter plot
  - `plot_kf_plus`: nonlinear Kalman filter plot
  - `plot_pf`: particle filter plot
  - `plot_sigmas`: Sigma point plot
  - `plot_adaptive`: adaptive plot
  - `plot_fusion`: data fusion plot
  - `plot_smoother`: smoother plot
- `simulators`: Simulation-related module
  - `datagen`: common data generation
  - `linear`: linear motion model
  - `maneuver`: maneuver model
  - `radar`: ground radar model
  - `robot`: robot model
  - `trajectory`: projectile model
- `cfg`: Simulation configuration interface
- `clutter`: Clutter-related module
- `tracker`: Tracking-related module
  - `associate`: association
  - `pda`: probabilistic data association
  - `estimators`: state estimation
  - `track*`: trackers with association
- `symbol`: Symbol derivation module
  - `datagen`: data generation
  - `models`: motion model

## Examples

- `filters-abcf.ipynb`: α-β-γ filtering
- `filters-bayes.ipynb`: Basics of Bayesian Statistics
- `filters-kf-basic.ipynb`: Basics of Kalman Filtering
- `filters-kf-design.ipynb`: Kalman Filter Design
- `filters-kf-plus.ipynb`: Nonlinear Kalman Filtering
- `filters-maneuver.ipynb`: Maneuvering Target Tracking
- `filters-pf.ipynb`: Particle Filtering
- `filters-smoothers.ipynb`: Smoothers
- `filters-task-fusion.ipynb`: Data Fusion
- `filters-task-tracking.ipynb`: Target Tracking
