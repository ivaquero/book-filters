# Kalman Filter Simulation Toolkit

![code size](https://img.shields.io/github/languages/code-size/ivaquero/blog-filters.svg)
![repo size](https://img.shields.io/github/repo-size/ivaquero/blog-filters.svg)

This project is the reorganization of the code in the book [Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python).

## Goals

- Provide a set of easy-to-understand introductory tutorials
- Build a filter simulation toolkit that is friendly to beginners

## Toolkit Structure

### `filters`: Filter-related module

- [ ] `bayes`: Bayesian statistics
- [ ] `fusion`: data fusion
- [ ] `ghk`: α-β-γ filtering
- [ ] `ghq`: Gaussian-Hermite numerical integration
- [ ] `imm`: interactive multiple models
- [ ] `kalman_ckf`: cubature Kalman filter
- [ ] `kalman_ekf`: extended Kalman filter
- [ ] `kalman_enkf`: ensemble Kalman filter
- [ ] `kalman_fm`: fading-memory filter
- [ ] `kalman_hinf`: H∞ filter
- [ ] `kalman_ukf`: unscented Kalman filter
- [ ] `kalman`: linear Kalman filter
- [ ] `lsq`: the least squares filter
- [ ] `particle`: particle filter
- [ ] `resamplers`: sampler
- [ ] `sigma_points`: Sigma point
- [ ] `smoothers`: smoother
- [ ] `solvers`: equation solvers (such as Runge-Kutta)
- [ ] `stats`: statistical indicators
- [ ] `helpers`: auxiliary tools

### `models`: Model-related module

- [ ] `const_acc`: constant acceleration model
- [ ] `const_vel`: constant velocity model
- [ ] `coord_turn`: coordinated rotation model
- [ ] `singer`: Singer model
- [ ] `noise`: model noise
- [ ] `ssmodel*`: model base class
- [ ] `pda`: probabilistic data association

### `plots`: Plot-related module

- [ ] `plot_common`: common plot (measurement, trajectory, residual)
- [ ] `plot_bayes`: Bayes statistical plot
- [ ] `plot_gh`: α-β-γ filter plot
- [ ] `plot_kf`: Kalman filter plot
- [ ] `plot_kf_plus`: nonlinear Kalman filter plot
- [ ] `plot_pf`: particle filter plot
- [ ] `plot_sigmas`: Sigma point plot
- [ ] `plot_fusion`: data fusion plot
- [ ] `plot_smoother`: smoother plot

### `simulators`: Simulation-related module

- [ ] `datagen`: common data generation
- [ ] `linear`: linear motion model
- [ ] `maneuver`: maneuver model
- [ ] `radar`: ground radar model
- [ ] `robot`: robot model
- [ ] `trajectory`: projectile model
- [ ] `cfg`: Simulation configuration interface

### `symbol`: Symbol derivation module

- [ ] `datagen`: data generation
- [ ] `models`: motion model
