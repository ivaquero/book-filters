# 滤波仿真教学工具箱

![code size](https://img.shields.io/github/languages/code-size/ivaquero/book-filters.svg)
![repo size](https://img.shields.io/github/repo-size/ivaquero/book-filters.svg)

本工程是 [Kalman-and-Bayesian-Filters-in-Python](https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python) 一书中代码的整合和再组织，并借鉴了 [EKF/UKF Toolbox for Matlab](https://github.com/EEA-sensors/ekfukf) 中的部分内容。

<p align="left">
<a href="README.md">English</a> |
<a href="README-CN.md">简体中文</a>
</p>

## 工程目标

- 构建一个对入门者友好的滤波仿真试验工具
- 形成一套通俗易懂的入门教程

## 要求

对于构建环境，有 3 个选项

- 安装 [Anaconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/)
- 安装 [Miniconda](https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/)
- 安装 [Miniforge](https://mirrors.tuna.tsinghua.edu.cn/github-release/conda-forge/miniforge/LatestRelease/)

对于选项 2 和 3，你需要在安装后运行以下命令

```bash
# 使用镜像
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
conda config --set show_channel_urls yes
# 安装包
conda install scipy matplotlib pandas
```

## 工程结构

- `filters`：滤波相关模块
  - `bayes`：Bayes 统计
  - `fusion`：数据融合
  - `ghk`：α-β-γ 滤波
  - `ghq`：Gaussian-Hermite 数值积分
  - `imm`：交互多模型
  - `kalman_ckf`：CKF
  - `kalman_ekf`：EKF
  - `kalman_enkf`：EnKF
  - `kalman_fm`：衰减滤波
  - `kalman_hinf`：H∞ 滤波
  - `kalman_ukf`：UKF
  - `kalman`：线性 Kalman 滤波
  - `lsq`：最小二乘滤波
  - `particle`：粒子滤波
  - `resamplers`：采样器
  - `sigma_points`：Sigma 点
  - `smoothers`：平滑器
  - `solvers`：方程求解器（如 Runge-Kutta）
  - `stats`：统计指标
  - `helpers`：辅助工具
- `models`：模型模块
  - `const_acc`：匀加速模型
  - `const_vel`：匀速模型
  - `coord_ture`：协同转角模型
  - `singer`：Singer 模型
  - `noise`：模型噪声
  - `ssmodel*`：模型基类
- `plots`：绘图相关模块
  - `plot_common`：常用绘图（量测、轨迹、残差）
  - `plot_bayes`：Bayes 统计绘图
  - `plot_nonlinear`：非线性统计绘图
  - `plot_gh`：α-β-γ 滤波绘图
  - `plot_kf`：Kalman 滤波绘图
  - `plot_kf_plus`：非线性 Kalman 滤波绘图
  - `plot_pf`：粒子滤波绘图
  - `plot_sigmas`：Sigma 点绘图
  - `plot_adaptive`：自适应绘图
  - `plot_fusion`：数据融合绘图
  - `plot_smoother`：平滑器绘图
- `simulators`：仿真示例相关模块
  - `datagen`：常见数据生成
  - `linear`：线性运动模型
  - `maneuver`：机动模型
  - `radar`：地面雷达模型
  - `robot`：机器人模型
  - `trajectory`：抛体模型
- `cfg`：仿真实验配置接口
- `clutter`：杂波模块
- `tracker`：关联跟踪模块
  - `associate`：关联
  - `pda`：概率互联
  - `estimators`：状态估计
  - `track*`：跟踪
- `symbol`：符号推导模块
  - `datagen`：数据生成
  - `models`：运动模型

## 工程示例

- `filters-abcf.ipynb`：α-β-γ 滤波
- `filters-bayes.ipynb`：Bayes 统计基础
- `filters-kf-basic.ipynb`：Kalman 滤波基础
- `filters-kf-design.ipynb`：Kalman 滤波设计
- `filters-kf-plus.ipynb`：非线性 Kalman 滤波
- `filters-maneuver.ipynb`：机动目标跟踪
- `filters-pf.ipynb`：粒子滤波
- `filters-smoothers.ipynb`：平滑器
- `filters-task-fusion.ipynb`：数据融合
- `filters-task-tracking.ipynb`：目标跟踪
