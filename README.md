# Automatic Verification of Intelligent Systems (2024)
Useful scripts for the course.
The course project's repository is [here](https://github.com/andrea-dimarco/high-dimensional-space-anomaly-detection).

- [Linear Difference Solver](./Equations/linear_difference_solver.m): MATLAB script that returns the solution for the models as **linear difference equations**, if some of the roots are complex and conjucate an additional $\alpha$ and $\phi$ parameters will be returned to put in the classic formula shown below where $Z_t$ is the random process, $N$ is the number of roots found, $m_i$ is the multiplicity of the $i$-th root, $b_ij$ is the weight but for our innntents and purposes just write it as $b_ij$ and $R_i$ is the $i$-th root
  $Z_t= \sum_{i=1}^{N} \sum_{j=0}^{m_i-1} b_{ij}t^jR_i^t$
  If the roots are complex and conjugated then write them like below where $R_i = (c\pm id)$ and $\alpha = \sqrt{c^2+d^2}$ and $\phi=tan^{-1}\left(\frac{d}{c}\right)$:
  $b_i\alpha^t cos \phi t$
  There is also the Python version of this script [here](./Equations/difference-equations.py), although is slightlu less accurate.

- [Stationary Time-Series Plot](time-series-plot.py): Python script that given the parameters of a (mean and variance) **_stationary_ model**, returns the plots for the time-series, the ACF $\rho_k$ and PACF $\gamma_k$.

- [Model Identification Plot](model-identification-plot.py): Python script that takes the string with the function results and plots the graph you vcan use to identify the model of the exercise.

- [ARMA Process](./Models/arma-process.py): Python cript that given the parameters for an ARMA(1,1) model and returns, simulates it and returns both the statistics computed over the simulation and with the formulas.

- [AR Process](./Models/moving-average-process.py): Python cript that given the parameters for an AR(1) or AR(2) model returns, simulates it and returns both the statistics computed over the simulation and with the formulas.

- [AR Process](./Models/auto-regressive-process.py): Python cript that given the parameters for an MA(q) or model returns, simulates it (only with q up to 2) and returns both the statistics computed over the simulation and with the formulas.
