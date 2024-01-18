# Automatic Verification of Intelligent Systems (2024)
Useful scripts for the course.
The course project's repository is [here](https://github.com/andrea-dimarco/high-dimensional-space-anomaly-detection).

- [Linear Difference Solver](linear_difference_solver.m): MATLAB script that returns the solution for the models as **linear difference equations**, if some of the roots are complex and conjucate an additional $\alpha$ and $\phi$ parameters will be returned to put in the classic formula shown below where $Z_t$ is the random process, $N$ is the number of roots found, $m_i$ is the multiplicity of the $i$-th root, $b_ij$ is the weight but for our innntents and purposes just write it as $b_ij$ and $R_i$ is the $i$-th root
  $Z_t=\sum_{i=1}^{N}\sum_{j=0}^{m_i-1}b_{ij}t^jR_i^t$
  If the roots are complex and conjugated then write them like below where $R_i = (c\pm id)$ and $\alpha = \sqrt{c^2+d^2}$ and $\phi=tan^{-1}\left(\frac{d}{c}\right)$:
  $b_i\alpha^t cos \phit$

- [Stationary Time-Series Plot](stationary-time-series-plot.py): Python script that given the parameters of a (mean and variance) **_stationary_ model**, returns the plots for the time-series, the ACF $\rho_k$ and PACF $\gamma_k$.
