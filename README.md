# Automatic Verification of Intelligent Systems (2024)
Useful scripts for the course.
The course project's repository is [here](https://github.com/andrea-dimarco/high-dimensional-space-anomaly-detection).

- [Linear Difference Solver](linear_difference_solver.m): MATLAB script that returns the solution for the models as **linear difference equations**, if some of the roots are complex and conjucate an additional $\alpha$ and $\phi$ parameters will be returned to put in the classic formula. $$ Z_t=\sum_{i=1}^{N}\sum_{j=0}^{m_i-1}b_{ij}t^jR_i^t $$

- [Stationary Time-Series Plot](stationary-time-series-plot.py): Python script that given the parameters of a (mean and variance) **_stationary_ model**, returns the plots for the time-series, the ACF $\rho_k$ and PACF $\gamma_k$.
