# Lab 3

### Run `BO_for_pid_tuning.py` for whole results on Bayesian Optimisation.

The `damping` could be set as `True` and `False` for whether turning the damping and delay on.

Figures will pop up as following order:
* Convergence plot for 3 acqusition function EI, PI and LCB.
* Results of `plot_objective()`
* Results of `plot_evaluations()`
* Results of different `length_scale` in fitting the new 1D Gaussian Process model.
* Results of 7 joints' control trajectories driving by the optimal P gains and D gains.

### Run `compare_GP_ZN.py` for checking difference between Gaussian Process optimal result and the result from Z-N method.

### Folders `report_figure` contains all figures for this lab.
* Folder `report_figure/no_damping` contains all results for no damping/delay config. Subfolders are made for each acquisition function.
* Folder `report_figure/damping` contains all results for damping config. Subfolders are made for each acquisition function.
* Folder `report_figure/compare_GP_ZN` contains 7 joints' moving trajectories for comparing GP and ZN P gains and D gains.