# Coursework

## All figures are in .zip file for each task, please upzip if want to check those results.

## Task 1:

Find task title and uncomment following code lines in `main()` of `task1.py`.
1. task 1.1 & task 1.2： hidden nodes for shallow MLP and deep MLP
2. task 1.3: learning rates
3. task 1.4：batch size
4. heatmap: for train loss, train time and test loss

All figures in folder `task1` with their folder name indicating the tasks. `experiment_results_final.csv` contains all data for grid search space (consist of the batch size, hidden nodes and learning rates).

## Task 2:

* For task 2.1, run `task_21.py` and all results are in folder `task2/task2_1`. Figures for different goal positions and joints are listed in the folder. All MLP models are in `task2/task2_1/model`. 
* For task 2.2, run `task_22.py` and all results are in folder `task2/task2_2`. Figures for different goal positions and joints are listed in the folder. All random forest models are in `task2/task2_2/model`. Other folders are named with the random forest `max_depth` setting.
* The record of computational efficiency is in `test_log.txt` and `training_log.txt` under each subtask folder.

## Task 3:

Specify the `neural_network_or_random_forest` and `task_id` at the `main()` in `task3.py`.

* For task 3.1 and 3.2, `task_id = "3-1"`. 
  * Set the `neural_network_or_random_forest="nerual_network"` for testing MLP and checking results in `task3/task3_1/nerual_network`.
  * Set the `neural_network_or_random_forest="random_forest"` for testing random forest model. Also specify the `depth` as 2 or 10 for a specified model. Results are in folders `task3/task3_1/random_forest_depth_2` and `task3/task3_1/random_forest_depth_10`.
  * The folder `task3/task3_1/comparison` contains all comparison figures for predicted and measured position/velocity, from running the `plot_task3_1_compare_MLP_RF()` function in `task3.py`.
  * The folder `task3/task3_2` contains the comparison figures for joint average position error and joint average velocity error.

* For task 3.3, `task_id = "3-3"`, `neural_network_or_random_forest="random_forest"`.
  * Set `filter = "EMA"` or `filter = "Gaussian"` for specify the smoothing filter. Also, `depth` should be set as 2 or 10.
  * If EMA filter, choose the `alpha` from [0.1, 0.01, 0.001]. Results are stored in `task3/task3_3/EMA`, with recorded videos (.webm).
  * If Gaussian filter, choose the `sigma` from [2, 75, 150]. Results are stored in `task3/task3_3/Gaussian`, with recorded videos (.webm).
  * The comparison between smoothed models are stored in `task3/task3_3/comparison`.
  * Other figures are mainly for displaying the average joint position error and average joint velocity error.
  * The `task3_3.xlsx` contains all data for each model, including the different smoothing effect.
