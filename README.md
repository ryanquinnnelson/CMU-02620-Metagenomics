# 02620-Metagenomics
Spring 2021 Machine Learning for Scientists group project

## Executing the code
### Using implemented models
Jupyter Notebook `example_execution_rqn_210513.ipynb` contains a sample run showing how to execute the sampling process and training for implemented models.

### Grid Search
Each Python module in `packages.gridsearch` contains the code for executing a grid search over a single model type.

### Plotting Results
Jupyter Notebooks `plot_results_rqn_210409.ipynb` and `results_plot_psk_042521.ipynb` contain the code for generating plots from the grid search results.

## Project Structure
Code for this project is subdivided into Python packages and Jupyter notebooks. 
Notebooks were used for testing and plotting results. 
Python packages were developed for sampling and encoding the data, implementing required models from scratch, 
and testing model performance via grid search.

### packages
- `generative_model`: Implements Naïve Bayes
- `linear_model`: Implements Multiclass Logistic Regression
- `metagenomics`: Implements fragment sampling and encoding
- `gridsearch`: Implements grid search functionality to test each model over a range of hyperparameters. 

### notebooks
- `example_execution_rqn_210513.ipynb`: Example run showing how to execute the sampling process and training for implemented models.
- `data_exploratory_rqn_210321.ipynb`: Explores original dataset and builds sequence datasets used by this project.
- `dev_logistic_regression_rqn_210327.ipynb`: Testing of logistic regression implementation
- `dev_sampling_encoding_rqn_210325.ipynb`: Testing of sampling and encoding
- `plot_results_rqn_210409.ipynb`: Plots results of grid search
- `practice_psk_040421.ipynb`: Testing of sampling and encoding
- `psk_testing_naivebayes_041821.ipynb`: Testing of Naïve Bayes implementation
- `results_plot_psk_042521.ipynb`: Plot results for grid search for SVM and Naïve Bayes
