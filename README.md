[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/U9FTc9i_)

# Codebase organization
The codebase is organized as follows:

- `implementations.py` contains the code for each method in the list of required ML functions, together with support methods useful to make the ML functions implementations more readable.
- `run.py` is the main script in the codebase, which allows you to train the model that performed best on the validator @ AIcrowd, among the different proposed ones. After the training, the script generates the corresponding submission file.
- `/model` folder contains two scripts that collects the main methods that allow to interact with the model: the first, `train.py`, collects the methods used for training flow and hyper parameters grid search, while the second, `predict.py`, encodes the functions used for the inference of the model, i.e. predicting the target class.
- `/data_preparation` collects all the scripts that realise the whole process of data preparation, from utility methods needed to load the dataset, filter features and pre-process using the two proposed pipeline, which are the basic one and the “informed” one. Moreover, `features_info.py` collects all the information extracted from the dataset’s documentation and needed to implement correctly the “informed” pre-processing.
- `/util` contains utilities needed mainly for plotting and evaluation.
- `/notebooks` collects three different notebooks each showing the complete flow of pre-processing, training and inference using the three proposed variants: the good, the bad and the ugly. Note that `run_UGLY.py` does basically the same as the main `run.py` script, just adding some plots.

To allow our results to be fully reproducible, this script, along with the three notebooks mentioned above, contains a call to a utility method that takes care of setting the correct seed for random numbers generation.

By running the `run.py` script you will be able to generate the same submission file already present in `/data`
