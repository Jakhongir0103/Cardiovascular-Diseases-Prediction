# "THE GOOD", "THE BAD" and "THE UGLY" for Cardiovascular Diseases Prediction on the Behavioral Risk Factor Surveillance System

## Codebase organization

📦Repository\
 ┣ 📂data\
 ┃ ┣ 📜submission.csv\
 ┃ ┣ 📜x_test.csv\
 ┃ ┣ 📜x_train.csv\
 ┃ ┗ 📜y_train.csv\
 ┣ 📂data_preparation\
 ┃ ┣ 📜data_loader.py\
 ┃ ┣ 📜features_info.py\
 ┃ ┣ 📜features_util.py\
 ┃ ┗ 📜preprocessing.py\
 ┣ 📂model\
 ┃ ┣ 📜predict.py\
 ┃ ┗ 📜train.py\
 ┣ 📂notebooks\
 ┃ ┣ 📜final-grid-search.ipynb\
 ┃ ┣ 📜heatmaps.ipynb\
 ┃ ┣ 📜run_BAD.ipynb\
 ┃ ┣ 📜run_GOOD.ipynb\
 ┃ ┣ 📜run_UGLY.ipynb\
 ┃ ┗ 📜THE-GOOD.pkl\
 ┣ 📂util\
 ┃ ┣ 📜custom_pca.py\
 ┃ ┣ 📜evaluation.py\
 ┃ ┣ 📜plotting.py\
 ┃ ┗ 📜util.py\
 ┣ 📜implementations.py\
 ┣ 📜README.md\
 ┗ 📜run.py

The codebase is organized as follows:

- `implementations.py` contains the code for each method in the list of required ML functions, together with support methods useful to make the ML functions implementations more readable.
- `run.py` is the main script in the codebase, which allows you to train the model that performed best on the validator @ AIcrowd, among the different proposed ones. After the training, the script generates the corresponding submission file.
- `/model` folder contains two scripts that collects the main methods that allow to interact with the model: the first, `train.py`, collects the methods used for training flow and hyper parameters grid search, while the second, `predict.py`, encodes the functions used for the inference of the model, i.e. predicting the target class.
- `/data_preparation` collects all the scripts that realise the whole process of data preparation, from utility methods needed to load the dataset, filter features and pre-process using the two proposed pipeline, which are the basic one and the “informed” one. Moreover, `features_info.py` collects all the information extracted from the dataset’s documentation and needed to implement correctly the “informed” pre-processing.
- `/util` contains utilities needed mainly for plotting and evaluation.
- `/notebooks` collects three different notebooks each showing the complete flow of pre-processing, training and inference using the three proposed variants: the good, the bad and the ugly. Note that `run_UGLY.py` does basically the same as the main `run.py` script, just adding some plots.

To allow our results to be fully reproducible, this script, along with the three notebooks mentioned above, contains a call to a utility method that takes care of setting the correct seed for random numbers generation.

## How to run

First clone our repo:

```bash
git clone https://github.com/epfml/ml-project-1-sarcastic-gradient-descent.git
```

Then, before running our model's training and inference, make sure the following files are inside the `/data/` directory:

- `x_train.csv` and `y_train.csv` are required to train the model
- `x_test.csv` is needed to inference and generate the submission

By running the `run.py` script you will be able to generate the same submission file already present in `/data`. Note that in the report we've described and compared three different approaches: the good, the bad and the ugly.
This main script actually uses "the ugly" which is the one that performed best on the validator, but other approaches presented more interpretability and other advantages discussed in the report.

```bash
python run.py
```

To explore the three different versions of the model proposed and described in the report (with different data pre-processing), you can run the three notebooks inside the `/notebook` directory, which also gives an overview of the results and analysis we’ve done via vizualisation.
