import sys

# Add the path to the util folder
sys.path.append("util/")

import numpy as np
from data_loader import *

TRAIN_DATA_PATH_X = "./data/x_train.csv"
TRAIN_DATA_PATH_Y = "./data/y_train.csv"
TEST_DATA_PATH = "./data/x_test.csv"


def main():
    # Load the data
    x_train, y_train, x_test, features_names = load_dataset(
        path_x_train=TRAIN_DATA_PATH_X,
        path_y_train=TRAIN_DATA_PATH_Y,
        path_x_test=TEST_DATA_PATH,
    )
    # Preprocess the data
    raise NotImplementedError

    # Train the model
    raise NotImplementedError

    # Make predictions on the test data
    raise NotImplementedError

    # Create a submission file with predictions
    raise NotImplementedError

    # Save predictions to a CSV file
    raise NotImplementedError


if __name__ == "__main__":
    main()
