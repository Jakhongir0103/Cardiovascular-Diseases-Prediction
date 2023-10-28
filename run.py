import os

import numpy as np

from util.data_loader import load_dataset, split_train_validation, create_csv_submission, change_negative_class
from util.features_util import keep_features, merge_features
from util.features_info import FEATURES_DICT, REPLACEMENT_LIST
from util.preprocessing import preprocessing_pipeline, basic_preprocessing_pipeline
from util.train import reg_logistic_regression
from util.evaluation import evaluation_summary, accuracy, f1_score
from util.util import init_random_seeds
from util.predict import predict, predict_no_labels

init_random_seeds()

BASE_PATH = os.path.dirname(os.getcwd()) + "/data"


def main():
    # Load data without subsampling
    x_train, y_train, x_sub, feature_names = load_dataset(path_x_train=BASE_PATH + "/x_train.csv",
                                                          path_y_train=BASE_PATH + "/y_train.csv",
                                                          path_x_test=BASE_PATH + "/x_test.csv",
                                                          sub_sample=False)
    # Create the feature index dictionary
    feature_index = dict(zip(feature_names, range(len(feature_names))))


    ############# Pre processing #############

    # Pre-process the selected features
    # Keeping only selected features both in train and validation set
    x_train_clean, clean_features, clean_feature_index = keep_features(x_train,
                                                                       FEATURES_DICT.keys(),
                                                                       feature_names,
                                                                       feature_index,
                                                                       verbose=False)
    # Call to the preprocessing pipeline both for the set of selected features
    x_train_clean_proc = preprocessing_pipeline(x_train_clean,
                                                where=clean_features,
                                                feature_index=clean_feature_index,
                                                nan_replacement=REPLACEMENT_LIST,
                                                normalize="mixed")

    # Call to a simplified version of the preprocessing pipeline on the rest of the features
    x_train_dirty_proc = basic_preprocessing_pipeline(x_train,
                                                      where=[f for f in feature_names if f not in clean_features],
                                                      feature_index=feature_index,
                                                      normalization="min-max")

    # Merge the results of the two preprocessing methods in one array
    x_train_merged = merge_features(x_big=x_train_dirty_proc,
                                    feature_index_big=feature_index,
                                    x_small=x_train_clean_proc,
                                    feature_index_small=clean_feature_index)


    ############# Data preparation #############
    # Translate labels from -1/1 to 0/1
    y_train = change_negative_class(y_train[:, 1], current=-1, new=0)

    # Split local data into train and validation
    (x_tr, x_va, y_tr, y_va) = split_train_validation(x_train_merged,
                                                      y_train,
                                                      valid_proportion=0.2)

    # Add bias column to the training and validation data
    x_tr_bias = np.append(x_tr, np.ones(shape=(x_tr.shape[0], 1)), axis=1)
    x_va_bias = np.append(x_va, np.ones(shape=(x_va.shape[0], 1)), axis=1)


    ############# Training #############
    w, train_loss, valid_loss = reg_logistic_regression(x_tr_bias,
                                                        y_tr,
                                                        x_va_bias,
                                                        y_va,
                                                        lambda_=0.00005,
                                                        max_iter=5000,
                                                        gamma=0.5,
                                                        batch_size=10000,
                                                        optimizer="sgd",
                                                        w=np.random.random(size=x_tr_bias.shape[1]),
                                                        all_losses=True)

    ############# Evaluation #############
    # Compute threshold that maximize the F1 score
    thresholds = np.linspace(0, 1, 200)
    accuracies = []
    f1_scores = []
    for threshold in thresholds:
        y_prediction = predict(x_va, w, threshold)
        accuracies.append(accuracy(y_va, y_prediction))
        f1_scores.append(f1_score(y_va, y_prediction))
    f1_scores = np.array(f1_scores)
    f1_scores = np.where(np.isnan(f1_scores), 0, f1_scores)
    opt_threshold = thresholds[np.argmax(f1_scores)]

    # Compute predictions for validation set
    predicted_y_va = predict(x_va_bias,
                             w,
                             threshold=opt_threshold,
                             negative_label=0)
    print("EVALUATION SUMMARY ON VALIDATION SET")
    evaluation_summary(y_va, predicted_y_va)
    print(" -> optimal threshold: {}".format(opt_threshold))


    ############# Load and process data for submission #############
    ids = x_sub[:, 0]
    # Keep only selected features in submission
    x_sub_clean, clean_features, clean_feature_index = keep_features(x_sub,
                                                                     FEATURES_DICT.keys(),
                                                                     feature_names,
                                                                     feature_index,
                                                                     verbose=False)

    x_sub_clean_proc = preprocessing_pipeline(x_sub_clean,
                                              where=clean_features,
                                              feature_index=clean_feature_index,
                                              nan_replacement=REPLACEMENT_LIST,
                                              normalize="mixed")

    x_sub_dirty_proc = basic_preprocessing_pipeline(x_sub,
                                                    where=[f for f in feature_names if f not in clean_features],
                                                    feature_index=feature_index,
                                                    normalization="min-max")

    x_sub_merged = merge_features(x_big=x_sub_dirty_proc,
                                  feature_index_big=feature_index,
                                  x_small=x_sub_clean_proc,
                                  feature_index_small=clean_feature_index)

    x_sub_bias = np.append(x_sub_merged, np.ones(shape=(x_sub_merged.shape[0], 1)), axis=1)

    # Compute predictions
    predicted_y_sub = predict(x_sub_bias,
                              w,
                              threshold=opt_threshold,
                              negative_label=-1)

    # Save predictions to csv file
    create_csv_submission(ids=ids,
                          y_pred=predicted_y_sub,
                          path=BASE_PATH + "/submission.csv")


if __name__ == "__main__":
    main()
