from typing import Union, List, Dict

import matplotlib.pyplot as plt
import numpy as np


def multiple_hists(data: np.ndarray, columns: Union[str, List[str]], feature_index: Dict[str, int]):

    if type(columns) == str:
        columns = [columns]

    num_plots_side = int(np.ceil(np.sqrt(len(columns))))
    fig, axes = plt.subplots(num_plots_side, num_plots_side, figsize=(14, 14))
    for n, column in enumerate(columns):
        ax = axes[n // num_plots_side][n % num_plots_side]
        ax.hist(data[:, feature_index[column]])
        ax.set_xlabel(column)
