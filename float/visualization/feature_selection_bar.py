from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Optional
import warnings

# Global color palette
# dark blue, teal, light blue, dark green, olive, yellow green, red, magenta, grey, black
_PALETTE = ['#003366', '#44aa99', '#88ccee', '#117733', '#999933', '#ddcc77', '#cc3311', '#ee3377', '#bbbbbb', '#000000']


def feature_selection_bar(selected_features: List[list],
                          model_names: List[str],
                          feature_names: list,
                          top_n_features: Optional[int] = None,
                          fig_size: tuple = (13, 5),
                          font_size: int = 16) -> Axes:
    """Returns a bar plot that shows the number of times a feature was selected (between multiple models).

    Args:
        selected_features:
            A list of lists, where each list corresponds to the selected feature vectors of one feature selection model.
        model_names: Names of the feature selection models. These labels will be used in the legend.
        feature_names: The names of all input features. The feature names will be used as x-tick labels.
        top_n_features:
            Specifies the top number of features to be displayed. If the attribute is None, we show all features in
            their original order. If the attribute is not None, we select the top features of the first provided model
            and compare it with the remaining models.
        fig_size: The figure size (length x height)
        font_size: The font size of the axis labels.

    Returns:
        Axes: The Axes object containing the bar plot.
    """
    width = 0.8
    n_models = len(selected_features)
    n_features = top_n_features if top_n_features else len(feature_names)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y')
    order = None
    name_idx = np.arange(n_features)

    for i in range(n_models):
        meas = np.array(selected_features[i]).flatten()
        uniques, counts = np.unique(meas, return_counts=True)

        if top_n_features:
            if order is None:  # We order the features according to the first provided model.
                order = np.argsort(counts)[::-1][:top_n_features]
                name_idx = uniques[order]
            y = []
            for ni in name_idx:  # Select correct counts
                count_i = np.argwhere(uniques == ni)
                if len(count_i) == 0:
                    y.append(0)
                else:
                    y.extend(counts[count_i[0]])

            if len(y) < top_n_features:
                n_features = len(y)
                warnings.warn('The reference model has only selected {} unique features. The number of displayed '
                              'features is automatically reset'.format(len(y)))
        else:
            y = np.zeros(n_features)
            y[uniques] = counts

        ax.bar(np.arange(n_features) - width / 2. + i / n_models * width,
               y,
               width=width / n_models,
               zorder=100,
               color=_PALETTE[i],
               label=model_names[i])

    plt.xticks(np.arange(n_features),
               labels=np.asarray(feature_names)[name_idx],
               rotation=20,
               ha='right')
    plt.ylabel('No. Times Selected', size=font_size, labelpad=1.5)
    plt.xlabel('Input Feature', size=font_size, labelpad=1.6)
    plt.legend(frameon=True, loc='best', fontsize=font_size * 0.8, borderpad=0.3, handletextpad=0.5)
    plt.margins(0.01, 0.03)
    plt.tick_params(axis='both', labelsize=font_size * 0.8, length=0)
    plt.tight_layout()
    return ax
