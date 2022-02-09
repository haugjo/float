"""Feature Weight Box Plot.

This function returns a box plot that illustrates the feature weights of one or multiple online feature selection
methods.

Copyright (C) 2022 Johannes Haug.
"""
from matplotlib.axes import Axes
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, List

# Global color palette
# dark blue, teal, light blue, dark green, olive, yellow green, red, magenta, grey, black
_PALETTE = ['#003366', '#44aa99', '#88ccee', '#117733', '#999933', '#ddcc77', '#cc3311', '#ee3377', '#bbbbbb', '#000000']


def feature_weight_box(feature_weights: List[list],
                       model_names: List[str],
                       feature_names: list,
                       top_n_features: Optional[int] = None,
                       fig_size: tuple = (13, 5),
                       font_size: int = 16) -> Axes:
    """Returns a box plot that shows the distribution of weights for the selected or all features.

    Args:
        feature_weights:
            A list of lists, where each list corresponds to the feature weights of one feature selection model.
        model_names: Names of the feature selection models. These labels will be used in the legend.
        feature_names: The names of all input features. The feature names will be used as x-tick labels.
        top_n_features:
            Specifies the top number of features to be displayed. If the attribute is None, we show all features in
            their original order. If the attribute is not None, we select the top features according to their median
            value.
        fig_size: The figure size (length x height)
        font_size: The font size of the axis labels.

    Returns:
        Axes: The Axes object containing the bar plot.
    """
    n_models = len(feature_weights)
    width = 0.6
    n_features = top_n_features if top_n_features else len(feature_names)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y')
    top_features_idx = None
    legend_lines = []
    top_feature_names = feature_names

    for i in range(n_models):
        weights_by_feature = [list(x) for x in zip(*feature_weights[i])]

        # get weights from top n features according to the median
        if top_n_features:
            if top_features_idx is None:
                medians_by_feature = np.median(weights_by_feature, axis=1)
                top_features_idx = np.argsort(-medians_by_feature)[:n_features]
            weights_by_feature = list(np.array(weights_by_feature)[top_features_idx])
            top_feature_names = list(np.array(feature_names)[top_features_idx])

        # draw the box plots
        color = _PALETTE[i]
        ax.boxplot(weights_by_feature,
                   positions=np.arange(n_features) - (width + 0.1) / 2 + i / n_models * (width + 0.1),
                   widths=width / n_models,
                   patch_artist=True,
                   boxprops=dict(facecolor='white', color=color, lw=2),
                   capprops=dict(color=color, lw=2),
                   whiskerprops=dict(color=color, lw=2),
                   flierprops=dict(color=color, markeredgecolor=color, lw=2),
                   medianprops=dict(color=_PALETTE[-3], lw=2),
                   )
        legend_lines.append(mlines.Line2D([], [], marker='|', linestyle='None', markersize=10, markeredgewidth=1.5,
                                          color=_PALETTE[i], label=model_names[i]))

    plt.xticks(np.arange(n_features),
               labels=np.asarray(top_feature_names),
               rotation=20,
               ha='right')
    plt.ylabel('Feature Weights', size=font_size, labelpad=1.5)
    plt.xlabel('Input Feature', size=font_size, labelpad=1.6)
    plt.legend(handles=legend_lines, frameon=True, loc='best', fontsize=font_size * 0.8, borderpad=0.3, handletextpad=0.5)
    plt.margins(0.01, 0.03)
    plt.tick_params(axis='both', labelsize=font_size * 0.8, length=0)
    plt.tight_layout()
    return ax
