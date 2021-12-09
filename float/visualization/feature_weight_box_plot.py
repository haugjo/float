from matplotlib.axes import Axes
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional

# Global color palette
# dark blue, teal, light blue, dark green, olive, yellow green, red, magenta, grey, black
_PALETTE = ['#003366', '#44aa99', '#88ccee', '#117733', '#999933', '#ddcc77', '#cc3311', '#ee3377', '#bbbbbb', '#000000']


def feature_weight_box_plot(feature_weights: list,
                            model_name: str,
                            feature_names: list,
                            top_n_features: Optional[int] = None,
                            fig_size: tuple = (13, 5),
                            font_size: int = 16) -> Axes:
    """Returns a box plot that shows the distribution of weights for the selected or all features.

    Args:
        feature_weights: A list, where each list corresponds the feature weights of one feature selection model.
        model_name: Name of the feature selection model. This label will be used in the legend.
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
    n_features = top_n_features if top_n_features else len(feature_names)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_axisbelow(True)
    ax.grid(True, axis='y')

    weights_by_feature = [list(x) for x in zip(*feature_weights)]

    # get weights from top n features according to the median
    if top_n_features:
        medians_by_feature = np.median(weights_by_feature, axis=1)
        top_features_idx = np.argsort(-medians_by_feature)[:n_features]
        weights_by_feature = list(np.array(weights_by_feature)[top_features_idx])
        feature_names = list(np.array(feature_names)[top_features_idx])

    # draw the box plots
    plt.boxplot(weights_by_feature)
    plt.xticks(np.arange(n_features),
               labels=np.asarray(feature_names),
               rotation=20,
               ha='right')
    plt.ylabel('Feature weights', size=font_size, labelpad=1.5)
    plt.xlabel('Input Feature', size=font_size, labelpad=1.6)
    legend_line = mlines.Line2D([], [], marker='|', linestyle='None', markersize=10, markeredgewidth=1,
                                color='black', label=model_name)
    plt.legend(handles=[legend_line], frameon=True, loc='best', fontsize=font_size * 0.8, borderpad=0.3, handletextpad=0.5)
    plt.margins(0.01, 0.03)
    plt.tick_params(axis='both', labelsize=font_size * 0.8, length=0)
    plt.tight_layout()
    return ax
