# class *FeatureSelector*
#### Description
Abstract base class for online feature selection methods.

#### Attributes
name (str)
: name of feature selection model

n_total_ftr (int)
: total number of features

n_selected_ftr (int)
: number of selected features

supports_multi_class (bool)
: True if model support multi-class classification, False otherwise

supports_streaming_features (bool)
: True if model supports streaming features, False otherwise

raw_weight_vector (np.array)
: current weights (as produced by feature selection model)

weights (list)
: absolute weights in all time steps

selection (list)
: indices of selected features in all time steps

comp_time (list)
: computation time in all time steps

_auto_scale (bool)
: indicator for scaling of weights

#### Functions
init
: Receives parameters of feature selection model.

weight_features
: Weights features.

select_features
: Selects features with highest absolute weights.

(substitute_remaining_features)
: Substitutes the features that are not part of the selected feature set.