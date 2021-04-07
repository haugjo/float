# class *ConceptDriftDetector*
#### Description
Abstract base class which serves as a wrapper for skmultiflow concept drift detection models.

#### Attributes
detector (skmultiflow.drift_detection.BaseDriftDetector)
: concept drift detector

#### Functions
init
: Receives a skmultiflow BaseDriftDetector object.

reset
: Resets the concept drift detector parameters.

detected_change
: Checks whether concept drift was detected or not.

detected_warning_zone
: If the concept drift detector supports the warning zone, this function will return whether it's inside the warning zone or not.

get_length_estimation
: Returns the length estimation.

add_element
: Adds the relevant data from a sample into the concept drift detector.s