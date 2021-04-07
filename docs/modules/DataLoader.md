# class *DataLoader*
#### Description
This class serves as a wrapper for the skmultiflow Stream module.

#### Attributes
stream (skmultiflow.data.base_stream.Stream)
: data stream object

file_path (str)
: path to a .csv file containing the data

target_column (int)
: index of the target column of the .csv file

#### Functions
init
: Receives either the path to a .csv-file (+ a target index) which is then mapped to a skmultiflow FileStream object OR a skmultiflow Stream object.

_evaluate_input	
: Evaluates the input data to check if it is in a valid format, i.e. if either a Stream object or a valid .csv file is provided. May be extended by further checks. This is a private function.

get_data
: Loads next batch of data from the Stream object. This is a wrapper for the skmultiflow Stream.next_sample(batch_size) function. The input is an integer value denoting the number of observations to load.