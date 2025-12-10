# Description
The source codes of RITS-I, RITS, BRITS-I, BRITS for health-care data imputation/classification

To run the code:
For test file:
python main.py --epochs 100 --batch_size 32 --model brits
For beijing airquality
python main_beijing.py --epochs 100 --batch_size 32 --model brits
For physionet2018 dataset (sepsis):
python main_spesis.py --epochs 100 --batch_size 32 --model brits


# Data Format
## Test data physionet 2012
Test data from physionet 2012:
In json folder, we provide the sample data (400 patients).
The data format is as follows:

* Each line in json/json is a string represents a python dict
* The structure of each dict is
    * forward
    * backward
    * label

    'forward' and 'backward' is a list of python dicts, which represents the input sequence in forward/backward directions. As an example for forward direction, each dict in the sequence contains:
    * values: list, indicating $x_t \in R^d$ (after elimination)
    * masks: list, indicating $m_t \in R^d$
    * deltas: list, indicating $\delta_t \in R^d$
    * evals: list, indicating $x_t \in R^d$ (before elimination)
    * eval_masks: list, indicating whether each value is an imputation ground-truth

## Beijing
Data Used for project can be found in Data/beijing_airquality/PRSA_Data_Aotizhongxin_20130301-20170228.csv. https://archive.ics.uci.edu/dataset/501/beijing+multi+site+air+quality+data is the original source of the data. Beijing handler and the makefile.py are files that make the beijing data be in the same form as the test data for model input.

## Physionet 2019
Physionet2019 data can be found in the training_setA folder where 20000 patients are. For our experiment we used the first 5000 patients and then from those selected patients with sequence length less than or equal to 50. Shorter sequences were padded and then same data processing steps were taken as in other datasets.

