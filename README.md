# Homework 1 Machine Learning: Supervised Learning

## Datasets

The datasets are located in the data/ directory.

### Credit Card

File data/craditcard.csv contain the dataset used for this homework.

As stated on the the report, a copy can also be downloaded at: https://www.kaggle.com/dalpozz/creditcardfraud/

### Starcraft II

File data/train.csv contain the dataset used for this homework.

As stated on the the report, a copy can also be downloaded at: https://www.kaggle.com/c/the-insa-starcraft-2-player-prediction-challenge/data

## How to use

To get the same result as we do on our report, you must take two steps:
 - generate the data
 - generate the plots 

### Generate data
 
To generate the data you will need to run the five python scripts corresponding to the five machine learning method studied here.

The five scripts to launch are:
 - src/tree.py
 - src/perceptron.py
 - src/boost.py
 - src/svm.py
 - src/knn.py
 
Each of the scripts take as argument the data set to use: "creditcard" or "starcraft"

For example you can launch the decision tree classifier on the starcraft dataset with:

    python3 -m src.tree starcraft

You can also get all the .pikle and csv files already generated at: https://github.com/anliec/ML_homework_supervised/tree/master/stats

### Generate the plots

To generate the plot given in the report you will have to run the following jupyter notebooks:
 - plot_tree.ipynb
 - plot_perceptron.ipynb
 - plot_boost.ipynb
 - plot_svm.ipynb
 - plot_knn.ipynb
 
Each one generate the plot for the given machine learning method from the associated .pickle files in the stats directory and try to write the generated plot into the graphs directory.

## Requirement 

 - Python 3 (tested with Python 3.6)
 - Keras
 - Sklearn
 - Pandas
 - Matplotlib
 - Seaborn
 - Numpy
 - jupyter
 
If you do not already have them installed, it can be done quickly using pip:

    pip3 install keras sklearn pandas matplotlib seaborn numpy 