import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import sys

from src.utils import get_data
from src.const import *


def tree(data_set, max_depth_values=(None,), min_samples_split_values=(2,)):
    x_train, y_train, x_test, y_test = get_data(data_set=data_set,
                                                batch_format="sklearn")

    max_depth_values = list(max_depth_values)
    min_samples_split_values = list(min_samples_split_values)

    data = []

    for max_depth in max_depth_values:
        for min_samples_split in min_samples_split_values:
            clf = DecisionTreeClassifier(max_depth=max_depth,
                                         min_samples_split=min_samples_split,
                                         random_state=None,
                                         class_weight="balanced"
                                         )
            clf.fit(X=np.array(x_train),
                    y=np.array(y_train))
            score = clf.score(X=x_test,
                              y=y_test)
            data.append((score, max_depth, min_samples_split))

    return pd.DataFrame(data, columns=["score", "max_depth", "min_sample_split"])


if __name__ == "__main__":
    df = tree("iris", [30, 35, 40], [2, 3, 4])
    print(df)
