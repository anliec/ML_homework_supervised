import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from src.utils import get_data
from src.const import *


def boost(data_set, max_depth_values=(5,), n_estimators_values=(100,), min_sample_split_values=(2,)):
    x_train, y_train, x_test, y_test = get_data(data_set=data_set,
                                                batch_format="sklearn")

    max_depth_values = list(max_depth_values)
    n_estimators_values = list(n_estimators_values)
    min_sample_split_values = list(min_sample_split_values)

    data = []

    for max_depth in max_depth_values:
        for n_estimators in n_estimators_values:
            for min_sample_split in min_sample_split_values:
                clf = GradientBoostingClassifier(max_depth=max_depth,
                                                 n_estimators=n_estimators,
                                                 min_samples_split=min_sample_split
                                                 )
                clf.fit(X=np.array(x_train),
                        y=np.array(y_train))
                score = clf.score(X=x_test,
                                  y=y_test)
                data.append((score, max_depth, n_estimators, min_sample_split))

    return pd.DataFrame(data, columns=["score", "max_depth", "n_estimators", "min_sample_split"])


if __name__ == "__main__":
    df = boost("iris", range(1, 5, 1), range(10, 200, 10), range(2, 10, 2))
    print(df)
