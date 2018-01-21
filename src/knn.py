import numpy as np
import pandas as pd
import os
import json
from sklearn.neighbors import KNeighborsClassifier

from src.utils import get_data
from src.const import *


def knn(data_set, n_neighbors_values=(5,), p_values=(2,), training_sizes=(-1,)):
    x_train, y_train, x_test, y_test = get_data(data_set=data_set,
                                                batch_format="sklearn")

    n_neighbors_values = list(n_neighbors_values)
    p_values = list(p_values)

    data = []
    data_dict = {}
    for n_neighbors in n_neighbors_values:
        data_dict[n_neighbors] = {}
        for p in p_values:
            data_dict[n_neighbors][p] = {}
            for train_size in training_sizes:
                clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                                           p=p,
                                           weights='uniform',
                                           n_jobs=-1)
                clf.fit(X=np.array(x_train[:train_size]),
                        y=np.array(y_train[:train_size]))
                score = clf.score(X=x_test,
                                  y=y_test)
                data.append((score, n_neighbors, p, train_size))
                data_dict[n_neighbors][p][train_size] = score

    data_frame = pd.DataFrame(data, columns=["score", "n_neighbors", "p", "train_size"])
    data_dict_indexes = {'n_neighbors': 0,
                         'p': 1,
                         'train_size': 2}
    return data_frame, data_dict, data_dict_indexes


if __name__ == "__main__":
    data_set_name = "creditcard"
    df, dd, ddi = knn(data_set_name,
                      n_neighbors_values=range(1, 112, 5),
                      p_values=[2],
                      training_sizes=[-1])
    if not os.path.exists("stats"):
        os.makedirs("stats")
    df.to_csv(path_or_buf="stats/knn_" + data_set_name + ".csv")
    json.dump(dd, open("stats/knn_" + data_set_name + "_dict.json", 'w'))
    json.dump(ddi, open("stats/knn_" + data_set_name + "_dict_indexes.json", 'w'))
