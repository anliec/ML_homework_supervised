import numpy as np
import pandas as pd
import os
import json
from sklearn.svm import SVC

from src.utils import get_data
from src.const import *


def svm(data_set, c_values=(1.0,), kernel_values=('rbf',), trainning_size=(-1,)):
    x_train, y_train, x_test, y_test = get_data(data_set=data_set,
                                                batch_format="sklearn")

    kernel_values = list(kernel_values)
    c_values = list(c_values)

    data = []
    data_dict = {}
    for c in c_values:
        data_dict[c] = {}
        for kernel in kernel_values:
            data_dict[c][kernel] = {}
            for train_size in trainning_size:
                clf = SVC(C=c,
                          kernel=kernel)
                clf.fit(X=np.array(x_train[:train_size]),
                        y=np.array(y_train[:train_size]))
                score = clf.score(X=x_test,
                                  y=y_test)
                data.append((score, c, kernel, train_size))
                data_dict[c][kernel][train_size] = score

    data_frame = pd.DataFrame(data, columns=["score", "c", "kernel", "train_size"])
    data_dict_indexes = {'c': 0,
                         'kernel': 1,
                         'train_size': 2}
    return data_frame, data_dict, data_dict_indexes


if __name__ == "__main__":
    data_set_name = "creditcard"
    df, dd, ddi = svm(data_set_name,
                      c_values=range(1, 5, 1),
                      kernel_values=["rbf"],
                      trainning_size=[-1])
    if not os.path.exists("stats"):
        os.makedirs("stats")
    df.to_csv(path_or_buf="stats/svm_" + data_set_name + ".csv")
    json.dump(dd, open("stats/svm_" + data_set_name + "_dict.json", 'w'))
    json.dump(ddi, open("stats/svm_" + data_set_name + "_dict_indexes.json", 'w'))

