import numpy as np
import pandas as pd
import os
import pickle
from sklearn.svm import SVC
import time

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
            for train_limit in trainning_size:
                clf = SVC(C=c,
                          kernel=kernel)
                print("fit for:", c, kernel, train_limit)
                start_time = time.time()
                clf.fit(X=np.array(x_train[:train_limit]),
                        y=np.array(y_train[:train_limit]).ravel())
                duration = time.time() - start_time
                print(duration)
                train_score = clf.score(X=np.array(x_train[:train_limit]),
                                        y=np.array(y_train[:train_limit]))
                test_score = clf.score(X=x_test,
                                       y=y_test.ravel())
                data.append((test_score, train_score, c, kernel, train_limit))
                data_dict[c][kernel][train_limit] = {'score': test_score,
                                                     'train_score': train_score}

    data_frame = pd.DataFrame(data, columns=["score", "train_score", "c", "kernel", "train_limit"])
    data_dict_indexes = {'c': 0,
                         'kernel': 1,
                         'train_limit': 2,
                         'score_type': 3}
    return data_frame, data_dict, data_dict_indexes


if __name__ == "__main__":
    data_set_name = "creditcard"
    df, dd, ddi = svm(data_set_name,
                      c_values=(0.2, 1.6, 1.0, 1.4, 1.8),
                      kernel_values=["rbf", "linear", "poly", "sigmoid"],
                      trainning_size=range(1000, 11001, 2500)
                      )
    # df, dd, ddi = svm(data_set_name,
    #                   c_values=np.array(range(2, 20, 2)) / 10,
    #                   kernel_values=["rbf", "linear", "poly", "sigmoid"],
    #                   trainning_size=range(500, 2001, 500)
    #                   )
    if not os.path.exists("stats"):
        os.makedirs("stats")
    df.to_csv(path_or_buf="stats/svm_" + data_set_name + ".csv")
    with open("stats/svm_" + data_set_name + "_dict.pikle", 'wb') as handle:
        pickle.dump(dd, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("stats/svm_" + data_set_name + "_dict_indexes.pikle", 'wb') as handle:
        pickle.dump(ddi, handle, protocol=pickle.HIGHEST_PROTOCOL)

