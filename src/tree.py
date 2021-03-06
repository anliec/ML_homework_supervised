import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import pickle
import os
import sys

from src.utils import get_data
from src.const import *


def tree(data_set, max_depth_values=(None,), min_samples_split_values=(2,), training_sizes=(-1,)):
    x_train, y_train, x_test, y_test = get_data(data_set=data_set,
                                                batch_format="sklearn")

    max_depth_values = list(max_depth_values)
    min_samples_split_values = list(min_samples_split_values)

    data = []
    data_dict = {}
    for max_depth in max_depth_values:
        data_dict[max_depth] = {}
        for min_samples_split in min_samples_split_values:
            data_dict[max_depth][min_samples_split] = {}
            for train_limit in training_sizes:
                clf = DecisionTreeClassifier(max_depth=max_depth,
                                             min_samples_split=min_samples_split,
                                             random_state=None,
                                             class_weight="balanced"
                                             )
                clf.fit(X=np.array(x_train[:train_limit]),
                        y=np.array(y_train[:train_limit]))
                train_score = clf.score(X=np.array(x_train[:train_limit]),
                                        y=np.array(y_train[:train_limit]))
                test_score = clf.score(X=x_test,
                                       y=y_test)
                data.append((test_score, train_score, max_depth, min_samples_split, train_limit))
                data_dict[max_depth][min_samples_split][train_limit] = {'score': test_score,
                                                                        'train_score': train_score}

    data_frame = pd.DataFrame(data, columns=["score", "train_score", "max_depth", "min_sample_split", "train_size"])
    data_dict_indexes = {'max_depth': 0,
                         'min_samples_split': 1,
                         'train_limit': 2,
                         'score_type': 3}
    return data_frame, data_dict, data_dict_indexes


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Wrong command line argument !")
        print()
        print("Expected call is:", sys.argv[0], "[dataset]")
        print("dataset: one of \"creditcard\" or \"starcraft\"")
        exit(1)
    data_set_name = sys.argv[1]
    if data_set_name == "creditcard":
        df, dd, ddi = tree(data_set_name,
                           max_depth_values=list(range(1, 10, 2)) + list(range(10, 41, 10)),
                           min_samples_split_values=range(2, 18, 5),
                           training_sizes=range(10000, 210001, 50000))
    elif data_set_name == "starcraft":
        df, dd, ddi = tree(data_set_name,
                           max_depth_values=list(range(1, 21, 5)) + list(range(23, 36, 2)) + [37, 40, 45],
                           min_samples_split_values=range(2, 23, 5),
                           training_sizes=range(500, 2001, 500))
    else:
        print("unknow dataset:", data_set_name)
        exit(1)
    if not os.path.exists("stats"):
        os.makedirs("stats")
    df.to_csv(path_or_buf="stats/tree_" + data_set_name + ".csv")
    with open("stats/tree_" + data_set_name + "_dict.pikle", 'wb') as handle:
        pickle.dump(dd, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("stats/tree_" + data_set_name + "_dict_indexes.pikle", 'wb') as handle:
        pickle.dump(ddi, handle, protocol=pickle.HIGHEST_PROTOCOL)
