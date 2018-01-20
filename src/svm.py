import numpy as np
import pandas as pd
from sklearn.svm import SVC

from src.utils import get_data
from src.const import *


def svm(data_set, c_values=(1.0,), kernel_values=('rbf',)):
    x_train, y_train, x_test, y_test = get_data(data_set=data_set,
                                                batch_format="sklearn")

    kernel_values = list(kernel_values)
    c_values = list(c_values)

    data = []

    for c in c_values:
        for kernel in kernel_values:
            clf = SVC(C=c,
                      kernel=kernel)
            clf.fit(X=np.array(x_train),
                    y=np.array(y_train))
            score = clf.score(X=x_test,
                              y=y_test)
            data.append((score, c, kernel))

    return pd.DataFrame(data, columns=["score", "c", "kernel"])


if __name__ == "__main__":
    df = svm("iris", range(1, 5, 1), ["rbf"])
    print(df)
