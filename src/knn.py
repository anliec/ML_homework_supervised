import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from src.utils import get_data
from src.const import *


def knn(data_set, n_neighbors_values=(5,), p_values=(2,)):
    x_train, y_train, x_test, y_test = get_data(data_set=data_set,
                                                batch_format="sklearn")

    n_neighbors_values = list(n_neighbors_values)
    p_values = list(p_values)

    data = []

    for n_neighbors in n_neighbors_values:
        for p in p_values:
            clf = KNeighborsClassifier(n_neighbors=n_neighbors,
                                       p=p,
                                       weights='uniform',
                                       n_jobs=-1)
            clf.fit(X=np.array(x_train),
                    y=np.array(y_train))
            score = clf.score(X=x_test,
                              y=y_test)
            data.append((score, n_neighbors, p))

    return pd.DataFrame(data, columns=["score", "n_neighbors", "p"])


if __name__ == "__main__":
    df = knn("iris", range(1, 112, 5))
    print(df)
