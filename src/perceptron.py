import numpy as np
import pandas as pd
import os
import json

from keras.layers import Dense, Activation, BatchNormalization, InputLayer, Dropout
from keras.models import Sequential

from src.utils import *


def get_model(feature_count, class_count, hidden_layer, use_batch_norm=True, optimizer='rmsprop', activation='sigmoid'):
    return_model = Sequential()
    return_model.add(InputLayer(input_shape=(feature_count,)))
    # return_model.add(Dropout(0.1))
    if use_batch_norm:
        return_model.add(BatchNormalization())
    for neurones in hidden_layer:
        return_model.add(Dense(neurones))
        return_model.add(Activation(activation))
        if use_batch_norm:
            return_model.add(BatchNormalization())
    # return_model.add(Dropout(0.1))
    return_model.add(Dense(class_count))
    return_model.add(Activation('softmax'))

    return_model.compile(optimizer=optimizer,
                         loss='categorical_crossentropy',
                         metrics=['categorical_accuracy'])

    # return_model.summary()

    return return_model


def perceptron(data_set, number_of_epoch, hidden_neurons=((NUMBER_HIDDEN_NEURONS,),),
               use_batch_norm_values=(True,), optimizer_values=('rmsprop',),
               activation_values=('sigmoid',), training_sizes=(-1,)):

    x_learn, y_learn, x_test, y_test = get_data(data_set=data_set,
                                                batch_format="keras")

    fit_dict = {}
    df = pd.DataFrame()
    for opt in optimizer_values:
        fit_dict[opt] = dict()
        for act in activation_values:
            fit_dict[opt][act] = dict()
            for use_batch_norm in use_batch_norm_values:
                fit_dict[opt][act][use_batch_norm] = dict()
                for layers in hidden_neurons:
                    fit_dict[opt][act][use_batch_norm][layers] = dict()
                    for train_limit in training_sizes:
                        number_of_feature = len(x_learn[0])
                        number_of_class = len(y_learn[0])
                        model = get_model(number_of_feature, number_of_class, layers, use_batch_norm, opt, act)
                        h = model.fit(x=np.array(x_learn[:train_limit]),
                                      y=np.array(y_learn[:train_limit]),
                                      batch_size=len(x_learn[:train_limit]),
                                      epochs=number_of_epoch,
                                      validation_data=(np.array(x_test),
                                                       np.array(y_test)),
                                      verbose=0
                                      )
                        epoch = h.epoch
                        h_values = h.history.values()
                        values = np.array([epoch, ] + list(h_values))
                        tmp = pd.DataFrame(data=values.T, columns=["epoch", ] + list(h.history.keys()))
                        tmp = tmp.assign(use_batch_norm=pd.Series([use_batch_norm] * number_of_epoch))
                        tmp = tmp.assign(optimizer=pd.Series([opt] * number_of_epoch))
                        tmp = tmp.assign(activation=pd.Series([act] * number_of_epoch))
                        tmp = tmp.assign(layers=pd.Series([str(layers)] * number_of_epoch))
                        tmp = tmp.assign(train_size=pd.Series([str(train_limit)] * number_of_epoch))
                        df = df.append(tmp)
                        # fit_dict[opt][act][use_batch_norm][layers][train_limit] = pd.DataFrame(data=values.T,
                        #                                                                        columns=["epoch", ] +
                        #                                                                                list(h.history.keys()))
                        fit_dict[opt][act][use_batch_norm][layers][train_limit] = dict()
                        for i, e in enumerate(epoch):
                            fit_dict[opt][act][use_batch_norm][layers][train_limit][e] = dict()
                            for k, v in h.history.items():
                                fit_dict[opt][act][use_batch_norm][layers][train_limit][e][k] = v[i]
    fit_dict_index = {'optimizer': 0,
                      'activation': 1,
                      'use_batch_norm': 2,
                      'layers': 3,
                      'train_limit': 4,
                      'epoch': 5,
                      'score_type': 6}
    return df, fit_dict, fit_dict_index


if __name__ == "__main__":
    data_set_name = "creditcard"
    dff, tt, tti = perceptron(data_set_name,
                              number_of_epoch=600,
                              hidden_neurons=((), (15,), (10,), (5,), (10, 5), (15, 5)),
                              use_batch_norm_values=(True, False),
                              optimizer_values=('rmsprop', 'adam'),
                              activation_values=('sigmoid', 'relu', 'linear', 'selu'),
                              training_sizes=(-1,) + tuple(range(500, 10000, 500))
                              )
    if not os.path.exists("stats"):
        os.makedirs("stats")
    dff.to_csv(path_or_buf="stats/per_" + data_set_name + ".csv")
    json.dump(tt, open("stats/per_" + data_set_name + "_dict.json", 'w'))
    json.dump(tti, open("stats/per_" + data_set_name + "_dict_indexes.json", 'w'))
