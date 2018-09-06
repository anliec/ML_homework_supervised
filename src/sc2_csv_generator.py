import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical

from src.utils import get_data

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = get_data("starcraft", "sklearn")
    # np.savetxt("data/starcraft_x_train.csv", x_train, delimiter=",")
    # np.savetxt("data/starcraft_y_train.csv", y_train, delimiter=",", fmt="%s", encoding="utf8")
    # np.savetxt("data/starcraft_x_test.csv", x_test, delimiter=",")
    # np.savetxt("data/starcraft_y_test.csv", y_test, delimiter=",", fmt="%s", encoding="utf8")

    header = ["race", 'count_s', 'count_hotkey50', 'count_hotkey40', 'count_hotkey52', 'count_hotkey42',
              'count_hotkey10', 'count_hotkey12', 'count_hotkey20', 'count_hotkey22', 'count_hotkey30',
              'count_hotkey60', 'count_hotkey62', 'count_hotkey32', 'count_Base', 'count_hotkey70', 'count_hotkey72',
              'count_hotkey00', 'count_hotkey90', 'count_hotkey80', 'count_SingleMineral', 'count_hotkey02',
              'count_hotkey82', 'count_hotkey92', 'count_hotkey91', 'count_hotkey01', 'count_hotkey41',
              'count_hotkey21', 'count_hotkey71', 'count_hotkey81', 'count_hotkey61', 'count_hotkey11',
              'count_hotkey51', 'count_hotkey31', 'first_time_s', 'first_time_hotkey50', 'first_time_hotkey40',
              'first_time_hotkey52', 'first_time_hotkey42', 'first_time_hotkey10', 'first_time_hotkey12',
              'first_time_hotkey20', 'first_time_hotkey22', 'first_time_hotkey30', 'first_time_hotkey60',
              'first_time_hotkey62', 'first_time_hotkey32', 'first_time_Base', 'first_time_hotkey70',
              'first_time_hotkey72', 'first_time_hotkey00', 'first_time_hotkey90', 'first_time_hotkey80',
              'first_time_SingleMineral', 'first_time_hotkey02', 'first_time_hotkey82', 'first_time_hotkey92',
              'first_time_hotkey91', 'first_time_hotkey01', 'first_time_hotkey41', 'first_time_hotkey21',
              'first_time_hotkey71', 'first_time_hotkey81', 'first_time_hotkey61', 'first_time_hotkey11',
              'first_time_hotkey51', 'first_time_hotkey31', 'line_position', 'avg_micro_apm', 'max_ap5s']

    df_x_train = pd.DataFrame(x_train, columns=header)
    df_x_test = pd.DataFrame(x_test, columns=header)
    df_x_all = df_x_train.append(df_x_test, ignore_index=True)

    df_x_train.to_csv("data/starcraft_x_train.csv", sep=',', encoding='utf-8')
    df_x_test.to_csv("data/starcraft_x_test.csv", sep=',', encoding='utf-8')
    df_x_all.to_csv("data/starcraft_x_all.csv", sep=',', encoding='utf-8')

    df_y_train = pd.DataFrame(y_train, columns=['label'])
    df_y_test = pd.DataFrame(y_test, columns=['label'])
    df_y_all = df_y_train.append(df_y_test, ignore_index=True)

    df_y_train.to_csv("data/starcraft_y_train.csv", sep=',', encoding='utf-8')
    df_y_test.to_csv("data/starcraft_y_test.csv", sep=',', encoding='utf-8')
    df_y_all.to_csv("data/starcraft_y_all.csv", sep=',', encoding='utf-8')

    lb = LabelEncoder()
    lb.fit(df_y_all)
    nl_y_all = lb.transform(df_y_all)
    nl_y_train = lb.transform(df_y_train)
    nl_y_test = lb.transform(df_y_test)

    df_nl_y_train = pd.DataFrame(nl_y_train, columns=['label'])
    df_nl_y_test = pd.DataFrame(nl_y_test, columns=['label'])
    df_nl_y_all = df_nl_y_train.append(df_nl_y_test, ignore_index=True)

    df_nl_y_train.to_csv("data/starcraft_nl_y_train.csv", sep=',', encoding='utf-8')
    df_nl_y_test.to_csv("data/starcraft_nl_y_test.csv", sep=',', encoding='utf-8')
    df_nl_y_all.to_csv("data/starcraft_nl_y_all.csv", sep=',', encoding='utf-8')

    y = to_categorical(df_nl_y_all.get('label'))

    sy = np.sum(y, axis=0)
    print(np.min(sy), np.max(sy), np.median(sy), np.mean(sy))

