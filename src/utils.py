import csv
import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle

from src.const import *

# data set agnostic tools


def get_data(data_set, batch_format):
    if data_set == "starcraft":
        csv_dict = read_csv("data/train.csv")

        players_dict, val_players_dict = split_training_set(csv_dict, VALIDATION_SPLIT)

        if batch_format is "sklearn":
            x_train, y_train = csv_set_to_sklearn_batch(players_dict)
            x_test, y_test = csv_set_to_sklearn_batch(val_players_dict)
        else:
            x_train, y_train, _ = csv_set_to_keras_batch(players_dict)
            x_test, y_test, _ = csv_set_to_keras_batch(val_players_dict)

        # shuffle train set
        x_train, y_train = shuffle(x_train, y_train, random_state=6)

    elif data_set == "creditcard":
        df = pd.read_csv("data/creditcard.csv", header=0)
        data = df.drop('Class', axis=1).values
        target = df.get('Class').values
        target = target.reshape((len(target), 1))
        x_train, x_test, y_train, y_test = train_test_split(data,
                                                            target,
                                                            test_size=VALIDATION_SPLIT,
                                                            shuffle=True)
        # limit the train set to 10 000 values
        # x_train = x_train[0:100000]
        # y_train = y_train[0:100000]
        # set the train set 50% of each class
        x_test_small = []
        y_test_small = []
        n_v, n_f = 0, 0
        for class_id, features in zip(y_test, x_test):
            if class_id == 1:
                n_f += 1
                x_test_small.append(features)
                y_test_small.append(class_id)
            elif n_v < n_f:
                n_v += 1
                x_test_small.append(features)
                y_test_small.append(class_id)
        x_test = x_test_small
        y_test = y_test_small

        # set the batch to a keras one if needed
        if batch_format == "keras":
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
    else:
        if data_set == "iris":
            data = load_iris()
        elif data_set == "cancer":
            data = load_breast_cancer()
        else:
            raise ValueError("%s data set is not implemented" % data_set)
        x_train, x_test, y_train, y_test = train_test_split(data.data,
                                                            data.target,
                                                            test_size=VALIDATION_SPLIT,
                                                            shuffle=True)
        if batch_format is "keras":
            y_train = to_categorical(y_train)
            y_test = to_categorical(y_test)
    print("Selected data set is", data_set, "with", len(y_train) + len(y_test),
          "data (train:", len(y_train), ", test:", len(y_test), ")")
    # normalize input data
    x_train, x_test = normalize_x_data(np.array(x_train), np.array(x_test))
    return x_train, np.array(y_train), x_test, np.array(y_test)


def normalize_x_data(x_train, x_test):
    # compute normalization on training set only
    x_min = np.min(x_train, axis=0)
    x_max = np.max(x_train, axis=0)
    # apply the same transform on both datasets
    epsilon = 0.0001
    x_train[:] = x_train[:] - x_min / (x_max - x_min + epsilon)
    x_test[:] = x_test[:] - x_min / (x_max - x_min + epsilon)

    return x_train, x_test


# data management tools


def consolidate_dict_data(dict_data, consolidate_argx, consolidate_argy, consolidate_argz,
                          argx_name="arg_x", argy_name="arg_y", filters=None):
    consolidate_dict = dict()
    for keys, values in dict_explore(dict_data):
        arg_x = keys[consolidate_argx]
        arg_y = keys[consolidate_argy]
        arg_z = keys[consolidate_argz]
        arg_xy = (arg_x, arg_y)
        if filters is None or are_keys_on_filters(keys, filters):
            if arg_z not in consolidate_dict:
                consolidate_dict[arg_z] = dict()
            if arg_xy not in consolidate_dict[arg_z]:
                consolidate_dict[arg_z][arg_xy] = list()
            consolidate_dict[arg_z][arg_xy].append(values)
    
    return_dict = dict()
    for arg_z, d in consolidate_dict.items():
        data = []
        for k, l in d.items():
            a = np.array(l)
            mean = np.mean(a)
            std = np.std(a)
            median = np.median(a)
            a_min = np.min(a)
            a_max = np.max(a)
            data.append([k[0], k[1], mean, median, std, a_min, a_max])
        df = pd.DataFrame(data, columns=[argx_name, argy_name, "mean", "median", "std", "min", "max"])
        return_dict[arg_z] = df
    return return_dict


def are_keys_on_filters(keys, filters):
    for key_index, key_value in filters:
        try:
            if keys[key_index] not in key_value:
                return False
        except TypeError:
            if keys[key_index] != key_value:
                return False
    return True


def dict_explore(dict_data, keys=None):
    if keys is None:
        keys = []
    if type(dict_data) == dict:
        for k, v in dict_data.items():
            keys.append(k)
            yield from dict_explore(v, keys)
            keys.pop()
    else:
        yield keys, dict_data


def to_seaborn_dataframe(consolidate_dict, wanted_value='median', value_name='median', consolidate_z_name='argz'):
    df = pd.DataFrame()
    for k, v in consolidate_dict.items():
        tmp = pd.DataFrame(data=v.get([v.columns[0], v.columns[1], wanted_value]),
                           columns=[v.columns[0], v.columns[1], value_name])
        tmp = tmp.assign(argz=[k] * len(tmp))
        tmp.rename(columns={'argz': consolidate_z_name}, inplace=True)
        df = df.append(tmp)
        df.columns = tmp.columns
    return df


# first data set tools


def read_csv(file_name, read_all=False, time_limit=GAME_TIME_STEP_LIMIT):
    players_action_dict = defaultdict(list)
    with open(file_name, 'r') as csvfile:
        number_of_line = sum(1 for _ in csvfile)  # count the number of line in the file
        csvfile.seek(0)  # set the file cursor back to file start
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line_number, row in enumerate(reader):
            action_list = row[2:]
            player_id = row[0]
            race = row[1]
            action_first_time_dict = {'s': time_limit, 'hotkey50': time_limit,
                                      'hotkey40': time_limit, 'hotkey52': time_limit,
                                      'hotkey42': time_limit, 'hotkey10': time_limit,
                                      'hotkey12': time_limit, 'hotkey20': time_limit,
                                      'hotkey22': time_limit, 'hotkey30': time_limit,
                                      'hotkey60': time_limit,
                                      'hotkey62': time_limit,
                                      'hotkey32': time_limit, 'Base': time_limit,
                                      'hotkey70': time_limit, 'hotkey72': time_limit,
                                      'hotkey00': time_limit,
                                      'hotkey90': time_limit,
                                      'hotkey80': time_limit, 'SingleMineral': time_limit,
                                      'hotkey02': time_limit, 'hotkey82': time_limit,
                                      'hotkey92': time_limit,
                                      'hotkey91': time_limit,
                                      'hotkey01': time_limit, 'hotkey41': time_limit,
                                      'hotkey21': time_limit, 'hotkey71': time_limit,
                                      'hotkey81': time_limit,
                                      'hotkey61': time_limit,
                                      'hotkey11': time_limit, 'hotkey51': time_limit,
                                      'hotkey31': time_limit}
            action_dict = {'s': 0, 'hotkey50': 0, 'hotkey40': 0, 'hotkey52': 0, 'hotkey42': 0, 'hotkey10': 0,
                           'hotkey12': 0, 'hotkey20': 0, 'hotkey22': 0, 'hotkey30': 0, 'hotkey60': 0, 'hotkey62': 0,
                           'hotkey32': 0, 'Base': 0, 'hotkey70': 0, 'hotkey72': 0, 'hotkey00': 0, 'hotkey90': 0,
                           'hotkey80': 0, 'SingleMineral': 0, 'hotkey02': 0, 'hotkey82': 0, 'hotkey92': 0, 'hotkey91': 0,
                           'hotkey01': 0, 'hotkey41': 0, 'hotkey21': 0, 'hotkey71': 0, 'hotkey81': 0, 'hotkey61': 0,
                           'hotkey11': 0, 'hotkey51': 0, 'hotkey31': 0}
            current_timestep = 0
            max_ap5s = 0
            ap5s = 0
            for i in action_list:
                current_action = i
                if current_action[0] == 't':
                    current_timestep += 5
                    max_ap5s = max_ap5s if max_ap5s > ap5s else ap5s
                    ap5s = 0
                    continue
                if current_timestep > time_limit:
                    break
                action_dict[current_action] += 1
                ap5s += 1
                if action_dict[current_action] == 1:
                    action_first_time_dict[current_action] = current_timestep

            # order action to be sure that every game as them in the same order
            ordered_action_dict = OrderedDict(sorted(action_dict.items()))
            ordered_action_first_time_dict = OrderedDict(sorted(action_first_time_dict.items()))

            # compute additional information
            relative_line_position = line_number  # / number_of_line
            mapm = len(action_list) * 1000 / (current_timestep + 1)  # + 1 to prevent division by 0
            other_info = (relative_line_position, mapm, max_ap5s)

            if current_timestep < 60 and not read_all:
                pass
                # print("discarded line %i, game too short: %is" % (line_number, current_timestep))
            else:
                players_action_dict[player_id].append((race, ordered_action_dict, ordered_action_first_time_dict, other_info))
    return players_action_dict


def split_training_set(source_dict, test_to_train_ratio=0.1):
    train_dict = {}
    test_dict = {}
    for player_name, player_game in source_dict.items():
        number_of_games = len(player_game)
        split_index = number_of_games - int(number_of_games * test_to_train_ratio) - 2
        train_game = player_game[:split_index]
        test_game = player_game[split_index:]
        train_dict[player_name] = train_game
        test_dict[player_name] = test_game
    return train_dict, test_dict


def csv_set_to_keras_batch(csv_dict):
    batch_input = []
    batch_output = []

    player_id_to_name_dict = {}
    for i, t in enumerate(csv_dict.items()):
        player_id_to_name_dict[i] = t[0]
        for race, action_dict, first_time_dict, other_info in t[1]:
            race_list = [0, 0, 0]

            if race == 'Zerg':
                race_list[2] = 1
            elif race == 'Protoss':
                race_list[1] = 1
            elif race == 'Terran':
                race_list[0] = 1
            else:
                print("unknown race:", race)
                exit(10)

            input_array = np.array(race_list
                                   + list(action_dict.values())
                                   + list(first_time_dict.values())
                                   + list(other_info)
                                   , dtype=int)
            output_array = np.zeros(shape=NUMBER_OF_PLAYERS, dtype=int)

            output_array[i] = 1

            batch_input.append(input_array)
            batch_output.append(output_array)
    return batch_input, batch_output, player_id_to_name_dict


def csv_set_to_sklearn_batch(csv_dict):
    batch_input = []
    batch_output = []

    for i, t in enumerate(csv_dict.items()):
        for race, action_dict, first_time_dict, other_info in t[1]:
            race_id = 0

            if race == 'Zerg':
                race_id = 3
            elif race == 'Protoss':
                race_id = 1
            elif race == 'Terran':
                race_id = 2
            else:
                print("unknown race:", race)
                exit(10)

            input_list = [race_id] + list(action_dict.values()) \
                                   + list(first_time_dict.values()) \
                                   + list(other_info)
            input_array = np.array(input_list
                                   , dtype=int)
            output_string = t[0]

            batch_input.append(input_array)
            batch_output.append(output_string)
    return batch_input, batch_output
