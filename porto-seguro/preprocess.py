import pandas as pd
import numpy as np


def split_vars(data):
    """
    split variable by type
    - binary
    - categorical
    - continuous

    :param data: data
    :return: binary, categorical, continuous variable
    """
    bin_vars = []
    cat_vars = []
    con_vars = []

    for col in data.columns:
        if 'cat' in col:
            cat_vars.append(col)
        elif 'bin' in col:
            bin_vars.append(col)
        else:
            con_vars.append(col)

    print("continuous, ordinal variables: ", con_vars[:5])
    print("binary variables: ", bin_vars[:5])
    print("catgorical variables: ", cat_vars[:5])
    print("\n")

    return bin_vars, cat_vars, con_vars


def select_bin_vars(bin_vars):
    """

    :param bin_vars: binary vars
    :return: selected binary_vars
    """
    ind_vars = []

    for feature in bin_vars:
        if 'ind' in feature:
            ind_vars.append(feature)
        else:
            pass
    return ind_vars


def shrink_cat_vars(train, cat_vars, shrink_size=10):
    """

    :param train_cat:
    :param shrink_size:
    :return:
    """

    shrinked_cat_vars = {} # key: column, value: shrink category list

    for col in cat_vars:
        if len(train[col].unique()) <= shrink_size:
            shrinked_cat_vars[col] = train[col].unique()
            continue
        else:
            target_by_col = train.groupby(col)['target'].mean().sort_values(ascending=False)
            shrinked_ratio = len(target_by_col) / shrink_size
            shrinked_cat_vars[col] = [target_by_col[int(i * shrinked_ratio):int((i+1) * shrinked_ratio)].index.values
                                      for i in range(shrink_size)]

    return shrinked_cat_vars


def shrink_cat(data, shrink_cat_vars_dict):
    """

    :param data:
    :param shrink_cat_vars_dict:
    :return:
    """
    for col, category in shrink_cat_vars_dict.items():
        cat_len = len(category)

        for i in range(cat_len):
            new_column = col + '_new_' + str(i)
            data[new_column] = 0

            if type(category) is list:
                data.loc[data[col].isin(category[i]), new_column] = 1
            else:
                data.loc[data[col] == category[i], new_column] = 1
    return data
