from argparse import ArgumentParser
import json

import numpy as np
import pandas as pd
from preprocess import split_vars, select_bin_vars, shrink_cat_vars, shrink_cat
from run import run_train, run_test

parser = ArgumentParser()

parser.add_argument('--objective', type=str, default='binary')
parser.add_argument('--boosting_type', type=str, default='gbdt')
parser.add_argument('--learning_rate', type=float, default=0.02)
parser.add_argument('--num_leaves', type=int, default=15)
parser.add_argument('--max_bin', type=int, default=256)
parser.add_argument('--feature_fraction', type=float, default=0.6)
parser.add_argument('--verbosity', type=int, default=0)
parser.add_argument('--is_unbalance', type=bool, default=False)
parser.add_argument('--max_drop', type=int, default=50)
parser.add_argument('--min_child_samples', type=int, default=10)
parser.add_argument('--min_child_weight', type=int, default=150)
parser.add_argument('--min_split_gain', type=int, default=0)
parser.add_argument('--subsample', type=float, default=0.9)
parser.add_argument('--seed', type=int, default=2018)
parser.add_argument('--num_boost_round', type=int, default=10000)


if __name__ == "__main__":

    print("start")

    # args
    args = parser.parse_args()
    params = vars(args)

    # 1. load data
    folder_loc = '/Users/lifesailor/.kaggle/porto-seguro/'

    train = pd.read_csv(folder_loc + 'train.csv')
    train_label = train['target']
    train_id = train['id']

    test = pd.read_csv(folder_loc + 'test.csv')
    test_id = test['id']

    del test['id']
    del train['id']
    print("loaded data.")

    # 2. feature
    # 2-1. split variables
    bin_vars, cat_vars, con_vars = split_vars(test)

    # 2-2. feature selection - binary, continuous
    features = []
    features += con_vars
    features += select_bin_vars(bin_vars)
    print("selected features: ", features)

    # 2-3. feature engineering - category
    train_cat = train[cat_vars].astype('O')
    test_cat = test[cat_vars].astype('O')

    # 10개 이상의 카테고리를 10개로 묶는다.
    shrink_cat_vars_dict = shrink_cat_vars(train, cat_vars, shrink_size=10)
    shrink_cat_train = shrink_cat(train_cat, shrink_cat_vars_dict)
    shrink_cat_test = shrink_cat(test_cat, shrink_cat_vars_dict)

    cat_features = [feature for feature in shrink_cat_train.columns if 'new' in feature]
    print("engineered features: ", cat_features)

    # 2-4. merge
    train = pd.concat([train[features], shrink_cat_train[cat_features]], axis=1).astype(float)
    test = pd.concat([test[features], shrink_cat_test[cat_features]], axis=1).astype(float)
    print(train.shape, test.shape)

    # 3. train and prediction
    fold = 10
    seed = 2018
    models = run_train(train, train_label, params, fold=fold, seed=seed)

    # 4. test
    prediction = run_test(test, models, fold)

    # 5. save
    test_submission = pd.DataFrame({'id': test_id, 'target': prediction})
    test_submission.to_csv('../porto-seguro/final.csv', index=False)
    print("saved result")











