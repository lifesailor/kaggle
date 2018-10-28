import numpy as np
import lightgbm as lgbm
from sklearn.model_selection import StratifiedKFold

from eval import gini_normalized


def evalerror(preds, dtrain):
    labels = dtrain.get_label()
    return 'gini', gini_normalized(labels, preds), True


def run_train(train, train_label, params, fold=10, seed=2018):
    NFOLDS = fold
    kfold = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=seed)
    kf = kfold.split(train, train_label)

    cv_train = np.zeros(len(train_label))

    models = []
    fold_scores = []

    for i, (train_fold, validate) in enumerate(kf):
        # 훈련/검증 데이터를 분리한다
        X_train, X_validate, label_train, label_validate = train.iloc[train_fold, :], train.iloc[validate, :], \
                                                           train_label[train_fold], train_label[validate]
        dtrain = lgbm.Dataset(X_train, label_train)
        dvalid = lgbm.Dataset(X_validate, label_validate, reference=dtrain)

        # 훈련 데이터를 학습하고, evalerror() 함수를 통해 검증 데이터에 대한 정규화 Gini 계수 점수를 기준으로 최적의 트리 개수를 찾는다.

        bst = lgbm.train(params, dtrain, valid_sets=dvalid, feval=evalerror, verbose_eval=100, early_stopping_rounds=100)
        models.append(bst)

        # 테스트 데이터에 대한 예측값을 cv_pred에 더한다.
        cv_train[validate] += bst.predict(X_validate)

        # 검증 데이터에 대한 평가 점수를 출력한다.
        score = gini_normalized(label_validate, cv_train[validate])
        fold_scores.append(score)

    print("cv score: ", fold_scores)
    print("validation score: ", gini_normalized(train_label, cv_train))

    return models


def run_test(test, models, fold):
    cv_pred = np.zeros(len(test))

    for model in models:
        cv_pred += model.predict(test, num_iteration=model.best_iteration)

    cv_pred /= fold
    return cv_pred
