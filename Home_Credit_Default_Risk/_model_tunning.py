"""
Define functions to tune model
"""

import pickle

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def grid_search(estimator, X_train, y_train, params_grid, scoring, cv,
                random_state=None, re_fit=False, pkl_out=None):
    """
    :param estimator: an estimator object which has fit, predict..., and other methods consistent with sklearn API
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param params_grid: dict, value grid of hyperparameters which are tuned.
    :param scoring: str, scoring metric
    :param cv: int
    :param random_state: int
    :param re_fit: whether to fit on the whole train set
    :param pkl_out: str or None, pickle file name
    :return: best_estimator, best_params, best_score
    """
    assert type(params_grid) == dict, "params_grid must be a dict"

    kfold = StratifiedKFold(n_splits=cv, random_state=random_state)
    gs = GridSearchCV(estimator=estimator, param_grid=params_grid, scoring=scoring,  cv=kfold)
    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    best_estimator = gs.best_estimator_

    # fit best_estimator on whole training data set
    if re_fit:
        best_estimator.fit(X_train, y_train)

    best_score = gs.best_score_

    results = {"best_estimator": best_estimator, "best_params": best_params, "best_score": best_score}

    if pkl_out is not None:
        pickle.dump(results, open(pkl_out, "wb"))

    return results


def randomized_search(estimator, X_train, y_train, params_grid, n_iter, scoring, cv,
                      random_state=None, re_fit=False, pkl_out=None):
    """
    :param estimator: an estimator object which has fit, predict... and other methods consistent with sklearn API
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param params_grid: dict, value grid of hyperparameters which are tuned.
    :param n_iter: int
    :param scoring: str, scoring metric
    :param cv: int, or an instance of StratifiedKFold
    :param random_state: int or RandomState instance or None
    :param re_fit: whether to fit on the whole train set
    :param pkl_out: str or None, pickle file name
    :return: best_estimator, best_params, best_score
    """
    assert type(params_grid) == dict, "params_grid must be a dict"

    kfold = StratifiedKFold(n_splits=cv, random_state=random_state)

    params = estimator.get_params()

    rs = RandomizedSearchCV(estimator=estimator, param_distributions=params_grid,
                            n_iter=n_iter, scoring=scoring, refit=False, cv=kfold,
                            random_state=random_state)
    rs.fit(X_train, y_train)

    best_params = rs.best_params_

    params.update(best_params)
    estimator.set_params(**params)

    # fit best_estimator on whole training data set
    if re_fit:
        estimator.fit(X_train, y_train)

    best_score = rs.best_score_

    results = {"best_estimator": estimator, "best_params": best_params, "best_score": best_score}

    if pkl_out is not None:
        pickle.dump(results, open(pkl_out, "wb"))

    return results


def tune_n_estimators_w_early_stopping(estimator, X_train, y_train,
                                       max_n_estimators=5000, eval_size=0.2,
                                       eval_metric="auc",
                                       early_stopping_rounds=50,
                                       verbose=False,
                                       random_state=None, pkl_out=None):
    """
    :param estimator: an estimator object which has fit, predict..., and other methods consistent with sklearn API
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param max_n_estimators: int
    :param eval_size: float, between 0 and 1
    :param eval_metric: str
    :param early_stopping_rounds: int
    :param verbose: bool
    :param random_state: int
    :param pkl_out: str or None, pickle file name
    :return: estimator
    """
    X_train_s, X_eval, y_train_s, y_eval = train_test_split(X_train, y_train, test_size=eval_size,
                                                            random_state=random_state)

    params = estimator.get_params()
    params.update(dict(n_estimators=max_n_estimators))
    estimator.set_params(**params)

    eval_set = [(X_eval, y_eval)]
    estimator.fit(X_train_s, y_train_s, eval_metric=eval_metric,
                  eval_set=eval_set, early_stopping_rounds=early_stopping_rounds,
                  verbose=verbose)

    params.update(dict(n_estimators=estimator.best_iteration))
    estimator.set_params(**params)
    estimator.fit(X_train, y_train)

    if pkl_out is not None:
        pickle.dump(estimator, open(pkl_out, "wb"))

    return estimator


def grid_search_stepwise(estimator, X_train, y_train, params_grid_steps,
                         scoring="roc_auc", cv=5,
                         random_state=None, pkl_out=None):
    """
    :param estimator: an estimator object which has fit, predict..., and other methods consistent with sklearn API
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param params_grid_steps: list of dict
    :param scoring: str
    :param cv: int
    :param random_state: int
    :param pkl_out: str or None, pickle file name
    :return: estimator
    """
    assert type(params_grid_steps) == list, "params_grid_steps must be a list"

    best_scores = []
    best_params = []
    for step, params_grid in enumerate(params_grid_steps):
        print("Doing grid search step %d" % step)
        results = grid_search(estimator, X_train, y_train, params_grid, scoring, cv, random_state=random_state)

        print("best_score:\n", results["best_score"])
        best_scores.append(results["best_score"])

        estimator = results["best_estimator"]

        print("best_params:\n", estimator.get_params())
        best_params.append(estimator.get_params())

    estimator.fit(X_train, y_train)
    results = {"best_estimator": estimator, "best_params": best_params, "best_scores": best_scores}

    if pkl_out is not None:
        pickle.dump(results, open(pkl_out, "wb"))

    return results
