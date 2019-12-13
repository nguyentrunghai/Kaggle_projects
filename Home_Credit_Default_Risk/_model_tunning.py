"""
Define functions to tune model
"""

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split


def grid_search(estimator, X_train, y_train, params_grid, scoring, cv, random_state=None):
    """
    :param estimator: an estimator object which has fit, predict..., and other methods consistent with sklearn API
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param params_grid: dict, value grid of hyperparameters which are tuned.
    :param scoring: str, scoring metric
    :param cv: int
    :param random_state: int
    :return: best_estimator, best_params, best_score
    """
    assert type(params_grid) == dict, "params_grid must be a dict"

    kfold = StratifiedKFold(n_splits=cv, random_state=random_state)
    gs = GridSearchCV(estimator=estimator, param_grid=params_grid, scoring=scoring,  cv=kfold)
    gs.fit(X_train, y_train)

    best_params = gs.best_params_
    print("Best parameters:", best_params)

    best_estimator = gs.best_estimator_
    # fit best_estimator on whole training dataset
    best_estimator.fit(X_train, y_train)
    print("Best estimator:", best_estimator)

    best_score = gs.best_score_
    print("Best score:", best_score)

    results = {"best_estimator": best_estimator, "best_params": best_params, "best_score": best_score}
    return results


def randomized_search(estimator, X_train, y_train, params_grid, n_iter, scoring, cv, random_state=None):
    """
    :param estimator: an estimator object which has fit, predict... and other methods consistent with sklearn API
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param params_grid: dict, value grid of hyperparameters which are tuned.
    :param n_iter: int
    :param scoring: str, scoring metric
    :param cv: int, or an instance of StratifiedKFold
    :param random_state: int or RandomState instance or None
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
    print("Best parameters:", best_params)

    params.update(best_params)
    estimator.set_params(**params)
    # fit best_estimator on whole training data set
    estimator.fit(X_train, y_train)
    print("Best estimator:", estimator)

    best_score = rs.best_score_
    print("Best score:", best_score)

    results = {"best_estimator": estimator, "best_params": best_params, "best_score": best_score}
    return results


def tune_n_estimators_w_early_stopping(estimator, X_train, y_train,
                                       max_n_estimators=5000, eval_size=0.2,
                                       eval_metric="roc_auc",
                                       early_stopping_rounds=50,
                                       random_state=None):
    """
    :param estimator: an estimator object which has fit, predict..., and other methods consistent with sklearn API
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param max_n_estimators: int
    :param eval_size: float, between 0 and 1
    :param eval_metric: str
    :param early_stopping_rounds: int
    :return: estimator
    """
    X_train_s, X_eval, y_train_s, y_eval = train_test_split(X_train, y_train, test_size=eval_size,
                                                            random_state=random_state)

    params = estimator.get_params()
    params.update(dict(n_estimators=max_n_estimators))
    estimator.set_params(**params)

    eval_set = [(X_eval, y_eval)]
    estimator.fit(X_train_s, y_train_s, eval_metric=eval_metric,
                  eval_set=eval_set, early_stopping_rounds=early_stopping_rounds)

    params.update(dict(n_estimators=estimator.best_iteration))
    estimator.set_params(**params)
    estimator.fit(X_train, y_train)
    return estimator


def grid_search_stepwise(estimator, X_train, y_train, params_grid_steps, scoring, cv, random_state=None):
    """
    :param estimator: an estimator object which has fit, predict..., and other methods consistent with sklearn API
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param params_grid_steps: list of dict
    :param scoring: str
    :param cv: int
    :param random_state: int
    :return: estimator
    """
    assert type(params_grid_steps) == list, "params_grid_steps must be a list"

    best_scores = []
    best_params = []
    for step, params_grid in enumerate(params_grid_steps):
        print("Doing grid search step %d" % step)
        results = grid_search(estimator, X_train, y_train, params_grid, scoring, cv, random_state=random_state)

        print("best_score:", results["best_score"])
        best_scores.append(results["best_score"])

        estimator = results["best_estimator"]

        print("best_params:", estimator.get_params())
        best_params.append(estimator.get_params())

    results = {"best_estimator": estimator, "best_params": best_params, "best_scores": best_scores}
    return results
