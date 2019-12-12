"""
Define functions to tune model
"""

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold


def grid_search(estimator, X_train, y_train, params_grid, scoring, cv, random_state=None):
    """
    :param estimator: name of the class from which estimator is constructed
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
    :param estimator_constructor: an estimatore object which has fit, predict..., and other methods
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

