"""
Define functions to tune model
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def grid_search(X_train, y_train, estimator_constructor,
                params_fixed, param_grid, scoring, cv):
    """
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param estimator_constructor: name of the class from which estimator is constructed
    :param params_fixed: dict, hyperparameters which are kept fixed. They are passed to the constructor
    :param param_grid: dict, value grid of hyperparameters which are tuned. They are passed to GridSearchCV
    :param scoring: str, scoring metric
    :param cv: int, or an instance of StratifiedKFold
    :return: best_estimator, best_params, best_score
    """
    estimator = estimator_constructor(**params_fixed)
    gs = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring,  cv=cv)
    gs.fit(X_train, y_train)

    best_estimator = gs.best_estimator_
    # fit best_estimator on whole training dataset
    best_estimator.fit(X_train, y_train)
    print("Best estimator:", best_estimator)

    best_params = gs.best_params_
    print("Best parameters:", best_params)

    best_score = gs.best_score_
    print("Best score:", best_score)

    results = {"best_estimator": best_estimator, "best_params": best_params, "best_score": best_score}
    return results


def randomized_search(X_train, y_train, estimator_constructor,
                      params_fixed, param_dist, n_iter,
                      scoring, cv,
                      random_state=None):
    """
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param estimator_constructor: name of the class from which estimator is constructed
    :param params_fixed: dict, hyperparameters which are kept fixed. They are passed to the constructor
    :param param_dist: dict, key are hyperparameter names, values are list of float ore a scipy distribution.
                        They are passed to RandomizedSearchCV.
    :param n_iter: int
    :param scoring: str, scoring metric
    :param cv: int, or an instance of StratifiedKFold
    :param random_state: int or RandomState instance or None
    :return: best_estimator, best_params, best_score
    """
    estimator = estimator_constructor(**params_fixed)

    rs = RandomizedSearchCV(estimator=estimator, param_distributions=param_dist,
                            n_iter=n_iter, scoring=scoring, refit=False, cv=cv,
                            random_state=random_state)
    rs.fit(X_train, y_train)

    best_estimator = rs.best_estimator_
    # fit best_estimator on whole training dataset
    best_estimator.fit(X_train, y_train)
    print("Best estimator:", best_estimator)

    best_params = rs.best_params_
    print("Best parameters:", best_params)

    best_score = rs.best_score_
    print("Best score:", best_score)

    results = {"best_estimator": best_estimator, "best_params": best_params, "best_score": best_score}
    return results