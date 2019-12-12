"""
Define functions to tune model
"""

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


def grid_search(estimator, X_train, y_train, params_grid, scoring, cv):
    """
    :param estimator: name of the class from which estimator is constructed
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param params_grid: dict, value grid of hyperparameters which are tuned. They are passed to GridSearchCV
    :param scoring: str, scoring metric
    :param cv: int, or an instance of StratifiedKFold
    :return: best_estimator, best_params, best_score
    """
    gs = GridSearchCV(estimator=estimator, param_grid=params_grid, scoring=scoring,  cv=cv)
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


def randomized_search(estimator, X_train, y_train, params_dist, n_iter, scoring, cv, random_state=None):
    """
    :param estimator_constructor: an estimatore object which has fit, predict..., and other methods
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param params_fixed: dict, hyperparameters which are kept fixed. They are passed to the constructor
    :param n_iter: int
    :param scoring: str, scoring metric
    :param cv: int, or an instance of StratifiedKFold
    :param random_state: int or RandomState instance or None
    :return: best_estimator, best_params, best_score
    """
    params = estimator.get_prarams()

    rs = RandomizedSearchCV(estimator=estimator, param_distributions=params_dist,
                            n_iter=n_iter, scoring=scoring, refit=False, cv=cv,
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
