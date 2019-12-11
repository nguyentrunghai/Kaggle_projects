"""
Define functions to tune model
"""

from sklearn.model_selection import GridSearchCV


def grid_search(X_train, y_train, estimator_constructor, fixed_params, param_grid, scoring, cv):
    """
    :param X_train: dataframe or array, training input features
    :param y_train: arryay, training labels
    :param estimator_constructor: name of the class from which estimator is constructed
    :param fixed_params: dict, hyperparameters which are kept fixed
    :param param_grid: dict, value grid of hyperparameters which are tunned
    :param scoring: str, scoring metric
    :param cv: int, or an instance of StratifiedKFold
    :return: best_estimator, best_params, best_score
    """
    estimator = estimator_constructor(**fixed_params)
    gs = GridSearchCV(estimator=estimator, param_grid=param_grid, scoring=scoring,  cv=cv, iif=False)
    gs.fit(X_train, y_train)

    best_estimator = gs.best_estimator_
    # fit best_estimator on whole training dataset
    best_estimator.fit(X_train, y_train)
    print("Best estimator:", best_estimator)

    best_params = gs.best_params_
    print("Best parameters:", best_params)

    best_score = gs.best_score_
    print("Best score:", best_score)

    return best_estimator, best_params, best_score
