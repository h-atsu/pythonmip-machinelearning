from .linear_regression import add_linear_regression_constr
from .logistic_regression import add_logistic_regression_constr


def sklearn_predictors():
    return {
        "LinearRegression": add_linear_regression_constr,
        "Ridge": add_linear_regression_constr,
        "Lasso": add_linear_regression_constr,
        "LogisticRegression": add_logistic_regression_constr,
    }
