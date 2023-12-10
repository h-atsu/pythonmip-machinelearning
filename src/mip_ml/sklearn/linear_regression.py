from .base_regressions import BaseSKlearnRegressionConstr


def add_linear_regression_constr(
    mip_model, linear_regression, input_vars, output_vars=None, **kwargs
):
    return LinearRegressionConstr(
        mip_model, linear_regression, input_vars, output_vars, **kwargs
    )


class LinearRegressionConstr():

    def __init__(self, mip_model, predictor, input_vars, output_vars=None, **kwargs):
        self._default_name = "lin_reg"
        BaseSKlearnRegressionConstr.__init__(
            self,
            mip_model,
            predictor,
            input_vars,
            output_vars,
            **kwargs,
        )

    def _mip_model(self, **kwargs):
        """Add the prediction constraints to Gurobi."""
        self._add_regression_constr()
