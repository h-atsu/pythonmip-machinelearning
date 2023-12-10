
from ..modeling import AbstractPredictorConstr


class BaseSKlearnRegressionConstr(AbstractPredictorConstr):
    """Predict a Gurobi variable using a Linear Regression that
    takes another Gurobi matrix variable as input.
    """

    def __init__(
        self,
        mip_model,
        predictor,
        input_vars,
        output_vars=None,
        output_type="",
        **kwargs,
    ):
        self._output_shape = 1
        AbstractPredictorConstr.__init__(
            self,
            mip_model,
            input_vars,
            output_vars,
            **kwargs,
        )

    def _add_regression_constr(self, output=None):
        """Add the prediction constraints to Gurobi."""
        if output is None:
            output = self.output
        coefs = self.predictor.coef_.reshape(-1, 1)
        intercept = self.predictor.intercept_
        self.gp_model.addConstr(output == self.input @
                                coefs + intercept, name="linreg")
