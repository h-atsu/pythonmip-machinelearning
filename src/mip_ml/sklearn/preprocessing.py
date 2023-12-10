import mip

from ..exceptions import NoModel


def add_standard_scaler_constr(mip_model, standard_scaler, input_vars, **kwargs):

    return StandardScalerConstr(mip_model, standard_scaler, input_vars, **kwargs)


class StandardScalerConstr():

    def __init__(self, mip_model, scaler, input_vars, **kwargs):
        self._default_name = "std_scaler"
        self._output_shape = scaler.n_features_in_
        super().__init__(mip_model, scaler, input_vars, **kwargs)

    def _mip_model(self, **kwargs):
        """Do the transformation on x."""
        _input = self._input
        output = self._output

        scale = self.transformer.scale_
        mean = self.transformer.mean_

        self.mip_model += _input - output * scale == mean, self._name_var("s")
        return self


def sklearn_transformers():
    """Return dictionary of Scikit Learn preprocessing objects."""
    return {
        "StandardScaler": add_standard_scaler_constr,
    }
