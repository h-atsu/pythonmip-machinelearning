import mip
import numpy as np

from ..exceptions import ParameterError

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


def _default_name(predictor):
    """Make a default name for predictor constraint.

    Parameters
    ----------
    predictor:
        Class of the predictor
    """
    return type(predictor).__name__.lower()


def _get_sol_values(values, columns=None, index=None):
    """Get solution values.

    This is complicated because of the column_transformer.
    In most cases we can just do values.X but if we have a column transformer with
    some constants that can't be translated to Gurobi variables we need to fill in missing values
    """
    if HAS_PANDAS:
        if isinstance(values, pd.DataFrame):
            rval = pd.DataFrame(
                data=_get_sol_values(values.to_numpy()),
                index=values.index,
                columns=values.columns,
            )
            for col in rval.columns:
                try:
                    rval[col] = rval[col].astype(np.float64)
                except ValueError:
                    pass
            return rval.convert_dtypes()
    if isinstance(values, np.ndarray):
        return np.array(
            [v.X if isinstance(v, gp.Var) else v for v in values.ravel()]
        ).reshape(values.shape)
    X = values.X
    if columns is not None and HAS_PANDAS:
        if isinstance(index, (pd.Index, pd.MultiIndex)):
            X = pd.DataFrame(data=X, columns=columns, index=index)
        else:
            X = pd.Series(data=X, columns=columns, name=index)
            raise NotImplementedError(
                "Input variables as pd.Series is not implemented")
    return X


def _dataframe_to_mvar(mip_model, df):
    pass


def validate_output_vars(mip_vars):
    pass


def validate_input_vars(mip_model, mip_vars):
    pass
