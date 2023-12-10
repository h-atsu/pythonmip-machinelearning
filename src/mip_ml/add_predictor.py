"""Define generic function that can add any known trained predictor."""

from .exceptions import NotRegistered
from .modeling.get_convertor import get_convertor
from .registered_predictors import registered_predictors


def add_predictor_constr(mip_model, predictor, input_vars, output_vars=None, **kwargs):

    convertors = registered_predictors()
    convertor = get_convertor(predictor, convertors)
    if convertor is None:
        raise NotRegistered(type(predictor).__name__)
    return convertor(mip_model, predictor, input_vars, output_vars, **kwargs)