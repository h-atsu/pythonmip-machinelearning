"""Define generic function that can add any known trained predictor."""
import sys


def sklearn_convertors():
    """Collect convertors for Scikit-learn objects."""
    if "sklearn" in sys.modules:
        from .sklearn import (  # pylint: disable=import-outside-toplevel
            add_pipeline_constr,
        )
        from .sklearn.predictors_list import (  # pylint: disable=import-outside-toplevel
            sklearn_predictors,
        )
        from .sklearn.preprocessing import (
            sklearn_transformers,  # pylint: disable-import-outside-toplevel
        )

        return (
            sklearn_predictors()
            | sklearn_transformers()
            | {
                "Pipeline": add_pipeline_constr,
            }
        )
    return {}


def registered_predictors():
    """Return the list of registered predictors."""
    convertors = {}
    convertors |= sklearn_convertors()
    return convertors
