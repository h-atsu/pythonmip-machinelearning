from ..exceptions import NoModel
from ..modeling.get_convertor import get_convertor
from .predictors_list import sklearn_predictors
from .preprocessing import sklearn_transformers


def add_pipeline_constr(mip_model, pipeline, input_vars, output_vars=None, **kwargs):

    return PipelineConstr(mip_model, pipeline, input_vars, output_vars, **kwargs)


class PipelineConstr():
    """Class to formulate a trained :external+sklearn:py:class:`sklearn.pipeline.Pipeline`
    in a gurobipy model.

    |ClassShort|
    """

    def __init__(self, gp_model, pipeline, input_vars, output_vars=None, **kwargs):
        self._steps = []
        self._default_name = "pipe"

    def _build_submodel(self, mip_model, *args, **kwargs):

        self._mip_model(**kwargs)
        assert self.output is not None
        assert self.input is not None
        # We can call validate only after the model is created
        self._validate()
        return self

    def _mip_model(self, **kwargs):
        pipeline = self.predictor
        gp_model = self.gp_model
        input_vars = self._input
        output_vars = self._output
        steps = self._steps
        transformers = sklearn_transformers()
        kwargs["validate_input"] = True

        for transformer in pipeline[:-1]:
            convertor = get_convertor(transformer, transformers)
            if convertor is None:
                raise NoModel(
                    self.predictor,
                    f"I don't know how to deal with that object: {transformer}",
                )
            steps.append(
                convertor(gp_model, transformer, input_vars, **kwargs))
            input_vars = steps[-1].output

        predictor = pipeline[-1]

        convertor = get_convertor(predictor)
        if convertor is None:
            raise NoModel(
                self.predictor,
                f"I don't know how to deal with that object: {predictor}",
            )
        steps.append(convertor(gp_model, predictor,
                     input_vars, output_vars, **kwargs))
        if self._output is None:
            self._output = steps[-1].output

    def print_stats(self, file=None):

        super().print_stats(file=file)
        print(file=file)
        print(f"Pipeline has {len(self._steps)} steps:", file=file)
        print(file=file)

        self._print_container_steps("Step", self._steps, file=file)

    @property
    def _has_solution(self):
        return self[-1]._has_solution

    @property
    def output(self):
        """Returns output variables of pipeline, i.e. output of its last step."""
        return self[-1].output

    @property
    def output_values(self):
        """Returns output values of pipeline in solution, i.e. output of its last step."""
        return self[-1].output_values

    @property
    def input(self):
        """Returns input variables of pipeline, i.e. input of its first step."""
        return self[0].input

    @property
    def input_values(self):
        """Returns input values of pipeline in solution, i.e. input of its first step."""
        return self[0].input_values

    def __getitem__(self, key):
        """Get an item from the pipeline steps."""
        return self._steps[key]

    def __iter__(self):
        """Iterate through pipeline steps."""
        return self._steps.__iter__()

    def __len__(self):
        """Get number of pipeline steps."""
        return self._steps.__len__()
