from functools import partial

from .preprocess import Preprocessor
from .superlearn import Superlearner
from .utils import make_binning_category, make_feature_filter, make_feature_saver


class Optimusle:
    def __init__(
        self,
        feature_saver,
        binning_specs,
        to_drop_cols,
        custom_fn,
        layer_cols,
        estimators,
        proba,
        propagate,
        all_cols_used_to_fit,
        layer_preprocessors,
    ):
        # preprocessing
        self.FEATURE_SAVER = feature_saver
        self.BINNING_SPECS = binning_specs
        self.TO_DROP_COLS = to_drop_cols
        self.CUSTOM_FN = custom_fn
        # layer attributes
        self.LAYER_COLS = layer_cols
        self.ESTIMATORS = estimators
        self.PROBA = proba
        self.PROPAGATE = propagate
        # for internal using
        self._ALL_COLS_USED_TO_FIT = all_cols_used_to_fit
        self._LAYER_PREPROCESSORS = layer_preprocessors

    # Inference properties
    @property
    def MODEL_CONFIG(self):
        self._MODEL_CONFIG = {
            "preprocessors": self.PREPROCESSES,
            "super_learner": self.SUPER_LEARNER,
        }
        return self._MODEL_CONFIG

    @property
    def PREPROCESSES(self):
        self._PREPROCESSES = {
            "feature_saver": self.FEATURE_SAVER,
            "binning_category": self.BINNING_SPECS,
            "feature_dropper": self.TO_DROP_COLS,
        }
        self._PREPROCESSES.update(self.CUSTOM_FN)
        return list(self._PREPROCESSES.items())

    @property
    def LAYER_PREPROCESSORS(self):
        return self._LAYER_PREPROCESSORS

    @property
    def SUPER_LEARNER(self):
        self._SUPER_LEARNER = {}
        for layer, c in self.LAYER_COLS.items():
            self._SUPER_LEARNER[layer] = {
                "columns": c,
                "estimator": self.ESTIMATORS[layer],
                "proba": self.PROBA[layer],
                "preprocessors": self.LAYER_PREPROCESSORS[layer],
            }
            if layer != "meta":
                self._SUPER_LEARNER[layer]["propagate"] = self.PROPAGATE[layer]
        return self._SUPER_LEARNER

    @property
    def ALL_COLS_USED_TO_FIT(self):
        return self._ALL_COLS_USED_TO_FIT

    def GenericProperty(name: str, required_type: tuple = None, pre_defined_funs=None):
        def _getter(self):
            if pre_defined_funs:
                return pre_defined_funs(getattr(self, f"_{name}"))
            return getattr(self, f"_{name}")

        def _setter(self, value):
            if required_type and not isinstance(value, required_type):
                raise TypeError(f"Expect to set {name} with {required_type}, but got {type(value)}.")
            setattr(self, f"_{name}", value)

        return property(_getter, _setter)

    # preprocessors
    FEATURE_SAVER = GenericProperty("FEATURE_SAVER", list, make_feature_saver)
    BINNING_SPECS = GenericProperty("BINNING_SPECS", list, make_binning_category)
    TO_DROP_COLS = GenericProperty(
        "TO_DROP_COLS",
        list,
        partial(make_feature_filter, copy=True, like=None, regex=None, exclude=True),
    )
    CUSTOM_FN = GenericProperty("CUSTOM_FN", dict)
    # SUBSCORE_SAVER = GenericProperty('SUBSCORE_SAVER', (list), make_feature_saver)

    # layer attributes
    LAYER_COLS = GenericProperty("LAYER_COLS", dict)
    ESTIMATORS = GenericProperty("ESTIMATORS", dict)
    PROBA = GenericProperty("PROBA", dict)
    PROPAGATE = GenericProperty("PROPAGATE", dict)

    @classmethod
    def from_pipeline(cls, pipeline):
        # make preprocssor
        preprocessor = Preprocessor(pipeline[:-1].steps)
        # make pipeliner
        super_learner = {}
        for layer in pipeline[-1].layers_:
            layer_name = layer.estimators[0][0]
            if layer_name == "proba":
                layer_name = "meta"
            super_learner[layer_name] = {}
            super_learner[layer_name]["columns"] = layer.preprocessors[-1].columns_
            super_learner[layer_name]["estimator"] = layer.estimators[0][1]
            super_learner[layer_name]["proba"] = layer.proba
            super_learner[layer_name]["preprocessors"] = layer.preprocessors.steps
            super_learner[layer_name]["propagate"] = layer.propagate
        super_learner = Superlearner(super_learner)

        return cls(
            preprocessor.feature_saver,
            preprocessor.binning_specs,
            preprocessor.to_drop_cols,
            preprocessor.custom_fn,
            super_learner.layer_columns,
            super_learner.estimators,
            super_learner.proba,
            super_learner.propagate,
            super_learner.all_cols_used_to_fit,
            super_learner.layer_preprocessors,
        )

    @classmethod
    def from_dict(cls, model_config):
        # make preprocessor
        preprocessor = Preprocessor(model_config.get("preprocessors"))
        # make pipeliner
        super_learner = Superlearner(model_config["super_learner"])

        return cls(
            preprocessor.feature_saver,
            preprocessor.binning_specs,
            preprocessor.to_drop_cols,
            preprocessor.custom_fn,
            super_learner.layer_columns,
            super_learner.estimators,
            super_learner.proba,
            super_learner.propagate,
            super_learner.all_cols_used_to_fit,
            super_learner.layer_preprocessors,
        )
