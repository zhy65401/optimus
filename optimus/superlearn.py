from bumblebee.checkpoint import Checkpoint
from bumblebee.clip import Clipper
from bumblebee.correlation import Correlation
from bumblebee.encoder import WoeEncoder
from bumblebee.impute import SimpleImputer
from bumblebee.project import Filter, LogOdds
from bumblebee.scale import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from .utils import (
    make_checkpoint,
    make_clipper,
    make_correlation,
    make_feature_filter,
    make_imputer,
    make_standard_scaler,
    make_to_logodds,
    make_woe_encoder,
)


class LayerPreprocessor:
    def __init__(
        self,
        layer_name,
        filter,
        logodds,
        woe_cols,
        clipper,
        imputer,
        scaler,
        correlation,
        checkpoint,
    ):
        self._name = layer_name

        self.FILTER = filter
        self.LOGODDS = logodds
        self.WOE_COLS = woe_cols
        self.CLIPPER = clipper
        self.IMPUTER = imputer
        self.SCALER = scaler
        self.CORRELATION = correlation
        self.CHECKPOINT = checkpoint

    @property
    def LAYER_PREPROCESSOR(self):
        steps = [
            (f"{self._name}_filter", self.FILTER),
            (f"{self._name}_logodds", self.LOGODDS),
            (f"{self._name}_woe_encoder", self.WOE_COLS),
            (f"{self._name}_clipper", self.CLIPPER),
            (f"{self._name}_imputer", self.IMPUTER),
            (f"{self._name}_scaler", self.SCALER),
            (f"{self._name}_correlation", self.CORRELATION),
            (f"{self._name}_checkpoint", self.CHECKPOINT),
        ]

        self._LAYER_PREPROCESSOR = [s for s in steps if s[1]]
        return self._LAYER_PREPROCESSOR

    @classmethod
    def from_pipeline(cls, layer_name, layer_columns, layer_preprocessors):
        filter = layer_columns
        logodds = None
        woe_cols = None
        clipper = None
        imputer = None
        scaler = None
        correlation = None
        checkpoint = None

        for _, step in layer_preprocessors:
            if isinstance(step, Checkpoint):
                checkpoint = step
            elif isinstance(step, Clipper):
                clipper = step
            elif isinstance(step, Correlation):
                correlation = step
            elif isinstance(step, Filter):
                filter = step
            elif isinstance(step, LogOdds):
                logodds = step
            elif isinstance(step, SimpleImputer):
                imputer = step
            elif isinstance(step, StandardScaler):
                scaler = step
            elif isinstance(step, WoeEncoder):
                woe_cols = step
            else:
                raise TypeError(f"Unrecognized preprocess step type in layer {layer_name}: {step}.")

        return cls(
            layer_name,
            filter,
            logodds,
            woe_cols,
            clipper,
            imputer,
            scaler,
            correlation,
            checkpoint,
        )

    @classmethod
    def from_layer_preprocessor(cls, layer_name, layer_preprocessor):

        return cls(
            layer_name,
            layer_preprocessor.FILTER,
            layer_preprocessor.LOGODDS,
            layer_preprocessor.WOE_COLS,
            layer_preprocessor.CLIPPER,
            layer_preprocessor.IMPUTER,
            layer_preprocessor.SCALER,
            layer_preprocessor.CORRELATION,
            layer_preprocessor.CHECKPOINT,
        )

    def CustomizedProperty(name: str, pre_defined_funs=None, **kwargs):
        def _getter(self):
            if pre_defined_funs:
                return pre_defined_funs(getattr(self, f"_{name}"), **kwargs)
            return getattr(self, f"_{name}")

        def _setter(self, value):
            setattr(self, f"_{name}", value)

        return property(_getter, _setter)

    FILTER = CustomizedProperty("FILTER", make_feature_filter)
    LOGODDS = CustomizedProperty("LOGODDS", make_to_logodds)
    WOE_COLS = CustomizedProperty("WOE_COLS", make_woe_encoder)
    CLIPPER = CustomizedProperty("CLIPPER", make_clipper)
    IMPUTER = CustomizedProperty("IMPUTER")
    SCALER = CustomizedProperty("SCALER")
    CORRELATION = CustomizedProperty("CORRELATION")
    CHECKPOINT = CustomizedProperty("CHECKPOINT")


class Superlearner:
    def __init__(self, super_learner):
        self.super_learner = super_learner

        self.estimators = dict()
        self.proba = dict()
        self.propagate = dict()
        self.layer_columns = dict()
        self.layer_preprocessors = dict()
        self.all_cols_used_to_fit = []

        self._generate_superlearner()

    def _generate_superlearner(self):
        subscore_cols = set()
        self.layer_preprocessors = {}
        for layer_name, layer in self.super_learner.items():
            self.layer_columns[layer_name] = {}

            # load columns
            self.layer_columns[layer_name] = layer["columns"]
            if layer_name != "meta":
                subscore_cols.add(layer_name)
            self.all_cols_used_to_fit.extend([c for c in layer["columns"] if c not in subscore_cols])

            # load estimator
            estimator = layer.get("estimator", "XGB")
            if estimator == "LR":
                estimator = LogisticRegression(
                    C=0.1,
                    penalty="l2",
                    solver="saga",
                    max_iter=10_000,
                    random_state=42,
                )
            elif estimator == "RF":
                estimator = RandomForestClassifier(
                    n_estimators=100,
                    criterion="gini",
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    min_weight_fraction_leaf=0.0,
                    max_features="sqrt",
                    max_leaf_nodes=None,
                    min_impurity_decrease=0.0,
                    bootstrap=True,
                    random_state=42,
                )
            elif estimator == "XGB":
                estimator = XGBClassifier(
                    scale_pos_weight=12,
                    min_child_weight=5,
                    n_estimators=200,
                    learning_rate=0.01,
                    gamma=5,
                    colsample_bytree=0.2,
                    reg_lambda=3,
                    eval_metric="auc",
                    objective="binary:logistic",
                    max_depth=3,
                    random_state=666,
                    use_label_encoder=False,
                )
            elif estimator == "LGBM":
                estimator = LGBMClassifier(
                    boosting_type="gbdt",
                    num_leaves=31,
                    max_depth=-1,
                    learning_rate=0.1,
                    n_estimators=100,
                    subsample_for_bin=200000,
                    objective=None,
                    class_weight=None,
                    min_split_gain=0.0,
                    min_child_weight=0.001,
                    min_child_samples=20,
                    subsample=1.0,
                    subsample_freq=0,
                    colsample_bytree=1.0,
                    reg_alpha=0.0,
                    reg_lambda=0.0,
                    random_state=42,
                    importance_type="split",
                )
            self.estimators[layer_name] = estimator

            # load proba
            self.proba[layer_name] = layer.get("proba", 1)

            # load propagate
            if layer_name != "meta":
                self.propagate[layer_name] = Filter(regex=r"^(?!({}))".format("|".join(layer["columns"])))

            # load preprocessors

            if layer.get("preprocessors") is not None:
                if isinstance(layer.get("preprocessors"), LayerPreprocessor):
                    self.layer_preprocessors[layer_name] = LayerPreprocessor.from_layer_preprocessor(
                        layer_name,
                        layer.get("preprocessors"),
                    )
                else:
                    self.layer_preprocessors[layer_name] = LayerPreprocessor.from_pipeline(
                        layer_name,
                        layer["columns"],
                        layer.get("preprocessors"),
                    )

            else:
                woe_columns = list(set(layer["columns"]) - subscore_cols)
                logodds = subscore_cols if layer_name == "meta" else None
                clipper = None
                imputer = make_imputer()
                scaler = make_standard_scaler()
                correlation = make_correlation()
                checkpoint = make_checkpoint()
                self.layer_preprocessors[layer_name] = LayerPreprocessor(
                    layer_name,
                    layer["columns"],
                    logodds,
                    woe_columns,
                    clipper,
                    imputer,
                    scaler,
                    correlation,
                    checkpoint,
                )
