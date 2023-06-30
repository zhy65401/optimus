from copy import deepcopy

import pandas as pd
from bumblebee.ensemble import SuperLearner
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .score import Scorer


class Pipeliner:
    def __init__(self, config):
        self.model_config = deepcopy(config.MODEL_CONFIG)
        self.pipeline = None
        self._scorer = Scorer()

        self._compile_config()

    def _compile_config(self):
        # compile preprocessors
        steps = self.model_config["preprocessors"].copy()
        # compile supler learner
        learner = SuperLearner(folds=5, random_state=42)
        for layer_name, sub_model in self.model_config["super_learner"].items():
            if layer_name != "meta":
                learner.add(
                    estimators=[(layer_name, sub_model["estimator"])],
                    proba=sub_model["proba"],
                    preprocessors=Pipeline(sub_model["preprocessors"].LAYER_PREPROCESSOR),
                    propagate=sub_model["propagate"],
                )
            else:
                learner.add_meta(
                    estimator=("proba", sub_model["estimator"]),
                    proba=sub_model["proba"],
                    preprocessor=Pipeline(sub_model["preprocessors"].LAYER_PREPROCESSOR),
                )
        steps.append(("super_learner", learner))
        self.pipeline = Pipeline(steps)

    def fit(self, X, y):
        self.pipeline.fit(X, y)

        return self

    def predict(self, X, y=None):
        check_is_fitted(self.pipeline[-1])
        pred_res = self.pipeline.predict(X)[["proba"]]
        pred_res["score"] = self._scorer.to_score(pred_res["proba"])

        return pred_res

    def transform(self, X):
        check_is_fitted(self.pipeline[-1])

        return pd.concat([X, self.predict(X)], axis=1)

    def fit_predict(self, X, y):

        return self.pipeline.fit(X, y).transform(X)
