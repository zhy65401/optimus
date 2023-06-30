import gzip
import pickle

import joblib
import pandas as pd
from bumblebee.correlation import Correlation
from bumblebee.encoder import WoeEncoder
from bumblebee.performance import Performance
from bumblebee.report import Reporter
from bumblebee.scorecard import ScoreCard
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from .score import Scorer


class Resourcer:
    def __init__(self, df, pipeliner, config, train_config):
        self.resource = None
        self.pipeliner = pipeliner

        self._df = df
        self._all_cols_used_to_fit = config.ALL_COLS_USED_TO_FIT.copy()
        self._label_col = train_config["label_col"]
        self._split_fn = train_config["split_fn"]
        self._tst_q = train_config["tst_q"]
        self._oot_q = train_config["oot_q"]
        self._user_col = train_config["user_col"]
        self._timestamp_col = train_config["timestamp_col"]
        self._split_basis = train_config["split_base"]
        self._deciles = train_config["deciles"]
        self._kwargs = train_config["kwargs"]

        # check columns exist
        self._check_df_cols()
        self.X, self.y, self.e = self._split_datasets()

        self._data_summary = self._generate_data_summary(self.y, self.e, self._timestamp_col)

    @staticmethod
    def default_split_fn(df, time_column=None, user_column=None, base="time", tst_q=0.2, oot_q=0.01):
        """Performs a split on df based on a time-based column or user group.

        Args:
            df (pd.DataFrame): data to perform train-test split on
            time_column (str): the time column name.
            user_column (str): the user_id column name.
            base (str): the base column for separation.
            This argument will be masked if time_column is not None.
            tst_q (float): the percentage of test set.
            oot_q (float): the percentage of out of sample evaluation set (for user base separation).

        Returns:
            tuple of bool pd.Series: indices of the train and test set from df
        """
        assert time_column, "`time_column` should not be None."
        assert time_column in df.columns, f"{time_column!r} is not found in columns of input dataframe(s)!"

        if base == "time":

            cut_point = df[time_column].quantile(1 - tst_q)
            is_fit = df[time_column] <= cut_point
            is_tst = df[time_column] > cut_point

            return {"fit": is_fit, "tst": is_tst}

        elif base == "user":
            assert user_column, "Both `time_column` and `user_column` should not be None."
            assert (
                user_column in df.columns
            ), f"{user_column!r} is not found in columns of input dataframe(s)!"
            cut_point = df[time_column].quantile(1 - oot_q)
            is_oot = df[time_column] > cut_point
            fit_user, tst_user = train_test_split(
                df[~is_oot][user_column].unique(),
                train_size=1 - tst_q,
                random_state=42,
                shuffle=True,
            )
            is_fit = df[user_column].isin(fit_user) & ~is_oot
            is_tst = df[user_column].isin(tst_user) & ~is_oot

            return {"fit": is_fit, "tst": is_tst, "oot": is_oot}

        else:
            raise ValueError(
                f"Separation base should be one of (`time`, `user`), but retrieve {base} instead.",
            )

    @classmethod
    def _get_model_name(cls, model):
        if isinstance(model, LogisticRegression):
            return "LR"
        if isinstance(model, XGBClassifier):
            return "XGB"
        if isinstance(model, LGBMClassifier):
            return "LGBM"
        if isinstance(model, RandomForestClassifier):
            return "RF"
        raise f"Unable to recognize model type of {model}."

    @classmethod
    def _generate_data_summary(cls, y_dict: dict, e_dict: dict, timestamp_col: str) -> pd.DataFrame:
        if y_dict.keys() != e_dict.keys():
            raise ValueError(
                f"Expected y_dict and e_dict to have the same keys; "
                f"Found {set(y_dict.keys())} and {set(e_dict.keys())} instead.",
            )
        splits = y_dict.keys()

        summary = pd.DataFrame.from_records(
            [
                {
                    "# observations": len(y_dict[split]),
                    "DR": y_dict[split].mean(),
                    "Start order time": e_dict[split][timestamp_col]
                    # FIXME: some business lines use timestamp64[ns]. Need to determine whether need to fix or not.
                    .astype("datetime64[s]").min().date(),
                    "End order time": e_dict[split][timestamp_col].astype("datetime64[s]").max().date(),
                }
                for split in splits
            ],
            index=[split.upper() for split in splits],
        )
        return summary

    def _check_df_cols(self):
        # check whether all columns in the model are provided
        assert self._label_col in self._df, f"{self._label_col} is not found in the df."
        assert self._df[self._label_col].nunique() <= 2, "Label is not binary!"
        assert all(
            c in self._df.columns for c in self._all_cols_used_to_fit
        ), f"{list(set(self._all_cols_used_to_fit) - set(self._df.columns))} are not in the df."

    def _split_datasets(self):
        X, y, e = {}, {}, {}

        if not self._split_fn:
            X["all"] = self._df[self._all_cols_used_to_fit]
            y["all"] = self._df[self._label_col]
            e["all"] = self._df.drop(self._all_cols_used_to_fit + [self._label_col], axis=1)
        elif self._split_fn == "default":
            self._splited_sets = self.default_split_fn(
                self._df,
                self._timestamp_col,
                self._user_col,
                self._split_basis,
                self._tst_q,
                self._oot_q,
            )
        else:
            self._splited_sets = self._split_fn(self._df, **self._kwargs)

        for k, v in self._splited_sets.items():
            X[k] = self._df[v][self._all_cols_used_to_fit]
            y[k] = self._df[v][self._label_col]
            e[k] = self._df[v].drop(self._all_cols_used_to_fit + [self._label_col], axis=1)
        return X, y, e

    def _construct_resources(self, pipeliner):
        self._perf_dict = {}
        self._scorecard_dict = {}
        self._scorer = Scorer()

        for k in self._splited_sets:
            pred = pipeliner.predict(self.X[k])
            self._perf_dict[k] = Performance(self.y[k], pred["proba"], pred["score"], deciles=self._deciles)

        # score card
        for layer in pipeliner.pipeline[-1].layers_:
            layer_name = layer.estimators[0][0]
            if layer_name == "proba":
                layer_name = "meta"
            self._scorecard_dict[layer_name] = ScoreCard(
                model=layer.estimators[0][1],
                scorer=self._scorer,
                checkpoint=layer.preprocessors[layer_name + "_checkpoint"],
                clipper=layer.preprocessors[layer_name + "_clipper"],
                scaler=layer.preprocessors[layer_name + "_scaler"],
                model_type=self._get_model_name(layer.estimators[0][1]),
            )

        self.resource = dict(
            model=pipeliner.pipeline,
            constant=dict(
                performance=self._perf_dict,
                score_card=self._scorecard_dict,
                summary=self._data_summary,
            ),
        )

        return self.resource


class ReportGenerator:
    def __init__(self):
        self.reporter = Reporter(render_toc=False)

    def generate_model_report(self, resource: dict, **kwargs):
        """
        Generate a whole model report for given model object.

        Args:
            resource: The resource dict with model and constant need to report
        """

        model_name = kwargs.get("model_name", "TBA")
        model_version = kwargs.get("model_name", "TBA")
        excluded_section_list = kwargs.get("excluded_section_list", [])
        pipe = resource.get("model")
        cons = resource.get("constant")
        assert pipe, "model is not found from the resource."
        assert cons, "consant is not found from the resource."

        section_list = set(
            [
                "Summary",
                "Performance",
                "Distribution",
                "Deciled Summary",
                "WOE",
                "Score Card",
                "Correlation",
            ],
        )

        if excluded_section_list:
            section_list -= set(excluded_section_list)

        layers = [(layer.estimators[0][0], layer) for layer in pipe[-1].layers_[:-1]] + [
            ("meta", pipe[-1].layers_[-1]),
        ]

        self.reporter = self.reporter.overview(name=model_name, version=model_version, level=1)
        if "Summary" in section_list:
            self.reporter = self.reporter.data_summary(cons["summary"], title="Dataset")
        if "Performance" in section_list:
            report_perf = cons["performance"]
            self.reporter = self.reporter.section("Performance", level=2)
            for name, perf in report_perf.items():
                self.reporter = self.reporter.performance(perf, title=name, level=3, ref_id="perf-" + name)
        if "Distribution" in section_list:
            self.reporter = self.reporter.section("Distribution", level=2)
            for name, perf in report_perf.items():
                if name == "fit":
                    continue
                self.reporter = self.reporter.distribution(
                    report_perf["fit"],
                    perf,
                    ref_label="fit",
                    act_label=name,
                    title=f"{name} Set v.s fit Set",
                    level=3,
                    ref_id=f"{name}-dist",
                )
        if "Deciled Summary" in section_list:
            self.reporter = self.reporter.section("Deciled Summary", level=2)
            for name, perf in report_perf.items():
                self.reporter = self.reporter.deciled_summary(perf, title=name, level=3, ref_id="dec-" + name)
        if "WOE" in section_list:
            self.reporter = self.reporter.section("WOE", level=2)
            for name, layer in layers:
                for s in layer.preprocessors.steps:
                    if isinstance(s[1], WoeEncoder):
                        self.reporter = self.reporter.woe(
                            s[1],
                            title=name,
                            level=3,
                            ref_id=name + "_woe",
                        )
        if "Score Card" in section_list:
            self.reporter = self.reporter.section("Score Card", level=2)
            for score_card_title, sc in cons["score_card"].items():
                self.reporter = self.reporter.scorecard(
                    sc,
                    title=score_card_title,
                    level=3,
                    ref_id=score_card_title + "_sc",
                    show_scorer_config=True,
                )
        if "Correlation" in section_list:
            self.reporter = self.reporter.section("Correlation", level=2)
            for name, layer in layers:
                for s in layer.preprocessors.steps:
                    if isinstance(s[1], Correlation):
                        self.reporter = self.reporter.correlation(
                            s[1],
                            vif_top=None,
                            title=name,
                            level=3,
                            ref_id=name + "_corr",
                        )
        self.reporter.display()

    def generate_model_comparison_report(
        self,
        target_resources: dict,
        data_subsets: list = None,
        **kwargs,
    ):
        """
        Generate and print the HTML display of the comparison report between multiple pipelines.

        NOTE: You can choose to compare the model from either features/data dimension. The report will alter accordingly.

        Args:
            target_resources: A dictionary of pipeine objects need to compare as target.
            data_subsets: A list names of data subsets.
        """
        model_name = kwargs.get("model_name", "Current Model")
        excluded_section_list = kwargs.get("excluded_section_list", [])
        pipe = dict()
        cons = dict()
        target_model_names = []
        for name, resource in target_resources.items():
            target_model_names += name
            assert resource.get("model") and resource.get(
                "constant",
            ), """some models' resources are not complete. Unable to find pipeline or constant key."""
            pipe[name] = resource.get("model")
            cons[name] = resource.get("constant")

        section_list = set(
            [
                "Summary",
                "Performance",
                "Cutoff",
            ],
        )

        if excluded_section_list:
            section_list -= set(excluded_section_list)

        self.reporter = self.reporter.overview(
            name=f"Performance comparison report between {model_name} and {', '.join(target_model_names)}",
            version="--",
            title="Model Comparison Summary",
            level=1,
        )
        if "Summary" in section_list:
            self.reporter = self.reporter.section(title="Dataset", level=2)
            if not data_subsets:
                data_subsets = ["fit"]
            for subset in data_subsets:
                self.reporter = self.reporter.section(title=subset, level=3)
                res = []
                for model_name, constant in cons.items():
                    try:
                        snippet = pd.DataFrame(constant["summary"].loc[subset.upper()]).transpose()
                        snippet.index = [model_name]
                        res.append(snippet)
                    except KeyError:
                        raise KeyError(f"{subset} subset is not in the model {model_name}.")
                self.reporter.components.append(
                    pd.concat(res).style.format(
                        {
                            "DR": "{:.2%}",
                        },
                    ),
                )

        if "Performance" in section_list:
            self.reporter = self.reporter.section(title="Performance", level=2)
            statistics = ["ROC_AUC", "PR_AUC", "GINI", "KS"]
            if not data_subsets:
                data_subsets = ["fit"]
            for subset in data_subsets:
                self.reporter = self.reporter.section(title=subset, level=3)
                res = []
                for model_name, constant in cons.items():
                    assert (
                        subset in constant["performance"]
                    ), f"{subset} subset performance is not in constant of model {model_name}."
                    report_perf = constant["performance"][subset]
                    res.append(
                        pd.DataFrame(
                            [[report_perf.roc_auc, report_perf.pr_auc, report_perf.gini, report_perf.ks]],
                            index=[model_name],
                            columns=statistics,
                        ),
                    )
                self.reporter.components.append(
                    pd.concat(res).style.format(
                        {
                            "ROC_AUC": "{:.4%}",
                            "PR_AUC": "{:.4%}",
                            "GINI": "{:.4%}",
                            "KS": "{:.4%}",
                        },
                    ),
                )

        if "Cutoff" in section_list:
            self.reporter = self.reporter.section(title="Cutoff", level=2)
            if not data_subsets:
                data_subsets = ["fit"]
            for subset in data_subsets:
                self.reporter = self.reporter.section(title=subset, level=3)
                statistics = ["min_score", "max_score", "total_percentage", "bad_rate", "bad_rate_above"]
                res = []
                for model_name, constant in cons.items():
                    assert (
                        subset in constant["performance"]
                    ), f"{subset} subset performance is not in constant of model {model_name}."
                    report_dec_summary = constant["performance"][subset].deciled_summary
                    min_score = report_dec_summary.iloc[-1].min_score
                    max_score = report_dec_summary.iloc[-1].max_score
                    total_percentage = report_dec_summary.iloc[-1].total_percentage
                    bad_rate = report_dec_summary.iloc[-1].bad_rate
                    bad_rate_above = report_dec_summary.iloc[-2].bad_rate_above
                    res.append(
                        pd.DataFrame(
                            [[min_score, max_score, total_percentage, bad_rate, bad_rate_above]],
                            index=[model_name],
                            columns=statistics,
                        ),
                    )
                self.reporter.components.append(
                    pd.concat(res).style.format(
                        {
                            "total_percentage": "{:.1%}",
                            "bad_rate": "{:.2%}",
                            "bad_rate_above": "{:.2%}",
                        },
                    ),
                )
        self.reporter.display()


class GzipResource:
    @staticmethod
    def load(path):
        with gzip.GzipFile(path, "rb") as f:
            return pickle.load(f)

    @staticmethod
    def dump(path, data):
        with gzip.GzipFile(path, "wb") as f:
            pickle.dump(data, f)


class JoblibResource:
    @staticmethod
    def load(path):
        return joblib.load(path)

    @staticmethod
    def dump(path, data):
        return joblib.dump(data, path, compress=3)
