from copy import deepcopy

import pandas as pd
from IPython.core.display import display
from sklearn import set_config

from .config import Optimusle
from .eda import EDAReport
from .pipeline import Pipeliner
from .resource import GzipResource, JoblibResource, ReportGenerator, Resourcer

__version__ = "0.2.0"
set_config(display="diagram")


class Optimus:
    def __init__(self, model_config: dict, df: pd.DataFrame = None, **kwargs):
        """
        A initialization of an Optimus instance.
        There are two ways to generate this object:
            a. Genearte an object with existing pipeline. (Use Optimus.from_pipeline)
            b. Generate a new Optimus object with given settings.

        Args:
            model_config: A dictionary that contains settings and columns of model.
                For example,
                model_config = {
                    (optional) 'preprocessors': [
                        ('feature_saver', FeatureSaver()),
                        ('feature_dropper', Filter(exclude=True)),
                        ('binning_category', BinningCategory()),
                        (... any other custom function)
                    ]
                    'super_learner': {
                        'user_info_sub_score': {
                            "columns": ('^user_info') OR (['user_info_age', 'user_info_gender', 'user_info_education']),
                            "estimator": 'RF',
                            (opt) "proba": (float)
                            (opt) "preprocessors": (list)
                            (opt) "propagate": (regex str)
                        },
                        'device_sub_score': {
                            "columns": ('^user_info') OR (['device_is_risky_brand']),
                            "estimator": 'XGB'
                        },
                        'meta': {
                            "columns": ['user_info_sub_score', 'device_sub_score', 'seon_email_sub_score', 'guardian_mobile_sub_score'],
                            "estimator": 'LR'
                        }
                    }
                }
            df: The dataframe used to fit and transform
            **kwargs: other arguments you want to pre-define for train_config

        *NOTE:
        [LAYER] can be 'LR', 'XGB', 'RF', 'LGBM' or any self-defined Layer object.

        Return :
            optimus: The Optimus instance
        """

        assert "super_learner" in model_config, "Key `super_learner` is compulsory in the config"
        self.config = Optimusle.from_dict(model_config)
        self.pipeliner = Pipeliner(self.config)
        self.resourcer = None

        self.set_train_config(**kwargs)
        self._df = df
        self._dataframe_summary()

    @classmethod
    def generate_report(
        cls,
        resources,
        resource_names: list = None,
        compare: bool = False,
        data_subsets: list = None,
        **kwargs,
    ):
        """
        Args:
            resources: a list or single resource dictionary for reporting or comparison. NOTE: both `model` and `constant` are neccessary in the resource.
            resource_names: a list of names of resources.
            compare: Whether do the comparison between resources.
            data_subsets: For comparison using, a list of subset need to compare. Must be the common subsets among all resources.
            kwargs:
                excluded_section_list: You can choose to exclude a subset of section from below list included in the report.
                [
                    "Summary",
                    "Performance",
                    "Distribution",
                    "Deciled Summary",
                    "WOE",
                    "Score Card",
                    "Correlation",
                    "Cutoff",
                ]
                model_name: displayed model name.
                model_version: displayed model version.
        """
        if compare:
            assert resource_names, "resource_names of each resource must be defined."
            target_resources = dict([(k, v) for k, v in zip(resource_names, resources)])
            ReportGenerator().generate_model_comparison_report(target_resources, data_subsets, **kwargs)
        else:
            if not isinstance(resources, list):
                resources = [resources]
            for resource in list(resources):
                assert "model" in resource, "`model` is not in resource dictionary"
                assert "constant" in resource, "`constant` is not in resource dictionary"
                ReportGenerator().generate_model_report(resource, **kwargs)

    @classmethod
    def from_pipeline(cls, pipeline, df=None, **kwargs):
        """
        Create Optimus object from sklearn pipeline object

        Args:
            pipeline: the sklearn pipeline for create.
            df: pre-loading dataframe.
            **kwargs: other arguments you want to pre-define for train_config.
        """
        # convert pipeline to config
        config = Optimusle.from_pipeline(deepcopy(pipeline))
        return cls(config.MODEL_CONFIG, df, **kwargs)

    @classmethod
    def from_gzipresource(cls, path, df=None, **kwargs):
        # convert gzip resource to config
        pipeline = GzipResource.load(path)
        config = Optimusle.from_pipeline(deepcopy(pipeline))
        return cls(config.MODEL_CONFIG, df, **kwargs)

    @classmethod
    def from_joblibresource(cls, path, df=None, **kwargs):
        # convert joblib resource to config
        pipeline = JoblibResource.load(path)
        config = Optimusle.from_pipeline(deepcopy(pipeline))
        return cls(config.MODEL_CONFIG, df, **kwargs)

    @classmethod
    def from_col_names(
        cls,
        cols=None,
        regex_patterns: dict = None,
        df: pd.DataFrame = None,
        **kwargs,
    ) -> dict:
        """
        Create Optimus object from a raw dataframe by inferring with the logic below:
            1. assign all columns into corresponding layer according to regex_patterns.
            2. assign all columns do not match to any given regex_patterns into meta layer.

        Args:
            cols: a list of df columns name for creating the Optimus object.
            regex_patterns: a dict of (layer_name, regex_pattern) pairs that the columns will be assigned accordingly.
            df: pre-loading dataframe.
            **kwargs: other arguments you want to pre-define for train_config.

        Return:
            optimus: Optimus object
        """
        model_config = {"super_learner": {}}
        assigned = set()
        if regex_patterns:
            for layer, p in regex_patterns.items():
                assert layer != "meta", "`meta` should not be self-defined!"
                matched_cols = cols[pd.Index(cols).str.match(p)].to_list()
                model_config["super_learner"][f"{layer}_sub_score"] = {
                    "columns": matched_cols,
                }
                assigned = assigned.union(set(matched_cols))
            model_config["super_learner"]["meta"] = {
                "columns": [f"{c}_sub_score" for c, _ in regex_patterns.items()],
            }
        else:
            model_config["super_learner"]["meta"] = {"columns": []}
        model_config["super_learner"]["meta"]["columns"].extend(set(cols) - assigned)

        return cls(model_config, df, **kwargs)

    @property
    def TRAIN_CONFIG(self):
        return self._TRAIN_CONFIG

    def set_train_config(
        self,
        label_col: str = "order_first_days_past_due",
        timestamp_col: str = "order_create_time",
        user_col: str = "user_id",
        split_base: str = "time",
        split_fn="default",
        tst_q=0.2,
        oot_q=0.01,
        deciles: list = None,
        **kwargs,
    ):
        """
        Config training related setting.

        Args:
            label_col: the label column name.
            timestamp_col: the timestamp column name.
            user_col: the user_id column name.
            split_base: the basis of splitting df for default fn. Can be `time` or `user`.
            split_fn: custom split function. The first argument should be df.
            tst_q: percentatge of test set.
            oot_q: percentatge of out of sample evaluation set (for user base separation).
            deciles: list of float for deciles.
            **kwargs: other arguments you want to insert in the customized splitting function.

        *NOTE: The default split_fn with split_base=='time' will only split df into `fit` and `tst` set and will split df into `fit`, `tst` and `oot` with split_base=='user'
        If using self-defined fn, the return of the fn must be a dictionary of Series of boolean.
        For example, {
            'fit': pd.Series([True, True, False, False, False]),
            'tst': pd.Series([False, False, True, True, False]),
            (other_set_name): pd.Series([False, False, False, False, True])
        }

        **NOTE: if this pipeline need to be trained, 'fit' must be in the return dictionary.
        """
        if deciles is None:
            deciles = [0, 0.05, 0.1, 0.2, 0.25, 0.5, 0.75, 0.8, 0.9, 0.95, 1]
        assert (split_fn == "default") or (
            split_fn.__code__.co_varnames[0] == "df"
        ), "`df` must be as the first arg of self-defined split func."
        train_config = {
            "label_col": label_col,
            "timestamp_col": timestamp_col,
            "user_col": user_col,
            "split_base": split_base,
            "split_fn": split_fn,
            "tst_q": tst_q,
            "oot_q": oot_q,
            "deciles": deciles,
            "kwargs": kwargs,
        }
        print("TRAINING CONFIG:")
        print(f"label_col: {label_col}")
        print(f"timestamp_col: {timestamp_col}")
        print(f"user_col: {user_col}")
        print(
            f'split_base: {train_config["split_base"] if train_config["split_fn"]=="default" else "not available"}',
        )
        print(f'split_fn: {"default" if train_config["split_fn"]=="default" else "custom"}')
        print(f'tst_q: {train_config["tst_q"] if train_config["split_fn"] == "default" else "--"}')
        print(f'oot_q: {train_config["oot_q"] if train_config["split_fn"] == "default" else "--"}')
        print(f'deciles: {train_config["deciles"]}')
        if kwargs:
            print(f"kwargs: {kwargs}")
        print("========================================================================")

        self._TRAIN_CONFIG = train_config

    def _dataframe_summary(self):
        print("DF SUMMARY:")
        if self._df is None:
            print("NOT LOADED")
            return
        if self.TRAIN_CONFIG["label_col"] not in self._df:
            print(f'Label column {self.TRAIN_CONFIG["label_col"]} not found!')
            return
        if self.TRAIN_CONFIG["timestamp_col"] not in self._df:
            print(f'Timestamp column {self.TRAIN_CONFIG["timestamp_col"]} not found!')
            return
        assert (
            self._df[self.TRAIN_CONFIG["label_col"]].nunique() <= 2
        ), f"Label {self.TRAIN_CONFIG['label_col']} is not binary!"
        print(f"SHAPE: {self._df.shape}")
        print(
            f"""Time period: {(self._df[self.TRAIN_CONFIG["timestamp_col"]].min(), self._df[self.TRAIN_CONFIG["timestamp_col"]].max()) if self.TRAIN_CONFIG["timestamp_col"] in self._df else "Warning: Timestamp col not found!"}""",
        )

    def generate_eda_report(
        self,
        df: pd.DataFrame = None,
        binning_specs: list = None,
        patterns: list = None,
        exclude_patterns: list = None,
        freq_mode="week",
        ignore_na=False,
        plot_time_trajectory=False,
        date_bound: tuple = None,
        figsize: tuple = (6, 6),
    ):
        """
        Generate EDA report.

        Args:
            df: the dataframe object that need to generate from.
            binning_specs: a list of tuples to define binnings, eg. [('xxx', [1,2,3,float('inf')]), ('xxx2', False)].
                           If leave it as None, auto-binning will apply.
            patterns: a list of regex patterns to indicate columns you want to include in the report. Union to columns defined in `binning_specs`
            exclude_patterns: a list of regex patterns to indicate columns you want to exclude from the report. Will not override columns defined in `binning_specs`
            freq_mode: the time axis of time trajectory plot. It can be one of {'month', 'week', 'day'}.
            ignore_na: the flag to indicate whether exclude NaN or None value in the dataframe.
            plot_time_trajectory: the flag to indicate whether plot time trajectory plot.
            date_bound: a pair of datetime in tuple to indicate the time period yoiu want to focus.
            figsize: the tuple to config the size of the plot.
        """
        if df is not None:
            self._df = df
        assert self._df is not None, "No dataframe is defined!"
        # Create a default binning_specs
        matched_cols_binning = dict(binning_specs or self.config.BINNING_SPECS.specs or [])
        patterns = patterns or []
        exclude_patterns = exclude_patterns or []

        excluded_cols = []
        # match all exclude_patterns
        for p in exclude_patterns:
            excluded_cols.extend(self._df.columns[self._df.columns.str.match(p)].to_list())
        # match all patterns beside those in exlude_patterns
        for p in patterns:
            for col in self._df.columns[self._df.columns.str.match(p)]:
                if col not in matched_cols_binning and col not in excluded_cols:
                    matched_cols_binning[col] = "auto"
        # otherwise, auto binning for all columns
        if not matched_cols_binning:
            matched_cols_binning = dict([(c, "auto") for c in self.config.ALL_COLS_USED_TO_FIT])
        # summarize all columns need auto-binning
        self._auto_binning_cols = [k for k, v in matched_cols_binning.items() if v == "auto"]
        print(f"[INFO] {len(self._auto_binning_cols)} columns will be auto binned")

        self.eda = EDAReport(
            self._df,
            matched_cols_binning,
            self._auto_binning_cols,
            freq_mode,
            self.TRAIN_CONFIG["label_col"],
            self.TRAIN_CONFIG["timestamp_col"],
            ignore_na,
            plot_time_trajectory,
            date_bound,
            figsize,
        )
        self.eda.display()
        self.woe_df = self.eda._woe_df

    def fit(self, df: pd.DataFrame = None, **kwargs):
        """
        Fit the pipeline with given setting.

        Args:
            df: The dataframe has to include all columns in the model_config/pipeline.
        Returns:
            self
        """
        print("Fitting...")
        if df is not None:
            self._df = df
        assert self._df is not None, "No dataframe is defined!"

        # Reload to sync everything
        self.config = self.config.from_dict(self.config.MODEL_CONFIG)
        self.pipeliner = Pipeliner(self.config)
        display(self.pipeliner.pipeline)
        self.resourcer = Resourcer(self._df, self.pipeliner, self.config, self.TRAIN_CONFIG)

        X, y = self.resourcer.X, self.resourcer.y
        assert "fit" in X, "Split fn must return X with dict key 'fit' for training."
        self.pipeliner.fit(X["fit"], y["fit"], **kwargs)

        return self

    def transform(self, df: pd.DataFrame = None):
        """
        Transform the df with fitted pipeline.

        Args:
            df: The dataframe has to include all columns in the model_config/pipeline.
        Returns:
            pred_df: The transformed dataframe including all scores.
        """
        print("Transforming...")
        if df is not None:
            self._df = df
        assert self._df is not None, "No dataframe is defined!"
        self.resourcer = Resourcer(self._df, self.pipeliner, self.config, self.TRAIN_CONFIG)

        X, y, e = self.resourcer.X, self.resourcer.y, self.resourcer.e
        pred_df = []
        for xv, yv, ev in zip(X.values(), y.values(), e.values()):
            pred_df.append(self.pipeliner.transform(pd.concat([xv, yv, ev], axis=1)))
        self.pred_df = pd.concat(pred_df).sort_index()
        self.resources = self.resourcer._construct_resources(self.pipeliner)
        print("Done")

        return self.pred_df.sort_index()

    def fit_transform(self, df: pd.DataFrame = None, **kwargs):
        if df is not None:
            self._df = df
        assert self._df is not None, "No dataframe is defined!"

        return self.fit(**kwargs).transform()

    def report(self, excluded_section_list=None, model_name=None, model_version=None):
        """
        Generate performace report for current pipeline.

        Args:
            excluded_section_list: the section excluded from the report.
            model_name, model_version: the overview content.
        """
        if self._df is None:
            raise ValueError("Pipeline has not transferred any df.")
        self.generate_report(
            self.resources,
            excluded_section_list=excluded_section_list,
            model_name=model_name,
            model_version=model_version,
        )
