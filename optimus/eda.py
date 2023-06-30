import altair as alt
import pandas as pd
from bumblebee.eda import EDATransformation
from IPython.display import display
from lutil.analytics import information_value


class EDAReport:
    def __init__(
        self,
        df,
        binning_specs,
        auto_binning_cols,
        freq,
        label_col,
        time_col,
        ignore_na,
        plot_time_trajectory,
        date_bound,
        figsize,
    ):
        self._df = df.copy()
        self._binning_specs = binning_specs
        self._freq = freq
        self._time_col = time_col
        self._label_col = label_col
        self._ignore_na = ignore_na
        self._plot_time_trajectory = plot_time_trajectory
        self._date_bound = date_bound
        self.figsize = figsize

        assert self._label_col in self._df.columns, "Label column is not in the df!"
        self.auto_binning(self._df[auto_binning_cols], self._df[self._label_col])

    def auto_binning(self, X: pd.DataFrame, y: pd.Series) -> dict:
        # columns checking
        assert y.nunique() == 2, "Label is not binary!"
        self._problematic_columns = X.columns[X.nunique() <= 1]
        if len(self._problematic_columns) > 0:
            print(
                f"[WARNING] Columns {list(self._problematic_columns)} have one category! Will be excluded from binning.",
            )
        # auto binning
        edat = EDATransformation().fit(X[X.columns[~X.columns.isin(self._problematic_columns)]], y)

        # retrieve auto-binning bins
        self._auto_binning_cols = edat.bins
        # removed problematic columns
        self._auto_binning_cols = [
            (k, v) for k, v in self._auto_binning_cols.items() if k not in self._problematic_columns
        ]
        # mapping the merged category label and original label
        self._merged_cat_mapping = {}
        self.bin_id_mapping = {}
        for k, v in self._auto_binning_cols:
            mappings = dict([(c, b) for b in v if isinstance(b, str) and "||" in b for c in b.split("||")])
            if mappings:
                self._merged_cat_mapping[k] = mappings
        # update category-merged columns in df
        self._df.replace(self._merged_cat_mapping, inplace=True)

        self._binning_specs.update(self._auto_binning_cols)
        self._woe_df = edat.woe_df

        # retrieve feature type
        self._feature_type = (
            edat.woe_df[["Variable_Name", "Data_Type"]].set_index("Variable_Name").to_dict()["Data_Type"]
        )
        # update auto-binning for categorical features to `False` for plotting
        for k, v in self._feature_type.items():
            if v == "Categorical":
                self._binning_specs[k] = False

    @classmethod
    def _plot_time_dependence(cls, tmp, freq_mode, feature_name):

        base_viz = (
            alt.Chart(tmp)
            .encode(
                color=alt.Color(
                    feature_name,
                    legend=alt.Legend(title=feature_name),
                    scale=alt.Scale(
                        scheme="set1",
                    ),  # use range=["#e60049","#0bb4ff",...] if you want to specify instead
                ),
            )
            .properties(
                width=550,
                height=180,
            )
        )
        line_dr = base_viz.mark_line(point=True).encode(
            x=alt.X(
                f"yearmonthdate(score_{freq_mode}):O",
                axis=alt.Axis(title=None, labels=False, ticks=True),
            ),
            y=alt.Y("dpd_mean:Q", axis=alt.Axis(title="Mean")),
            tooltip=[f"score_{freq_mode}", feature_name, "dpd_mean"],
        )
        bar_cnt = base_viz.mark_bar().encode(
            x=alt.X(
                f"yearmonthdate(score_{freq_mode}):O",
                axis=alt.Axis(title=f"Score {freq_mode}", labelAngle=325),
            ),
            y=alt.Y("count:Q", axis=alt.Axis(title="Count")),
            tooltip=[f"score_{freq_mode}", feature_name, "count"],
        )

        return alt.vconcat(line_dr, bar_cnt).properties(
            title={
                "text": feature_name,
                "fontSize": 16,
                "color": "#656565",
                "align": "center",
                "anchor": "middle",
            },
        )

    @classmethod
    def _display_trajectory(
        cls,
        time: pd.Series,
        label: pd.Series,
        feature: pd.Series,
        date_bound: tuple = None,
        bins=None,
        freq_mode: str = None,
    ):
        """
        Display the time-dependent (weekly) plot with given bins for features.

        Args:
            time: The x axis of trajectory plot. Generally it's the timestamp column.
            label: The y axis of trajectory plot. Generally it's the label column. Note: This have to be binary.
            feature: The level of trajectory plot. Generally it's the feature column.
            date_threshold: The tuple include upper time bound and lower time bound.
            bins: Same as lutils.analytics.information_value binning. This is the binning accordance for level.
            freq_mode: The frequency of x axis if x axis is the timestamp. Should be one of values: ['day', 'week', 'month']
        """
        if bins:
            try:
                feature = pd.cut(feature, bins=bins, right=False)
            except ValueError as e:
                print(e)

        if freq_mode == "day":
            freq = time.dt.date
        elif freq_mode == "week":
            freq = (time - pd.to_timedelta(time.dt.dayofweek, unit="d")).dt.date
        elif freq_mode == "month":
            freq = (time.dt.to_period("M")).dt.to_timestamp()
        else:
            freq = time
        freq.name = "freq"

        tmp = pd.concat(
            [
                feature.astype("string").fillna("N.A.").astype("category"),
                label,
                freq.astype("datetime64[s]"),
            ],
            axis=1,
        )

        tmp = (
            tmp.groupby(["freq", feature.name])
            .agg(
                dpd_mean=(label.name, "mean"),
                count=(label.name, "count"),
            )
            .round(
                {"dpd_mean": 5},
            )
            .reset_index()
        )
        tmp.columns = [f"score_{freq_mode}", feature.name, "dpd_mean", "count"]
        if date_bound:
            tmp = tmp[tmp[f"score_{freq_mode}"].between(*date_bound)].reset_index(drop=True)

        return cls._plot_time_dependence(tmp, freq_mode, feature.name)

    def display(self):
        res = (
            self._woe_df[
                [
                    "Variable_Name",
                    "Information_Value",
                    "Bin_Label",
                    "Non_Event",
                    "Event",
                    "Count",
                    "Non_Event_Rate",
                    "Event_Rate",
                    "Non_Event_Distribution",
                    "Event_Distribution",
                    "WOE",
                ]
            ]
            .rename({"Information_Value": "Information_Value (%)"}, axis=1)
            .copy()
            .reset_index(drop=True)
        )
        res["Information_Value (%)"] = res["Information_Value (%)"].apply(lambda x: round(x * 100, 2))
        res = res.set_index(["Variable_Name", "Information_Value (%)", "Bin_Label"]).sort_index(
            level=1,
            ascending=False,
        )
        res.columns = [
            [
                "Count",
                "Count",
                "Count",
                "Rate",
                "Rate",
                "Distribution",
                "Distribution",
                "WOE",
            ],
            [
                "Good",
                "Bad",
                "Total",
                "Good",
                "Bad",
                "Good",
                "Bad",
                "",
            ],
        ]
        # IV plot
        if len(self._binning_specs) <= 5:
            for col, binning in self._binning_specs.items():
                assert col in self._df, f"{col} is not in the df!"
                display(
                    _=information_value(
                        x=self._df[col],
                        y=self._df[self._label_col],
                        binning=binning,
                        ignore_na=self._ignore_na,
                        title_feature=col,
                        bin_label_precision=10,
                        figsize=self.figsize,
                    ),
                )
        else:
            display(
                res.style.background_gradient(
                    subset=[("Rate", "Bad")],
                    cmap="Reds",
                )
                .bar(
                    [("WOE", "")],
                    color=["red", "green"],
                    align="zero",
                    vmin=-1,
                    vmax=1,
                )
                .format(
                    {
                        ("Count", "Bad"): "{:.0f}",
                        ("Count", "Good"): "{:.0f}",
                        ("Rate", "Bad"): "{:.2%}",
                        ("Rate", "Good"): "{:.2%}",
                        ("Distribution", "Bad"): "{:.2%}",
                        ("Distribution", "Good"): "{:.2%}",
                        ("WOE", ""): "{:.4f}",
                    },
                ),
            )

        # time trajectory plot
        if self._plot_time_trajectory:
            for col, binning in self._binning_specs.items():
                display(
                    self._display_trajectory(
                        self._df[self._time_col],
                        self._df[self._label_col],
                        self._df[col],
                        self._date_bound,
                        bins=binning,
                        freq_mode=self._freq,
                    ),
                )
