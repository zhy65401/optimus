import numpy as np
from bumblebee.binning import BinningCategory
from bumblebee.checkpoint import Checkpoint
from bumblebee.clip import Clipper
from bumblebee.correlation import Correlation
from bumblebee.encoder import WoeEncoder
from bumblebee.feature import FeatureSaver
from bumblebee.impute import SimpleImputer
from bumblebee.project import Filter, LogOdds
from bumblebee.scale import StandardScaler


def make_feature_saver(feature_names=None):
    if isinstance(feature_names, FeatureSaver):
        return feature_names
    return FeatureSaver(feature_names=feature_names)


def make_feature_filter(items=None, like=None, regex=None, exclude=False, copy=True):
    if not items:
        return None
    if isinstance(items, Filter):
        return items
    return Filter(copy=copy, items=items, like=like, regex=regex, exclude=exclude)


def make_binning_category(specs=None, cardinality_cutoff=5, prefix=None, copy=False):
    if isinstance(specs, BinningCategory):
        return specs
    return BinningCategory(specs=specs, cardinality_cutoff=cardinality_cutoff, prefix=prefix, copy=copy)


def make_woe_encoder(
    feature_names=None,
    woe_prefix=None,
    treat_missing="skip",
    woe_bins=None,
    copy=True,
):
    if not feature_names:
        return None
    if isinstance(feature_names, WoeEncoder):
        return feature_names
    return WoeEncoder(
        feature_names=feature_names,
        woe_prefix=woe_prefix,
        treat_missing=treat_missing,
        woe_bins=woe_bins,
        copy=copy,
    )


def make_imputer(strategy="mean", missing_values=np.nan, copy=False):
    return SimpleImputer(strategy=strategy, missing_values=missing_values, copy=copy)


def make_checkpoint():
    return Checkpoint()


def make_standard_scaler(copy=False):
    return StandardScaler(copy=copy)


def make_correlation(include_vif=True):
    return Correlation(include_vif=include_vif)


def make_to_logodds(cols=None):
    if isinstance(cols, LogOdds):
        return cols
    return LogOdds(cols=cols or [])


def make_clipper(specs=None, woe_columns=None, warn=False, copy=False):
    if isinstance(specs, Clipper):
        return specs
    return Clipper(specs=specs, woe_columns=woe_columns, warn=warn, copy=copy)
