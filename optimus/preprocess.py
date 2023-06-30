from bumblebee.binning import BinningCategory
from bumblebee.feature import FeatureSaver
from bumblebee.project import Filter

from .utils import make_feature_saver


class Preprocessor:
    def __init__(self, preprocesses):
        self.preprocesses = preprocesses
        self.feature_saver = []
        self.binning_specs = []
        self.to_drop_cols = []
        self.custom_fn = {}

        self._generate_preprocessors()

    def _generate_preprocessors(self):
        preprocesses = []
        if not self.preprocesses:
            # dummy settings for building pipeline
            preprocesses = [
                ("feature_saver", make_feature_saver()),
            ]

        else:
            for name, value in self.preprocesses:
                if isinstance(value, FeatureSaver):
                    self.feature_saver = value.feature_names or []
                    preprocesses.append((name, value))
                elif name == "feature_saver":
                    self.feature_saver = value
                    preprocesses.append((name, self.feature_saver))
                elif isinstance(value, BinningCategory):
                    self.binning_specs = value.specs or []
                    preprocesses.append((name, value))
                elif name == "binning_category":
                    self.binning_specs = value
                    preprocesses.append((name, self.binning_specs))
                elif isinstance(value, Filter):
                    self.to_drop_cols = value.items or []
                    preprocesses.append((name, value))
                elif name == "feature_dropper":
                    self.to_drop_cols = value
                    preprocesses.append((name, self.to_drop_cols))
                else:
                    self.custom_fn[name] = value
                    preprocesses.append((name, value))
        self.preprocesses = preprocesses
