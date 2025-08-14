#!/usr/bin/env python
# Version: 0.1.0
# Created: 2024-04-07
# Author: ["Hanyuan Zhang"]

import numpy as np
import pandas as pd
from termcolor import cprint
from pandas.api.types import is_numeric_dtype
from itertools import combinations

from sklearn.base import BaseEstimator, TransformerMixin
    

class Common:
    
    @staticmethod
    def _adjust_bins(bins):
        outBins = bins.copy()
        outBins[0] = float('-inf')
        outBins[-1] = float('inf')
        
        return outBins

    @staticmethod
    def _append_bins(bins):
        outBins = bins.copy()
        pre = [float('-inf')] if outBins[0] != float('-inf') else []
        suf = [float('inf')] if outBins[-1] != float('inf') else []
        
        return pre + outBins + suf
    
    
class QCut(BaseEstimator, TransformerMixin):
    def __init__(self, target_bin_cnt: int, copy: bool=True):
        """Qcut binnning strategy. This strategy is only applicable to numerical features.

        Args:
            target_bin_cnt (int): The expected output bins number. Will feed to q.
            copy (bool, optional): Defaults to True.
        """
        self.target_bin_cnt = target_bin_cnt
        self.bins = None
        self.copy = copy
    
    def fit(self, X, y=None):
        if not is_numeric_dtype(X):
            raise TypeError(f'Feature `{X.name}` is not a numerical feature, but the qcut strategy is selected.')
        _, self.bins = pd.qcut(X, q=self.target_bin_cnt, duplicates='drop', retbins=True)
        self.bins = Common._adjust_bins(self.bins)
        return self
    
    def transform(self, X):
        if self.copy:
            outX = X.copy(deep=True)
        else:
            outX = X
        outX = pd.cut(outX, self.bins, include_lowest=True)
        return outX


class SimpleCut(BaseEstimator, TransformerMixin):
    def __init__(self, copy: bool=True):
        """
        This binning method is only used for strategy rule setting, and will focus on the cutting points within the head and tail quantiles. 
        Binning quantile will be set as [0, .02, .04, .1, .9, .96, .98, 1]

        Args:
            copy (bool, optional): Defaults to True.
        """
        self.bins = None
        self.copy = copy
    
    def fit(self, X, y=None):
        if not is_numeric_dtype(X):
            raise TypeError(f'Feature `{X.name}` is not a numerical feature, but the simple_cut strategy is selected.')
        _, self.bins = pd.qcut(X, [0, .02, .04, .05, .95, .96, .98, 1], duplicates='drop', retbins=True)
        self.bins = Common._adjust_bins(self.bins)
        
        return self
    
    def transform(self, X):
        if self.copy:
            outX = X.copy(deep=True)
        else:
            outX = X
        outX = pd.cut(outX, self.bins, include_lowest=True)
        return outX


class ChiMergeCut(BaseEstimator, TransformerMixin):
    def __init__(self, target_intervals: int, initial_intervals=False, copy: bool=True):
        """
        The idea of this binning method is to keep merging contingent two bins if they are not significantly different in label distribution
        until reach the target_intervals.

        Args:
            target_intervals (int): The expected output bins number.
            initial_intervals (int or False, optional): Initial bins. This is used for pre-binning high cardinality features. If False, no pre-binning process will apply. Defaults to False.
            copy (bool, optional): Defaults to True.
        """
        self.target_intervals = target_intervals
        self.initial_intervals = initial_intervals
        self.bins = None
        self.copy = copy
        
    @staticmethod
    def chisquare(arr):
        # Construct the expected frequency table E, where E_ij = ft_cat_i * (label_cat_j / total) = ft_cat_i * P(label_cat_j)
        # Here, ft_cat_i is the frequency of feature category i, label_cat_j is the frequency of label category j.
        cls_cnt = np.sum(arr, axis=0, keepdims=True)
        bin_cnt = np.sum(arr, axis=1, keepdims=True)
        cls_dist = cls_cnt / np.sum(cls_cnt)
        E_ = np.matmul(bin_cnt, cls_dist)
        # Calculate the chi-square value
        # chi2 = sum((O_ij - E_ij)^2 / E_ij), where O_ij is the observed frequency.
        return np.sum(
            np.divide((arr - E_) ** 2, E_, out=np.zeros_like(E_), where=E_ != 0)
        )

    def fit(self, X, y):
        X = np.ravel(X)
        y = np.ravel(y)

        if X.size != y.size:
            raise ValueError("X and y have different sizes.")

        if self.initial_intervals:
            index, bins = pd.qcut(
                X,
                self.initial_intervals,
                labels=False,
                retbins=True,
                duplicates="drop",
            )
            # sum up the y values for each bin
            df_y = pd.get_dummies(y).groupby(index, observed=True).sum()
            # add the last endpoint to ensure every bin appears
            bins = np.append(bins[df_y.index], bins[-1])
        else:
            index, bins = X, np.append(float('-inf'), np.sort(np.unique(X)))
            df_y = pd.get_dummies(y).groupby(index, observed=True).sum()
        y = df_y.to_numpy()

        while len(y) > self.target_intervals:
            # calculate the chi-square value for each pair of adjacent bins
            chi2 = np.array([self.chisquare(y[i : i + 2, :]) for i in range(len(y) - 1)])
            # merge the pair of bins with the smallest chi-square value and update the y values
            pos = np.argmin(chi2)
            y[pos, :] += y[pos + 1, :]
            y = np.delete(y, pos + 1, axis=0)
            bins = np.delete(bins, pos + 1)
        # add the last endpoint to ensure every bin appears
        self.bins = Common._adjust_bins(bins)

        return self

    def transform(self, X):
        if self.copy:
            outX = X.copy(deep=True)
        else:
            outX = X
        outX = pd.cut(outX, self.bins, include_lowest=True)
        return outX


class BestKSCut(BaseEstimator, TransformerMixin):
    def __init__(self, target_bin_cnt: int, min_bin_rate: float, copy: bool=True):
        """
        This binning method has two steps:
            1. Greedily looking for all of initial knots which achieve the best KS until reaching the `min_bin_rate`.
            2. Among all of initial knots, traversing all of knots combination with `target_bin_cnt + 1` knots and the result is the one with the highest IV and monotonic WOE.
            
        NOTE:
        This binning method performs well but with the low efficiency (O(n^2)).
        Since the combination number could be too large with too small `min_bin_rate`, need to take care of the choice of this parameter. In general, do not exceed 5%.

        Args:
            target_bin_cnt (int): The expected output bins number.
            min_bin_rate (float): The minimum bin rate of each bin.
            copy (bool, optional): Defaults to True.
        """
        
        self.target_bin_cnt = target_bin_cnt
        self.min_bin_rate = min_bin_rate
        self.bins = None
        self._eps = np.finfo(float).eps
        self.copy = copy
    
    def _stat_bins(self, X, y):
        # Calculate the KS value for each bin
        df = pd.concat([X, y], axis=1).rename({X.name: 'bin', y.name: 'label'}, axis=1)
        total_good = len(X) - sum(y)
        total_bad = sum(y)
        df_bins = df.groupby('bin', observed=True).agg(
            bin_cnt=('label', 'count'),
            good_rate=('label', lambda x: (len(x) - sum(x)) / len(X)),
            bad_rate=('label', lambda x: sum(x) / len(X)),
            good_dist=('label', lambda x: (len(x) - sum(x)) / (total_good + self._eps)),
            bad_dist=('label', lambda x: sum(x) / (total_bad + self._eps)),
        )
        df_bins['ks'] = np.abs(np.cumsum(df_bins['bad_dist']) - np.cumsum(df_bins['good_dist']))
        
        return df_bins.reset_index()
    
    def cut_bins(self, X, y, left_idx, right_idx):
        left = self.init_bins[left_idx]
        right = self.init_bins[right_idx]
        bin_X, bin_y = X[X.between(left, right, inclusive='right')], y[X.between(left, right, inclusive='right')]
        # Reach the min_bin_rate no matter the position of knot, abort
        if len(bin_X) < 2*self._total*self.min_bin_rate:
            return []
        stat_bin = self._stat_bins(bin_X, bin_y)
        # Empty bin, abort
        if left >= right:
            return []
        # Label not binary, abort
        if (stat_bin['good_rate'].sum()==0) or (stat_bin['bad_rate'].sum()==0):
            return []
        
        best_ks_knot = stat_bin.loc[stat_bin['ks'].argmax(), 'bin']
        best_ks_knot_idx = self.init_bins.index(best_ks_knot)
        
        # No better KS than keeping the bin, abort
        if best_ks_knot == right_idx:
            return []
        # If one of sub-bins hit the min_bin_rate, reject this knot
        # Default inclusive_mode is (left, right], but need to change to [minimum, right] for including the lowest.
        inclusive_mode = 'both' if left == self.init_bins[0] else 'right'
        left_bin_total = stat_bin.loc[stat_bin['bin'].between(left, best_ks_knot, inclusive=inclusive_mode), 'bin_cnt'].sum()
        right_bin_total = stat_bin.loc[stat_bin['bin'].between(best_ks_knot, right, inclusive='right'), 'bin_cnt'].sum()
        if (left_bin_total < self._total*self.min_bin_rate) or (right_bin_total < self._total*self.min_bin_rate):
            return []
        
        knots = []
        knots.extend(self.cut_bins(bin_X, bin_y, left_idx, best_ks_knot_idx))
        knots.append(best_ks_knot_idx)
        knots.extend(self.cut_bins(bin_X, bin_y, best_ks_knot_idx, right_idx))
        
        return knots

    def _evaluate_iv(self, ids):
        lst_df = []
        lst_df.append(self._stat_df.loc[:ids[0]])
        lst_df.extend([self._stat_df.loc[ids[i-1] + 1 : ids[i]] for i in range(1, len(ids))])
        lst_df.append(self._stat_df.loc[ids[-1] + 1:])
        ratio_good = pd.Series(list(map(lambda x: float(sum(x["good_dist"])), lst_df)))
        ratio_bad = pd.Series(list(map(lambda x: float(sum(x["bad_dist"])), lst_df)))
        # if WOE is not monotonic, rejust this combination
        lst_woe = list(np.log((ratio_good + self._eps) / (ratio_bad + self._eps)))
        if sorted(lst_woe) != lst_woe and sorted(lst_woe, reverse=True) != lst_woe:
            return -1
        lst_iv = (ratio_good - ratio_bad) * np.log((ratio_good + self._eps) / (ratio_bad + self._eps))
        return sum(lst_iv)
    
    def fit(self, X, y):
        inX = X.copy(deep=True)
        if not is_numeric_dtype(inX):
            raise TypeError(f'Feature `{inX.name}` is not a numerical feature, but the BestKS strategy is selected.')
        self._total = len(inX)
        self.init_bins = sorted(inX.unique())
        if len(self.init_bins) <= self.target_bin_cnt:
            cprint(f"[WARNING] {inX.name}: Feature unique number is less than target bin!", "yellow")
            self.bins = np.array([-float('inf'), float('inf')], dtype=float)
            return self
        # Recursively cut bins to achieve mono WOE
        bins_idx = self.cut_bins(inX, y, 0, len(self.init_bins)-1)
        best_ks_knots_ = [self.init_bins[idx] for idx in bins_idx]
        if best_ks_knots_:
            inX = pd.cut(inX, Common._append_bins(best_ks_knots_), include_lowest=True)
            self._stat_df = self._stat_bins(inX, y)
            # In case there is no proper knots combination (not monotonic or too many combinations)
            for target_knot_cnt in range(self.target_bin_cnt, 2, -1):
                # The best KS bins number is fewer than expected]
                lst_comb = [list(lst_knot_idx) for lst_knot_idx in combinations(range(len(best_ks_knots_)), min(len(best_ks_knots_), target_knot_cnt-1))]
                # Control the efficiency. If there are too many combinations, try a smaller target_bins.
                if len(lst_comb) > 10_000:
                    cprint(f"[WARNING] {inX.name}: Too many combination ({len(lst_comb)}) at target_bin_cnt=={target_knot_cnt}. Try with a larger min_bin_rate.", "yellow")
                    continue
                lst_iv = [i for i in map(lambda x: self._evaluate_iv(x), lst_comb)]
                if max(lst_iv) > 0:
                    self.bins = np.array(Common._append_bins([best_ks_knots_[i] for i in lst_comb[np.argmax(lst_iv)]]), dtype=float)
                    return self
        # At least have a (-inf, inf) bin
        cprint(f"[WARNING] {inX.name}: No proper bin combination exists!", "yellow")
        self.bins = np.array([-float('inf'), float('inf')], dtype=float)
        return self
    
    def transform(self, X):
        if self.copy:
            outX = X.copy(deep=True)
        else:
            outX = X
        outX = pd.cut(outX, self.bins, include_lowest=True)
        return outX


class WOEMerge(BaseEstimator, TransformerMixin):
    def __init__(self, target_bin_cnt: int, min_bin_rate: float, split_symbol: str='||', copy=True):
        """The idea of this merge strategy is keep merging bins with samilar weight-of-evidence value
         until `target_bin_cnt` is reached. This strategy is only applicable to categorical features.

        Args:
            target_bin_cnt (int): The expected output bins number.
            min_bin_rate (float): The minimum bin rate of each bin.
            split_symbol (str): The split symbol to separate each categorical bin. Defaults to `||`.
            copy (bool, optional): _description_. Defaults to True.
        """
        self.target_bin_cnt = target_bin_cnt
        self.min_bin_rate = min_bin_rate
        self.bins = None
        self.cat_others = []
        self.copy = copy
        self.split_symbol = split_symbol
        self._eps = np.finfo(float).eps
        
    def _get_woe(self, X, y):
        df = pd.concat([X, y], axis=1).rename({X.name: 'bin', y.name: 'label'}, axis=1)
        total_good = len(X)-sum(y)
        total_bad = sum(y)
        
        return df.groupby('bin', observed=True).agg(
            woe=('label', lambda x: np.log(((len(x)-sum(x))/total_good) / (sum(x)/total_bad+self._eps)))
        ).sort_values('woe', ascending=False).to_dict()['woe']
        
    def _cat_bin_mapping(self, x, bin_list, cat_others):
        if pd.isna(x) or not x:
            return '__N.A__'
        if x in cat_others:
            return '__OTHERS__'
        for val in bin_list:
            if x in val.split(self.split_symbol):
                return val
        return '__N.A__'
        
    def _pre_merge(self, X):
        """
        Pre-merge the bins that fewer than min_bin_rate to decrease the cardinality in feature.
        """
        cutoff_count = np.ceil(self.min_bin_rate * len(X))
        cat_count = pd.Series(X).value_counts()
        self.cat_others = cat_count[cat_count < cutoff_count].index.values
        mask_others = pd.Series(X).isin(self.cat_others).values

        if np.count_nonzero(~mask_others) == 0:
            cprint(f"[WARNING] {X.name}: All categories are moved to `others` bin!", "yellow")

        return mask_others
    
    def _category_merger(self, X, y):
        while X.nunique() > self.target_bin_cnt:
            df_woe = self._get_woe(X, y)
            df_woe['woe_diff'] = df_woe.woe.diff()
            min_diff_idx = df_woe.woe_diff.argmax()
            to_merge_cols = [df_woe.index[min_diff_idx-1], df_woe.index[min_diff_idx]]
            # assign new name to merged categories
            if "__OTHERS__" in to_merge_cols:
                new_merge_name = "__OTHERS__"
                self.cat_others.extend([n for n in to_merge_cols if n != "__OTHERS__"])
            else:
                new_merge_name = self.split_symbol.join(to_merge_cols)
            mask_others = X.isin(to_merge_cols)
            # overwrite with new merged name
            X.loc[mask_others] = new_merge_name
            
        return list(X.unique())
        
    def fit(self, X, y):
        inX = X.copy(deep=True)
        if is_numeric_dtype(X):
            inX = inX.astype(str)
        masked_others = self._pre_merge(inX)
        inX.loc[masked_others] = "__OTHERS__"
        merged_categories: list = self._category_merger(inX, y)
        self.bins = merged_categories
        
        return self
    
    def transform(self, X, y=None):
        if self.copy:
            outX = X.copy(deep=True).astype(str)
        else:
            outX = X.astype(str)
        return outX.map(lambda x: self._cat_bin_mapping(x, self.bins, self.cat_others))