#!/usr/bin/env python
# Version: 0.1.0
# Created: 2024-04-07
# Author: ["Hanyuan Zhang"]

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin


class Calibration(TransformerMixin):
    """
    A score calibration helper.

    Score Types
    -----------
    'mega_score':
         mega_score calibration applies to Ascore and Bscore
         score scale 300~1000, 850 -> 0.1%, 500 -> 12.8%

    'sub_score':
         sub_score calibration applies to the sub-score of Ascore and Bscore
         score scale 0~100, 95 -> 0.1%, 10 -> 12.8%

    'self-defining':
         self-defining score scale
         must input mapping_base, score_cap, score_floor in fit

    'probability':
         only calibrate probability

    Notes
    -----
    Must check the probability calibration process:
         calling Calibration.calibrate_detail and Calibration.get_calibrate_plot()

    Must check the distribution and risk level of calibration results:
         calling Calibration.compare_calibrate_result(df_score, df_label)

    """

    def __init__(self, n_bins=25, n_degree=1, score_type='mega_score', 
        mapping_base=None, score_cap=None, score_floor=None):

        self.mapping_base = mapping_base
        self.score_cap = score_cap
        self.score_floor = score_floor
        self.score_type = score_type
        self.n_bins = n_bins
        self.n_degree = n_degree

        self.calibrate_detail = None
        self.calibrate_coef = None
        self.mapping_intercept = None
        self.mapping_slope = None

    def fit(self, df_prob, df_label):
        """
        Fit the calibration model. The purpose is to calibrate the output from a classifier to a score, so that each score bin will have a similar bad rate.
        The idea is to build a linear regression between the ln(odds(y_hat)) and the real bad rate in each bin in the training set.
        The transformation process will compute the estimated bad rate with a given ln(odds(y_hat)) and hence map to a score with self-defined mapping base.
        
        Arguments:
        df_prob : pd.DataFrame
            The predicted output from the classifier.
        df_label : pd.DataFrame
            The true labels of the data.
        """
        if self.mapping_base is not None:
            self.score_type = 'self-defining'
            logging.warning('self-defining score type, input mapping_base, score_cap, and score_floor')
            self.mapping_base, self.score_cap, self.score_floor = self.mapping_base, self.score_cap, self.score_floor
            self.mapping_slope, self.mapping_intercept = self.__set_mapping_base(self.mapping_base)

        elif self.score_type == 'probability':
            logging.warning('probability score type, only probability calibration')

        elif self.score_type in ['mega_score', 'sub_score']:
            self.mapping_base, self.score_cap, self.score_floor = self.__set_default_score_base(self.score_type)
            self.mapping_slope, self.mapping_intercept = self.__set_mapping_base(self.mapping_base)

        else:
            raise Exception('unknown score type, expect mega_score, sub_score, probability, and self-defining')

        lst_prob = self.__check_type(df_prob)
        lst_label = self.__check_type(df_label)

        df_data = pd.DataFrame({'yprob': lst_prob, 'label': lst_label,
                                'lnodds_prob': [self.prob2lnodds(x) for x in lst_prob]})
        df_data['lnodds_prob_bin'] = pd.qcut(df_data['lnodds_prob'], self.n_bins, duplicates='drop')

        df_cal = df_data.groupby('lnodds_prob_bin').agg(total=('label', 'count'),
                                                        bad_rate=('label', 'mean'),
                                                        lnodds_prob_mean_x=('lnodds_prob', 'mean'))
        df_cal['adj_bad_rate'] = df_cal.apply(lambda x: max(x['bad_rate'], 1 / x['total'], 0.0001), axis=1)
        df_cal['lnodds_bad_rate_y'] = df_cal['adj_bad_rate'].apply(lambda x: self.prob2lnodds(x))

        lst_col = ['total', 'bad_rate', 'adj_bad_rate', 'lnodds_prob_mean_x', 'lnodds_bad_rate_y']
        self.calibrate_detail = df_cal[lst_col]
        self.calibrate_coef = np.polyfit(df_cal['lnodds_prob_mean_x'], df_cal['lnodds_bad_rate_y'], self.n_degree)
        return self

    def transform(self, df_prob):
        """
        Transform the predicted output from the classifier to a score.
        
        Arguments:
        df_prob : pd.DataFrame
            The predicted output from the classifier.
        """
        lst_prob = self.__check_type(df_prob)
        lst_lnodds_prob = [self.prob2lnodds(x) for x in lst_prob]
        lst_lnodds_cal_prob = [np.poly1d(self.calibrate_coef)(x) for x in lst_lnodds_prob]

        if self.score_type == 'probability':
            lst_cal_prob = [self.lnodds2prob(x) for x in lst_lnodds_cal_prob]
            return np.array(lst_cal_prob)

        else:
            lst_score = [self.mapping_intercept + self.mapping_slope * x for x in lst_lnodds_cal_prob]
            lst_score = [max(x, self.score_floor) for x in lst_score]
            lst_score = [min(x, self.score_cap) for x in lst_score]
            return np.array(lst_score)

    def compare_calibrate_result(self, df_score, df_label, bins=None):
        """
        Compare the calibration result with the original score.
        
        Arguments:
        df_score : pd.DataFrame
            The predicted output from the classifier.
        df_label : pd.DataFrame
            The true labels of the data.
        bins : list
            The bins to be used for the calibration result.
        """
        if bins is None:
            if self.score_type == 'mega_score':
                bins = [0, 300, 400, 500, 550, 600, 650, 700, 750, 800, 850, 1000]
            elif self.score_type == 'sub_score':
                bins = [0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
            else:
                raise Exception('bins is required for self-defining score type')

        lst_score = self.__check_type(df_score)
        lst_label = self.__check_type(df_label)
        df_data = pd.DataFrame({'score': lst_score, 'label': lst_label})
        df_data['score_bin'] = pd.cut(df_data['score'], bins)

        eps = np.finfo(np.float32).eps
        df_res = df_data.groupby('score_bin').agg(
            total=('label', 'count'), 
            total_pct=('label', lambda x: len(x) / (len(df_data) + eps)), 
            total_good=('label', lambda x: len(x) - sum(x)),
            good_rate=('label', lambda x: (len(x) - sum(x)) / (len(x) + eps)),
            total_bad=('label', 'sum'), 
            bad_rate=('label', 'mean')
        )
        df_res = df_res.reset_index()
        df_res['score_max'] = df_res['score_bin'].apply(lambda x: x.right)
        df_res['good_dist'] = df_res['total_good'] / (df_res['total_good'].sum() + eps)
        df_res['bad_dist'] = df_res['total_bad'] / (df_res['total_bad'].sum() + eps)

        if self.score_type == 'probability':
            df_res['exp_bad_rate'] = df_res['score_max']
        else:
            df_res['exp_bad_rate'] = df_res['score_max'].apply(
                lambda x: self.lnodds2prob((x - self.mapping_intercept) / self.mapping_slope))
        
        df_res['iv'] = ((df_res["bad_dist"] - df_res["good_dist"]) * np.log((df_res["bad_dist"] + eps) / (df_res["good_dist"] + eps))).sum()
        df_res['cum_good_dist'] = df_res['good_dist'].cumsum()
        df_res['cum_bad_dist'] = df_res['bad_dist'].cumsum()
        df_res['ks'] = (df_res['cum_bad_dist'] - df_res['cum_good_dist']).abs()
        # Odds ratio between the odds before (inclusive) and after the bin to decide score cut-off
        # if max(odds_ratio) >= max(inv_odds_ratio), take the bin (a,b] that achieve the max odds_ratio and set score <= b
        # if max(odds_ratio) < max(inv_odds_ratio), take the bin (a,b] that achieve the max inv_odds_ratio and set score > a
        df_res['odds_bef'] = df_res['total_bad'].cumsum() / (df_res['total_good'].cumsum() + eps)
        df_res['odds_aft'] = (df_res['total_bad'] - df_res['total_bad'].cumsum()) / (df_res['total_good'] - df_res['total_good'].cumsum() + eps)
        df_res['odds_ratio'] = df_res['odds_bef'] / df_res['odds_aft']
        df_res['inv_odds_ratio'] = df_res['odds_aft'] / df_res['odds_bef']
        

        lst_col = ['score_bin', 'score_max', 'total', 'total_pct', 'total_good', 'good_rate', 'good_dist', 'total_bad', 'bad_rate', 'bad_dist', 'exp_bad_rate', 'odds_ratio', 'inv_odds_ratio', 'ks', 'iv']
        df_res = df_res[lst_col]
        return df_res

    def get_bad_rate(self, score_min, score_max, step):
        """
        Get the bad rate for a given score range.
        
        Arguments:
        score_min : int
            The minimum score.
        score_max : int
            The maximum score.
        step : int
            The step size for the score range.
        """
        if self.score_type == 'probability':
            raise Exception('probability score type, no score mapping process')

        ary_score = np.arange(score_min, score_max, step)
        ary_lnodds = (ary_score - self.mapping_intercept) / self.mapping_slope
        ary_bad_rate = self.lnodds2prob(ary_lnodds)
        return pd.DataFrame({'score': ary_score, 'bad_rate': ary_bad_rate, 'lnodds': ary_lnodds})

    def get_calibrate_plot(self):
        """
        Plot the calibration result. The x-axis is the ln(odds(y_hat)) and the y-axis is the bad rate.
        Ideally, the points should be close to the diagonal line.
        """
        x = self.calibrate_detail['lnodds_prob_mean_x']
        y_actual = self.calibrate_detail['lnodds_bad_rate_y']

        y_pred = np.poly1d(self.calibrate_coef)(x)
        f, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, y_actual, 'o', x, y_pred, '-', label='1d')
        ax.set_xlabel('lnodds_prob_mean'), ax.set_ylabel('lnodds_bad_rate')
        return ax

    @classmethod
    def __set_default_score_base(cls, score_type):
        if score_type == 'mega_score':
            mapping_base = {
                500: 0.128,
                550: 0.0671,
                600: 0.0341,
                650: 0.017,
                700: 0.0084,
                750: 0.0041,
                800: 0.002,
                850: 0.001
            }
            score_cap = 1000
            score_floor = 300

        elif score_type == 'sub_score':
            mapping_base = {
                10: 0.128,
                15: 0.0987,
                20: 0.0755,
                25: 0.0574,
                30: 0.0434,
                35: 0.0327,
                40: 0.0246,
                45: 0.0185,
                50: 0.0138,
                55: 0.0104,
                60: 0.0077,
                65: 0.0058,
                70: 0.0043,
                75: 0.0032,
                80: 0.0024,
                85: 0.0018,
                90: 0.0013,
                95: 0.001
            }
            score_cap = 100
            score_floor = 0

        else:
            raise Exception('unknown score type, only mega_score and sub_score available')

        return mapping_base, score_cap, score_floor

    @classmethod
    def __set_mapping_base(cls, dict_base):
        # Linear regression for mapping base
        lst_score = sorted(dict_base.keys())
        lst_bad_rate = sorted(dict_base.values(), reverse=True)
        lst_lnodds_bad_rate = [cls.prob2lnodds(x) for x in lst_bad_rate]

        score_max, score_min = lst_score[-1], lst_score[0]
        lnodds_max, lnodds_min = lst_lnodds_bad_rate[-1], lst_lnodds_bad_rate[0]

        slope = (score_max - score_min) / (lnodds_max - lnodds_min)
        intercept = score_max - slope * lnodds_max
        return slope, intercept

    @classmethod
    def __check_type(cls, data):
        if isinstance(data, (list, pd.Series, np.ndarray)):
            lst_data = list(data)
        elif isinstance(data, pd.DataFrame):
            lst_data = data[data.columns.item()].tolist()
        else:
            raise TypeError('Expected data type: DataFrame, List, Series or Array')
        return lst_data

    @classmethod
    def prob2lnodds(cls, prob):
        if prob == 0:
            lnodds = np.log(np.finfo(float).eps)
        elif prob == 1:
            lnodds = np.log(prob / (1 - prob + np.finfo(float).eps))
        else:
            lnodds = np.log(prob / (1 - prob))
        return lnodds

    @classmethod
    def lnodds2prob(cls, lnodds):
        prob = 1 - 1 / (np.exp(lnodds) + 1)
        return prob