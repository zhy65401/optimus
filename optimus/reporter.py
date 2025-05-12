#!/usr/bin/env python
# Version: 0.1.0
# Created: 2024-04-07
# Author: ["Hanyuan Zhang"]

import pandas as pd
from pandas.api.types import is_numeric_dtype
from openpyxl.utils import get_column_letter

from .metrics import Metrics


class Reporter():
    def __init__(self, report_path):
        self.report_path = report_path
    
    @classmethod
    def _set_col_format(cls, worksheet, col_ids, format):
        for col_idx in col_ids:
            for cell in worksheet[col_idx]:
                    cell.number_format = format
    
    @classmethod
    def _set_col_width(cls, worksheet, col_ids, width):
        for col_idx in col_ids:
            worksheet.column_dimensions[col_idx].width = width
            
    def _format_overview_stats_df(self, worksheet):
        perc_cols = ['C', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
        Reporter._set_col_format(worksheet, perc_cols, '0.000%')
        w15 = ['F', 'G', 'H', 'I', 'J', 'K', 'L']
        w20 = ['A', 'B', 'C', 'D', 'E']
        Reporter._set_col_width(worksheet, w15, 15)
        Reporter._set_col_width(worksheet, w20, 20)
        
    def _format_overview_perf_df(self, worksheet):
        perc_cols = ['B', 'C', 'D', 'E', 'F', 'G']
        Reporter._set_col_format(worksheet, perc_cols, '0.000%')
        w20 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
        Reporter._set_col_width(worksheet, w20, 20)
    
    def _format_feat_df(self, worksheet):
        perc_cols = ['D', 'F', 'G', 'H', 'S', 'U', 'W', 'X', 'Y']
        Reporter._set_col_format(worksheet, perc_cols, '0.000%')
        w40 = ['A']
        w25 = ['G', 'H', 'I', 'J']
        w20 = ['B', 'C', 'D', 'E', 'F', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W']
        w15 = ['X', 'Y']
        Reporter._set_col_width(worksheet, w40, 40)
        Reporter._set_col_width(worksheet, w25, 25)
        Reporter._set_col_width(worksheet, w20, 20)
        Reporter._set_col_width(worksheet, w15, 10)
            
    def _format_woe_df(self, worksheet):
        num_cols_4dec = ['N', 'O', 'Q']
        perc_cols = ['D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'P', 'S', 'T']
        Reporter._set_col_format(worksheet, num_cols_4dec, '0.0000')
        Reporter._set_col_format(worksheet, perc_cols, '0.000%')
        w40 = ['A']
        w20 = ['B']
        w15 = [get_column_letter(i) for i in range(3, worksheet.max_column+1)]
        Reporter._set_col_width(worksheet, w40, 40)
        Reporter._set_col_width(worksheet, w20, 20)
        Reporter._set_col_width(worksheet, w15, 15)
    
    def _format_feature_selection_overview_df(self, worksheet):
        w40 = ['A']
        Reporter._set_col_width(worksheet, w40, 40)
        
    def _format_feature_selection_df(self, worksheet):
        w20 = ['C', 'D', 'E']
        w40 = ['B']
        Reporter._set_col_width(worksheet, w20, 20)
        Reporter._set_col_width(worksheet, w40, 40)
                    
    def _format_tuning_df(self, worksheet):
        perc_cols = ['P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X']
        Reporter._set_col_format(worksheet, perc_cols, '0.000%')
        w15 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'T', 'U', 'V', 'W', 'X']
        w20 = ['P', 'Q', 'R', 'S']
        w40 = ['L', 'M', 'N', 'O']
        Reporter._set_col_width(worksheet, w15, 15)
        Reporter._set_col_width(worksheet, w20, 20)
        Reporter._set_col_width(worksheet, w40, 40)
        
    def _format_calibration_reg_df(self, worksheet):
        w15 = ['A', 'B', 'C', 'D']
        w20 = ['E', 'F']
        Reporter._set_col_width(worksheet, w15, 15)
        Reporter._set_col_width(worksheet, w20, 20)
        
    def _format_calibration_scorecard_df(self, worksheet):
        perc_cols = ['E', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        Reporter._set_col_format(worksheet, perc_cols, '0.000%')
        w15 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']
        Reporter._set_col_width(worksheet, w15, 15)
    
    def _stat_feat(self, X, y, woe_df, missing_values):
        feature_summaries = []
        for column in X.columns:
            total = len(X)
            missing = X[column].apply(lambda x: x in missing_values).sum()
            unique = X[column].nunique()
            unique_without_missing = X.loc[X[column].apply(lambda x: x not in missing_values), column].nunique()
            bad_rate_with_missing = y.mean()
            bad_rate_without_missing = y[X[column].apply(lambda x: x not in missing_values)].mean()
            
            X_normal = X.loc[X[column].apply(lambda x: x not in missing_values)]
            mode = X_normal[column].mode().iloc[0] if len(X_normal[column].mode()) > 0 else None
            mode_rate = (X_normal[column]==mode).mean()
            second_mode = X_normal[column].value_counts().index[1] if len(X_normal[column].value_counts()) > 1 else None
            second_mode_rate = (X_normal[column]==second_mode).mean() if second_mode else None
            third_mode = X_normal[column].value_counts().index[2] if len(X_normal[column].value_counts()) > 2 else None
            third_mode_rate = (X_normal[column]==third_mode).mean() if third_mode else None
            
            # for numerical features
            is_numerical = is_numeric_dtype(X_normal[column])
            desc_stats = X_normal[column].describe() if is_numerical else None
            stats_min = desc_stats['min'] if is_numerical else None
            stats_25 = desc_stats['25%'] if is_numerical else None
            stats_mean = desc_stats['mean'] if is_numerical else None
            stats_75 = desc_stats['75%'] if is_numerical else None
            stats_max = desc_stats['max'] if is_numerical else None
            stats_std = desc_stats['std'] if is_numerical else None
            zero = (X_normal[column] == 0).sum() if is_numerical else None
            zero_rate = zero / total if is_numerical else None
            cva = stats_std / stats_mean if is_numerical else None
            

            feature_summary = {
                'Feature': column,
                'total': total,
                '#missing': missing,
                '%missing': missing / total,
                '#zero': zero,
                '%zero': zero_rate,
                'Bad Rate (with missing)': bad_rate_with_missing,
                'Bad Rate (without missing)': bad_rate_without_missing,
                'Unique (with missing)': unique,
                'Unique (without missing)': unique_without_missing,
                'min': stats_min,
                '25%': stats_25,
                'mean': stats_mean,
                '75%': stats_75,
                'max': stats_max,
                'std': stats_std,
                'cva': cva,
                'mode': mode,
                'mode_rate': mode_rate,
                'second_mode': second_mode,
                'second_mode_rate': second_mode_rate,
                'third_mode': third_mode,
                'third_mode_rate': third_mode_rate,
            }

            feature_summaries.append(feature_summary)

        summary_df = pd.DataFrame(feature_summaries).set_index('Feature')
        return pd.concat([summary_df, woe_df[['KS', 'IV']].reset_index().groupby('feature_name').agg(
            KS=(('KS', ''), 'max'),
            IV=(('IV', 'bin'), 'sum')
        )], axis=1)
        
    def _stat_perf(self, gp, target_label):
        y_true = gp[target_label]
        y_bm_proba = gp['bm_proba']
        y_proba = gp['proba']
        if y_true.nunique() == 1:
            return pd.DataFrame({
                'BM_AUC': [None],
                'AUC': [None],
                'BM_KS': [None],
                'KS': [None],
                'Gini': [None],
                'IV': [None],
            })
        return pd.DataFrame({
            'BM_AUC': [Metrics.get_auc(y_true, y_bm_proba)],
            'AUC': [Metrics.get_auc(y_true, y_proba)],
            'BM_KS': [Metrics.get_ks(y_true, y_bm_proba)],
            'KS': [Metrics.get_ks(y_true, y_proba)],
            'Gini': [Metrics.get_gini(y_true, y_proba)],
            'IV': [Metrics.get_iv(y_true, y_proba)],
        })
        
    def generate_sample_overview_report(self, writer, res_all, label, id_col):
        df_basic_summary = res_all.groupby('sample_type').agg(
            sample_size=(id_col, 'count'),
            sample_size_pct=(id_col, lambda x: x.count() / res_all.shape[0]),
            earliest_due_date=('due_date', 'min'),
            lastest_due_date=('due_date', 'max'),
            dpd7=('dpd', lambda x: (x >= 7).mean()),
            dpd14=('dpd', lambda x: (x >= 14).mean()),
            dpd30=('dpd', lambda x: (x >= 30).mean()),
            dpd60=('dpd', lambda x: (x >= 60).mean()),
            dpd90=('dpd', lambda x: (x >= 90).mean()),
        )
        df_basic_summary['earliest_due_date'] = df_basic_summary['earliest_due_date'].astype(str)
        df_basic_summary['lastest_due_date'] = df_basic_summary['lastest_due_date'].astype(str)
        if 'score' in res_all.columns:
            df_psi = pd.DataFrame([
                Metrics.get_psi(res_all[res_all['sample_type']=="train"]['score'], res_all[res_all['sample_type']=="train"]['score']),
                Metrics.get_psi(res_all[res_all['sample_type']=="train"]['score'], res_all[res_all['sample_type']=="test"]['score']),
                Metrics.get_psi(res_all[res_all['sample_type']=="train"]['score'], res_all[res_all['sample_type']=="extra"]['score'])
            ], index=['train', 'test', 'extra'], columns=['PSI'])
            df_basic_summary = pd.concat([df_basic_summary, df_psi], axis=1)
        df_basic_summary.sort_index(ascending=False).to_excel(
            writer, sheet_name='Sample Overview - Statistics', freeze_panes=(1,1)
        )
        self._format_overview_stats_df(writer.sheets['Sample Overview - Statistics'])
        
        if 'proba' in res_all.columns and 'bm_proba' in res_all.columns:
            res_all.groupby('sample_type').apply(
                self._stat_perf, target_label=label
            ).droplevel(level=1).sort_index(ascending=False).to_excel(
                writer, sheet_name='Sample Overview - Performance', freeze_panes=(1,1)
            )
            self._format_overview_perf_df(writer.sheets['Sample Overview - Performance'])
        
    def generate_single_feature_eda_report(self, writer, X, y, woe_df, missing_values, prefix=""):
        feature_names = woe_df.reset_index()['feature_name'].unique().tolist()
        assert X.shape[1] == len(feature_names), f"The number of feature dataframe columns ({X.shape[1]}) is not match the woe_df ({len(feature_names)})!"
        feat_df = self._stat_feat(X, y, woe_df, missing_values)
        feat_df.to_excel(writer, sheet_name=f'{prefix} - Feature Overview', freeze_panes=(1,1))
        woe_df.to_excel(writer, sheet_name=f'{prefix} - Feature Binning Report', freeze_panes=(2,2))
        self._format_feat_df(writer.sheets[f'{prefix} - Feature Overview'])
        self._format_woe_df(writer.sheets[f'{prefix} - Feature Binning Report'])
        
    def generate_feature_selection_report(self, writer, original_cols, selectors):
        original_cols = pd.DataFrame(original_cols, columns=['feature']).set_index('feature')
        for name, selector in selectors.steps:
            original_cols.loc[selector.selected_features, name] = 1
            original_cols[name] = original_cols[name].fillna(0)
        original_cols.sort_values([i[0] for i in selectors.steps][::-1], ascending=False).to_excel(writer, sheet_name='Feature Selection - Overview')
        self._format_feature_selection_overview_df(writer.sheets['Feature Selection - Overview'])
        for name, selector in selectors.steps:
            if name == "Corr":
                selector.detail['after'].to_excel(writer, sheet_name=f"Feature Selection - {name}", freeze_panes=(1,1))
            else:
                selector.detail.to_excel(writer, sheet_name=f"Feature Selection - {name}", freeze_panes=(1,1))
                self._format_feature_selection_df(writer.sheets[f"Feature Selection - {name}"])
    
    def generate_model_tuning_report(self, writer, tune_results):
        if tune_results is not None:
            tune_results.to_excel(writer, sheet_name='Model Tuning', freeze_panes=(1,1))
            self._format_tuning_df(writer.sheets['Model Tuning'])
            
    def generate_calibration_report(self, writer, calibrate_detail, scorecard):
        calibrate_detail.to_excel(writer, sheet_name='Calibration - Regression', freeze_panes=(1,1))
        for name, df_scorecard in scorecard.items():
            df_scorecard.to_excel(writer, sheet_name=f'Calibration - {name} Score Card', freeze_panes=(1,1))
            self._format_calibration_scorecard_df(writer.sheets[f'Calibration - {name} Score Card'])
        self._format_calibration_reg_df(writer.sheets['Calibration - Regression'])
                
    def generate_report(self, performance, id_col, **kwargs):
        df_res = performance['df_res']
        writer = pd.ExcelWriter(self.report_path, engine='openpyxl')
        if kwargs.get("include_overview", True):
            self.generate_sample_overview_report(writer, df_res, performance['label'], id_col)
        if 'woe_df' in performance:
            for sample_type in performance['woe_df']:
                X = df_res.loc[df_res['sample_type']==sample_type, performance['original_cols']].copy()
                y = df_res.loc[df_res['sample_type']==sample_type, performance['label']].copy()
                self.generate_single_feature_eda_report(writer, X, y, performance['woe_df'][sample_type], performance['missing_values'], sample_type)
        if 'original_cols' in performance and 'feature_selection' in performance:
            self.generate_feature_selection_report(writer, performance['original_cols'], performance['feature_selection'])
        if 'tune_results' in performance:
            self.generate_model_tuning_report(writer, performance['tune_results'])
        if 'calibrate_detail' in performance and 'scorecard' in performance:
            self.generate_calibration_report(writer, performance['calibrate_detail'], performance['scorecard'])
        writer.close()
            