import os
import pyspark.sql.functions as F
from myproject.analysis.utils.utils_datasets import Utils
from transforms.api import transform_df, Input, Output, configure, transform
from myproject.analysis.utils.utils_feature_extraction import group_by_correlation, compress_with_pls_per_vehicle
from myproject.analysis.utils.utils_correlation import calc_corr_per_vehicle

dir_save = r"feature_extraction/improved_correlation_filter_with_pls"

colin_lim = 0.9
corr_lim = 0.8
target = 'speicher_soh_%'


################# SPEARMAN ##################
@transform_df(
    Output(os.path.join(dir_save, "feat_corr_filt_spearman_with_pls")),
    df_field_corr=Input("rid_input_field_correlation"),
    df_labor_corr=Input("rid_input_lab_correlation"),
    df_source=Input("data_field_lab")
)
def imprvd_correlation_filt_spearman_with_pls_ezlhist(df_field_corr, df_labor_corr, df_source):
    corr_pdf_target = df_labor_corr.toPandas()
    corr_pdf_features = df_field_corr.toPandas()

    corr_dict = group_by_correlation(corr_pdf_features, corr_pdf_target, corr_lim, colin_lim, target)

    df = compress_with_pls_per_vehicle(df_source, corr_dict, target)
    return df


@transform_df(
    Output(os.path.join(dir_save, "corr_feat_corr_filt_spearman_with_pls")),
    source_df=Input(os.path.join(dir_save, "feat_corr_filt_spearman_with_pls")))
def compute_spearman_spearman_filt_ezlhist(source_df):
    col_names = Utils.get_numeric_cols_aging(source_df)
    corr_df = calc_corr_per_vehicle(source_df, col_names, method='spearman', filt_nan=True)
    return corr_df


@transform_df(
    Output(os.path.join(dir_save, "feat_corr_filt_spearman_with_pls_pvfit")),
    df_field_corr=Input("rid_input_field_correlation_pvfit"),
    df_labor_corr=Input("rid_input_lab_correlation_pvfit"),
    df_source=Input("data_field_lab")
)
def imprvd_correlation_filt_spearman_with_pls_pvfit(df_field_corr, df_labor_corr, df_source):
    corr_pdf_target = df_labor_corr.toPandas()
    corr_pdf_features = df_field_corr.toPandas()

    corr_dict = group_by_correlation(corr_pdf_features, corr_pdf_target, corr_lim, colin_lim, target)

    df = compress_with_pls_per_vehicle(df_source, corr_dict, target)
    return df


@transform_df(
    Output(os.path.join(dir_save, "corr_feat_corr_filt_spearman_with_pls_pvfit")),
    source_df=Input(os.path.join(dir_save, "feat_corr_filt_spearman_with_pls_pvfit")))
def compute_spearman_spearman_filt_pvfit(source_df):
    col_names = Utils.get_numeric_cols_aging(source_df)
    corr_df = calc_corr_per_vehicle(source_df, col_names, method='spearman', filt_nan=True)
    return corr_df

